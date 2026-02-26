"""
Table-level disaggregation service.
File: app/core/table_disaggregation_service.py

Supports three input modes:
  1. periods    – explicit list of {date, quantity} from frontend
  2. single     – single date + quantity
  3. range      – date_from + date_to, reads totals from DB per period
"""

import uuid
import logging
from typing import Dict, Any, List, Optional, Tuple

from psycopg2 import sql

from app.core.database import get_db_manager
from app.core.exceptions import (
    DatabaseException,
    ValidationException,
    NotFoundException,
)

logger = logging.getLogger(__name__)


class TableDisaggregationService:
    """
    Disaggregates aggregated quantities into lower-level member records
    using one of three strategies: own_ratio, key_figure, equal.
    """

    # ──────────────────────────────────────────────────────────────────────
    # Public entry-point
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def disaggregate_table_data(
        database_name: str,
        request,        # DisaggregateTableRequest
        user_email: str,
    ) -> Dict[str, Any]:
        """
        Disaggregate one or more periods into lower-level member rows
        and upsert them into the target table.

        Mode is auto-detected from request fields:
          periods    → explicit {date, quantity} list from frontend
          date       → single cell
          date_from  → range, reads totals from DB
        """
        db_manager = get_db_manager()

        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # ── 1. Resolve dynamic field names ─────────────────────
                    date_field, target_field = TableDisaggregationService._get_field_names(cursor)

                    # ── 2. Validate tables ─────────────────────────────────
                    TableDisaggregationService._assert_table_exists(cursor, request.target_table)
                    if request.method == "key_figure":
                        TableDisaggregationService._assert_table_exists(cursor, request.key_figure_table)

                    # ── 3. Resolve member master_ids ───────────────────────
                    member_ids = TableDisaggregationService._resolve_member_ids(
                        cursor, request.master_data
                    )
                    if not member_ids:
                        raise NotFoundException("Master data members", str(request.master_data))

                    # ── 4. Calculate ratios once (shared across all periods)
                    ratios = TableDisaggregationService._calculate_ratios(
                        cursor=cursor,
                        method=request.method,
                        member_ids=member_ids,
                        target_table=request.target_table,
                        key_figure_table=request.key_figure_table,
                        target_field=target_field,
                        date_field=date_field,
                        ratio_date_from=request.ratio_date_from,
                        ratio_date_to=request.ratio_date_to,
                    )

                    # ── 5. Resolve periods list ────────────────────────────
                    periods = TableDisaggregationService._resolve_periods(
                        cursor=cursor,
                        request=request,
                        member_ids=member_ids,
                        target_field=target_field,
                        date_field=date_field,
                    )

                    if not periods:
                        return {
                            "status": "success",
                            "method": request.method,
                            "target_table": request.target_table,
                            "mode": TableDisaggregationService._detect_mode(request),
                            "periods_processed": 0,
                            "members_count": len(member_ids),
                            "records_upserted": 0,
                            "message": "No data found for the given date range.",
                            "periods": [],
                        }

                    # ── 6. Cache table structure checks once ───────────────
                    has_audit = TableDisaggregationService._has_update_audit_columns(
                        cursor, request.target_table
                    )
                    has_type = TableDisaggregationService._has_type_column(
                        cursor, request.target_table
                    )
                    # Cache UOM/unit_price per master_id across all periods
                    sales_info_cache: Dict[str, Tuple] = {}

                    # ── 7. Disaggregate + upsert each period ───────────────
                    total_upserted = 0
                    period_results = []

                    for plan_date, period_qty in periods:
                        distributed = TableDisaggregationService._apply_ratios(
                            ratios=ratios,
                            total_quantity=period_qty,
                        )
                        upserted = TableDisaggregationService._upsert_members(
                            cursor=cursor,
                            target_table=request.target_table,
                            distributed=distributed,
                            plan_date=plan_date,
                            date_field=date_field,
                            target_field=target_field,
                            user_email=user_email,
                            has_audit=has_audit,
                            has_type=has_type,
                            sales_info_cache=sales_info_cache,
                        )
                        total_upserted += upserted
                        period_results.append({
                            "date": str(plan_date),
                            "total_quantity": period_qty,
                            "records_upserted": upserted,
                            "distribution": [
                                {
                                    "master_id": str(mid),
                                    "ratio": round(ratio, 6),
                                    "quantity": round(qty, 4),
                                }
                                for mid, ratio, qty in distributed
                            ],
                        })

                    conn.commit()

                    mode = TableDisaggregationService._detect_mode(request)
                    return {
                        "status": "success",
                        "method": request.method,
                        "target_table": request.target_table,
                        "mode": mode,
                        "periods_processed": len(periods),
                        "members_count": len(member_ids),
                        "records_upserted": total_upserted,
                        "periods": period_results,
                    }

                finally:
                    cursor.close()

        except (ValidationException, NotFoundException):
            raise
        except Exception as e:
            logger.error(f"disaggregate_table_data failed: {e}", exc_info=True)
            raise DatabaseException(f"Failed to disaggregate table data: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Mode detection
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_mode(request) -> str:
        if request.periods is not None:
            return "periods"
        if request.date is not None:
            return "single"
        return "range"

    # ──────────────────────────────────────────────────────────────────────
    # Period resolution — handles all 3 input modes
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_periods(
        cursor,
        request,
        member_ids: List[str],
        target_field: str,
        date_field: str,
    ) -> List[Tuple[Any, float]]:
        """
        Returns [(date, quantity), ...] based on the active input mode.

        Mode 1 – periods array  : directly convert to (date, qty) tuples
        Mode 2 – single date    : return [(request.date, request.quantity)]
        Mode 3 – date range     : query DB for existing period totals
        """
        # Mode 1 — explicit periods from frontend
        if request.periods is not None:
            return [(p.date, p.quantity) for p in request.periods]

        # Mode 2 — single date
        if request.date is not None:
            return [(request.date, request.quantity if request.quantity is not None else 0.0)]

        # Mode 3 — range: read existing totals from source table
        source_table = (
            request.key_figure_table
            if request.method == "key_figure"
            else request.target_table
        )
        return TableDisaggregationService._fetch_period_totals(
            cursor=cursor,
            table_name=source_table,
            member_ids=member_ids,
            target_field=target_field,
            date_field=date_field,
            date_from=request.date_from,
            date_to=request.date_to,
        )

    @staticmethod
    def _fetch_period_totals(
        cursor,
        table_name: str,
        member_ids: List[str],
        target_field: str,
        date_field: str,
        date_from,
        date_to,
    ) -> List[Tuple[Any, float]]:
        """
        Read distinct dates + group totals from table within the date range.
        Used only for range mode.
        """
        placeholders = ",".join(["%s"] * len(member_ids))
        params: List[Any] = list(member_ids) + [date_from, date_to]

        query = f"""
            SELECT "{date_field}",
                   SUM(CAST("{target_field}" AS DOUBLE PRECISION)) AS total_qty
            FROM {table_name}
            WHERE master_id IN ({placeholders})
              AND "{date_field}" BETWEEN %s AND %s
            GROUP BY "{date_field}"
            ORDER BY "{date_field}" ASC
        """
        cursor.execute(query, params)
        return [
            (row[0], float(row[1]) if row[1] is not None else 0.0)
            for row in cursor.fetchall()
        ]

    # ──────────────────────────────────────────────────────────────────────
    # Ratio calculation
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _calculate_ratios(
        cursor,
        method: str,
        member_ids: List[str],
        target_table: str,
        key_figure_table: Optional[str],
        target_field: str,
        date_field: str,
        ratio_date_from,
        ratio_date_to,
    ) -> Dict[str, float]:
        """
        Returns {master_id: ratio} summing to 1.0.
        Falls back to equal distribution if source table has no data.
        """
        if method == "equal":
            return TableDisaggregationService._equal_ratios(member_ids)

        ratio_table = target_table if method == "own_ratio" else key_figure_table

        raw = TableDisaggregationService._fetch_member_totals(
            cursor=cursor,
            table_name=ratio_table,
            member_ids=member_ids,
            target_field=target_field,
            date_field=date_field,
            date_from=ratio_date_from,
            date_to=ratio_date_to,
        )

        total = sum(raw.values())
        if not raw or total == 0:
            logger.warning(
                f"No data in '{ratio_table}' for ratio calculation "
                f"(method={method}). Falling back to equal distribution."
            )
            return TableDisaggregationService._equal_ratios(member_ids)

        return {mid: v / total for mid, v in raw.items()}

    @staticmethod
    def _equal_ratios(member_ids: List[str]) -> Dict[str, float]:
        ratio = 1.0 / len(member_ids)
        return {mid: ratio for mid in member_ids}

    @staticmethod
    def _fetch_member_totals(
        cursor,
        table_name: str,
        member_ids: List[str],
        target_field: str,
        date_field: str,
        date_from,
        date_to,
    ) -> Dict[str, float]:
        placeholders = ",".join(["%s"] * len(member_ids))
        params: List[Any] = list(member_ids)

        date_clause = ""
        if date_from:
            date_clause += f' AND "{date_field}" >= %s'
            params.append(date_from)
        if date_to:
            date_clause += f' AND "{date_field}" <= %s'
            params.append(date_to)

        query = f"""
            SELECT master_id, SUM(CAST("{target_field}" AS DOUBLE PRECISION))
            FROM {table_name}
            WHERE master_id IN ({placeholders})
            {date_clause}
            GROUP BY master_id
        """
        cursor.execute(query, params)

        result: Dict[str, float] = {}
        for master_id, total in cursor.fetchall():
            if total is not None and total > 0:
                result[str(master_id)] = float(total)

        for mid in member_ids:
            if mid not in result:
                result[mid] = 0.0

        return result

    # ──────────────────────────────────────────────────────────────────────
    # Apply ratios
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_ratios(
        ratios: Dict[str, float],
        total_quantity: float,
    ) -> List[Tuple[str, float, float]]:
        """
        Returns [(master_id, ratio, distributed_qty), ...].
        Rounding drift absorbed by the highest-ratio member.
        Zero quantity is fully supported.
        """
        items = [(mid, r, round(r * total_quantity, 4)) for mid, r in ratios.items()]

        if total_quantity != 0 and items:
            distributed_sum = sum(q for _, _, q in items)
            drift = round(total_quantity - distributed_sum, 4)
            if drift != 0:
                max_idx = max(range(len(items)), key=lambda i: items[i][1])
                mid, r, q = items[max_idx]
                items[max_idx] = (mid, r, round(q + drift, 4))

        return items

    # ──────────────────────────────────────────────────────────────────────
    # Upsert
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _upsert_members(
        cursor,
        target_table: str,
        distributed: List[Tuple[str, float, float]],
        plan_date,
        date_field: str,
        target_field: str,
        user_email: str,
        has_audit: bool,
        has_type: bool,
        sales_info_cache: Dict[str, Tuple],
    ) -> int:
        id_column = f"{target_table}_id"
        upserted = 0

        for master_id, _ratio, quantity in distributed:
            if master_id not in sales_info_cache:
                sales_info_cache[master_id] = (
                    TableDisaggregationService._resolve_sales_info(cursor, master_id)
                )
            uom, unit_price = sales_info_cache[master_id]
            record_id = str(uuid.uuid4())

            if has_type and has_audit:
                cursor.execute(
                    f"""
                    INSERT INTO {target_table}
                        ({id_column}, master_id, "{date_field}", "{target_field}",
                         uom, unit_price, type, created_by)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (master_id, "{date_field}")
                    DO UPDATE SET
                        "{target_field}" = EXCLUDED."{target_field}",
                        uom              = EXCLUDED.uom,
                        unit_price       = EXCLUDED.unit_price,
                        type             = EXCLUDED.type,
                        updated_at       = CURRENT_TIMESTAMP,
                        updated_by       = %s
                    """,
                    (record_id, master_id, plan_date, quantity,
                     uom, unit_price, "disaggregated", user_email, user_email),
                )
            elif has_audit:
                cursor.execute(
                    f"""
                    INSERT INTO {target_table}
                        ({id_column}, master_id, "{date_field}", "{target_field}",
                         uom, unit_price, created_by)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (master_id, "{date_field}")
                    DO UPDATE SET
                        "{target_field}" = EXCLUDED."{target_field}",
                        uom              = EXCLUDED.uom,
                        unit_price       = EXCLUDED.unit_price,
                        updated_at       = CURRENT_TIMESTAMP,
                        updated_by       = %s
                    """,
                    (record_id, master_id, plan_date, quantity,
                     uom, unit_price, user_email, user_email),
                )
            else:
                # Immutable tables like forecast_data
                cursor.execute(
                    f"""
                    INSERT INTO {target_table}
                        ({id_column}, master_id, "{date_field}", "{target_field}",
                         uom, unit_price, created_by)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (master_id, "{date_field}")
                    DO UPDATE SET
                        "{target_field}" = EXCLUDED."{target_field}",
                        uom              = EXCLUDED.uom,
                        unit_price       = EXCLUDED.unit_price
                    """,
                    (record_id, master_id, plan_date, quantity,
                     uom, unit_price, user_email),
                )
            upserted += 1

        return upserted

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_field_names(cursor) -> Tuple[str, str]:
        cursor.execute(
            "SELECT date_field_name, target_field_name FROM field_catalogue_metadata LIMIT 1"
        )
        row = cursor.fetchone()
        if not row:
            raise ValidationException(
                "Field catalogue not finalized. Please finalize your field catalogue first."
            )
        return row[0], row[1]   # date_field, target_field

    @staticmethod
    def _assert_table_exists(cursor, table_name: str) -> None:
        cursor.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = %s
            )
            """,
            (table_name,),
        )
        if not bool(cursor.fetchone()[0]):
            raise NotFoundException("Table", table_name)

    @staticmethod
    def _resolve_member_ids(cursor, master_data: Dict[str, Any]) -> List[str]:
        if not master_data:
            raise ValidationException("master_data must not be empty")
        clauses = []
        params = []
        for key, value in master_data.items():
            clauses.append(sql.SQL("{} = %s").format(sql.Identifier(key)))
            params.append(value)
        query = sql.SQL(
            "SELECT master_id FROM master_data WHERE {}"
        ).format(sql.SQL(" AND ").join(clauses))
        cursor.execute(query, params)
        return [str(row[0]) for row in cursor.fetchall()]

    @staticmethod
    def _has_update_audit_columns(cursor, table_name: str) -> bool:
        cursor.execute(
            """
            SELECT COUNT(*) FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name   = %s
              AND column_name IN ('updated_at', 'updated_by')
            """,
            (table_name,),
        )
        return cursor.fetchone()[0] == 2

    @staticmethod
    def _has_type_column(cursor, table_name: str) -> bool:
        cursor.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name   = %s
                  AND column_name  = 'type'
            )
            """,
            (table_name,),
        )
        return bool(cursor.fetchone()[0])

    @staticmethod
    def _resolve_sales_info(cursor, master_id: str) -> Tuple:
        cursor.execute(
            """
            SELECT uom, unit_price FROM sales_data
            WHERE master_id = %s
            ORDER BY created_at DESC LIMIT 1
            """,
            (master_id,),
        )
        row = cursor.fetchone()
        return (row[0] if row else "UNIT", row[1] if row else 0.0)