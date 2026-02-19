"""
Dashboard Data Service - UPDATED WITH NEW DISAGGREGATION LOGIC.
Aggregates sales, forecast for UI dashboards.

CHANGES:
- New method _calculate_table_ratios to get ratios from the same table
"""

import logging
from typing import Dict, Any, List, Optional

from psycopg2 import sql

from app.core.database import get_db_manager
from app.core.exceptions import AppException, DatabaseException, ValidationException, NotFoundException
from app.core.sales_data_service import SalesDataService
from app.schemas.sales_data import SalesDataQueryRequest, AggregatedDataQueryRequest

logger = logging.getLogger(__name__)


class DashboardService:
    """Service for aggregating sales, forecast, and final plan data."""

    @staticmethod
    def _table_exists(cursor, table_name: str) -> bool:
        cursor.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = %s
            )
            """,
            (table_name,),
        )
        return bool(cursor.fetchone()[0])

    @staticmethod
    def get_all_data_export(
        database_name: str,
        request: SalesDataQueryRequest,
    ) -> Dict[str, Any]:
        """
        Retrieve all master data with related sales, forecast, and final plan
        records for the provided date range (no pagination).
        
        UPDATED: Now dynamically fetches all tenant's custom tables instead of hardcoding product_manager.
        """
        db_manager = get_db_manager()
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()

                date_field, target_field = SalesDataService._get_field_names(cursor)

                start_date = request.from_date if request.from_date else "2025-01-01"
                end_date = request.to_date if request.to_date else "2026-03-31"

                master_where = "WHERE 1=1"
                master_params: List[Any] = []
                if request.filters:
                    for f in request.filters:
                        if f.values:
                            placeholders = ",".join(["%s"] * len(f.values))
                            master_where += (
                                f' AND "{f.field_name}" IN ({placeholders})'
                            )
                            master_params.extend(f.values)

                count_query = (
                    f"SELECT COUNT(*) FROM public.master_data {master_where}"
                )
                cursor.execute(count_query, master_params)
                total_master_count = cursor.fetchone()[0]

                order_by = "master_id ASC"
                try:
                    cursor.execute(
                        """
                        SELECT column_name FROM information_schema.columns
                        WHERE table_name = 'master_data'
                        AND column_name IN ('product', 'customer')
                        """
                    )
                    existing_cols = [r[0] for r in cursor.fetchall()]
                    if "product" in existing_cols and "customer" in existing_cols:
                        order_by = "product, customer ASC"
                    elif "product" in existing_cols:
                        order_by = "product ASC"
                except Exception:
                    pass

                master_id_query = f"""
                    SELECT master_id
                    FROM public.master_data
                    {master_where}
                    ORDER BY {order_by}
                """
                cursor.execute(master_id_query, master_params)
                master_ids = [row[0] for row in cursor.fetchall()]

                if not master_ids:
                    return {"records": [], "total_count": total_master_count}

                placeholders = ",".join(["%s"] * len(master_ids))

                cursor.execute(
                    f"""
                    SELECT * FROM public.master_data
                    WHERE master_id IN ({placeholders})
                    """,
                    master_ids,
                )
                col_names = [desc[0] for desc in cursor.description]
                exclude_fields = {
                    "master_id",
                    "uom",
                    "tenant_id",
                    "created_at",
                    "created_by",
                    "updated_at",
                    "updated_by",
                    "deleted_at",
                }
                master_id_idx = col_names.index("master_id")
                master_data_map = {}
                for row in cursor.fetchall():
                    m_id = row[master_id_idx]
                    master_data_map[m_id] = {
                        col_names[i]: row[i]
                        for i in range(len(col_names))
                        if col_names[i] not in exclude_fields
                    }

                # ========================================================================
                # CHANGED: Get dynamic tables from metadata instead of hardcoding
                # ========================================================================
                from app.core.dynamic_table_service import DynamicTableService
                
                dynamic_tables = DynamicTableService.get_tenant_dynamic_tables(
                    database_name=database_name
                )

                # Build results structure dynamically
                results = []
                results_map = {}
                for master_id in master_ids:
                    entry = {
                        "master_data": master_data_map.get(master_id, {}),
                        "sales_data": [],
                        "forecast_data": [],
                    }
                    
                    # Add all dynamic tables to the structure
                    for table in dynamic_tables:
                        entry[table['table_name']] = []
                    
                    results.append(entry)
                    results_map[master_id] = entry

                def add_table_records(
                    table_name: str,
                    id_column: str,
                    target_key: str,
                    id_key: str,
                ) -> None:
                    if not DashboardService._table_exists(cursor, table_name):
                        return
                    query = f"""
                        SELECT {id_column}, master_id, "{date_field}",
                            "{target_field}", uom
                        FROM public.{table_name}
                        WHERE master_id IN ({placeholders})
                        AND "{date_field}" BETWEEN %s AND %s
                        ORDER BY master_id, "{date_field}" ASC
                    """
                    cursor.execute(query, master_ids + [start_date, end_date])
                    for row in cursor.fetchall():
                        record_id, master_id, data_date, quantity, uom = row
                        results_map[master_id][target_key].append(
                            {
                                id_key: str(record_id),
                                "date": str(data_date),
                                "UOM": uom,
                                "Quantity": float(quantity)
                                if quantity is not None
                                else 0.0,
                            }
                        )

                # Add sales and forecast data (core tables)
                add_table_records(
                    table_name="sales_data",
                    id_column="sales_id",
                    target_key="sales_data",
                    id_key="sales_id",
                )
                add_table_records(
                    table_name="forecast_data",
                    id_column="forecast_data_id",
                    target_key="forecast_data",
                    id_key="forecast_data_id",
                )
                
                # ========================================================================
                # CHANGED: Dynamically add all tenant's planning tables
                # ========================================================================
                for table in dynamic_tables:
                    table_name = table['table_name']
                    id_column = f"{table_name}_id"
                    add_table_records(
                        table_name=table_name,
                        id_column=id_column,
                        target_key=table_name,
                        id_key=id_column,
                    )

                return {"records": results, "total_count": total_master_count}

        except Exception as e:
            logger.error(f"Error in get_all_data_export: {str(e)}")
            raise DatabaseException(
                f"Failed to retrieve dashboard data: {str(e)}"
            )

    @staticmethod
    def get_all_data_ui(
        database_name: str,
        request: SalesDataQueryRequest,
    ) -> Dict[str, Any]:
        """
        Retrieve master data with paginated filters and fetch sales/forecast/final plan
        records for the provided date range.
        
        UPDATED: Now dynamically fetches all tenant's custom tables instead of hardcoding product_manager.
        """
        db_manager = get_db_manager()
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()

                date_field, target_field = SalesDataService._get_field_names(cursor)

                start_date = request.from_date if request.from_date else "2025-01-01"
                end_date = request.to_date if request.to_date else "2026-03-31"

                master_where = "WHERE 1=1"
                master_params: List[Any] = []
                if request.filters:
                    for f in request.filters:
                        if f.values:
                            placeholders = ",".join(["%s"] * len(f.values))
                            master_where += (
                                f' AND "{f.field_name}" IN ({placeholders})'
                            )
                            master_params.extend(f.values)

                count_query = (
                    f"SELECT COUNT(*) FROM public.master_data {master_where}"
                )
                cursor.execute(count_query, master_params)
                total_master_count = cursor.fetchone()[0]

                offset = (request.page - 1) * request.page_size
                order_by = "master_id ASC"
                try:
                    cursor.execute(
                        """
                        SELECT column_name FROM information_schema.columns
                        WHERE table_name = 'master_data'
                        AND column_name IN ('product', 'customer')
                        """
                    )
                    existing_cols = [r[0] for r in cursor.fetchall()]
                    if "product" in existing_cols and "customer" in existing_cols:
                        order_by = "product, customer ASC"
                    elif "product" in existing_cols:
                        order_by = "product ASC"
                except Exception:
                    pass

                master_id_query = f"""
                    SELECT master_id
                    FROM public.master_data
                    {master_where}
                    ORDER BY {order_by}
                    LIMIT %s OFFSET %s
                """
                cursor.execute(
                    master_id_query,
                    master_params + [request.page_size, offset],
                )
                master_ids = [row[0] for row in cursor.fetchall()]

                if not master_ids:
                    return {"records": [], "total_count": total_master_count}

                placeholders = ",".join(["%s"] * len(master_ids))

                cursor.execute(
                    f"""
                    SELECT * FROM public.master_data
                    WHERE master_id IN ({placeholders})
                    """,
                    master_ids,
                )
                col_names = [desc[0] for desc in cursor.description]
                exclude_fields = {
                    "master_id",
                    "uom",
                    "tenant_id",
                    "created_at",
                    "created_by",
                    "updated_at",
                    "updated_by",
                    "deleted_at",
                }
                master_id_idx = col_names.index("master_id")
                master_data_map = {}
                for row in cursor.fetchall():
                    m_id = row[master_id_idx]
                    master_data_map[m_id] = {
                        col_names[i]: row[i]
                        for i in range(len(col_names))
                        if col_names[i] not in exclude_fields
                    }

                # ========================================================================
                # CHANGED: Get dynamic tables from metadata instead of hardcoding
                # ========================================================================
                from app.core.dynamic_table_service import DynamicTableService
                
                dynamic_tables = DynamicTableService.get_tenant_dynamic_tables(
                    database_name=database_name
                )

                # Build results structure dynamically
                results = []
                results_map = {}
                for master_id in master_ids:
                    entry = {
                        "master_data": master_data_map.get(master_id, {}),
                        "sales_data": [],
                        "forecast_data": [],
                    }
                    
                    # Add all dynamic tables to the structure
                    for table in dynamic_tables:
                        entry[table['table_name']] = []
                    
                    results.append(entry)
                    results_map[master_id] = entry

                def add_table_records(
                    table_name: str,
                    id_column: str,
                    target_key: str,
                    id_key: str,
                ) -> None:
                    if not DashboardService._table_exists(cursor, table_name):
                        return
                    query = f"""
                        SELECT {id_column}, master_id, "{date_field}",
                            "{target_field}", uom
                        FROM public.{table_name}
                        WHERE master_id IN ({placeholders})
                        AND "{date_field}" BETWEEN %s AND %s
                        ORDER BY master_id, "{date_field}" ASC
                    """
                    cursor.execute(query, master_ids + [start_date, end_date])
                    for row in cursor.fetchall():
                        record_id, master_id, data_date, quantity, uom = row
                        results_map[master_id][target_key].append(
                            {
                                id_key: str(record_id),
                                "date": str(data_date),
                                "UOM": uom,
                                "Quantity": float(quantity)
                                if quantity is not None
                                else 0.0,
                            }
                        )

                # Add sales and forecast data (core tables)
                add_table_records(
                    table_name="sales_data",
                    id_column="sales_id",
                    target_key="sales_data",
                    id_key="sales_id",
                )
                add_table_records(
                    table_name="forecast_data",
                    id_column="forecast_data_id",
                    target_key="forecast_data",
                    id_key="forecast_data_id",
                )
                
                # ========================================================================
                # CHANGED: Dynamically add all tenant's planning tables
                # ========================================================================
                for table in dynamic_tables:
                    table_name = table['table_name']
                    id_column = f"{table_name}_id"
                    add_table_records(
                        table_name=table_name,
                        id_column=id_column,
                        target_key=table_name,
                        id_key=id_column,
                    )

                return {"records": results, "total_count": total_master_count}

        except Exception as e:
            logger.error(f"Error in get_all_data_ui: {str(e)}")
            raise DatabaseException(
                f"Failed to retrieve dashboard data: {str(e)}"
            )

    @staticmethod
    def get_aggregated_data_ui(
        database_name: str,
        request: AggregatedDataQueryRequest,
    ) -> Dict[str, Any]:
        """
        Retrieve aggregated master data based on aggregated_fields and fetch
        aggregated sales/forecast/final plan records.
        
        UPDATED: Now dynamically fetches all tenant's custom tables instead of hardcoding product_manager.
        """
        db_manager = get_db_manager()
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()

                date_field, target_field = SalesDataService._get_field_names(cursor)

                start_date = request.from_date if request.from_date else "2025-01-01"
                end_date = request.to_date if request.to_date else "2026-03-31"

                # Build where clause for master data filters
                master_where = "WHERE 1=1"
                master_params: List[Any] = []
                if request.filters:
                    for f in request.filters:
                        if f.values:
                            placeholders = ",".join(["%s"] * len(f.values))
                            master_where += (
                                f' AND "{f.field_name}" IN ({placeholders})'
                            )
                            master_params.extend(f.values)

                # Identify aggregated fields
                agg_fields = request.aggregated_fields
                if not agg_fields:
                    raise ValidationException("aggregated_fields are required")

                agg_fields_str = ",".join([f'"{f}"' for f in agg_fields])

                # Count unique combinations of aggregated fields
                count_query = f"""
                    SELECT COUNT(*) FROM (
                        SELECT {agg_fields_str}
                        FROM public.master_data
                        {master_where}
                        GROUP BY {agg_fields_str}
                    ) as agg_groups
                """
                cursor.execute(count_query, master_params)
                total_groups_count = cursor.fetchone()[0]

                # Paginate groups
                offset = (request.page - 1) * request.page_size
                groups_query = f"""
                    SELECT {agg_fields_str}
                    FROM public.master_data
                    {master_where}
                    GROUP BY {agg_fields_str}
                    ORDER BY {agg_fields_str}
                    LIMIT %s OFFSET %s
                """
                cursor.execute(
                    groups_query,
                    master_params + [request.page_size, offset],
                )
                groups = cursor.fetchall()

                if not groups:
                    return {"records": [], "total_count": total_groups_count}

                # ========================================================================
                # CHANGED: Get dynamic tables from metadata instead of hardcoding
                # ========================================================================
                from app.core.dynamic_table_service import DynamicTableService
                
                dynamic_tables = DynamicTableService.get_tenant_dynamic_tables(
                    database_name=database_name
                )

                results = []
                for group in groups:
                    group_data = dict(zip(agg_fields, group))
                    
                    # Find all master_ids for this group
                    group_conditions = []
                    group_vals = []
                    for f, val in group_data.items():
                        if val is None:
                            group_conditions.append(f'"{f}" IS NULL')
                        else:
                            group_conditions.append(f'"{f}" = %s')
                            group_vals.append(val)
                    
                    group_where = " AND ".join(group_conditions)
                    master_id_query = f"""
                        SELECT master_id
                        FROM public.master_data
                        {master_where} AND {group_where}
                    """
                    cursor.execute(master_id_query, master_params + group_vals)
                    master_ids = [row[0] for row in cursor.fetchall()]

                    if not master_ids:
                        continue

                    placeholders = ",".join(["%s"] * len(master_ids))
                    
                    entry = {
                        "master_data": group_data,
                        "sales_data": [],
                        "forecast_data": [],
                    }
                    
                    # ========================================================================
                    # CHANGED: Dynamically add all tenant's planning tables
                    # ========================================================================
                    for table in dynamic_tables:
                        entry[table['table_name']] = []

                    def add_aggregated_table_records(
                        table_name: str,
                        target_key: str,
                    ) -> None:
                        if not DashboardService._table_exists(cursor, table_name):
                            return
                        query = f"""
                            SELECT "{date_field}", SUM(CAST("{target_field}" AS DOUBLE PRECISION)), MAX(uom)
                            FROM public.{table_name}
                            WHERE master_id IN ({placeholders})
                            AND "{date_field}" BETWEEN %s AND %s
                            GROUP BY "{date_field}"
                            ORDER BY "{date_field}" ASC
                        """
                        cursor.execute(query, master_ids + [start_date, end_date])
                        for row in cursor.fetchall():
                            data_date, quantity, uom = row
                            entry[target_key].append(
                                {
                                    "date": str(data_date),
                                    "UOM": uom,
                                    "Quantity": float(quantity)
                                    if quantity is not None
                                    else 0.0,
                                }
                            )

                    # Add sales and forecast data (core tables)
                    add_aggregated_table_records("sales_data", "sales_data")
                    add_aggregated_table_records("forecast_data", "forecast_data")
                    
                    # Add all tenant planning tables (This includes final_plan and custom ones)
                    for table in dynamic_tables:
                        add_aggregated_table_records(table['table_name'], table['table_name'])
                    
                    results.append(entry)

                return {"records": results, "total_count": total_groups_count}

        except AppException as e:
            raise e
        except Exception as e:
            logger.error(f"Error in get_aggregated_data_ui: {str(e)}")
            raise DatabaseException(
                f"Failed to retrieve aggregated dashboard data: {str(e)}"
            )

    @staticmethod
    def _resolve_master_id(cursor, master_data: Dict[str, Any]) -> str:
        if not master_data:
            raise ValidationException("master_data is required")
        from psycopg2 import sql
        clauses = []
        params = []
        for key, value in master_data.items():
            clauses.append(sql.SQL("{} = %s").format(sql.Identifier(key)))
            params.append(value)
        query = sql.SQL("SELECT master_id FROM master_data WHERE {} LIMIT 1").format(
            sql.SQL(" AND ").join(clauses)
        )
        cursor.execute(query, params)
        row = cursor.fetchone()
        if not row:
            raise NotFoundException("Master data", str(master_data))
        return row[0]

    @staticmethod
    def _resolve_sales_info(cursor, master_id: str) -> tuple[str, Any]:
        cursor.execute(
            """
            SELECT uom, unit_price
            FROM sales_data
            WHERE master_id = %s
            ORDER BY created_at DESC LIMIT 1
            """,
            (master_id,),
        )
        sales_row = cursor.fetchone()
        uom = sales_row[0] if sales_row else "UNIT"
        unit_price = sales_row[1] if sales_row else 0
        return uom, unit_price
