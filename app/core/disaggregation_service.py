"""
Disaggregation service.
Handles forecast disaggregation, data breakdown to granular levels, and ratio calculations.

CHANGES:
- disaggregate_data: now dynamically loads all tenant planning tables from dynamic_tables
  metadata instead of hardcoding final_plan, product_manager. Each dynamic table gets its
  own ratio calculation from its own data (same logic as before for final_plan/product_manager).
- disaggregate_forecast: still writes to final_plan (mandatory table) - unchanged in intent
  but now validates final_plan is registered in dynamic_tables.
- _upsert_disaggregated_data: now accepts any dynamic table name and builds the upsert
  generically instead of branching on hardcoded table names.
- _calculate_disaggregation_ratios: unchanged - already accepts source_table param.
- _build_ratios_dict: unchanged.
- UOM is resolved per master_id throughout.
"""

import uuid
import json
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple
from psycopg2 import sql

from app.core.database import get_db_manager
from app.core.exceptions import (
    DatabaseException,
    ValidationException,
    NotFoundException
)
from app.schemas.forecasting import (
    DisaggregationRequest,
    DisaggregateDataRequest
)

logger = logging.getLogger(__name__)


class DisaggregationService:
    """Service for forecast disaggregation operations."""

    @staticmethod
    def disaggregate_forecast(
        tenant_id: str,
        database_name: str,
        request: DisaggregationRequest,
        user_email: str
    ) -> Dict[str, Any]:
        """
        Disaggregate an existing forecast run to a more granular level.
        Writes disaggregated results into final_plan (the mandatory planning table).
        """
        from app.core.forecasting_service import ForecastingService
        
        # 1. Get original forecast run to understand source level
        forecast_run = ForecastingService.get_forecast_run(
            tenant_id, database_name, request.forecast_run_id
        )

        source_filters = forecast_run.get('forecast_filters', {})
        source_level = source_filters.get('aggregation_level', 'product')
        source_interval = source_filters.get('interval', 'MONTHLY')

        # Convert filters from list of SalesDataFilter to dict for internal processing
        filters_dict = {}
        if request.filters:
            for filter_item in request.filters:
                filters_dict[filter_item.field_name] = filter_item.values

        # 2. Calculate historical ratios using the same interval as the source forecast
        from app.core.aggregation_service import AggregationService
        
        ratios_df = AggregationService.calculate_historical_ratios(
            tenant_id=tenant_id,
            database_name=database_name,
            source_aggregation_level=source_level,
            target_aggregation_level=request.target_aggregation_level,
            interval=source_interval,
            history_start=request.history_start,
            history_end=request.history_end,
            filters=filters_dict
        )
        
        if ratios_df.empty:
            raise ValidationException("No historical data found to calculate disaggregation ratios")

        db_manager = get_db_manager()
        disaggregated_count = 0
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Get date_field and target_field from field catalogue
                    from app.core.sales_data_service import SalesDataService
                    date_field, target_field = SalesDataService._get_field_names(cursor)
                    
                    # 3. Get best algorithm results for this run
                    cursor.execute("""
                        SELECT fam.mapping_id, fam.algorithm_id
                        FROM forecast_algorithms_mapping fam
                        LEFT JOIN forecast_results fr ON fam.mapping_id = fr.mapping_id
                        WHERE fam.forecast_run_id = %s
                          AND fam.execution_status = 'Completed'
                        GROUP BY fam.mapping_id, fam.algorithm_id
                        ORDER BY AVG(fr.accuracy_metric) DESC
                        LIMIT 1
                    """, (request.forecast_run_id,))
                    
                    mapping_row = cursor.fetchone()
                    if not mapping_row:
                        raise NotFoundException("Forecast Results", request.forecast_run_id)
                    
                    mapping_id, algorithm_id = mapping_row
                    
                    # 4. Fetch aggregated results (only future forecast as requested)
                    cursor.execute("""
                        SELECT date, value, type, metadata
                        FROM forecast_results
                        WHERE mapping_id = %s
                          AND type = 'future_forecast'
                    """, (mapping_id,))
                    
                    agg_results = cursor.fetchall()
                    
                    source_cols = AggregationService._get_aggregation_columns(
                        tenant_id, database_name, source_level
                    )
                    target_cols = AggregationService._get_aggregation_columns(
                        tenant_id, database_name, request.target_aggregation_level
                    )

                    for row in agg_results:
                        f_date, f_value, f_type, f_metadata = row
                        if isinstance(f_metadata, str):
                            f_metadata = json.loads(f_metadata)
                        
                        entity_filter = f_metadata.get('entity_filter', {})
                        
                        # Match this aggregated result with relevant ratios
                        mask = pd.Series(True, index=ratios_df.index)
                        for col, val in entity_filter.items():
                            if col in ratios_df.columns:
                                mask &= (ratios_df[col].astype(str) == str(val))
                        
                        relevant_ratios = ratios_df[mask]
                        
                        for _, ratio_row in relevant_ratios.iterrows():
                            disagg_value = float(f_value) * float(ratio_row['allocation_factor'])
                            
                            # Create new granular entity filter to resolve master_id
                            new_entity_filter = {col: ratio_row[col] for col in target_cols}
                            
                            # Resolve master_id for the granular entity
                            master_id = DisaggregationService._resolve_master_id(
                                cursor, new_entity_filter
                            )
                            
                            # Get UOM from sales_data for this specific master_id
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
                            uom = sales_row[0] if sales_row else 'EA'
                            unit_price = sales_row[1] if sales_row else None
                            
                            # Insert granular result into final_plan table
                            final_plan_id = str(uuid.uuid4())
                            
                            # Build disaggregation_level string from target columns
                            disaggregation_level = '-'.join(target_cols)
                            source_aggregation_level = '-'.join(source_cols)
                            
                            cursor.execute(f"""
                                INSERT INTO final_plan
                                (final_plan_id, master_id, "{date_field}", "{target_field}",
                                 uom, unit_price, type, disaggregation_level, 
                                 source_aggregation_level, source_forecast_run_id, created_by)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                final_plan_id,
                                master_id,
                                f_date,
                                disagg_value,
                                uom,
                                unit_price,
                                'disaggregated',
                                disaggregation_level,
                                source_aggregation_level,
                                request.forecast_run_id,
                                user_email
                            ))
                            disaggregated_count += 1
                    
                    conn.commit()
                    return {
                        "status": "success",
                        "disaggregated_records": disaggregated_count,
                        "message": (
                            f"Successfully disaggregated forecast into {disaggregated_count} "
                            "granular records in final_plan table"
                        )
                    }
                    
                finally:
                    cursor.close()
        except Exception as e:
            logger.error(f"Disaggregation failed: {str(e)}")
            if isinstance(e, (ValidationException, NotFoundException)):
                raise
            raise DatabaseException(f"Failed to disaggregate forecast: {str(e)}")

    @staticmethod
    def get_disaggregated_forecast_data(
        tenant_id: str,
        database_name: str,
        forecast_run_id: str,
        aggregation_level: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieve disaggregated forecast data for a given forecast run.
        Returns detailed forecast records with dimension breakdowns.
        """
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        SELECT forecast_run_id, version_id, forecast_filters, 
                               created_at, run_status
                        FROM forecast_runs
                        WHERE forecast_run_id = %s
                    """, (forecast_run_id,))
                    
                    run_info = cursor.fetchone()
                    if not run_info:
                        raise NotFoundException("Forecast Run", forecast_run_id)
                    
                    run_id, version_id, forecast_filters, created_at, run_status = run_info
                    
                    query = """
                        SELECT 
                            fr.result_id,
                            fr.date,
                            fr.value,
                            fr.type,
                            fr.metadata,
                            fam.algorithm_id,
                            a.algorithm_name,
                            fr.accuracy_metric,
                            fr.created_at
                        FROM forecast_results fr
                        JOIN forecast_algorithms_mapping fam ON fr.mapping_id = fam.mapping_id
                        JOIN algorithms a ON fam.algorithm_id = a.algorithm_id
                        WHERE fr.forecast_run_id = %s
                          AND fr.metadata IS NOT NULL
                    """
                    
                    params = [forecast_run_id]
                    
                    if aggregation_level:
                        query += " AND fr.metadata::jsonb->>'aggregation_level' = %s"
                        params.append(aggregation_level)
                    
                    query += " ORDER BY fr.date, fr.value DESC"
                    
                    cursor.execute(query, params)
                    results = cursor.fetchall()
                    
                    disaggregated_records = []
                    for row in results:
                        result_id, date, value, f_type, metadata, algo_id, algo_name, accuracy, rec_created_at = row
                        
                        if isinstance(metadata, str):
                            try:
                                metadata_dict = json.loads(metadata)
                            except json.JSONDecodeError:
                                metadata_dict = {}
                        else:
                            metadata_dict = metadata if metadata else {}
                        
                        record = {
                            "result_id": result_id,
                            "date": date.isoformat() if hasattr(date, 'isoformat') else str(date),
                            "value": float(value),
                            "type": f_type,
                            "algorithm_id": algo_id,
                            "algorithm_name": algo_name,
                            "accuracy_metric": float(accuracy) if accuracy else None,
                            "created_at": rec_created_at.isoformat() if hasattr(rec_created_at, 'isoformat') else str(rec_created_at),
                            "dimensions": metadata_dict.get('entity_filter', {}),
                            "disaggregated_from": metadata_dict.get('disaggregated_from'),
                            "additional_metadata": {
                                k: v for k, v in metadata_dict.items()
                                if k not in ['entity_filter', 'disaggregated_from']
                            }
                        }
                        disaggregated_records.append(record)
                    
                    if disaggregated_records:
                        total_value = sum(r['value'] for r in disaggregated_records)
                        avg_value = total_value / len(disaggregated_records)
                        min_value = min(r['value'] for r in disaggregated_records)
                        max_value = max(r['value'] for r in disaggregated_records)
                    else:
                        total_value = avg_value = min_value = max_value = 0
                    
                    return {
                        "forecast_run_id": run_id,
                        "status": run_status,
                        "created_at": created_at.isoformat() if hasattr(created_at, 'isoformat') else str(created_at),
                        "record_count": len(disaggregated_records),
                        "summary": {
                            "total_value": total_value,
                            "average_value": avg_value,
                            "min_value": min_value,
                            "max_value": max_value,
                            "unique_dates": len(set(r['date'] for r in disaggregated_records)) if disaggregated_records else 0
                        },
                        "disaggregated_records": disaggregated_records
                    }
                    
                finally:
                    cursor.close()
                    
        except (ValidationException, NotFoundException):
            raise
        except Exception as e:
            logger.error(f"Failed to retrieve disaggregated forecast data: {str(e)}")
            raise DatabaseException(f"Failed to retrieve disaggregated forecast data: {str(e)}")

    @staticmethod
    def disaggregate_data(
        tenant_id: str,
        database_name: str,
        request: DisaggregateDataRequest,
        user_email: str = 'system',
        page: int = 1,
        page_size: int = 50
    ) -> Dict[str, Any]:
        """
        Disaggregate sales, forecast, and ALL tenant dynamic planning tables to the
        lowest granular level and upsert results into each respective table.

        CHANGED: Instead of hardcoding final_plan and product_manager, this method now
        dynamically loads all planning tables registered in dynamic_tables metadata.

        Ratio logic per table type:
        - forecast_data  → ratios calculated from sales_data
        - every dynamic table (final_plan, product_manager, any custom table)
                         → ratios calculated from that same table's own data

        FIX: user_email param added so audit trail records real user, not 'system'.
        FIX: existing_tables set computed once to avoid 3x duplicate information_schema
             queries per table.

        Returns data in dashboard format: grouped by master_data dimensions with
        separate arrays per table.
        """
        from app.core.aggregation_service import AggregationService
        from app.core.forecasting_service import ForecastingService
        from app.core.dynamic_table_service import DynamicTableService

        db_manager = get_db_manager()

        try:
            # Get field names from metadata
            target_field, date_field = ForecastingService._get_field_names(tenant_id, database_name)

            # Auto-determine target aggregation level as all dimension fields
            target_columns = AggregationService._get_dimension_fields(tenant_id, database_name)
            target_aggregation_level = '-'.join(target_columns)

            # Convert filters
            filters_dict = {}
            if request.filters:
                for filter_item in request.filters:
                    if len(filter_item.values) == 1:
                        filters_dict[filter_item.field_name] = filter_item.values[0]
                    else:
                        filters_dict[filter_item.field_name] = filter_item.values

            # Determine source aggregation level from filters
            source_columns = []
            if filters_dict:
                for key in filters_dict.keys():
                    if key in target_columns:
                        source_columns.append(key)

            if not source_columns:
                source_columns = [target_columns[0]]

            source_aggregation_level = '-'.join(source_columns)

            logger.info(f"Disaggregating from {source_aggregation_level} to {target_aggregation_level}")

            # ================================================================
            # CHANGED: Load all dynamic tables from metadata once
            # ================================================================
            dynamic_tables = DynamicTableService.get_tenant_dynamic_tables(
                database_name=database_name,
                include_mandatory=True
            )
            # Build a simple list of table names for dynamic planning tables
            # (excludes the core immutable tables: sales_data, forecast_data)
            dynamic_table_names = [t['table_name'] for t in dynamic_tables]

            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # ============================================================
                    # FIX: Compute which dynamic tables physically exist once,
                    # reused for ratio calculation, source data fetch, and upsert.
                    # Avoids 3x duplicate information_schema queries per table.
                    # ============================================================
                    if dynamic_table_names:
                        placeholders_tables = ','.join(['%s'] * len(dynamic_table_names))
                        cursor.execute(f"""
                            SELECT table_name FROM information_schema.tables
                            WHERE table_schema = 'public'
                            AND table_name IN ({placeholders_tables})
                        """, dynamic_table_names)
                        existing_tables: set = {row[0] for row in cursor.fetchall()}
                    else:
                        existing_tables = set()

                    # Warn about any registered but missing tables
                    missing_tables = []
                    for table_name in dynamic_table_names:
                        if table_name not in existing_tables:
                            logger.warning(
                                f"Dynamic table '{table_name}' registered in metadata "
                                "but does not exist physically — skipping"
                            )
                            missing_tables.append(table_name)
                    
                    # FIX: If too many tables are missing, raise error to alert user
                    # that something is misconfigured
                    if missing_tables and len(missing_tables) >= len(dynamic_table_names):
                        raise ValidationException(
                            f"All registered dynamic tables are missing: {missing_tables}. "
                            "Please check your table configuration."
                        )

                    # ============================================================
                    # CHANGED: Calculate ratios for each table independently
                    # forecast_data uses sales_data as ratio source
                    # every dynamic table uses its own data as ratio source
                    # ============================================================
                    forecast_ratios_df = DisaggregationService._calculate_disaggregation_ratios(
                        conn=conn,
                        source_columns=source_columns,
                        target_columns=target_columns,
                        filters=filters_dict,
                        date_field=date_field,
                        target_field=target_field,
                        interval=request.interval,
                        date_from=request.date_from,
                        date_to=request.date_to,
                        source_table='sales_data'
                    )

                    # Build ratios for each dynamic table using its own data
                    dynamic_table_ratios: Dict[str, pd.DataFrame] = {}
                    for table_name in dynamic_table_names:
                        if table_name not in existing_tables:
                            dynamic_table_ratios[table_name] = pd.DataFrame()
                            continue

                        dynamic_table_ratios[table_name] = DisaggregationService._calculate_disaggregation_ratios(
                            conn=conn,
                            source_columns=source_columns,
                            target_columns=target_columns,
                            filters=filters_dict,
                            date_field=date_field,
                            target_field=target_field,
                            interval=request.interval,
                            date_from=request.date_from,
                            date_to=request.date_to,
                            source_table=table_name
                        )

                    # Check if we have at least some data
                    has_any_data = (
                        not forecast_ratios_df.empty
                        or any(not df.empty for df in dynamic_table_ratios.values())
                    )
                    if not has_any_data:
                        return {
                            "records": [],
                            "total_count": 0,
                            "summary": {"message": "No data found for ratio calculation"}
                        }

                    # Build filter clause for master_data
                    where_conditions = []
                    where_params = []

                    if filters_dict:
                        for col, val in filters_dict.items():
                            if isinstance(val, list):
                                placeholders = ', '.join(['%s'] * len(val))
                                where_conditions.append(f'm."{col}" IN ({placeholders})')
                                where_params.extend(val)
                            else:
                                where_conditions.append(f'm."{col}" = %s')
                                where_params.append(val)

                    where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else "WHERE 1=1"

                    # Get ALL target dimension combinations
                    target_cols_sql = ', '.join([f'm."{col}"' for col in target_columns])

                    count_query = f"""
                        SELECT COUNT(*) FROM (
                            SELECT DISTINCT {target_cols_sql}
                            FROM master_data m
                            {where_clause}
                        ) as dim_groups
                    """
                    cursor.execute(count_query, where_params)
                    total_combinations = cursor.fetchone()[0]

                    groups_query = f"""
                        SELECT DISTINCT {target_cols_sql}
                        FROM master_data m
                        {where_clause}
                        ORDER BY {target_cols_sql}
                    """
                    cursor.execute(groups_query, where_params)
                    dimension_groups = cursor.fetchall()

                    if not dimension_groups:
                        return {
                            "records": [],
                            "total_count": total_combinations,
                            "summary": {"message": "No dimension combinations found"}
                        }

                    # Build ratios lookup dicts for fast access
                    forecast_ratios_dict = DisaggregationService._build_ratios_dict(
                        forecast_ratios_df, target_columns, source_columns
                    )
                    # CHANGED: build dict per dynamic table
                    dynamic_ratios_dicts: Dict[str, Dict] = {}
                    for table_name, ratios_df in dynamic_table_ratios.items():
                        dynamic_ratios_dicts[table_name] = DisaggregationService._build_ratios_dict(
                            ratios_df, target_columns, source_columns
                        )

                    # Fetch SOURCE-LEVEL data for each table
                    start_date = request.date_from or "1900-01-01"
                    end_date = request.date_to or "2099-12-31"

                    source_cols_sql = ', '.join([f'm."{col}"' for col in source_columns])

                    # Build filter for source-level data
                    source_filter_conditions = []
                    source_filter_params = []

                    if filters_dict:
                        for col, val in filters_dict.items():
                            if isinstance(val, list):
                                placeholders_f = ', '.join(['%s'] * len(val))
                                source_filter_conditions.append(f'm."{col}" IN ({placeholders_f})')
                                source_filter_params.extend(val)
                            else:
                                source_filter_conditions.append(f'm."{col}" = %s')
                                source_filter_params.append(val)

                    source_where = " AND ".join(source_filter_conditions) if source_filter_conditions else "1=1"

                    # Build the same date_trunc expression used in ratio calculation
                    # so that data_date values match the period keys in ratios_dict.
                    from app.core.aggregation_service import AggregationService
                    date_trunc_expr_f = AggregationService._get_date_trunc_expr(
                        request.interval, f'f."{date_field}"'
                    )
                    date_trunc_expr_t = AggregationService._get_date_trunc_expr(
                        request.interval, f't."{date_field}"'
                    )

                    # Fetch forecast_data at SOURCE level (always from forecast_data table)
                    forecast_query = f"""
                        SELECT {date_trunc_expr_f} AS period,
                            SUM(CAST(f."{target_field}" AS DOUBLE PRECISION)),
                            MAX(f.uom), {source_cols_sql}
                        FROM forecast_data f
                        JOIN master_data m ON f.master_id = m.master_id
                        WHERE {source_where}
                        AND f."{date_field}" BETWEEN %s AND %s
                        GROUP BY {date_trunc_expr_f}, {source_cols_sql}
                        ORDER BY {date_trunc_expr_f} ASC
                    """
                    cursor.execute(forecast_query, source_filter_params + [start_date, end_date])
                    forecast_source_data = cursor.fetchall()

                    # ============================================================
                    # CHANGED: Fetch source-level data for each dynamic table
                    # ============================================================
                    dynamic_source_data: Dict[str, List] = {}
                    for table_name in dynamic_table_names:
                        if table_name not in existing_tables:
                            dynamic_source_data[table_name] = []
                            continue

                        # Use type filter only for tables that have the 'type' column
                        # (all dynamic tables created by DynamicTableService have it)
                        dyn_query = f"""
                            SELECT {date_trunc_expr_t} AS period,
                                SUM(CAST(t."{target_field}" AS DOUBLE PRECISION)),
                                MAX(t.uom), {source_cols_sql}
                            FROM {table_name} t
                            JOIN master_data m ON t.master_id = m.master_id
                            WHERE {source_where}
                            AND t."{date_field}" BETWEEN %s AND %s
                            AND (t.type IS NULL OR t.type != 'disaggregated')
                            GROUP BY {date_trunc_expr_t}, {source_cols_sql}
                            ORDER BY {date_trunc_expr_t} ASC
                        """
                        cursor.execute(dyn_query, source_filter_params + [start_date, end_date])
                        dynamic_source_data[table_name] = cursor.fetchall()

                    # ============================================================
                    # Build results for ALL target dimension groups
                    # ============================================================
                    all_results = []

                    for group in dimension_groups:
                        group_data = dict(zip(target_columns, group))
                        target_key = tuple(str(v) for v in group)

                        # Find master_ids for this target group
                        group_conditions = []
                        group_vals = []
                        for col, val in group_data.items():
                            if val is None:
                                group_conditions.append(f'm."{col}" IS NULL')
                            else:
                                group_conditions.append(f'm."{col}" = %s')
                                group_vals.append(val)

                        group_where_str = " AND ".join(group_conditions)
                        master_id_query = f"SELECT master_id FROM master_data m WHERE {group_where_str}"
                        cursor.execute(master_id_query, group_vals)
                        master_ids = [row[0] for row in cursor.fetchall()]

                        if not master_ids:
                            continue

                        # Build the result entry with static sections + dynamic table sections
                        entry: Dict[str, Any] = {
                            "master_data": group_data,
                            "sales_data": [],
                            "forecast_data": [],
                        }

                        # CHANGED: add a key for each dynamic table dynamically
                        for table_name in dynamic_table_names:
                            entry[table_name] = []

                        # Fetch SALES data (always granular — no disaggregation needed)
                        placeholders = ",".join(["%s"] * len(master_ids))
                        sales_query = f"""
                            SELECT "{date_field}", SUM(CAST("{target_field}" AS DOUBLE PRECISION)), MAX(uom)
                            FROM sales_data
                            WHERE master_id IN ({placeholders})
                            AND "{date_field}" BETWEEN %s AND %s
                            GROUP BY "{date_field}"
                            ORDER BY "{date_field}" ASC
                        """
                        cursor.execute(sales_query, master_ids + [start_date, end_date])
                        for row in cursor.fetchall():
                            data_date, quantity, uom = row
                            entry["sales_data"].append({
                                "date": str(data_date),
                                "UOM": uom or "UNIT",
                                "Quantity": float(quantity) if quantity is not None else 0.0,
                            })

                        # Disaggregate FORECAST using forecast_ratios_dict (sales_data ratios)
                        num_source_cols = len(source_columns)
                        for row in forecast_source_data:
                            data_date = row[0]
                            source_quantity = row[1]
                            uom = row[2]
                            source_vals = row[3:3 + num_source_cols]

                            if source_quantity is None or source_quantity == 0:
                                continue

                            # Period key matches what DATE_TRUNC produced in the ratio query.
                            # data_date is already at period granularity (grouped by date_field
                            # in the source fetch query), so str() of it forms a consistent key.
                            source_key = tuple(str(v) for v in source_vals) + (str(data_date),)

                            if (
                                target_key in forecast_ratios_dict
                                and source_key in forecast_ratios_dict[target_key]
                            ):
                                allocation_factor = forecast_ratios_dict[target_key][source_key]
                                disaggregated_qty = float(source_quantity) * allocation_factor
                                entry["forecast_data"].append({
                                    "date": str(data_date),
                                    "UOM": uom or "UNIT",
                                    "Quantity": round(disaggregated_qty, 4),
                                })

                        # CHANGED: Disaggregate each dynamic table using its own ratios dict
                        for table_name in dynamic_table_names:
                            table_ratios_dict = dynamic_ratios_dicts.get(table_name, {})
                            source_rows = dynamic_source_data.get(table_name, [])

                            for row in source_rows:
                                data_date = row[0]
                                source_quantity = row[1]
                                uom = row[2]
                                source_vals = row[3:3 + num_source_cols]

                                if source_quantity is None or source_quantity == 0:
                                    continue

                                # Include period in source_key to match _build_ratios_dict
                                source_key = tuple(str(v) for v in source_vals) + (str(data_date),)

                                if (
                                    target_key in table_ratios_dict
                                    and source_key in table_ratios_dict[target_key]
                                ):
                                    allocation_factor = table_ratios_dict[target_key][source_key]
                                    disaggregated_qty = float(source_quantity) * allocation_factor
                                    entry[table_name].append({
                                        "date": str(data_date),
                                        "UOM": uom or "UNIT",
                                        "Quantity": round(disaggregated_qty, 4),
                                    })

                        all_results.append(entry)

                    # ============================================================
                    # CHANGED: Upsert into forecast_data and all dynamic tables
                    # ============================================================
                    upserted_counts: Dict[str, int] = {}

                    # Upsert forecast_data
                    upserted_counts["forecast_data"] = DisaggregationService._upsert_disaggregated_data(
                        cursor=cursor,
                        results=all_results,
                        target_table="forecast_data",
                        target_columns=target_columns,
                        date_field=date_field,
                        target_field=target_field,
                        user_email=user_email
                    )

                    # Upsert each dynamic table — skip physically missing ones
                    for table_name in dynamic_table_names:
                        if table_name not in existing_tables:
                            upserted_counts[table_name] = 0
                            continue

                        upserted_counts[table_name] = DisaggregationService._upsert_disaggregated_data(
                            cursor=cursor,
                            results=all_results,
                            target_table=table_name,
                            target_columns=target_columns,
                            date_field=date_field,
                            target_field=target_field,
                            user_email=user_email
                        )

                    conn.commit()
                    total_upserted = sum(upserted_counts.values())
                    logger.info(f"Upserted {total_upserted} records total: {upserted_counts}")

                    # Paginate the response records
                    offset = (page - 1) * page_size
                    paginated_results = all_results[offset: offset + page_size]

                    return {
                        "records": paginated_results,
                        "total_count": total_combinations,
                        "summary": {
                            "source_aggregation_level": source_aggregation_level,
                            "target_aggregation_level": target_aggregation_level,
                            "target_tables": ["forecast_data"] + dynamic_table_names,
                            "records_upserted": upserted_counts,
                            "total_records_upserted": total_upserted,
                            "total_combinations": total_combinations,
                            "date_range": {
                                "from": request.date_from,
                                "to": request.date_to
                            },
                            "interval": request.interval,
                            "ratio_calculation_method": {
                                "forecast_data": "sales_data ratios",
                                **{
                                    table_name: f"{table_name} own data ratios"
                                    for table_name in dynamic_table_names
                                }
                            }
                        }
                    }

                finally:
                    cursor.close()

        except Exception as e:
            logger.error(f"Failed to disaggregate data: {str(e)}", exc_info=True)
            if isinstance(e, (ValidationException, NotFoundException)):
                raise
            raise DatabaseException(f"Failed to disaggregate data: {str(e)}")

    @staticmethod
    def _resolve_master_id(cursor, entity_filter: Dict[str, Any]) -> str:
        """Resolve master_id from entity filter."""
        if not entity_filter:
            raise ValidationException("Entity filter is required for master data lookup")

        clauses = []
        params = []
        for key, value in entity_filter.items():
            clauses.append(sql.SQL("{} = %s").format(sql.Identifier(key)))
            params.append(value)
        query = sql.SQL("SELECT master_id FROM master_data WHERE {} LIMIT 1").format(
            sql.SQL(" AND ").join(clauses)
        )
        cursor.execute(query, params)
        row = cursor.fetchone()
        if not row:
            raise ValidationException(
                f"No master data found for entity_filter: {entity_filter}"
            )
        return row[0]

    @staticmethod
    def _build_ratios_dict(
        ratios_df: pd.DataFrame,
        target_columns: List[str],
        source_columns: List[str]
    ) -> Dict[tuple, Dict[tuple, float]]:
        """
        Build a nested dictionary for fast ratio lookups.

        Keys include period when present in ratios_df (i.e. after interval fix), so
        that per-period allocation factors don't overwrite each other in the dict.

        Returns:
            {target_key: {(source_key, period): allocation_factor}}
            where period is included only when the 'period' column exists in ratios_df.
        """
        ratios_dict: Dict[tuple, Dict[tuple, float]] = {}
        if ratios_df.empty:
            return ratios_dict

        has_period = 'period' in ratios_df.columns

        for _, row in ratios_df.iterrows():
            target_key = tuple(str(row[col]) for col in target_columns)
            source_key = tuple(str(row[col]) for col in source_columns)
            if has_period:
                # Include period in source_key so per-period factors are stored separately
                source_key = source_key + (str(row['period']),)
            allocation_factor = float(row['allocation_factor'])

            if target_key not in ratios_dict:
                ratios_dict[target_key] = {}
            ratios_dict[target_key][source_key] = allocation_factor

        return ratios_dict

    @staticmethod
    def _calculate_disaggregation_ratios(
        conn,
        source_columns: List[str],
        target_columns: List[str],
        filters: Optional[Dict[str, Any]],
        date_field: str,
        target_field: str,
        interval: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        source_table: str = 'sales_data'
    ) -> pd.DataFrame:
        """
        Calculate disaggregation ratios based on historical data from the specified source table.

        Accepts source_table parameter to use different tables for ratio calculation:
        - 'sales_data'      : for forecast_data disaggregation
        - 'final_plan'      : for final_plan disaggregation
        - any dynamic table : uses that table's own data for ratio calculation

        FIX: interval is now used — ratios are aggregated at the correct time granularity
        (MONTHLY, WEEKLY, etc.) via DATE_TRUNC before summing, so the allocation factors
        reflect the same periodicity as the source forecast data.

        Returns DataFrame with target-level combinations and their allocation factors.
        Returns empty DataFrame if the source table has no data.
        """
        from app.core.aggregation_service import AggregationService

        # Build filter clause
        filter_clause = ""
        filter_params = []

        if filters:
            filter_conditions = []
            for col, val in filters.items():
                if col not in ["interval", "date_from", "date_to"]:
                    if isinstance(val, list):
                        placeholders = ', '.join(['%s'] * len(val))
                        filter_conditions.append(f'm."{col}" IN ({placeholders})')
                        filter_params.extend(val)
                    else:
                        filter_conditions.append(f'm."{col}" = %s')
                        filter_params.append(val)

            if filter_conditions:
                filter_clause = "AND " + " AND ".join(filter_conditions)

        # Add date filters
        if date_from:
            filter_clause += f' AND s."{date_field}" >= %s'
            filter_params.append(date_from)

        if date_to:
            filter_clause += f' AND s."{date_field}" <= %s'
            filter_params.append(date_to)

        target_cols_sql = ', '.join([f'm."{col}"' for col in target_columns])

        # FIX: Use DATE_TRUNC so ratios respect the requested interval granularity.
        # This ensures a MONTHLY interval sums each month's data before computing
        # allocation factors, matching how source forecast data is periodised.
        # The period column is included in SELECT and GROUP BY so that allocation
        # factors are calculated per-period, not collapsed across all time.
        date_trunc_expr = AggregationService._get_date_trunc_expr(interval, f's."{date_field}"')

        query = f"""
            SELECT
                {target_cols_sql},
                {date_trunc_expr} AS period,
                SUM(CAST(s."{target_field}" AS numeric)) as target_volume
            FROM {source_table} s
            JOIN master_data m ON s.master_id = m.master_id
            WHERE 1=1 {filter_clause}
            GROUP BY {target_cols_sql}, {date_trunc_expr}
            ORDER BY {target_cols_sql}, {date_trunc_expr}
        """

        logger.debug(f"Ratio calculation query for {source_table} (interval={interval}): {query}")
        logger.debug(f"Params: {filter_params}")

        df_target = pd.read_sql_query(query, conn, params=filter_params)

        if df_target.empty:
            logger.warning(f"No data found in {source_table} for ratio calculation")
            return pd.DataFrame()

        # FIX: group by source_columns + ['period'] so source totals are computed
        # per period, not collapsed across all time. Without this, a tenant with
        # MONTHLY interval would get allocation factors averaged over the entire
        # history range rather than per month — producing wrong disaggregations.
        groupby_cols = source_columns + ['period']
        source_totals = df_target.groupby(groupby_cols)['target_volume'].sum().reset_index()
        source_totals.rename(columns={'target_volume': 'source_total_volume'}, inplace=True)

        # Merge on source_columns + period so each period gets its own allocation factor
        ratios_df = pd.merge(df_target, source_totals, on=groupby_cols)
        ratios_df['allocation_factor'] = ratios_df['target_volume'] / ratios_df['source_total_volume']
        ratios_df['allocation_factor'] = ratios_df['allocation_factor'].fillna(0)

        sort_columns = target_columns + source_columns + ['period']
        ratios_df = ratios_df.sort_values(by=sort_columns).reset_index(drop=True)

        logger.info(f"Calculated {len(ratios_df)} disaggregation ratios from {source_table}")

        return ratios_df

    @staticmethod
    def _upsert_disaggregated_data(
        cursor,
        results: List[Dict[str, Any]],
        target_table: str,
        target_columns: List[str],
        date_field: str,
        target_field: str,
        user_email: str
    ) -> int:
        """
        Upsert disaggregated data into the specified target table.

        CHANGED: No longer branches on hardcoded table names (final_plan, product_manager,
        forecast_data). Instead uses a single generic UPSERT that works for any table
        that follows the standard dynamic table schema (master_id, date_field, target_field,
        uom, unit_price, created_by, updated_at, updated_by columns).

        FIX: Checks whether the target table has a 'type' column before including it in
        the INSERT. forecast_data does NOT have a 'type' column (it was created without
        one in schema_manager). All tables created by DynamicTableService DO have it.
        The old generic upsert crashed on forecast_data because it always inserted 'type'.

        Args:
            cursor: Database cursor
            results: List of disaggregated result entries (keyed by table_name)
            target_table: Target table name (any dynamic table or forecast_data)
            target_columns: List of dimension columns (used to resolve master_id)
            date_field: Date field name
            target_field: Target field name (quantity)
            user_email: User email for auditing

        Returns:
            Number of records upserted
        """
        upserted_count = 0
        id_column = f"{target_table}_id"

        # FIX: Check once whether this table has a 'type' column.
        # forecast_data was created without it; all DynamicTableService tables have it.
        cursor.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'public'
                AND table_name = %s
                AND column_name = 'type'
            )
        """, (target_table,))
        has_type_column = bool(cursor.fetchone()[0])

        for entry in results:
            master_data = entry.get("master_data", {})
            if not master_data:
                continue

            # Resolve master_id for this dimension combination
            try:
                master_id = DisaggregationService._resolve_master_id(cursor, master_data)
            except ValidationException:
                logger.warning(f"Could not resolve master_id for {master_data}, skipping")
                continue

            # Get UOM and unit_price from sales_data for this specific master_id
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
            unit_price = sales_row[1] if sales_row else 0.0

            # Get the data list for this specific table from the entry
            data_list = entry.get(target_table, [])
            if not data_list:
                continue

            for data_point in data_list:
                data_date = data_point.get("date")
                quantity = data_point.get("Quantity", 0)

                if not data_date or quantity == 0:
                    continue

                record_id = str(uuid.uuid4())

                if has_type_column:
                    # Tables created by DynamicTableService (final_plan, product_manager,
                    # any custom table) — include 'type' column in INSERT.
                    cursor.execute(f"""
                        INSERT INTO {target_table}
                        ({id_column}, master_id, "{date_field}", "{target_field}",
                         uom, unit_price, type, created_by)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (master_id, "{date_field}")
                        DO UPDATE SET
                            "{target_field}" = EXCLUDED."{target_field}",
                            uom = EXCLUDED.uom,
                            unit_price = EXCLUDED.unit_price,
                            type = EXCLUDED.type,
                            updated_at = CURRENT_TIMESTAMP,
                            updated_by = %s
                    """, (
                        record_id,
                        master_id,
                        data_date,
                        quantity,
                        uom,
                        unit_price,
                        'disaggregated',
                        user_email,
                        user_email
                    ))
                else:
                    # forecast_data — no 'type', 'updated_at', or 'updated_by' columns.
                    cursor.execute(f"""
                        INSERT INTO {target_table}
                        ({id_column}, master_id, "{date_field}", "{target_field}",
                         uom, unit_price, created_by)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (master_id, "{date_field}")
                        DO UPDATE SET
                            "{target_field}" = EXCLUDED."{target_field}",
                            uom = EXCLUDED.uom,
                            unit_price = EXCLUDED.unit_price
                    """, (
                        record_id,
                        master_id,
                        data_date,
                        quantity,
                        uom,
                        unit_price,
                        user_email
                    ))

                upserted_count += 1

        return upserted_count