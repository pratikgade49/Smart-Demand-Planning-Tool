"""
Disaggregation service.
Handles forecast disaggregation, data breakdown to granular levels, and ratio calculations.
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
                        # Filter ratios_df by entity_filter (which contains source level values)
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
                            
                            # Get UOM from sales_data
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
                        "message": f"Successfully disaggregated forecast into {disaggregated_count} granular records in final_plan table"
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
                    # Get forecast run details
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
                    
                    # Get disaggregated results
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
                    
                    # Apply optional aggregation level filter
                    if aggregation_level:
                        query += " AND fr.metadata::jsonb->>'aggregation_level' = %s"
                        params.append(aggregation_level)
                    
                    query += " ORDER BY fr.date, fr.value DESC"
                    
                    cursor.execute(query, params)
                    results = cursor.fetchall()
                    
                    disaggregated_records = []
                    for row in results:
                        result_id, date, value, f_type, metadata, algo_id, algo_name, accuracy, created_at = row
                        
                        # Parse metadata
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
                            "created_at": created_at.isoformat() if hasattr(created_at, 'isoformat') else str(created_at),
                            "dimensions": metadata_dict.get('entity_filter', {}),
                            "disaggregated_from": metadata_dict.get('disaggregated_from'),
                            "additional_metadata": {
                                k: v for k, v in metadata_dict.items() 
                                if k not in ['entity_filter', 'disaggregated_from']
                            }
                        }
                        disaggregated_records.append(record)
                    
                    # Summary statistics
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
        page: int = 1,
        page_size: int = 50
    ) -> Dict[str, Any]:
        """
        Disaggregate sales, forecast, and final plan data to the lowest granular level
        and upsert results into the specified target table.

        Uses table-specific ratios for final_plan and product_manager:
        - forecast_data: uses sales_data ratios
        - final_plan: uses existing final_plan ratios
        - product_manager: uses existing product_manager ratios

        Returns data in dashboard format: grouped by master_data dimensions with separate arrays.
        """
        from app.core.aggregation_service import AggregationService
        from app.core.forecasting_service import ForecastingService
        
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

            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Calculate ratios for each table type separately
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
                    
                    final_plan_ratios_df = DisaggregationService._calculate_disaggregation_ratios(
                        conn=conn,
                        source_columns=source_columns,
                        target_columns=target_columns,
                        filters=filters_dict,
                        date_field=date_field,
                        target_field=target_field,
                        interval=request.interval,
                        date_from=request.date_from,
                        date_to=request.date_to,
                        source_table='final_plan'
                    )
                    
                    product_manager_ratios_df = DisaggregationService._calculate_disaggregation_ratios(
                        conn=conn,
                        source_columns=source_columns,
                        target_columns=target_columns,
                        filters=filters_dict,
                        date_field=date_field,
                        target_field=target_field,
                        interval=request.interval,
                        date_from=request.date_from,
                        date_to=request.date_to,
                        source_table='product_manager'
                    )
                    
                    if forecast_ratios_df.empty and final_plan_ratios_df.empty and product_manager_ratios_df.empty:
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
                    
                    # Build ratios lookups for each table type
                    forecast_ratios_dict = DisaggregationService._build_ratios_dict(
                        forecast_ratios_df, target_columns, source_columns
                    )
                    final_plan_ratios_dict = DisaggregationService._build_ratios_dict(
                        final_plan_ratios_df, target_columns, source_columns
                    )
                    product_manager_ratios_dict = DisaggregationService._build_ratios_dict(
                        product_manager_ratios_df, target_columns, source_columns
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
                    
                    # Fetch forecast_data at SOURCE level
                    forecast_query = f"""
                        SELECT f."{date_field}", SUM(CAST(f."{target_field}" AS DOUBLE PRECISION)),
                            MAX(f.uom), {source_cols_sql}
                        FROM forecast_data f
                        JOIN master_data m ON f.master_id = m.master_id
                        WHERE {source_where}
                        AND f."{date_field}" BETWEEN %s AND %s
                        GROUP BY f."{date_field}", {source_cols_sql}
                        ORDER BY f."{date_field}" ASC
                    """
                    cursor.execute(forecast_query, source_filter_params + [start_date, end_date])
                    forecast_source_data = cursor.fetchall()
                    
                    # Fetch final_plan at SOURCE level
                    final_plan_query = f"""
                        SELECT fp."{date_field}", SUM(CAST(fp."{target_field}" AS DOUBLE PRECISION)),
                            MAX(fp.uom), {source_cols_sql}
                        FROM final_plan fp
                        JOIN master_data m ON fp.master_id = m.master_id
                        WHERE {source_where}
                        AND fp."{date_field}" BETWEEN %s AND %s
                        AND (fp.type IS NULL OR fp.type != 'disaggregated')
                        GROUP BY fp."{date_field}", {source_cols_sql}
                        ORDER BY fp."{date_field}" ASC
                    """
                    cursor.execute(final_plan_query, source_filter_params + [start_date, end_date])
                    final_plan_source_data = cursor.fetchall()

                    # Fetch product_manager at SOURCE level
                    product_manager_query = f"""
                        SELECT pm."{date_field}", SUM(CAST(pm."{target_field}" AS DOUBLE PRECISION)),
                            MAX(pm.uom), {source_cols_sql}
                        FROM product_manager pm
                        JOIN master_data m ON pm.master_id = m.master_id
                        WHERE {source_where}
                        AND pm."{date_field}" BETWEEN %s AND %s
                        AND (pm.type IS NULL OR pm.type != 'disaggregated')
                        GROUP BY pm."{date_field}", {source_cols_sql}
                        ORDER BY pm."{date_field}" ASC
                    """
                    cursor.execute(product_manager_query, source_filter_params + [start_date, end_date])
                    product_manager_source_data = cursor.fetchall()
                    
                    # Build results for ALL target dimension groups
                    all_results = []
                    
                    for group in dimension_groups:
                        group_data = dict(zip(target_columns, group))
                        target_key = tuple(str(v) for v in group)
                        
                        # Find master_ids for this target group (for sales data only)
                        group_conditions = []
                        group_vals = []
                        for col, val in group_data.items():
                            if val is None:
                                group_conditions.append(f'm."{col}" IS NULL')
                            else:
                                group_conditions.append(f'm."{col}" = %s')
                                group_vals.append(val)
                        
                        group_where = " AND ".join(group_conditions)
                        master_id_query = f"SELECT master_id FROM master_data m WHERE {group_where}"
                        cursor.execute(master_id_query, group_vals)
                        master_ids = [row[0] for row in cursor.fetchall()]
                        
                        if not master_ids:
                            continue
                        
                        entry = {
                            "master_data": group_data,
                            "sales_data": [],
                            "forecast_data": [],
                            "final_plan": [],
                            "product_manager": [],
                        }
                        
                        # Fetch SALES data (already granular)
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
                        
                        # Disaggregate FORECAST using forecast_ratios_dict
                        for row in forecast_source_data:
                            num_source_cols = len(source_columns)
                            data_date = row[0]
                            source_quantity = row[1]
                            uom = row[2]
                            source_vals = row[3:3+num_source_cols]
                            
                            if source_quantity is None or source_quantity == 0:
                                continue
                            
                            source_key = tuple(str(v) for v in source_vals)
                            
                            if target_key in forecast_ratios_dict and source_key in forecast_ratios_dict[target_key]:
                                allocation_factor = forecast_ratios_dict[target_key][source_key]
                                disaggregated_qty = float(source_quantity) * allocation_factor
                                
                                entry["forecast_data"].append({
                                    "date": str(data_date),
                                    "UOM": uom or "UNIT",
                                    "Quantity": round(disaggregated_qty, 4),
                                })
                        
                        # Disaggregate FINAL PLAN using final_plan_ratios_dict
                        for row in final_plan_source_data:
                            num_source_cols = len(source_columns)
                            data_date = row[0]
                            source_quantity = row[1]
                            uom = row[2]
                            source_vals = row[3:3+num_source_cols]
                            
                            if source_quantity is None or source_quantity == 0:
                                continue
                            
                            source_key = tuple(str(v) for v in source_vals)
                            
                            if target_key in final_plan_ratios_dict and source_key in final_plan_ratios_dict[target_key]:
                                allocation_factor = final_plan_ratios_dict[target_key][source_key]
                                disaggregated_qty = float(source_quantity) * allocation_factor
                                
                                entry["final_plan"].append({
                                    "date": str(data_date),
                                    "UOM": uom or "UNIT",
                                    "Quantity": round(disaggregated_qty, 4),
                                })

                        # Disaggregate PRODUCT MANAGER using product_manager_ratios_dict
                        for row in product_manager_source_data:
                            num_source_cols = len(source_columns)
                            data_date = row[0]
                            source_quantity = row[1]
                            uom = row[2]
                            source_vals = row[3:3+num_source_cols]
                            
                            if source_quantity is None or source_quantity == 0:
                                continue
                            
                            source_key = tuple(str(v) for v in source_vals)
                            
                            if target_key in product_manager_ratios_dict and source_key in product_manager_ratios_dict[target_key]:
                                allocation_factor = product_manager_ratios_dict[target_key][source_key]
                                disaggregated_qty = float(source_quantity) * allocation_factor
                                
                                entry["product_manager"].append({
                                    "date": str(data_date),
                                    "UOM": uom or "UNIT",
                                    "Quantity": round(disaggregated_qty, 4),
                                })
                        
                        all_results.append(entry)

                    # Upsert to all three tables
                    upserted_counts = {}
                    for table_name in ["forecast_data", "product_manager", "final_plan"]:
                        count = DisaggregationService._upsert_disaggregated_data(
                            cursor=cursor,
                            results=all_results,
                            target_table=table_name,
                            target_columns=target_columns,
                            date_field=date_field,
                            target_field=target_field,
                            user_email='system'
                        )
                        upserted_counts[table_name] = count

                    conn.commit()
                    total_upserted = sum(upserted_counts.values())
                    logger.info(f"Upserted {total_upserted} records total: {upserted_counts}")

                    # Paginate the response records
                    offset = (page - 1) * page_size
                    paginated_results = all_results[offset : offset + page_size]

                    return {
                        "records": paginated_results,
                        "total_count": total_combinations,
                        "summary": {
                            "source_aggregation_level": source_aggregation_level,
                            "target_aggregation_level": target_aggregation_level,
                            "target_tables": ["forecast_data", "product_manager", "final_plan"],
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
                                "final_plan": "final_plan existing ratios",
                                "product_manager": "product_manager existing ratios"
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
        
        Returns:
            {target_key: {source_key: allocation_factor}}
        """
        ratios_dict = {}
        for _, row in ratios_df.iterrows():
            target_key = tuple(str(row[col]) for col in target_columns)
            source_key = tuple(str(row[col]) for col in source_columns)
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
        Calculate disaggregation ratios based on data from specified source table.

        Accepts source_table parameter to use different tables for ratio calculation:
        - 'sales_data': for forecast_data disaggregation
        - 'final_plan': for final_plan disaggregation  
        - 'product_manager': for product_manager disaggregation

        Returns DataFrame with target-level combinations and their allocation factors.
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
        
        # Get date truncation expression
        date_trunc = AggregationService._get_date_trunc_expr(interval, date_field)
        
        # Query for target-level aggregated data FROM THE SPECIFIED SOURCE TABLE
        target_cols_sql = ', '.join([f'm."{col}"' for col in target_columns])
        source_cols_sql = ', '.join([f'm."{col}"' for col in source_columns])
        
        query = f"""
            SELECT
                {target_cols_sql},
                SUM(CAST(s."{target_field}" AS numeric)) as target_volume
            FROM {source_table} s
            JOIN master_data m ON s.master_id = m.master_id
            WHERE 1=1 {filter_clause}
            GROUP BY {target_cols_sql}
            ORDER BY {target_cols_sql}
        """
        
        logger.debug(f"Ratio calculation query for {source_table}: {query}")
        logger.debug(f"Params: {filter_params}")
        
        df_target = pd.read_sql_query(query, conn, params=filter_params)
        
        if df_target.empty:
            logger.warning(f"No data found in {source_table} for ratio calculation")
            return pd.DataFrame()
        
        # Calculate source-level totals
        source_totals = df_target.groupby(source_columns)['target_volume'].sum().reset_index()
        source_totals.rename(columns={'target_volume': 'source_total_volume'}, inplace=True)
        
        # Merge and calculate allocation factors
        ratios_df = pd.merge(df_target, source_totals, on=source_columns)
        ratios_df['allocation_factor'] = ratios_df['target_volume'] / ratios_df['source_total_volume']
        ratios_df['allocation_factor'] = ratios_df['allocation_factor'].fillna(0)

        # Sort ratios_df for consistent processing order
        sort_columns = target_columns + source_columns
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

        Args:
            cursor: Database cursor
            results: List of disaggregated result entries
            target_table: Target table name (final_plan, product_manager, forecast_data)
            target_columns: List of dimension columns
            date_field: Date field name
            target_field: Target field name
            user_email: User email for auditing

        Returns:
            Number of records upserted
        """
        upserted_count = 0

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

            # Get UOM and unit_price from sales data
            cursor.execute(
                f"""
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

            # Process only the data type that matches the target table
            data_type = target_table if target_table in ["forecast_data", "final_plan", "product_manager"] else "forecast_data"
            data_list = entry.get(data_type, [])
            if not data_list:
                continue

            for data_point in data_list:
                data_date = data_point.get("date")
                quantity = data_point.get("Quantity", 0)

                if not data_date or quantity == 0:
                    continue

                # Build upsert query based on target table
                if target_table == "final_plan":
                    # Insert into final_plan table
                    final_plan_id = str(uuid.uuid4())
                    cursor.execute(f"""
                        INSERT INTO final_plan
                        (final_plan_id, master_id, "{date_field}", "{target_field}",
                         uom, unit_price, type, created_by)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (master_id, "{date_field}")
                        DO UPDATE SET
                            "{target_field}" = EXCLUDED."{target_field}",
                            uom = EXCLUDED.uom,
                            unit_price = EXCLUDED.unit_price,
                            type = EXCLUDED.type,
                            updated_at = CURRENT_TIMESTAMP,
                            updated_by = EXCLUDED.created_by
                    """, (
                        final_plan_id,
                        master_id,
                        data_date,
                        quantity,
                        uom,
                        unit_price,
                        'disaggregated',
                        user_email
                    ))

                elif target_table == "product_manager":
                    # Insert into product_manager table
                    cursor.execute(f"""
                        INSERT INTO product_manager
                        (master_id, "{date_field}", "{target_field}",
                         uom, unit_price, type, created_by)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (master_id, "{date_field}")
                        DO UPDATE SET
                            "{target_field}" = EXCLUDED."{target_field}",
                            uom = EXCLUDED.uom,
                            unit_price = EXCLUDED.unit_price
                    """, (
                        master_id,
                        data_date,
                        quantity,
                        uom,
                        unit_price,
                        'disaggregated',
                        user_email
                    ))

                elif target_table == "forecast_data":
                    # Insert into forecast_data table
                    cursor.execute(f"""
                        INSERT INTO forecast_data
                        (master_id, "{date_field}", "{target_field}", uom, unit_price, created_by)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (master_id, "{date_field}")
                        DO UPDATE SET
                            "{target_field}" = EXCLUDED."{target_field}",
                            uom = EXCLUDED.uom,
                            unit_price = EXCLUDED.unit_price
                    """, (
                        master_id,
                        data_date,
                        quantity,
                        uom,
                        unit_price,
                        user_email
                    ))

                upserted_count += 1

        return upserted_count
