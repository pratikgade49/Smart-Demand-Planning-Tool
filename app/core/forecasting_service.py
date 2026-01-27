"""
Forecasting business logic service.
Handles forecast run creation, execution, and result management.
"""

import uuid
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import logging
from psycopg2.extras import Json
import warnings
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from app.core.database import get_db_manager
from app.core.resource_monitor import performance_tracker
from app.config import settings
from app.core.exceptions import (
    DatabaseException,
    ValidationException,
    NotFoundException
)
from app.schemas.forecasting import (
    ForecastRunCreate,
    ForecastRunResponse,
    AlgorithmMappingResponse,
    ForecastResultResponse
)

logger = logging.getLogger(__name__)


class ForecastingService:
    """Service for forecasting operations."""

    # Supported forecast intervals
    FORECAST_INTERVALS = ["WEEKLY", "MONTHLY", "QUARTERLY", "YEARLY"]

    @staticmethod
    def get_available_aggregation_levels(
        tenant_id: str,
        database_name: str
    ) -> List[Dict[str, Any]]:
        """
        Get available aggregation levels based on tenant's master data structure.
        
        Returns list of available fields and their combinations for aggregation.
        """
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Get finalized field catalogue
                    cursor.execute("""
                        SELECT fields_json
                        FROM field_catalogue
                        WHERE status = 'FINALIZED'
                        ORDER BY created_at DESC
                        LIMIT 1
                    """)
                    
                    result = cursor.fetchone()
                    if not result:
                        return []
                    
                    import json
                    fields_json = result[0]
                    if isinstance(fields_json, str):
                        fields = json.loads(fields_json)
                    else:
                        fields = fields_json
                    
                    # Get all non-characteristic fields (base dimensions)
                    base_fields = [
                        {
                            "field_name": field["field_name"],
                            "data_type": field["data_type"],
                            "is_characteristic": field.get("is_characteristic", False),
                            "parent_field_name": field.get("parent_field_name")
                        }
                        for field in fields
                    ]
                    
                    # Single field aggregations
                    single_levels = [
                        {
                            "level_name": field["field_name"],
                            "fields": [field["field_name"]],
                            "description": f"Forecast by {field['field_name']}"
                        }
                        for field in base_fields
                    ]
                    
                    # Generate common 2-field combinations
                    two_field_combos = []
                    for i, field1 in enumerate(base_fields):
                        for field2 in base_fields[i+1:]:
                            combo_name = f"{field1['field_name']}-{field2['field_name']}"
                            two_field_combos.append({
                                "level_name": combo_name,
                                "fields": [field1["field_name"], field2["field_name"]],
                                "description": f"Forecast by {field1['field_name']} and {field2['field_name']}"
                            })

                    # Generate common 3-field combinations
                    three_field_combos = []
                    for i, field1 in enumerate(base_fields):
                        for j, field2 in enumerate(base_fields[i+1:], i+1):
                            for field3 in base_fields[j+1:]:
                                combo_name = f"{field1['field_name']}-{field2['field_name']}-{field3['field_name']}"
                                three_field_combos.append({
                                    "level_name": combo_name,
                                    "fields": [field1["field_name"], field2["field_name"], field3["field_name"]],
                                    "description": f"Forecast by {field1['field_name']}, {field2['field_name']}, and {field3['field_name']}"
                                })

                    # Limit combinations to reasonable numbers
                    two_field_combos = two_field_combos[:10]
                    three_field_combos = three_field_combos[:5]  # Fewer 3-field combos due to higher complexity

                    # Combine multi-dimension levels
                    multi_dimension_levels = two_field_combos + three_field_combos

                    return {
                        "single_dimension": single_levels,
                        "multi_dimension": multi_dimension_levels,
                        "all_fields": [f["field_name"] for f in base_fields]
                    }
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"Failed to get aggregation levels: {str(e)}")
            return []

    @staticmethod
    def create_forecast_run(
        tenant_id: str,
        database_name: str,
        request: ForecastRunCreate,
        user_email: str
    ) -> Dict[str, Any]:
        """
        Create a new forecast run.
        
        Args:
            tenant_id: Tenant identifier
            database_name: Tenant's database name
            request: Forecast run creation request
            user_email: Email of user creating the run
            
        Returns:
            Created forecast run details
        """
        forecast_run_id = str(uuid.uuid4())
        db_manager = get_db_manager()

        try:
            # Validate forecast dates
            if request.forecast_end <= request.forecast_start:
                raise ValidationException("Forecast end date must be after start date")

            # Validate version exists
            ForecastingService._validate_version_exists(
                tenant_id, database_name, request.version_id
            )

            # Validate algorithms exist
            algorithm_ids = [algo.algorithm_id for algo in request.algorithms]
            ForecastingService._validate_algorithms_exist(
                database_name, algorithm_ids
            )

            # Validate aggregation level if specified in filters
            if request.forecast_filters:
                agg_level = request.forecast_filters.get("aggregation_level")
                if agg_level:
                    # Validate aggregation level exists in tenant's master data
                    ForecastingService._validate_aggregation_level(
                        tenant_id, database_name, agg_level
                    )

                interval = request.forecast_filters.get("interval")
                if interval and interval not in ForecastingService.FORECAST_INTERVALS:
                    raise ValidationException(
                        f"Invalid forecast interval. Must be one of: {ForecastingService.FORECAST_INTERVALS}"
                    )

            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Insert forecast run
                    cursor.execute("""
                        INSERT INTO forecast_runs
                        (forecast_run_id, version_id, forecast_filters,
                         forecast_start, forecast_end, history_start, history_end,
                         run_status, created_by)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        forecast_run_id,
                        str(request.version_id),
                        Json(request.forecast_filters) if request.forecast_filters is not None else None,
                        request.forecast_start,
                        request.forecast_end,
                        request.history_start,
                        request.history_end,
                        "Pending",
                        user_email
                    ))

                    # Insert algorithm mappings
                    for algo in request.algorithms:
                        mapping_id = str(uuid.uuid4())
                        
                        # Get algorithm name
                        cursor.execute(
                            "SELECT algorithm_name FROM algorithms WHERE algorithm_id = %s",
                            (algo.algorithm_id,)
                        )
                        algo_name = cursor.fetchone()[0]

                        cursor.execute("""
                            INSERT INTO forecast_algorithms_mapping
                            (mapping_id, forecast_run_id, algorithm_id, 
                             algorithm_name, custom_parameters, execution_order, created_by)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            mapping_id,
                            forecast_run_id,
                            algo.algorithm_id,
                            algo_name,
                            Json(algo.custom_parameters) if algo.custom_parameters is not None else None,
                            algo.execution_order,
                            user_email
                        ))

                    # Log audit entry
                    cursor.execute("""
                        INSERT INTO forecast_audit_log
                        (forecast_run_id, action, entity_type, 
                         entity_id, performed_by, details)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        forecast_run_id,
                        "Created",
                        "ForecastRun",
                        forecast_run_id,
                        user_email,
                        Json({"filters": request.forecast_filters})
                    ))

                    conn.commit()
                    logger.info(f"Forecast run created: {forecast_run_id}")

                    # Fetch and return created run
                    return ForecastingService.get_forecast_run(
                        tenant_id, database_name, forecast_run_id
                    )

                finally:
                    cursor.close()

        except (ValidationException, NotFoundException):
            raise
        except Exception as e:
            logger.error(f"Failed to create forecast run: {str(e)}")
            raise DatabaseException(f"Failed to create forecast run: {str(e)}")

    @staticmethod
    def get_forecast_run(
        tenant_id: str,
        database_name: str,
        forecast_run_id: str
    ) -> Dict[str, Any]:
        """Get forecast run details by ID."""
        db_manager = get_db_manager()

        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Get forecast run
                    cursor.execute("""
                        SELECT forecast_run_id, version_id, forecast_filters,
                               forecast_start, forecast_end, history_start, history_end,
                               run_status, run_progress,
                               total_records, processed_records,
                               failed_records, error_message, created_at, updated_at,
                               started_at, completed_at, created_by
                        FROM forecast_runs
                        WHERE forecast_run_id = %s
                    """, (forecast_run_id,))

                    result = cursor.fetchone()
                    if not result:
                        raise NotFoundException("Forecast Run", forecast_run_id)

                    (run_id, version_id, filters, start, end, hist_start, hist_end,
                     status, progress, total, processed, failed, error_msg,
                     created_at, updated_at, started_at, completed_at, created_by) = result

                    # Get algorithm mappings
                    cursor.execute("""
                        SELECT mapping_id, algorithm_id, algorithm_name, custom_parameters,
                               execution_order, execution_status, started_at, completed_at,
                               error_message, created_at, updated_at
                        FROM forecast_algorithms_mapping
                        WHERE forecast_run_id = %s
                        ORDER BY execution_order
                    """, (forecast_run_id,))

                    algorithms = []
                    for row in cursor.fetchall():
                        algorithms.append({
                            "mapping_id": str(row[0]),
                            "forecast_run_id": forecast_run_id,
                            "algorithm_id": row[1],
                            "algorithm_name": row[2],
                            "custom_parameters": row[3],
                            "execution_order": row[4],
                            "execution_status": row[5],
                            "started_at": row[6].isoformat() if row[6] else None,
                            "completed_at": row[7].isoformat() if row[7] else None,
                            "error_message": row[8],
                            "created_at": row[9].isoformat() if row[9] else None,
                            "updated_at": row[10].isoformat() if row[10] else None
                        })

                    return {
                        "forecast_run_id": str(run_id),
                        "version_id": str(version_id),
                        "forecast_filters": filters,
                        "forecast_start": start.isoformat() if start else None,
                        "forecast_end": end.isoformat() if end else None,
                        "history_start": hist_start.isoformat() if hist_start else None,
                        "history_end": hist_end.isoformat() if hist_end else None,
                        "run_status": status,
                        "run_progress": progress,
                        "total_records": total,
                        "processed_records": processed,
                        "failed_records": failed,
                        "error_message": error_msg,
                        "created_at": created_at.isoformat() if created_at else None,
                        "updated_at": updated_at.isoformat() if updated_at else None,
                        "started_at": started_at.isoformat() if started_at else None,
                        "completed_at": completed_at.isoformat() if completed_at else None,
                        "created_by": created_by,
                        "algorithms": algorithms
                    }

                finally:
                    cursor.close()

        except NotFoundException:
            raise
        except Exception as e:
            logger.error(f"Failed to get forecast run: {str(e)}")
            raise DatabaseException(f"Failed to get forecast run: {str(e)}")

    @staticmethod
    def list_forecast_runs(
        tenant_id: str,
        database_name: str,
        version_id: Optional[str] = None,
        status: Optional[str] = None,
        page: int = 1,
        page_size: int = 50
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List forecast runs with optional filters."""
        db_manager = get_db_manager()

        try:
            where_clauses = ["1=1"]
            params = []

            if version_id:
                where_clauses.append("version_id = %s")
                params.append(version_id)
            
            if status:
                where_clauses.append("run_status = %s")
                params.append(status)

            where_sql = " AND ".join(where_clauses)

            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Get total count
                    cursor.execute(f"SELECT COUNT(*) FROM forecast_runs WHERE {where_sql}", params)
                    total_count = cursor.fetchone()[0]

                    # Get paginated results
                    offset = (page - 1) * page_size
                    cursor.execute(f"""
                        SELECT forecast_run_id, version_id, forecast_start, forecast_end,
                               run_status, run_progress, total_records, processed_records,
                               failed_records, created_at, created_by
                        FROM forecast_runs
                        WHERE {where_sql}
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s
                    """, params + [page_size, offset])

                    runs = []
                    for row in cursor.fetchall():
                        runs.append({
                            "forecast_run_id": str(row[0]),
                            "version_id": str(row[1]),
                            "forecast_start": row[2].isoformat() if row[2] else None,
                            "forecast_end": row[3].isoformat() if row[3] else None,
                            "run_status": row[4],
                            "run_progress": row[5],
                            "total_records": row[6],
                            "processed_records": row[7],
                            "failed_records": row[8],
                            "created_at": row[9].isoformat() if row[9] else None,
                            "created_by": row[10]
                        })

                    return runs, total_count

                finally:
                    cursor.close()

        except Exception as e:
            logger.error(f"Failed to list forecast runs: {str(e)}")
            raise DatabaseException(f"Failed to list forecast runs: {str(e)}")

    @staticmethod
    def get_aggregation_combinations(
        tenant_id: str,
        database_name: str,
        aggregation_level: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all distinct combinations of aggregation level fields from filtered data.
        
        Args:
            tenant_id: Tenant identifier
            database_name: Tenant's database name
            aggregation_level: Aggregation level string (e.g., "customer-location")
            filters: Filters to apply before finding combinations
            
        Returns:
            List of dictionaries, each containing one unique combination of aggregation fields
        """
        db_manager = get_db_manager()
        
        try:
            # Get aggregation columns
            agg_columns = ForecastingService._get_aggregation_columns(
                tenant_id, database_name, aggregation_level
            )
            
            # Build filter clause
            filter_clause = ""
            filter_params = []
            if filters:
                filter_conditions = []
                for col, val in filters.items():
                    if col not in ["aggregation_level", "interval", "selected_external_factors"]:
                        if isinstance(val, list):
                            placeholders = ', '.join(['%s'] * len(val))
                            filter_conditions.append(f'm."{col}" IN ({placeholders})')
                            filter_params.extend(val)
                        else:
                            filter_conditions.append(f'm."{col}" = %s')
                            filter_params.append(val)
                
                if filter_conditions:
                    filter_clause = "AND " + " AND ".join(filter_conditions)
            
            with db_manager.get_tenant_connection(database_name) as conn:
                # Get distinct combinations
                query = f"""
                    SELECT DISTINCT {', '.join([f'm."{col}"' for col in agg_columns])}
                    FROM sales_data s
                    JOIN master_data m ON s.master_id = m.master_id
                    WHERE 1=1 {filter_clause}
                    ORDER BY {', '.join([f'm."{col}"' for col in agg_columns])}
                """
                
                logger.info(f"Fetching aggregation combinations for level: {aggregation_level}")
                logger.debug(f"SQL: {query}")
                logger.debug(f"Params: {filter_params}")
                
                df = pd.read_sql_query(query, conn, params=filter_params)
                
                # Convert rows to list of dictionaries
                combinations = []
                for _, row in df.iterrows():
                    combo = {col: row[col] for col in agg_columns}
                    combinations.append(combo)
                
                logger.info(f"Found {len(combinations)} distinct aggregation combinations")
                return combinations
        
        except Exception as e:
            logger.error(f"Failed to get aggregation combinations: {str(e)}")
            raise DatabaseException(f"Failed to get aggregation combinations: {str(e)}")

    @staticmethod
    def _get_dimension_fields(tenant_id: str, database_name: str) -> List[str]:
        """
        Get all dimension fields (non-characteristic, non-target, non-date) from field catalogue.
        These are the fields that can be used for aggregation.
        
        Args:
            tenant_id: Tenant identifier
            database_name: Tenant's database name
            
        Returns:
            List of dimension field names
        """
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Get finalized field catalogue
                    cursor.execute("""
                        SELECT fields_json
                        FROM field_catalogue
                        WHERE status = 'FINALIZED'
                        ORDER BY created_at DESC
                        LIMIT 1
                    """)
                    
                    result = cursor.fetchone()
                    if not result:
                        logger.warning("No finalized field catalogue found")
                        return []
                    
                    import json
                    fields_json = result[0]
                    if isinstance(fields_json, str):
                        fields = json.loads(fields_json)
                    else:
                        fields = fields_json
                    
                    # Get all base dimension fields (non-characteristic, non-target, non-date)
                    dimension_fields = []
                    for field in fields:
                        is_characteristic = field.get("is_characteristic", False)
                        is_target = field.get("is_target_variable", False)
                        is_date = field.get("is_date_field", False)
                        
                        # Only include base dimensions (not characteristics, target, or date)
                        if not is_characteristic and not is_target and not is_date:
                            dimension_fields.append(field["field_name"])
                    
                    logger.info(f"Retrieved {len(dimension_fields)} dimension fields: {dimension_fields}")
                    return dimension_fields
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"Failed to get dimension fields: {str(e)}")
            return []
    @staticmethod
    def prepare_aggregated_data(
        tenant_id: str,
        database_name: str,
        aggregation_level: str,
        interval: str,
        filters: Optional[Dict[str, Any]] = None,
        specific_combination: Optional[Dict[str, Any]] = None,
        history_start: Optional[str] = None,
        history_end: Optional[str] = None
    ) -> Tuple[pd.DataFrame, str]:
        """
        Prepare aggregated historical data for forecasting.
        
        Args:
            tenant_id: Tenant identifier
            database_name: Tenant's database name
            aggregation_level: Aggregation level (e.g., "customer-location")
            interval: Forecast interval (WEEKLY, MONTHLY, etc.)
            filters: Base filters to apply
            specific_combination: If provided, filter to this specific aggregation combination
                                 (e.g., {"customer": "CUST001", "location": "LOC001"})
            history_start: Start date for historical data
            history_end: End date for historical data
        
        Returns:
            Tuple of (DataFrame with aggregated data, date_field_name)
        """
        db_manager = get_db_manager()

        try:
            # Get dynamic field names from metadata
            target_field_name, date_field_name = ForecastingService._get_field_names(
                tenant_id, database_name
            )
            
            logger.info(f"Using dynamic fields - Target: {target_field_name}, Date: {date_field_name}")
            
            # Get aggregation columns
            agg_columns = ForecastingService._get_aggregation_columns(
                tenant_id, database_name, aggregation_level
            )
            
            # Build filter clause with base filters
            filter_clause = ""
            filter_params = []
            if filters:
                filter_conditions = []
                for col, val in filters.items():
                    if col not in ["aggregation_level", "interval", "selected_external_factors"]:
                        if isinstance(val, list):
                            placeholders = ', '.join(['%s'] * len(val))
                            filter_conditions.append(f'm."{col}" IN ({placeholders})')
                            filter_params.extend(val)
                        else:
                            filter_conditions.append(f'm."{col}" = %s')
                            filter_params.append(val)
                
                if filter_conditions:
                    filter_clause = "AND " + " AND ".join(filter_conditions)
            
            # ✅ NEW: Add specific combination filters if provided
            if specific_combination:
                for col, val in specific_combination.items():
                    filter_clause += f' AND m."{col}" = %s'
                    filter_params.append(val)

            # ✅ NEW: Add history date range filters if provided
            if history_start:
                filter_clause += f' AND s."{date_field_name}" >= %s'
                filter_params.append(history_start)
            
            if history_end:
                filter_clause += f' AND s."{date_field_name}" <= %s'
                filter_params.append(history_end)

            with db_manager.get_tenant_connection(database_name) as conn:
                # Build SQL based on interval - using DYNAMIC field names
                date_trunc = ForecastingService._get_date_trunc_expr(interval, date_field_name)
                
                query = f"""
                    SELECT
                        {date_trunc} as "{date_field_name}",
                        {', '.join([f'm."{col}"' for col in agg_columns])},
                        SUM(CAST(s."{target_field_name}" AS numeric)) as total_quantity,
                        COUNT(DISTINCT s.sales_id) as transaction_count,
                        AVG(s.unit_price) as avg_price
                    FROM sales_data s
                    JOIN master_data m ON s.master_id = m.master_id
                    WHERE 1=1 {filter_clause}
                    GROUP BY {date_trunc}, {', '.join([f'm."{col}"' for col in agg_columns])}
                    ORDER BY "{date_field_name}"
                """

                logger.info(f"Executing query with filters: {filters}, combination: {specific_combination}")
                logger.debug(f"SQL: {query}")
                logger.debug(f"Params: {filter_params}")

                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    df = pd.read_sql_query(query, conn, params=filter_params)

                logger.info(f"Prepared {len(df)} aggregated records for forecasting")
                return df, date_field_name

        except Exception as e:
            logger.error(f"Failed to prepare aggregated data: {str(e)}")
            raise DatabaseException(f"Failed to prepare data: {str(e)}")

    @staticmethod
    def _get_aggregation_columns(
        tenant_id: str,
        database_name: str,
        aggregation_level: str
    ) -> List[str]:
        """
        Get master data columns for aggregation level dynamically.
        
        Args:
            tenant_id: Tenant identifier
            database_name: Tenant's database name
            aggregation_level: Aggregation level string (e.g., "product" or "product-location")
        
        Returns:
            List of column names to aggregate by
        """
        # Parse aggregation level - can be single field or hyphen-separated combination
        fields = [field.strip() for field in aggregation_level.split('-')]
        
        # Validate fields exist in master data
        db_manager = get_db_manager()
        with db_manager.get_tenant_connection(database_name) as conn:
            cursor = conn.cursor()
            try:
                # Get available columns from master_data table
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'master_data' 
                    AND table_schema = 'public'
                    AND column_name NOT IN ('master_id', 'tenant_id', 'created_at', 
                                           'created_by', 'updated_at', 'updated_by')
                """)
                
                available_columns = [row[0] for row in cursor.fetchall()]
                
                # Validate requested fields exist
                invalid_fields = [f for f in fields if f not in available_columns]
                if invalid_fields:
                    raise ValidationException(
                        f"Invalid aggregation fields: {invalid_fields}. "
                        f"Available fields: {available_columns}"
                    )
                
                return fields
                
            finally:
                cursor.close()

    @staticmethod
    def _validate_aggregation_level(
        tenant_id: str,
        database_name: str,
        aggregation_level: str
    ) -> None:
        """Validate that aggregation level contains valid fields."""
        try:
            ForecastingService._get_aggregation_columns(
                tenant_id, database_name, aggregation_level
            )
        except Exception as e:
            raise ValidationException(f"Invalid aggregation level: {str(e)}")

    @staticmethod
    def _get_date_trunc_expr(interval: str, date_field_name: str = "date") -> str:
        """
        Get SQL date truncation expression for interval.
        """
        interval_map = {
            "WEEKLY": f"DATE_TRUNC('week', s.\"{date_field_name}\"::timestamp)::date",
            "MONTHLY": f"DATE_TRUNC('month', s.\"{date_field_name}\"::timestamp)::date",
            "QUARTERLY": f"DATE_TRUNC('quarter', s.\"{date_field_name}\"::timestamp)::date",
            "YEARLY": f"DATE_TRUNC('year', s.\"{date_field_name}\"::timestamp)::date"
        }
        
        return interval_map.get(interval, f"DATE_TRUNC('month', s.\"{date_field_name}\"::timestamp)::date")

    @staticmethod
    def _validate_version_exists(
        tenant_id: str,
        database_name: str,
        version_id: str
    ) -> None:
        """Validate that forecast version exists."""
        db_manager = get_db_manager()
        
        with db_manager.get_tenant_connection(database_name) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "SELECT 1 FROM forecast_versions WHERE version_id = %s",
                    (version_id,)
                )
                if not cursor.fetchone():
                    raise NotFoundException("Forecast Version", version_id)
            finally:
                cursor.close()

    @staticmethod
    def _validate_algorithms_exist(
        database_name: str,
        algorithm_ids: List[int]
    ) -> None:
        """Validate that all algorithms exist."""
        db_manager = get_db_manager()
        
        with db_manager.get_tenant_connection(database_name) as conn:
            cursor = conn.cursor()
            try:
                placeholders = ', '.join(['%s'] * len(algorithm_ids))
                cursor.execute(
                    f"SELECT algorithm_id FROM algorithms WHERE algorithm_id IN ({placeholders})",
                    algorithm_ids
                )
                found_ids = [row[0] for row in cursor.fetchall()]
                
                missing_ids = set(algorithm_ids) - set(found_ids)
                if missing_ids:
                    raise ValidationException(
                        f"Algorithm(s) not found: {missing_ids}"
                    )
            finally:
                cursor.close()

    @staticmethod
    def _get_field_names(tenant_id: str, database_name: str) -> Tuple[str, str]:
        """
        Get target and date field names from field catalogue metadata.
        
        Args:
            tenant_id: Tenant identifier
            database_name: Tenant's database name
            
        Returns:
            Tuple of (target_field_name, date_field_name)
        """
        db_manager = get_db_manager()
        
        with db_manager.get_tenant_connection(database_name) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    SELECT date_field_name, target_field_name
                    FROM field_catalogue_metadata
                """)
                
                result = cursor.fetchone()
                if not result:
                    logger.error(f"No field catalogue metadata found for tenant {tenant_id}")
                    raise ValidationException(
                        "Field catalogue not finalized. Please finalize your field catalogue first."
                    )
                
                date_field, target_field = result[0], result[1]
                logger.debug(f"Retrieved field names - Date: '{date_field}', Target: '{target_field}'")
                return target_field, date_field
                
            finally:
                cursor.close()

    @staticmethod
    def save_forecast_results(
        tenant_id: str,
        database_name: str,
        forecast_run_id: str,
        user_email: str,
        entity_identifier: Optional[str] = None,
        aggregation_level: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Save future forecast results into forecast_data table.
        
        Args:
            tenant_id: Tenant identifier
            database_name: Tenant's database name
            forecast_run_id: The ID of the forecast run to save
            user_email: Email of user performing the action
            entity_identifier: Optional specific entity to save (hyphen-separated)
            aggregation_level: Required if entity_identifier is provided
            
        Returns:
            Status and count of saved records
        """
        db_manager = get_db_manager()
        
        try:
            target_field, date_field = ForecastingService._get_field_names(tenant_id, database_name)
            
            # If entity_identifier provided, parse it to filter metadata
            entity_fields = None
            if entity_identifier and aggregation_level:
                from app.core.forecast_comparison_service import ForecastComparisonService
                entity_fields = ForecastComparisonService.parse_entity_identifier(
                    entity_identifier, aggregation_level
                )
            
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Get best performing algorithm's mapping_id for this run
                    cursor.execute("""
                        SELECT fam.mapping_id
                        FROM forecast_algorithms_mapping fam
                        LEFT JOIN forecast_results fr ON fam.mapping_id = fr.mapping_id
                        WHERE fam.forecast_run_id = %s
                          AND fam.execution_status = 'Completed'
                        GROUP BY fam.mapping_id
                        ORDER BY AVG(fr.accuracy_metric) DESC
                        LIMIT 1
                    """, (forecast_run_id,))
                    
                    mapping_row = cursor.fetchone()
                    if not mapping_row:
                        raise NotFoundException("Forecast Results", forecast_run_id)
                    
                    mapping_id = mapping_row[0]

                    # Fetch future_forecast results
                    query = """
                        SELECT date, value, metadata
                        FROM forecast_results
                        WHERE mapping_id = %s AND type = 'future_forecast'
                    """
                    params = [mapping_id]
                    
                    # If we have specific entity fields, filter by metadata JSONB
                    if entity_fields:
                        import json
                        query += " AND metadata->'entity_filter' @> %s::jsonb"
                        params.append(json.dumps(entity_fields))
                    
                    cursor.execute(query, params)
                    results = cursor.fetchall()
                    
                    if not results:
                        return {
                            "status": "success",
                            "message": "No forecast results found matching criteria",
                            "saved_count": 0
                        }
                    
                    master_id_cache = {}
                    sales_info_cache = {}

                    saved_count = 0
                    for row in results:
                        forecast_date, forecast_value, metadata = row
                        if isinstance(metadata, str):
                            try:
                                metadata = json.loads(metadata)
                            except json.JSONDecodeError:
                                metadata = {}
                        entity_filter = (
                            metadata.get("entity_filter")
                            if isinstance(metadata, dict)
                            else None
                        )
                        if not entity_filter and entity_fields:
                            entity_filter = entity_fields
                        if not isinstance(entity_filter, dict) or not entity_filter:
                            raise ValidationException(
                                "Entity filter missing for forecast result"
                            )
                        entity_key = tuple(
                            sorted((k, str(v)) for k, v in entity_filter.items())
                        )
                        if entity_key not in master_id_cache:
                            master_id_cache[entity_key] = ForecastingService._resolve_master_id(
                                cursor,
                                entity_filter,
                            )
                        master_id = master_id_cache[entity_key]

                        if master_id not in sales_info_cache:
                            sales_info_cache[master_id] = ForecastingService._resolve_sales_info(
                                cursor,
                                master_id,
                            )
                        uom, unit_price = sales_info_cache[master_id]

                        # Save aggregated forecast record (one per date)
                        cursor.execute(f"""
                            UPDATE forecast_data
                            SET "{target_field}" = %s,
                                uom = %s,
                                unit_price = %s,
                                created_at = CURRENT_TIMESTAMP,
                                created_by = %s
                            WHERE master_id = %s
                              AND "{date_field}" = %s
                        """, (
                            forecast_value,
                            uom,
                            unit_price,
                            user_email,
                            master_id,
                            forecast_date,
                        ))
                        if cursor.rowcount == 0:
                            cursor.execute(f"""
                                INSERT INTO forecast_data
                                (master_id, "{date_field}", "{target_field}", uom, unit_price, created_by)
                                VALUES (%s, %s, %s, %s, %s, %s)
                            """, (
                                master_id,
                                forecast_date,
                                forecast_value,
                                uom,  # Fetched UOM
                                unit_price,  # Fetched unit price
                                user_email
                            ))
                        saved_count += 1
                    
                    conn.commit()
                    logger.info(f"Saved {saved_count} forecast records to forecast_data for run {forecast_run_id}")
                    
                    return {
                        "status": "success",
                        "message": f"Successfully saved {saved_count} forecast records",
                        "saved_count": saved_count
                    }
                    
                finally:
                    cursor.close()
        except Exception as e:
            logger.error(f"Failed to save forecast results: {str(e)}")
            if isinstance(e, (ValidationException, NotFoundException)):
                raise
            raise DatabaseException(f"Failed to save forecast results: {str(e)}")

    @staticmethod
    def _resolve_master_id(cursor, entity_filter: Dict[str, Any]) -> str:
        if not entity_filter:
            raise ValidationException("Entity filter is required for master data lookup")
        from psycopg2 import sql
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
