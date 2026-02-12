"""
Aggregation service.
Handles data aggregation, aggregation level management, and ratio calculations for forecasting.
"""

import json
import pandas as pd
import logging
import warnings
from typing import Dict, Any, List, Optional, Tuple

from app.core.database import get_db_manager
from app.core.exceptions import (
    DatabaseException,
    ValidationException,
    NotFoundException
)

logger = logging.getLogger(__name__)


class AggregationService:
    """Service for aggregation operations."""

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
            agg_columns = AggregationService._get_aggregation_columns(
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
        from app.core.forecasting_service import ForecastingService
        
        db_manager = get_db_manager()

        try:
            # Get dynamic field names from metadata
            target_field_name, date_field_name = ForecastingService._get_field_names(
                tenant_id, database_name
            )
            
            logger.info(f"Using dynamic fields - Target: {target_field_name}, Date: {date_field_name}")
            
            # Get aggregation columns
            agg_columns = AggregationService._get_aggregation_columns(
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
            
            # Add specific combination filters if provided
            if specific_combination:
                for col, val in specific_combination.items():
                    filter_clause += f' AND m."{col}" = %s'
                    filter_params.append(val)

            # Add history date range filters if provided
            if history_start:
                filter_clause += f' AND s."{date_field_name}" >= %s'
                filter_params.append(history_start)
            
            if history_end:
                filter_clause += f' AND s."{date_field_name}" <= %s'
                filter_params.append(history_end)

            with db_manager.get_tenant_connection(database_name) as conn:
                # Build SQL based on interval - using DYNAMIC field names
                date_trunc = AggregationService._get_date_trunc_expr(interval, date_field_name)
                
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

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    df = pd.read_sql_query(query, conn, params=filter_params)

                logger.info(f"Prepared {len(df)} aggregated records for forecasting")
                return df, date_field_name

        except Exception as e:
            logger.error(f"Failed to prepare aggregated data: {str(e)}")
            raise DatabaseException(f"Failed to prepare data: {str(e)}")

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
                    
                    return dimension_fields
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"Failed to get dimension fields: {str(e)}")
            return []

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
            AggregationService._get_aggregation_columns(
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
    def calculate_historical_ratios(
        tenant_id: str,
        database_name: str,
        source_aggregation_level: str,
        target_aggregation_level: str,
        interval: str = "MONTHLY",
        history_start: Optional[str] = None,
        history_end: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Calculate historical split ratios from source level to target level.
        
        Args:
            interval: Forecast interval (WEEKLY, MONTHLY, QUARTERLY, YEARLY) to match the forecast granularity
        """
        # Get target level aggregated data using the same interval as the forecast
        df_target, _ = AggregationService.prepare_aggregated_data(
            tenant_id=tenant_id,
            database_name=database_name,
            aggregation_level=target_aggregation_level,
            interval=interval,
            filters=filters,
            history_start=history_start,
            history_end=history_end
        )

        if df_target.empty:
            return pd.DataFrame()

        # Source columns
        source_cols = AggregationService._get_aggregation_columns(
            tenant_id, database_name, source_aggregation_level
        )
        
        # Target columns
        target_cols = AggregationService._get_aggregation_columns(
            tenant_id, database_name, target_aggregation_level
        )

        # Group by source columns and calculate total volume
        source_totals = df_target.groupby(source_cols)['total_quantity'].sum().reset_index()
        source_totals.rename(columns={'total_quantity': 'source_total_volume'}, inplace=True)

        # Group by target columns and calculate total volume
        target_totals = df_target.groupby(target_cols)['total_quantity'].sum().reset_index()
        target_totals.rename(columns={'total_quantity': 'target_total_volume'}, inplace=True)

        # Join to calculate ratios
        ratios_df = pd.merge(target_totals, source_totals, on=source_cols)
        ratios_df['allocation_factor'] = ratios_df['target_total_volume'] / ratios_df['source_total_volume']
        
        # Handle division by zero
        ratios_df['allocation_factor'] = ratios_df['allocation_factor'].fillna(0)

        return ratios_df
