"""
Forecast Comparison Service.
Handles comparison of multiple forecast scenarios for a specific entity.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import date
import pandas as pd
import numpy as np

from app.core.database import get_db_manager
from app.core.exceptions import ValidationException, NotFoundException, DatabaseException

logger = logging.getLogger(__name__)


class ForecastComparisonService:
    """Service for comparing forecast scenarios."""

    @staticmethod
    def parse_entity_identifier(
        entity_identifier: str,
        aggregation_level: str
    ) -> Dict[str, str]:
        """
        Parse entity identifier into field values.
        
        Example:
            entity_identifier="1001-Loc1"
            aggregation_level="product-location"
            Returns: {"product": "1001", "location": "Loc1"}
        
        Args:
            entity_identifier: Hyphen-separated entity values
            aggregation_level: Aggregation level definition
            
        Returns:
            Dictionary mapping field names to values
            
        Raises:
            ValidationException: If identifier doesn't match aggregation level
        """
        fields = aggregation_level.split('-')
        values = entity_identifier.split('-')
        
        if len(fields) != len(values):
            raise ValidationException(
                f"Entity identifier '{entity_identifier}' does not match "
                f"aggregation level '{aggregation_level}'. "
                f"Expected {len(fields)} values, got {len(values)}"
            )
        
        entity_dict = {}
        for field, value in zip(fields, values):
            entity_dict[field.strip()] = value.strip()
        
        logger.info(f"Parsed entity identifier: {entity_dict}")
        return entity_dict

    @staticmethod
    def get_historical_data(
        database_name: str,
        entity_fields: Dict[str, str],
        interval: str
    ) -> List[Dict[str, Any]]:
        """
        Get historical data for a specific entity.
        
        Args:
            database_name: Tenant's database name
            entity_fields: Dictionary of field:value pairs
            interval: Time interval (MONTHLY, WEEKLY, etc.)
            
        Returns:
            List of historical data points
        """
        db_manager = get_db_manager()
        
        try:
            # Get dynamic field names (target and date)
            target_field, date_field = ForecastComparisonService._get_field_names(
                database_name
            )
            
            # Build WHERE clause for entity fields
            where_conditions = []
            params = []

            for field, value in entity_fields.items():
                where_conditions.append(f'm."{field}" = %s')
                params.append(value)

            where_clause = " AND ".join(where_conditions)
            
            # Build date truncation based on interval
            date_trunc_expr = ForecastComparisonService._get_date_trunc_expr(interval, date_field)
            
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    query = f"""
                        SELECT
                            {date_trunc_expr} as period,
                            SUM(CAST(s."{target_field}" AS numeric)) as actual_quantity,
                            COUNT(DISTINCT s.sales_id) as transaction_count,
                            AVG(s.unit_price) as avg_price
                        FROM sales_data s
                        JOIN master_data m ON s.master_id = m.master_id
                        WHERE {where_clause}
                        GROUP BY period
                        ORDER BY period
                    """
                    
                    cursor.execute(query, params)
                    
                    historical_data = []
                    for row in cursor.fetchall():
                        historical_data.append({
                            "date": row[0].isoformat() if row[0] else None,
                            "actual_quantity": float(row[1]) if row[1] else 0,
                            "transaction_count": row[2] if row[2] else 0,
                            "avg_price": float(row[3]) if row[3] else None
                        })
                    
                    logger.info(f"Retrieved {len(historical_data)} historical data points")
                    return historical_data
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"Failed to get historical data: {str(e)}")
            raise DatabaseException(f"Failed to retrieve historical data: {str(e)}")

    @staticmethod
    def find_matching_forecast_runs(
        database_name: str,
        entity_fields: Dict[str, str],
        aggregation_level: str,
        interval: str
    ) -> List[Dict[str, Any]]:
        """
        Find all forecast runs matching the entity and criteria.
        
        Args:
            database_name: Tenant's database name
            entity_fields: Dictionary of field:value pairs
            aggregation_level: Aggregation level
            interval: Time interval
            
        Returns:
            List of matching forecast run metadata
        """
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Build JSONB filter conditions
                    jsonb_conditions = [
                        "fr.forecast_filters->>'aggregation_level' = %s",
                        "fr.forecast_filters->>'interval' = %s"
                    ]
                    params = [aggregation_level, interval]

                    # Add entity field filters
                    for field, value in entity_fields.items():
                        jsonb_conditions.append(f"fr.forecast_filters->>'{field}' = %s")
                        params.append(value)

                    jsonb_where = " AND ".join(jsonb_conditions)

                    query = f"""
                        SELECT
                            fr.forecast_run_id,
                            fr.version_id,
                            fv.version_name,
                            fv.version_type,
                            fr.forecast_filters,
                            fr.forecast_start,
                            fr.forecast_end,
                            fr.run_status,
                            fr.created_at,
                            fr.created_by
                        FROM forecast_runs fr
                        JOIN forecast_versions fv ON fr.version_id = fv.version_id
                        WHERE {jsonb_where}
                          AND fr.run_status = 'Completed'
                        ORDER BY fr.created_at DESC
                    """
                    
                    cursor.execute(query, params)
                    
                    forecast_runs = []
                    for row in cursor.fetchall():
                        forecast_runs.append({
                            "forecast_run_id": str(row[0]),
                            "version_id": str(row[1]),
                            "version_name": row[2],
                            "version_type": row[3],
                            "forecast_filters": row[4],
                            "forecast_start": row[5].isoformat() if row[5] else None,
                            "forecast_end": row[6].isoformat() if row[6] else None,
                            "run_status": row[7],
                            "created_at": row[8].isoformat() if row[8] else None,
                            "created_by": row[9]
                        })
                    
                    logger.info(f"Found {len(forecast_runs)} matching forecast runs")
                    return forecast_runs
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"Failed to find forecast runs: {str(e)}")
            raise DatabaseException(f"Failed to find matching forecast runs: {str(e)}")

    @staticmethod
    def get_forecast_metadata(
        database_name: str,
        forecast_run_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed metadata for forecast runs including algorithm and external factors.
        
        Args:
            database_name: Tenant's database name
            forecast_run_ids: List of forecast run IDs
            
        Returns:
            Dictionary mapping forecast_run_id to metadata
        """
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    metadata = {}
                    
                    for run_id in forecast_run_ids:
                        # Get best performing algorithm for this run
                        cursor.execute("""
                            SELECT 
                                fam.algorithm_id,
                                fam.algorithm_name,
                                fam.custom_parameters,
                                AVG(fr.accuracy_metric) as avg_accuracy
                            FROM forecast_algorithms_mapping fam
                            LEFT JOIN forecast_results fr ON fam.mapping_id = fr.mapping_id
                            WHERE fam.forecast_run_id = %s
                              AND fam.execution_status = 'Completed'
                            GROUP BY fam.algorithm_id, fam.algorithm_name, fam.custom_parameters
                            ORDER BY avg_accuracy DESC
                            LIMIT 1
                        """, (run_id,))
                        
                        algo_row = cursor.fetchone()
                        
                        if not algo_row:
                            logger.warning(f"No completed algorithms found for run {run_id}")
                            continue
                        
                        # Get forecast run details
                        cursor.execute("""
                            SELECT 
                                fr.forecast_filters,
                                fr.forecast_start,
                                fr.forecast_end
                            FROM forecast_runs fr
                            WHERE fr.forecast_run_id = %s
                        """, (run_id,))
                        
                        run_row = cursor.fetchone()
                        
                        if not run_row:
                            continue
                        
                        forecast_filters = run_row[0] or {}
                        selected_factors = forecast_filters.get('selected_external_factors', [])
                        
                        metadata[run_id] = {
                            "algorithm_id": algo_row[0],
                            "algorithm_name": algo_row[1],
                            "custom_parameters": algo_row[2],
                            "avg_accuracy": float(algo_row[3]) if algo_row[3] else None,
                            "external_factors": selected_factors if selected_factors else [],
                            "forecast_start": run_row[1].isoformat() if run_row[1] else None,
                            "forecast_end": run_row[2].isoformat() if run_row[2] else None
                        }
                    
                    logger.info(f"Retrieved metadata for {len(metadata)} forecast runs")
                    return metadata
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"Failed to get forecast metadata: {str(e)}")
            raise DatabaseException(f"Failed to retrieve forecast metadata: {str(e)}")

    @staticmethod
    def get_forecast_results(
        database_name: str,
        forecast_run_ids: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get forecast results for multiple runs (best performing algorithm only).
        
        Args:
            database_name: Tenant's database name
            forecast_run_ids: List of forecast run IDs
            
        Returns:
            Dictionary mapping forecast_run_id to list of forecast data points
        """
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    results = {}
                    
                    for run_id in forecast_run_ids:
                        # Get best performing algorithm's mapping_id
                        cursor.execute("""
                            SELECT fam.mapping_id
                            FROM forecast_algorithms_mapping fam
                            LEFT JOIN forecast_results fr ON fam.mapping_id = fr.mapping_id
                            WHERE fam.forecast_run_id = %s
                              AND fam.execution_status = 'Completed'
                            GROUP BY fam.mapping_id
                            ORDER BY AVG(fr.accuracy_metric) DESC
                            LIMIT 1
                        """, (run_id,))
                        
                        mapping_row = cursor.fetchone()
                        
                        if not mapping_row:
                            logger.warning(f"No results found for run {run_id}")
                            results[run_id] = []
                            continue
                        
                        mapping_id = mapping_row[0]
                        
                        # Get forecast results
                        cursor.execute("""
                            SELECT
                                date,
                                value,
                                type,
                                accuracy_metric,
                                confidence_interval_lower,
                                confidence_interval_upper
                            FROM forecast_results
                            WHERE mapping_id = %s
                            ORDER BY date
                        """, (mapping_id,))

                        forecast_data = []
                        for row in cursor.fetchall():
                            forecast_data.append({
                                "date": row[0].isoformat() if row[0] else None,
                                "forecast_quantity": float(row[1]) if row[1] else 0,
                                "type": row[2],
                                "accuracy_metric": float(row[3]) if row[3] else None,
                                "confidence_interval_lower": float(row[4]) if row[4] else None,
                                "confidence_interval_upper": float(row[5]) if row[5] else None
                            })
                        
                        results[run_id] = forecast_data
                        logger.info(f"Retrieved {len(forecast_data)} forecast points for run {run_id}")
                    
                    return results
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"Failed to get forecast results: {str(e)}")
            raise DatabaseException(f"Failed to retrieve forecast results: {str(e)}")

    @staticmethod
    def calculate_comparison_metrics(
        forecast_data: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Calculate pairwise comparison metrics between forecasts.
        
        Args:
            forecast_data: Dictionary mapping forecast_run_id to forecast values
            
        Returns:
            Comparison metrics including differences and correlations
        """
        try:
            run_ids = list(forecast_data.keys())
            
            if len(run_ids) < 2:
                return {
                    "period_overlap": None,
                    "pairwise_differences": []
                }
            
            # Find common date range
            all_dates = set()
            for run_id in run_ids:
                dates = {point['date'] for point in forecast_data[run_id]}
                if not all_dates:
                    all_dates = dates
                else:
                    all_dates = all_dates.intersection(dates)
            
            if not all_dates:
                return {
                    "period_overlap": None,
                    "pairwise_differences": []
                }
            
            sorted_dates = sorted(all_dates)
            
            # Calculate pairwise comparisons
            pairwise_differences = []
            
            for i in range(len(run_ids)):
                for j in range(i + 1, len(run_ids)):
                    run_1 = run_ids[i]
                    run_2 = run_ids[j]
                    
                    # Get aligned values
                    values_1 = []
                    values_2 = []
                    
                    data_1_map = {p['date']: p['forecast_quantity'] for p in forecast_data[run_1]}
                    data_2_map = {p['date']: p['forecast_quantity'] for p in forecast_data[run_2]}
                    
                    for date_str in sorted_dates:
                        if date_str in data_1_map and date_str in data_2_map:
                            values_1.append(data_1_map[date_str])
                            values_2.append(data_2_map[date_str])
                    
                    if not values_1 or not values_2:
                        continue
                    
                    # Convert to numpy arrays
                    arr_1 = np.array(values_1)
                    arr_2 = np.array(values_2)
                    
                    # Calculate metrics
                    differences = arr_2 - arr_1
                    avg_difference = float(np.mean(differences))
                    avg_abs_difference = float(np.mean(np.abs(differences)))
                    
                    # Percentage difference
                    avg_val_1 = float(np.mean(arr_1))
                    percentage_difference = (avg_difference / avg_val_1 * 100) if avg_val_1 != 0 else 0
                    
                    # Correlation
                    correlation = float(np.corrcoef(arr_1, arr_2)[0, 1]) if len(arr_1) > 1 else 1.0
                    
                    # RMSE between forecasts
                    rmse = float(np.sqrt(np.mean(differences ** 2)))
                    
                    pairwise_differences.append({
                        "forecast_1": run_1,
                        "forecast_2": run_2,
                        "average_difference": round(avg_difference, 2),
                        "average_absolute_difference": round(avg_abs_difference, 2),
                        "percentage_difference": round(percentage_difference, 2),
                        "correlation": round(correlation, 4),
                        "rmse": round(rmse, 2),
                        "comparison_points": len(values_1)
                    })
            
            return {
                "period_overlap": {
                    "start": sorted_dates[0] if sorted_dates else None,
                    "end": sorted_dates[-1] if sorted_dates else None,
                    "total_periods": len(sorted_dates)
                },
                "pairwise_differences": pairwise_differences
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate comparison metrics: {str(e)}")
            return {
                "period_overlap": None,
                "pairwise_differences": []
            }

    @staticmethod
    def compare_forecasts(
        database_name: str,
        entity_identifier: str,
        aggregation_level: str,
        interval: str,
        forecast_run_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Main comparison method - orchestrates all comparison steps.
        
        Args:
            database_name: Tenant's database name
            entity_identifier: Entity identifier (e.g., "1001-Loc1")
            aggregation_level: Aggregation level (e.g., "product-location")
            interval: Time interval (e.g., "MONTHLY")
            forecast_run_ids: Optional list of specific runs to compare
            
        Returns:
            Complete comparison data structure
        """
        logger.info(
            f"Starting forecast comparison for entity '{entity_identifier}' "
            f"at level '{aggregation_level}' with interval '{interval}'"
        )
        
        # Step 1: Parse entity identifier
        entity_fields = ForecastComparisonService.parse_entity_identifier(
            entity_identifier, aggregation_level
        )
        
        # Step 2: Get historical data
        historical_data = ForecastComparisonService.get_historical_data(
            database_name, entity_fields, interval
        )
        
        # Step 3: Find matching forecast runs
        matching_runs = ForecastComparisonService.find_matching_forecast_runs(
            database_name, entity_fields, aggregation_level, interval
        )
        
        if not matching_runs:
            raise NotFoundException(
                "Forecast Runs",
                f"No completed forecast runs found for entity '{entity_identifier}'"
            )
        
        # Step 4: Filter to specific runs if requested
        if forecast_run_ids:
            matching_run_ids = {run['forecast_run_id'] for run in matching_runs}
            invalid_ids = set(forecast_run_ids) - matching_run_ids
            
            if invalid_ids:
                raise ValidationException(
                    f"Invalid forecast_run_ids: {invalid_ids}. "
                    f"These runs don't match the specified entity."
                )
            
            matching_runs = [
                run for run in matching_runs 
                if run['forecast_run_id'] in forecast_run_ids
            ]
        
        run_ids = [run['forecast_run_id'] for run in matching_runs]
        
        # Step 5: Get forecast metadata
        forecast_metadata = ForecastComparisonService.get_forecast_metadata(
            database_name, run_ids
        )
        
        # Step 6: Get forecast results
        forecast_data = ForecastComparisonService.get_forecast_results(
            database_name, run_ids
        )
        
        # Step 7: Calculate comparison metrics
        comparison_metrics = ForecastComparisonService.calculate_comparison_metrics(
            forecast_data
        )
        
        # Step 8: Enrich forecast metadata
        available_forecasts = []
        for run in matching_runs:
            run_id = run['forecast_run_id']
            metadata = forecast_metadata.get(run_id, {})
            
            # Build forecast name
            forecast_name = f"{run['version_name']} - {metadata.get('algorithm_name', 'Unknown')}"
            if metadata.get('external_factors'):
                forecast_name += f" (with {len(metadata['external_factors'])} factors)"
            
            available_forecasts.append({
                "forecast_run_id": run_id,
                "forecast_name": forecast_name,
                "version_name": run['version_name'],
                "version_type": run['version_type'],
                "algorithm_name": metadata.get('algorithm_name'),
                "algorithm_id": metadata.get('algorithm_id'),
                "external_factors": metadata.get('external_factors', []),
                "forecast_period": {
                    "start": metadata.get('forecast_start') or run['forecast_start'],
                    "end": metadata.get('forecast_end') or run['forecast_end']
                },
                "accuracy_metrics": {
                    "accuracy": metadata.get('avg_accuracy')
                },
                "created_at": run['created_at'],
                "created_by": run['created_by']
            })
        
        # Step 9: Build final response
        response = {
            "entity": {
                "identifier": entity_identifier,
                "aggregation_level": aggregation_level,
                "interval": interval,
                "field_values": entity_fields
            },
            "historical_data": historical_data,
            "available_forecasts": available_forecasts,
            "forecast_data": forecast_data,
            "comparison_matrix": comparison_metrics
        }
        
        logger.info(
            f"Comparison completed: {len(available_forecasts)} forecasts, "
            f"{len(historical_data)} historical points"
        )
        
        return response

    @staticmethod
    def _get_field_names(database_name: str) -> Tuple[str, str]:
        """Get target and date field names from metadata."""
        db_manager = get_db_manager()

        with db_manager.get_tenant_connection(database_name) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    SELECT target_field_name, date_field_name
                    FROM field_catalogue_metadata
                """)

                result = cursor.fetchone()
                if not result or len(result) < 2:
                    raise ValidationException(
                        "Field catalogue not finalized. Please finalize your field catalogue first."
                    )

                return result[0], result[1]
            finally:
                cursor.close()

    @staticmethod
    def _get_date_trunc_expr(interval: str, date_field_name: str) -> str:
        """Get SQL date truncation expression."""
        interval_map = {
            "WEEKLY": f"DATE_TRUNC('week', s.\"{date_field_name}\"::timestamp)::date",
            "MONTHLY": f"DATE_TRUNC('month', s.\"{date_field_name}\"::timestamp)::date",
            "QUARTERLY": f"DATE_TRUNC('quarter', s.\"{date_field_name}\"::timestamp)::date",
            "YEARLY": f"DATE_TRUNC('year', s.\"{date_field_name}\"::timestamp)::date"
        }
        
        return interval_map.get(interval, f"DATE_TRUNC('month', s.\"{date_field_name}\"::timestamp)::date")