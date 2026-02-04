"""
Enhanced External Factors Service with FRED Integration.
Works with existing tenant-specific external_factors tables.
"""

import uuid
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
import logging
import pandas as pd
from psycopg2.extras import Json

from app.core.database import get_db_manager
from app.core.exceptions import DatabaseException, NotFoundException, ValidationException
from app.schemas.forecasting import ExternalFactorCreate

logger = logging.getLogger(__name__)


class ExternalFactorsService:
    """Enhanced service for external factors with FRED integration."""

    @staticmethod
    def create_factor(
        tenant_id: str,
        database_name: str,
        request: ExternalFactorCreate,
        user_email: str
    ) -> Dict[str, Any]:
        """
        Create or update a single external factor record.
        Uses upsert on (factor_name, date).
        """
        db_manager = get_db_manager()

        try:
            try:
                record_date = datetime.fromisoformat(request.date).date()
            except ValueError:
                raise ValidationException("date must be in YYYY-MM-DD format")

            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    factor_id = str(uuid.uuid4())
                    cursor.execute("""
                        INSERT INTO external_factors
                        (factor_id, date, factor_name, factor_value, unit, source, created_by, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (factor_name, date)
                        DO UPDATE SET
                            factor_value = EXCLUDED.factor_value,
                            unit = EXCLUDED.unit,
                            source = EXCLUDED.source,
                            updated_at = %s,
                            updated_by = %s,
                            deleted_at = NULL
                    """, (
                        factor_id,
                        record_date,
                        request.factor_name,
                        request.factor_value,
                        request.unit,
                        request.source,
                        user_email,
                        datetime.utcnow(),
                        datetime.utcnow(),
                        user_email
                    ))
                    conn.commit()

                    # Fetch and return the upserted record
                    cursor.execute("""
                        SELECT factor_id, date, factor_name, factor_value,
                               unit, source, created_at, created_by, updated_at, updated_by
                        FROM external_factors
                        WHERE factor_name = %s AND date = %s
                    """, (request.factor_name, record_date))
                    row = cursor.fetchone()
                    if not row:
                        raise DatabaseException("Failed to retrieve created factor")

                    return {
                        "factor_id": str(row[0]),
                        "date": row[1].isoformat() if row[1] else None,
                        "factor_name": row[2],
                        "factor_value": float(row[3]) if row[3] is not None else None,
                        "unit": row[4],
                        "source": row[5],
                        "created_at": row[6].isoformat() if row[6] else None,
                        "created_by": row[7],
                        "updated_at": row[8].isoformat() if row[8] else None,
                        "updated_by": row[9]
                    }

                finally:
                    cursor.close()

        except (ValidationException, NotFoundException):
            raise
        except Exception as e:
            logger.error(f"Failed to create external factor: {str(e)}")
            raise DatabaseException(f"Failed to create external factor: {str(e)}")

    @staticmethod
    def bulk_import_from_fred(
        tenant_id: str,
        database_name: str,
        series_configs: List[Dict[str, Any]],
        start_date: date,
        end_date: date,
        user_email: str,
        fred_api_key: str
    ) -> Dict[str, Any]:
        """
        Bulk import external factors from FRED API into tenant's external_factors table.
        
        Args:
            tenant_id: Tenant identifier
            database_name: Tenant's database name
            series_configs: List of FRED series configurations
            start_date: Start date for data
            end_date: End date for data
            user_email: User performing import
            fred_api_key: FRED API key
            
        Returns:
            Import summary with success/failure counts
        """
        from app.core.fred_api_service import FREDAPIService
        
        fred_service = FREDAPIService(fred_api_key)
        db_manager = get_db_manager()
        
        total_imported = 0
        total_updated = 0
        failed_series = []
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    for config in series_configs:
                        series_id = config.get("series_id")
                        factor_name = config.get("factor_name", series_id)
                        
                        try:
                            logger.info(f"Importing FRED series {series_id} as '{factor_name}'")
                            
                            # Fetch data from FRED
                            df = fred_service.get_series_data(
                                series_id=series_id,
                                start_date=start_date,
                                end_date=end_date
                            )
                            
                            if df.empty:
                                logger.warning(f"No data returned for series {series_id}")
                                failed_series.append({
                                    "series_id": series_id,
                                    "error": "No data available for date range"
                                })
                                continue
                            
                            # Get series info for metadata
                            series_info = fred_service.get_series_info(series_id)
                            
                            # Bulk insert with conflict handling
                            inserted = 0
                            updated = 0
                            
                            for _, row in df.iterrows():
                                factor_id = str(uuid.uuid4())
                                record_date = row['date'].date()
                                
                                # Try insert, update on conflict
                                cursor.execute("""
                                    INSERT INTO external_factors
                                    (factor_id, date, factor_name, factor_value,
                                     unit, source, created_by, created_at)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                                    ON CONFLICT (factor_name, date) 
                                    DO UPDATE SET 
                                        factor_value = EXCLUDED.factor_value,
                                        unit = EXCLUDED.unit,
                                        source = EXCLUDED.source,
                                        updated_at = %s,
                                        updated_by = %s
                                    RETURNING (xmax = 0) AS inserted
                                """, (
                                    factor_id,
                                    record_date,
                                    factor_name,
                                    float(row['value']),
                                    series_info.get('units', 'N/A'),
                                    f"FRED: {series_id}",
                                    user_email,
                                    datetime.utcnow(),
                                    datetime.utcnow(),
                                    user_email
                                ))
                                
                                # Check if inserted or updated
                                result = cursor.fetchone()
                                if result and result[0]:
                                    inserted += 1
                                else:
                                    updated += 1
                            
                            total_imported += inserted
                            total_updated += updated
                            
                            logger.info(
                                f"Imported {factor_name}: {inserted} new, {updated} updated records"
                            )
                            
                        except Exception as e:
                            logger.error(f"Failed to import {series_id}: {str(e)}")
                            failed_series.append({
                                "series_id": series_id,
                                "factor_name": factor_name,
                                "error": str(e)
                            })
                    
                    conn.commit()
                    logger.info(f"Bulk import completed: {total_imported} inserted, {total_updated} updated")
                    
                finally:
                    cursor.close()
            
            return {
                "total_inserted": total_imported,
                "total_updated": total_updated,
                "total_records": total_imported + total_updated,
                "successful_series": len(series_configs) - len(failed_series),
                "failed_series": failed_series,
                "import_date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Bulk import from FRED failed: {str(e)}")
            raise DatabaseException(f"Failed to import from FRED: {str(e)}")

    @staticmethod
    def forecast_future_factors(
        tenant_id: str,
        database_name: str,
        factor_names: List[str],
        forecast_start: date,
        forecast_end: date,
        user_email: str
    ) -> Dict[str, Any]:
        """
        Forecast future values for selected external factors.
        Uses historical data to extrapolate future values.
        """
        from app.core.forecast_execution_service import ForecastExecutionService
        
        db_manager = get_db_manager()
        total_forecasted = 0
        successful_factors = []
        failed_factors = []
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    for factor_name in factor_names:
                        try:
                            # Get historical data
                            cursor.execute("""
                                SELECT date, factor_value, unit, source
                                FROM external_factors
                                WHERE factor_name = %s 
                                AND deleted_at IS NULL
                                ORDER BY date
                            """, (factor_name,))
                            
                            rows = cursor.fetchall()
                            
                            if len(rows) < 3:
                                logger.warning(f"Insufficient historical data for {factor_name}")
                                failed_factors.append({
                                    "factor_name": factor_name,
                                    "error": "Insufficient historical data (need at least 3 data points)"
                                })
                                continue
                            
                            # Convert to DataFrame
                            df = pd.DataFrame(rows, columns=['date', 'value', 'unit', 'source'])
                            df['date'] = pd.to_datetime(df['date'])
                            df['value'] = df['value'].astype(float)
                            
                            interval_type = ExternalFactorsService._detect_factor_interval(df['date'])
                            forecast_dates = ExternalFactorsService._generate_forecast_dates(
                                forecast_start,
                                forecast_end,
                                interval_type
                            )
                            periods = len(forecast_dates)
                            
                            if periods <= 0:
                                failed_factors.append({
                                    "factor_name": factor_name,
                                    "error": "Invalid forecast date range"
                                })
                                continue
                            
                            forecast_values, _ = ForecastExecutionService.linear_regression_forecast(
                                data=pd.DataFrame({'total_quantity': df['value'].values}),
                                periods=periods
                            )
                            
                            # Insert forecasted values
                            unit = rows[0][2] if rows else 'N/A'
                            source = f"Forecasted from historical data"
                            
                            inserted = 0
                            for forecast_date, forecast_value in zip(forecast_dates, forecast_values):
                                factor_id = str(uuid.uuid4())
                                
                                cursor.execute("""
                                    INSERT INTO external_factors
                                    (factor_id, date, factor_name, factor_value,
                                     unit, source, created_by, created_at)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                                    ON CONFLICT (factor_name, date)
                                    DO UPDATE SET
                                        factor_value = EXCLUDED.factor_value,
                                        source = EXCLUDED.source,
                                        updated_at = %s,
                                        updated_by = %s
                                """, (
                                    factor_id,
                                    forecast_date,
                                    factor_name,
                                    float(forecast_value),
                                    unit,
                                    source,
                                    user_email,
                                    datetime.utcnow(),
                                    datetime.utcnow(),
                                    user_email
                                ))
                                inserted += 1
                            
                            total_forecasted += inserted
                            successful_factors.append(factor_name)
                            
                            logger.info(f"Forecasted {inserted} values for {factor_name}")
                            
                        except Exception as e:
                            logger.error(f"Failed to forecast {factor_name}: {str(e)}")
                            failed_factors.append({
                                "factor_name": factor_name,
                                "error": str(e)
                            })
                    
                    conn.commit()
                    
                finally:
                    cursor.close()
            
            return {
                "total_forecasted": total_forecasted,
                "successful_factors": successful_factors,
                "failed_factors": failed_factors,
                "forecast_date_range": {
                    "start": forecast_start.isoformat(),
                    "end": forecast_end.isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Factor forecasting failed: {str(e)}")
            raise DatabaseException(f"Failed to forecast factors: {str(e)}")

    @staticmethod
    def _detect_factor_interval(dates: pd.Series) -> str:
        if dates is None or len(dates) < 2:
            return "DAILY"
        ordered_dates = dates.sort_values().reset_index(drop=True)
        inferred = None
        try:
            inferred = pd.infer_freq(pd.DatetimeIndex(ordered_dates))
        except (ValueError, TypeError):
            inferred = None
        if inferred:
            freq_key = inferred.upper()
            freq_mappings = [
                ("QS", "QUARTERLY"),
                ("Q", "QUARTERLY"),
                ("MS", "MONTHLY"),
                ("M", "MONTHLY"),
                ("W", "WEEKLY"),
                ("D", "DAILY"),
                ("AS", "YEARLY"),
                ("A", "YEARLY")
            ]
            for prefix, interval in freq_mappings:
                if freq_key.startswith(prefix):
                    return interval
        diffs = ordered_dates.diff().dropna().dt.days.abs()
        if diffs.empty:
            return "DAILY"
        median_diff = diffs.median()
        if median_diff >= 300:
            return "YEARLY"
        if median_diff >= 80:
            return "QUARTERLY"
        if median_diff >= 25:
            return "MONTHLY"
        if median_diff >= 6:
            return "WEEKLY"
        return "DAILY"

    @staticmethod
    def _generate_forecast_dates(
        start_date_value: date,
        end_date_value: date,
        interval_type: str
    ) -> List[date]:
        if not start_date_value or not end_date_value or start_date_value > end_date_value:
            return []
        interval = (interval_type or "DAILY").upper()
        dates: List[date] = []
        current = start_date_value
        while current <= end_date_value:
            dates.append(current)
            if interval == "WEEKLY":
                current += timedelta(weeks=1)
            elif interval == "MONTHLY":
                current += relativedelta(months=1)
            elif interval == "QUARTERLY":
                current += relativedelta(months=3)
            elif interval == "YEARLY":
                current += relativedelta(years=1)
            else:
                current += timedelta(days=1)
        return dates

    @staticmethod
    def get_factors_for_forecast_run(
        tenant_id: str,
        database_name: str,
        selected_factors: Optional[List[str]],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Get external factors for forecast run with optional selection.
        
        Args:
            tenant_id: Tenant identifier
            database_name: Tenant's database name
            selected_factors: List of factor names to include (None/empty = no factors)
            start_date: Optional start date filter (None = no filtering)
            end_date: Optional end date filter (None = no filtering)
            
        Returns:
            DataFrame with factors pivoted by name
        """
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Check if any factors are selected
                    if not selected_factors or len(selected_factors) == 0:
                        logger.info("No external factors selected for this forecast run")
                        return pd.DataFrame()
                    
                    logger.info(f"Fetching external factors: {selected_factors}")
                    
                    # Build the query
                    placeholders = ', '.join(['%s'] * len(selected_factors))
                    params = []
                    
                    # Build optional date filters
                    date_conditions = []
                    if start_date is not None:
                        date_conditions.append("date >= %s")
                        params.append(start_date)
                    
                    if end_date is not None:
                        date_conditions.append("date <= %s")
                        params.append(end_date)
                    
                    date_filter = ""
                    if date_conditions:
                        date_filter = "AND " + " AND ".join(date_conditions)
                    
                    # Add selected factors to params
                    params.extend(selected_factors)
                    
                    # Build final query
                    query = f"""
                        SELECT date, factor_name, factor_value
                        FROM external_factors
                        WHERE deleted_at IS NULL
                        {date_filter}
                        AND factor_name IN ({placeholders})
                        ORDER BY date, factor_name
                    """
                    
                    # Execute query
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    
                    # Log results
                    if not rows:
                        # Debug: Check what's actually in the table
                        cursor.execute("""
                            SELECT DISTINCT factor_name 
                            FROM external_factors 
                            WHERE deleted_at IS NULL
                        """)
                        available = [r[0] for r in cursor.fetchall()]
                        
                        logger.warning(
                            f"No external factor data found! "
                            f"Requested: {selected_factors}, "
                            f"Available in DB: {available}"
                        )
                        return pd.DataFrame()
                    
                    logger.info(f"Fetched {len(rows)} external factor records")
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(rows, columns=['date', 'factor_name', 'factor_value'])
                    df['date'] = pd.to_datetime(df['date'])
                    df['factor_value'] = df['factor_value'].astype(float)
                    
                    # Pivot: each factor becomes a column
                    df_pivot = df.pivot_table(
                        index='date',
                        columns='factor_name',
                        values='factor_value',
                        aggfunc='mean'
                    ).reset_index()
                    
                    date_range_msg = ""
                    if start_date or end_date:
                        date_range_msg = f" (filtered: {start_date or 'start'} to {end_date or 'end'})"
                    
                    logger.info(
                        f"Prepared {len(df_pivot)} date records with "
                        f"{len(df_pivot.columns) - 1} factors: {list(df_pivot.columns[1:])}"
                        f"{date_range_msg}"
                    )
                    
                    if not df_pivot.empty:
                        logger.info(
                            f"Factor data date range: {df_pivot['date'].min()} to {df_pivot['date'].max()}"
                        )
                    
                    return df_pivot
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"Failed to get factors for forecast: {str(e)}", exc_info=True)
            raise DatabaseException(f"Failed to retrieve factors: {str(e)}")

    @staticmethod
    def get_available_factors_summary(
        tenant_id: str,
        database_name: str
    ) -> List[Dict[str, Any]]:
        """
        Get summary of all available external factors for tenant.
        """
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        SELECT 
                            factor_name,
                            MIN(date) as earliest_date,
                            MAX(date) as latest_date,
                            COUNT(*) as data_points,
                            MAX(unit) as unit,
                            MAX(source) as source,
                            AVG(factor_value) as avg_value,
                            MIN(factor_value) as min_value,
                            MAX(factor_value) as max_value
                        FROM external_factors
                        WHERE deleted_at IS NULL
                        GROUP BY factor_name
                        ORDER BY factor_name
                    """)
                    
                    factors = []
                    for row in cursor.fetchall():
                        factors.append({
                            "factor_name": row[0],
                            "earliest_date": row[1].isoformat() if row[1] else None,
                            "latest_date": row[2].isoformat() if row[2] else None,
                            "data_points": row[3],
                            "unit": row[4],
                            "source": row[5],
                            "avg_value": round(float(row[6]), 4) if row[6] else None,
                            "min_value": round(float(row[7]), 4) if row[7] else None,
                            "max_value": round(float(row[8]), 4) if row[8] else None
                        })
                    
                    return factors
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"Failed to get factors summary: {str(e)}")
            raise DatabaseException(f"Failed to get factors summary: {str(e)}")

    @staticmethod
    def list_factors(
        tenant_id: str,
        database_name: str,
        factor_name: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        page: int = 1,
        page_size: int = 50
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        List external factors with optional filters and pagination.

        Args:
            tenant_id: Tenant identifier
            database_name: Tenant's database name
            factor_name: Optional filter by factor name
            date_from: Optional start date filter (YYYY-MM-DD)
            date_to: Optional end date filter (YYYY-MM-DD)
            page: Page number (1-based)
            page_size: Number of records per page

        Returns:
            Tuple of (factors list, total count)
        """
        db_manager = get_db_manager()

        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Build WHERE conditions
                    conditions = ["deleted_at IS NULL"]
                    params = []

                    if factor_name:
                        conditions.append("factor_name = %s")
                        params.append(factor_name)

                    if date_from:
                        conditions.append("date >= %s")
                        params.append(date_from)

                    if date_to:
                        conditions.append("date <= %s")
                        params.append(date_to)

                    where_clause = " AND ".join(conditions)

                    # Get total count
                    count_query = f"SELECT COUNT(*) FROM external_factors WHERE {where_clause}"
                    cursor.execute(count_query, params)
                    total_count = cursor.fetchone()[0]

                    # Get paginated results
                    offset = (page - 1) * page_size
                    query = f"""
                        SELECT
                            factor_id,
                            date,
                            factor_name,
                            factor_value,
                            unit,
                            source,
                            created_by,
                            created_at,
                            updated_by,
                            updated_at
                        FROM external_factors
                        WHERE {where_clause}
                        ORDER BY date DESC, factor_name, created_at DESC
                        LIMIT %s OFFSET %s
                    """
                    params.extend([page_size, offset])

                    cursor.execute(query, params)
                    rows = cursor.fetchall()

                    factors = []
                    for row in rows:
                        factors.append({
                            "factor_id": row[0],
                            "date": row[1].isoformat() if row[1] else None,
                            "factor_name": row[2],
                            "factor_value": float(row[3]) if row[3] is not None else None,
                            "unit": row[4],
                            "source": row[5],
                            "created_by": row[6],
                            "created_at": row[7].isoformat() if row[7] else None,
                            "updated_by": row[8],
                            "updated_at": row[9].isoformat() if row[9] else None
                        })

                    return factors, total_count

                finally:
                    cursor.close()

        except Exception as e:
            logger.error(f"Failed to list factors: {str(e)}")
            raise DatabaseException(f"Failed to list factors: {str(e)}")

    @staticmethod
    def delete_factor_by_name(
        tenant_id: str,
        database_name: str,
        factor_name: str
    ) -> int:
        """
        Soft delete all records for a specific factor.
        Returns number of records deleted.
        """
        db_manager = get_db_manager()

        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        UPDATE external_factors
                        SET deleted_at = %s
                        WHERE factor_name = %s
                        AND deleted_at IS NULL
                    """, (datetime.utcnow(), factor_name))

                    deleted_count = cursor.rowcount
                    conn.commit()

                    logger.info(f"Soft deleted {deleted_count} records for factor '{factor_name}'")
                    return deleted_count

                finally:
                    cursor.close()

        except Exception as e:
            logger.error(f"Failed to delete factor: {str(e)}")
            raise DatabaseException(f"Failed to delete factor: {str(e)}")
