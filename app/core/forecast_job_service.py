"""
Forecast Job Service - Manages async forecast execution and job tracking.
Handles job status, results storage, and background execution.
"""

import uuid
import logging
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Any, Optional
from enum import Enum
import json

from app.core.database import get_db_manager
from app.core.exceptions import NotFoundException, ValidationException, DatabaseException
from app.core.forecasting_service import ForecastingService

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Forecast job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ForecastJobService:
    """Service for managing forecast job lifecycle and results."""

    @staticmethod
    def create_job(
        tenant_id: str,
        database_name: str,
        request_data: Dict[str, Any],
        user_email: str
    ) -> Dict[str, Any]:
        """
        Create a new forecast job record.
        
        Args:
            tenant_id: Tenant identifier
            database_name: Database name for tenant
            request_data: The forecast request data
            user_email: Email of user who initiated the job
        
        Returns:
            Job details including job_id
        """
        db_manager = get_db_manager()
        job_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                with conn.cursor() as cursor:
                    # Create forecast_jobs table if it doesn't exist
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS forecast_jobs (
                            job_id UUID PRIMARY KEY,
                            tenant_id TEXT NOT NULL,
                            status VARCHAR(20) NOT NULL DEFAULT 'pending',
                            request_data JSONB NOT NULL,
                            result_data JSONB,
                            error_message TEXT,
                            selected_metrics TEXT[],
                            started_at TIMESTAMP,
                            completed_at TIMESTAMP,
                            created_at TIMESTAMP NOT NULL,
                            created_by TEXT NOT NULL,
                            updated_at TIMESTAMP NOT NULL
                        )
                    """)
                    
                    cursor.execute("""
                        INSERT INTO forecast_jobs 
                        (job_id, tenant_id, status, request_data, selected_metrics, created_at, created_by, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        job_id,
                        tenant_id,
                        JobStatus.PENDING.value,
                        json.dumps(request_data, default=str),
                        request_data.get('selected_metrics', ['mape', 'accuracy']),  #   Extract and store
                        created_at,
                        user_email,
                        created_at
                    ))
                                        
                    conn.commit()
            
            logger.info(f"Created forecast job {job_id} for tenant {tenant_id}")
            
            return {
                "job_id": job_id,
                "status": JobStatus.PENDING.value,
                "created_at": created_at.isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to create forecast job: {str(e)}", exc_info=True)
            raise

    @staticmethod
    def get_job_status(
        tenant_id: str,
        database_name: str,
        job_id: str
    ) -> Dict[str, Any]:
        """
        Get the current status and details of a forecast job.
        
        Args:
            tenant_id: Tenant identifier
            database_name: Database name for tenant
            job_id: Job identifier
        
        Returns:
            Job details including status and results if completed
        """
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT 
                            job_id, 
                            status, 
                            result_data, 
                            error_message, 
                            created_at, 
                            started_at, 
                            completed_at
                        FROM forecast_jobs
                        WHERE job_id = %s AND tenant_id = %s
                    """, (job_id, tenant_id))
                    
                    row = cursor.fetchone()
                    
                    if not row:
                        return {
                            "job_id": job_id,
                            "status": "not_found",
                            "error": "Job not found"
                        }
                    
                    job_id, status, result_data, error_msg, created_at, started_at, completed_at = row
                    
                    response = {
                        "job_id": str(job_id),
                        "status": status,
                        "created_at": created_at.isoformat() if created_at else None,
                        "started_at": started_at.isoformat() if started_at else None,
                        "completed_at": completed_at.isoformat() if completed_at else None
                    }
                    
                    if result_data:
                        response["result"] = result_data
                    
                    if error_msg:
                        response["error"] = error_msg
                    
                    return response
        
        except Exception as e:
            logger.error(f"Failed to get job status: {str(e)}", exc_info=True)
            raise

    @staticmethod
    def get_forecast_results(
        tenant_id: str,
        database_name: str,
        job_id: str
    ) -> Dict[str, Any]:
        """
        Get forecast results for a job ID by combining job result data and
        forecast results from the database.
        """
        db_manager = get_db_manager()

        with db_manager.get_tenant_connection(database_name) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT result_data
                    FROM forecast_jobs
                    WHERE job_id = %s AND tenant_id = %s
                    """,
                    (job_id, tenant_id)
                )
                row = cursor.fetchone()

        if not row:
            raise NotFoundException("forecast_job", job_id)

        result_data = row[0]
        if not result_data:
            raise NotFoundException("forecast_results", job_id)

        if isinstance(result_data, str):
            result_data = json.loads(result_data)

        forecast_runs = result_data.get("forecast_runs") or []
        if not isinstance(forecast_runs, list):
            forecast_runs = []

        run_ids = [
            str(run.get("forecast_run_id"))
            for run in forecast_runs
            if run.get("forecast_run_id")
        ]

        results_by_run_type: Dict[str, Dict[str, list]] = {
            run_id: {} for run_id in run_ids
        }
        mapping_ids_by_run: Dict[str, str] = {}
        if run_ids:
            with db_manager.get_tenant_connection(database_name) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT forecast_run_id, mapping_id, type, date, value, metadata
                        FROM forecast_results
                        WHERE forecast_run_id = ANY(%s::uuid[])
                        ORDER BY forecast_run_id, mapping_id, type, date
                        """,
                        (run_ids,)
                    )
                    for forecast_run_id, mapping_id, result_type, date, value, metadata in cursor.fetchall():
                        run_id = str(forecast_run_id)
                        if mapping_id:
                            mapping_ids_by_run.setdefault(run_id, str(mapping_id))
                        if not result_type:
                            continue
                        
                        result_item = {
                            "date": date.isoformat() if date else None,
                            "quantity": float(value) if value is not None else None
                        }
                        
                        # Include accuracy metrics if available in metadata
                        if metadata and isinstance(metadata, dict) and 'test_metrics' in metadata:
                            result_item["metrics"] = metadata['test_metrics']
                        
                        results_by_run_type.setdefault(run_id, {}).setdefault(
                            result_type, []
                        ).append(result_item)

        results = []
        for run in forecast_runs:
            run_id = run.get("forecast_run_id")
            if not run_id:
                continue
            run_id_str = str(run_id)
            entry = {
                "mapping_id": mapping_ids_by_run.get(run_id_str, ""),
                "entity_filter": run.get("entity_filter") or {}
            }
            for result_type, items in results_by_run_type.get(run_id_str, {}).items():
                entry[result_type] = items
            results.append(entry)

        return {
            "job_id": job_id,
            "results": results
        }

    @staticmethod
    def get_mapping_details(
        tenant_id: str,
        database_name: str,
        mapping_id: str
    ) -> Dict[str, Any]:
        """
        Get algorithm mapping details for a mapping_id.
        """
        db_manager = get_db_manager()

        with db_manager.get_tenant_connection(database_name) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT mapping_id, algorithm_name, custom_parameters
                    FROM forecast_algorithms_mapping
                    WHERE mapping_id = %s
                    """,
                    (mapping_id,)
                )
                row = cursor.fetchone()

        if not row:
            raise NotFoundException("forecast_algorithm_mapping", mapping_id)

        mapping_id_value, algorithm_name, custom_parameters = row
        return {
            "mapping_id": str(mapping_id_value),
            "algorithm_name": algorithm_name,
            "custom_parameters": custom_parameters,
        }

    @staticmethod
    def copy_forecast_results(
        tenant_id: str,
        database_name: str,
        job_id: str,
        user_email: str,
        batch_size: int = 500
    ) -> Dict[str, Any]:
        """
        Copy forecast results for all runs in a job into forecast_data.
        Updates are matched on master_id and date.
        """
        db_manager = get_db_manager()

        with db_manager.get_tenant_connection(database_name) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT result_data
                    FROM forecast_jobs
                    WHERE job_id = %s AND tenant_id = %s
                    """,
                    (job_id, tenant_id)
                )
                row = cursor.fetchone()

        if not row:
            raise NotFoundException("forecast_job", job_id)

        result_data = row[0]
        if not result_data:
            raise NotFoundException("forecast_results", job_id)

        if isinstance(result_data, str):
            result_data = json.loads(result_data)

        forecast_runs = result_data.get("forecast_runs") or []
        if not isinstance(forecast_runs, list):
            forecast_runs = []

        run_ids = [
            str(run.get("forecast_run_id"))
            for run in forecast_runs
            if run.get("forecast_run_id")
        ]

        if not run_ids:
            return {
                "status": "success",
                "message": "No forecast runs found for this job",
                "saved_count": 0,
                "processed_runs": 0
            }

        try:
            target_field, date_field = ForecastingService._get_field_names(
                tenant_id, database_name
            )

            with db_manager.get_tenant_connection(database_name) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        WITH ranked AS (
                            SELECT
                                fam.forecast_run_id,
                                fam.mapping_id,
                                AVG(fr.accuracy_metric) AS avg_accuracy,
                                ROW_NUMBER() OVER (
                                    PARTITION BY fam.forecast_run_id
                                    ORDER BY AVG(fr.accuracy_metric) DESC NULLS LAST
                                ) AS rn
                            FROM forecast_algorithms_mapping fam
                            LEFT JOIN forecast_results fr
                                ON fam.mapping_id = fr.mapping_id
                            WHERE fam.forecast_run_id = ANY(%s::uuid[])
                              AND fam.execution_status = 'Completed'
                            GROUP BY fam.forecast_run_id, fam.mapping_id
                        )
                        SELECT forecast_run_id, mapping_id
                        FROM ranked
                        WHERE rn = 1
                        """,
                        (run_ids,)
                    )
                    mapping_rows = cursor.fetchall()
                    if not mapping_rows:
                        raise NotFoundException("forecast_results", job_id)

                    mapping_ids = [
                        row[1] for row in mapping_rows if row[1] is not None
                    ]
                    if not mapping_ids:
                        return {
                            "status": "success",
                            "message": "No mapping IDs found for this job",
                            "saved_count": 0,
                            "processed_runs": 0
                        }

                    run_entity_filters = {}
                    for run in forecast_runs:
                        run_id = run.get("forecast_run_id")
                        if run_id:
                            run_entity_filters[str(run_id)] = (
                                run.get("entity_filter") or {}
                            )

                    mapping_entity_filters = {
                        str(mapping_id): run_entity_filters.get(
                            str(run_id),
                            {}
                        )
                        for run_id, mapping_id in mapping_rows
                        if mapping_id is not None
                    }

                    cursor.execute(
                        """
                        SELECT mapping_id, date, value, metadata
                        FROM forecast_results
                        WHERE mapping_id = ANY(%s::uuid[])
                          AND type = 'future_forecast'
                        ORDER BY date
                        """,
                        (mapping_ids,)
                    )
                    results = cursor.fetchall()

                    if not results:
                        return {
                            "status": "success",
                            "message": "No future forecast results found",
                            "saved_count": 0,
                            "processed_runs": len(mapping_ids)
                        }

                    saved_count = 0
                    batch = []
                    master_id_cache = {}
                    sales_info_cache = {}
                    for mapping_id, forecast_date, forecast_value, metadata in results:
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
                        if not entity_filter:
                            entity_filter = mapping_entity_filters.get(
                                str(mapping_id),
                                {}
                            )
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

                        batch.append((
                            master_id,
                            forecast_date,
                            forecast_value,
                            uom,
                            unit_price,
                            user_email
                        ))
                        if len(batch) >= batch_size:
                            saved_count += ForecastJobService._bulk_upsert_forecast_data(
                                cursor,
                                target_field,
                                date_field,
                                batch
                            )
                            conn.commit()
                            batch = []

                    if batch:
                        saved_count += ForecastJobService._bulk_upsert_forecast_data(
                            cursor,
                            target_field,
                            date_field,
                            batch
                        )
                        conn.commit()

            return {
                "status": "success",
                "message": f"Successfully saved {saved_count} forecast records",
                "saved_count": saved_count,
                "processed_runs": len(mapping_ids)
            }

        except (ValidationException, NotFoundException):
            raise
        except Exception as e:
            logger.error(f"Failed to copy forecast results: {str(e)}")
            raise DatabaseException(f"Failed to copy forecast results: {str(e)}")

    @staticmethod
    def _bulk_upsert_forecast_data(
        cursor,
        target_field: str,
        date_field: str,
        rows: list
    ) -> int:
        from psycopg2.extras import execute_values

        template = (
            "(%s::uuid, %s::date, %s::numeric, "
            "%s::varchar, %s::numeric, %s::varchar)"
        )
        update_sql = f"""
            WITH data(
                master_id,
                forecast_date,
                forecast_value,
                uom,
                unit_price,
                created_by
            ) AS (
                VALUES %s
            ),
            dedup AS (
                SELECT
                    master_id,
                    forecast_date,
                    MAX(forecast_value) AS forecast_value,
                    MAX(uom) AS uom,
                    MAX(unit_price) AS unit_price,
                    MAX(created_by) AS created_by
                FROM data
                GROUP BY master_id, forecast_date
            )
            UPDATE forecast_data fd
            SET "{target_field}" = dedup.forecast_value,
                uom = dedup.uom,
                unit_price = dedup.unit_price,
                created_at = CURRENT_TIMESTAMP,
                created_by = dedup.created_by
            FROM dedup
            WHERE fd.master_id = dedup.master_id
              AND fd."{date_field}" = dedup.forecast_date
        """
        execute_values(
            cursor,
            update_sql,
            rows,
            template=template,
            page_size=len(rows),
        )

        insert_sql = f"""
            WITH data(
                master_id,
                forecast_date,
                forecast_value,
                uom,
                unit_price,
                created_by
            ) AS (
                VALUES %s
            ),
            dedup AS (
                SELECT
                    master_id,
                    forecast_date,
                    MAX(forecast_value) AS forecast_value,
                    MAX(uom) AS uom,
                    MAX(unit_price) AS unit_price,
                    MAX(created_by) AS created_by
                FROM data
                GROUP BY master_id, forecast_date
            )
            INSERT INTO forecast_data (
                master_id,
                "{date_field}",
                "{target_field}",
                uom,
                unit_price,
                created_by
            )
            SELECT
                dedup.master_id,
                dedup.forecast_date,
                dedup.forecast_value,
                dedup.uom,
                dedup.unit_price,
                dedup.created_by
            FROM dedup
            LEFT JOIN forecast_data fd
              ON fd.master_id = dedup.master_id
             AND fd."{date_field}" = dedup.forecast_date
            WHERE fd.master_id IS NULL
        """
        execute_values(
            cursor,
            insert_sql,
            rows,
            template=template,
            page_size=len(rows),
        )
        return len(rows)

    @staticmethod
    def update_job_status(
        tenant_id: str,
        database_name: str,
        job_id: str,
        status: JobStatus,
        result_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Update job status and optionally store results.
        
        Args:
            tenant_id: Tenant identifier
            database_name: Database name for tenant
            job_id: Job identifier
            status: New job status
            result_data: Result data (if completed successfully)
            error_message: Error message (if failed)
            **kwargs: Additional arguments (e.g., metadata)
        """
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                with conn.cursor() as cursor:
                    updated_at = datetime.utcnow()
                    
                    # Determine started_at and completed_at based on status
                    started_at = None
                    completed_at = None
                    
                    if status == JobStatus.RUNNING:
                        # Check if metadata is provided
                        metadata = kwargs.get('metadata')
                        if metadata:
                            # Store metadata in result_data for RUNNING state
                            result_json = json.dumps(metadata, default=str)
                            cursor.execute("""
                                UPDATE forecast_jobs
                                SET status = %s, started_at = %s, updated_at = %s, result_data = %s
                                WHERE job_id = %s AND tenant_id = %s
                            """, (status.value, updated_at, updated_at, result_json, job_id, tenant_id))
                        else:
                            cursor.execute("""
                                UPDATE forecast_jobs
                                SET status = %s, started_at = %s, updated_at = %s
                                WHERE job_id = %s AND tenant_id = %s
                            """, (status.value, updated_at, updated_at, job_id, tenant_id))
                    
                    elif status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                        result_json = json.dumps(result_data, default=str) if result_data else None
                        cursor.execute("""
                            UPDATE forecast_jobs
                            SET status = %s, result_data = %s, error_message = %s, 
                                completed_at = %s, updated_at = %s
                            WHERE job_id = %s AND tenant_id = %s
                        """, (
                            status.value,
                            result_json,
                            error_message,
                            updated_at,
                            updated_at,
                            job_id,
                            tenant_id
                        ))
                    
                    else:
                        cursor.execute("""
                            UPDATE forecast_jobs
                            SET status = %s, updated_at = %s
                            WHERE job_id = %s AND tenant_id = %s
                        """, (status.value, updated_at, job_id, tenant_id))
                    
                    conn.commit()
            
            logger.info(f"Updated forecast job {job_id} status to {status.value}")
        
        except Exception as e:
            logger.error(f"Failed to update job status: {str(e)}", exc_info=True)
            raise

    @staticmethod
    def check_duplicate_running_job(
        tenant_id: str,
        database_name: str,
        new_request_data: Dict[str, Any]
    ) -> bool:
        """
        Check if there's a running job with the same forecast parameters.

        Args:
            tenant_id: Tenant identifier
            database_name: Database name for tenant
            new_request_data: The new forecast request data

        Returns:
            True if duplicate running job found, False otherwise
        """
        db_manager = get_db_manager()

        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT request_data
                        FROM forecast_jobs
                        WHERE tenant_id = %s AND status = 'running'
                    """, (tenant_id,))

                    rows = cursor.fetchall()

                    for row in rows:
                        existing_request_data = row[0]

                        # Compare key fields that determine the forecast sub-items
                        if (existing_request_data.get('forecast_filters') == new_request_data.get('forecast_filters') and
                            existing_request_data.get('forecast_start') == new_request_data.get('forecast_start') and
                            existing_request_data.get('forecast_end') == new_request_data.get('forecast_end') and
                            existing_request_data.get('algorithms') == new_request_data.get('algorithms')):
                            return True

                    return False

        except Exception as e:
            logger.error(f"Failed to check for duplicate running job: {str(e)}", exc_info=True)
            return False  # On error, allow creation to avoid blocking

    @staticmethod
    def get_user_jobs(
        tenant_id: str,
        database_name: str,
        user_email: str,
        limit: int = 50,
        created_from: Optional[datetime] = None,
        created_to: Optional[datetime] = None,
        timezone_name: Optional[str] = None
    ) -> list:
        """
        Get all forecast jobs for a specific user.
        
        Args:
            tenant_id: Tenant identifier
            database_name: Database name for tenant
            user_email: User email
            limit: Maximum number of jobs to return
        
        Returns:
            List of job summaries
        """
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                with conn.cursor() as cursor:
                    where_clauses = ["tenant_id = %s", "created_by = %s"]
                    params = [tenant_id, user_email]

                    if created_from:
                        where_clauses.append("created_at >= %s")
                        params.append(created_from)
                    if created_to:
                        where_clauses.append("created_at <= %s")
                        params.append(created_to)

                    where_sql = " AND ".join(where_clauses)
                    params.append(limit)

                    cursor.execute(f"""
                        SELECT 
                            job_id, 
                            status, 
                            request_data,
                            created_at, 
                            started_at, 
                            completed_at
                        FROM forecast_jobs
                        WHERE {where_sql}
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, params)
                    
                    rows = cursor.fetchall()
                    
                    algorithm_ids = set()
                    raw_rows = []
                    for job_id, status, request_data, created_at, started_at, completed_at in rows:
                        raw_rows.append(
                            (job_id, status, request_data, created_at, started_at, completed_at)
                        )
                        if isinstance(request_data, dict):
                            algo_id = request_data.get("algorithm_id")
                            if algo_id is not None:
                                algorithm_ids.add(algo_id)
                            algorithms = request_data.get("algorithms")
                            if isinstance(algorithms, list):
                                for algo in algorithms:
                                    if not isinstance(algo, dict):
                                        continue
                                    list_algo_id = algo.get("algorithm_id")
                                    if list_algo_id is not None:
                                        algorithm_ids.add(list_algo_id)

                    algorithm_names = {}
                    if algorithm_ids:
                        cursor.execute(
                            """
                            SELECT algorithm_id, algorithm_name
                            FROM algorithms
                            WHERE algorithm_id = ANY(%s)
                            """,
                            (list(algorithm_ids),)
                        )
                        algorithm_names = {row[0]: row[1] for row in cursor.fetchall()}

                    jobs = []
                    target_tz = (
                        ZoneInfo(timezone_name)
                        if timezone_name
                        else timezone.utc
                    )

                    def to_tz(dt: Optional[datetime]) -> Optional[str]:
                        if dt is None:
                            return None
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        return dt.astimezone(target_tz).isoformat()
                    for job_id, status, request_data, created_at, started_at, completed_at in raw_rows:
                        request_payload = request_data
                        if isinstance(request_payload, dict):
                            algo_id = request_payload.get("algorithm_id")
                            if algo_id is not None and "algorithm_name" not in request_payload:
                                request_payload = dict(request_payload)
                                request_payload["algorithm_name"] = algorithm_names.get(algo_id)
                            algorithms = request_payload.get("algorithms")
                            if isinstance(algorithms, list):
                                updated_algorithms = []
                                for algo in algorithms:
                                    if not isinstance(algo, dict):
                                        continue
                                    algo_id = algo.get("algorithm_id")
                                    if algo_id is None:
                                        updated_algorithms.append(algo)
                                        continue
                                    if "algorithm_name" in algo and algo["algorithm_name"]:
                                        updated_algorithms.append(algo)
                                        continue
                                    updated_algo = dict(algo)
                                    updated_algo["algorithm_name"] = algorithm_names.get(algo_id)
                                    updated_algorithms.append(updated_algo)
                                if updated_algorithms:
                                    request_payload = dict(request_payload)
                                    request_payload["algorithms"] = updated_algorithms
                        jobs.append({
                            "job_id": str(job_id),
                            "status": status,   
                            "request_data": request_payload,
                            "created_at": to_tz(created_at),
                            "started_at": to_tz(started_at),
                            "completed_at": to_tz(completed_at)
                        })
                    
                    return jobs
        
        except Exception as e:
            logger.error(f"Failed to get user jobs: {str(e)}", exc_info=True)
            return []
