"""
Forecast Job Service - Manages async forecast execution and job tracking.
Handles job status, results storage, and background execution.
"""

import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum
import json

from app.core.database import get_db_manager

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
        created_at = datetime.utcnow()
        
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
                            started_at TIMESTAMP,
                            completed_at TIMESTAMP,
                            created_at TIMESTAMP NOT NULL,
                            created_by TEXT NOT NULL,
                            updated_at TIMESTAMP NOT NULL
                        )
                    """)
                    
                    cursor.execute("""
                        INSERT INTO forecast_jobs 
                        (job_id, tenant_id, status, request_data, created_at, created_by, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        job_id,
                        tenant_id,
                        JobStatus.PENDING.value,
                        json.dumps(request_data, default=str),
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
    def get_user_jobs(
        tenant_id: str,
        database_name: str,
        user_email: str,
        limit: int = 50
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
                    cursor.execute("""
                        SELECT 
                            job_id, 
                            status, 
                            created_at, 
                            started_at, 
                            completed_at
                        FROM forecast_jobs
                        WHERE tenant_id = %s AND created_by = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (tenant_id, user_email, limit))
                    
                    rows = cursor.fetchall()
                    
                    jobs = []
                    for job_id, status, created_at, started_at, completed_at in rows:
                        jobs.append({
                            "job_id": str(job_id),
                            "status": status,
                            "created_at": created_at.isoformat() if created_at else None,
                            "started_at": started_at.isoformat() if started_at else None,
                            "completed_at": completed_at.isoformat() if completed_at else None
                        })
                    
                    return jobs
        
        except Exception as e:
            logger.error(f"Failed to get user jobs: {str(e)}", exc_info=True)
            return []
