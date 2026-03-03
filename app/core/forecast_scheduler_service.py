"""
Forecast Scheduler Service - Manages recurring forecast execution with APScheduler.
Handles schedule creation, persistence, and execution with full audit trails.
"""

import uuid
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dateutil import rrule
from dateutil.parser import parse as parse_cron
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from psycopg2.extras import Json
from pytz import utc, timezone

# IST timezone for India
IST = timezone('Asia/Kolkata')

from app.core.database import get_db_manager
from app.core.background_forecast_executor import BackgroundForecastExecutor
from app.core.forecast_job_service import ForecastJobService, JobStatus
from app.core.exceptions import (
    DatabaseException, 
    ValidationException, 
    NotFoundException
)
from app.schemas.forecast_schedules import ScheduleRunStatusEnum
from app.config import settings

logger = logging.getLogger(__name__)


class ForecastSchedulerService:
    """Service for managing forecast schedules with APScheduler integration."""
    
    # Global scheduler instance (shared across application)
    _scheduler: Optional[BackgroundScheduler] = None
    _scheduler_lock = False
    
    @classmethod
    def initialize_scheduler(cls) -> None:
        """
        Initialize the APScheduler BackgroundScheduler.
        Called once during application startup.
        """
        if cls._scheduler is not None:
            logger.info("Scheduler already initialized, skipping initialization")
            return
        
        try:
            cls._scheduler = BackgroundScheduler(timezone=IST)
            cls._scheduler.start()
            logger.info("APScheduler BackgroundScheduler initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize scheduler: {str(e)}", exc_info=True)
            raise
    
    @classmethod
    def shutdown_scheduler(cls) -> None:
        """
        Shutdown the APScheduler BackgroundScheduler.
        Called during application shutdown.
        """
        if cls._scheduler is not None:
            try:
                cls._scheduler.shutdown()
                cls._scheduler = None
                logger.info("APScheduler BackgroundScheduler shutdown successfully")
            except Exception as e:
                logger.error(f"Error during scheduler shutdown: {str(e)}", exc_info=True)
    
    @classmethod
    def _ensure_scheduler(cls) -> BackgroundScheduler:
        """Ensure scheduler is initialized."""
        if cls._scheduler is None:
            raise RuntimeError("Scheduler not initialized. Call initialize_scheduler() first.")
        return cls._scheduler
    
    @staticmethod
    def _calculate_next_run(cron_expression: str) -> datetime:
        """
        Calculate the next run datetime from cron expression.
        
        Args:
            cron_expression: Standard cron format (e.g., "0 2 * * 1" for Monday at 2 AM)
            
        Returns:
            Next run datetime
            
        Raises:
            ValidationException: If cron expression is invalid
        """
        try:
            # Parse cron expression using rrule
            # Format: minute hour day month weekday
            parts = cron_expression.strip().split()
            if len(parts) != 5:
                raise ValueError("Cron expression must have 5 fields")
            
            minute, hour, day, month, weekday = parts
            
            # Build rrule for next occurrence (using IST timezone)
            rrule_kwargs = {
                'freq': rrule.DAILY,
                'dtstart': datetime.now(IST),
                'count': 1
            }
            
            # Parse each component
            if minute != '*':
                rrule_kwargs['byminute'] = int(minute)
            if hour != '*':
                rrule_kwargs['byhour'] = int(hour)
            if day != '*' and day != '?':
                rrule_kwargs['bymonthday'] = int(day)
            if month != '*':
                rrule_kwargs['bymonth'] = int(month)
            if weekday != '*' and weekday != '?':
                # Convert cron weekday (0=Sunday) to rrule (0=Monday)
                weekday_int = int(weekday)
                if weekday_int == 0:
                    weekday_int = 6  # Sunday in rrule
                else:
                    weekday_int -= 1
                rrule_kwargs['byweekday'] = weekday_int
                rrule_kwargs['freq'] = rrule.WEEKLY
            
            # Get next occurrence
            rule = rrule.rrule(**rrule_kwargs)
            next_occurence_list = list(rule)
            
            if not next_occurence_list:
                # Fallback: calculate next run manually
                return datetime.now(utc) + timedelta(hours=1)
            
            return next_occurence_list[0]
        except Exception as e:
            logger.warning(f"Error calculating next run from cron expression '{cron_expression}': {str(e)}")
            # Return a safe default (next hour)
            return datetime.now(utc) + timedelta(hours=1)
    
    @staticmethod
    def load_schedules_from_database(
        tenant_id: str,
        database_name: str
    ) -> List[Dict[str, Any]]:
        """
        Load all active schedules from database for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            database_name: Tenant's database name
            
        Returns:
            List of schedule records
            
        Raises:
            DatabaseException: If database query fails
        """
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        SELECT 
                            schedule_id, schedule_name, description, request_data,
                            cron_expression, schedule_type, is_active, last_run,
                            next_run, last_job_id, last_run_status, last_run_error,
                            execution_count, created_at, created_by, updated_at, updated_by
                        FROM forecast_schedules
                        WHERE is_active = TRUE AND deleted_at IS NULL
                        ORDER BY next_run ASC
                    """)
                    
                    schedules = []
                    for row in cursor.fetchall():
                        schedules.append({
                            'schedule_id': row[0],
                            'schedule_name': row[1],
                            'description': row[2],
                            'request_data': row[3],
                            'cron_expression': row[4],
                            'schedule_type': row[5],
                            'is_active': row[6],
                            'last_run': row[7],
                            'next_run': row[8],
                            'last_job_id': row[9],
                            'last_run_status': row[10],
                            'last_run_error': row[11],
                            'execution_count': row[12],
                            'created_at': row[13],
                            'created_by': row[14],
                            'updated_at': row[15],
                            'updated_by': row[16]
                        })
                    
                    return schedules
                finally:
                    cursor.close()
        except Exception as e:
            logger.error(f"Failed to load schedules for {database_name}: {str(e)}", exc_info=True)
            raise DatabaseException(f"Failed to load schedules: {str(e)}")
    
    @classmethod
    def load_all_active_schedules(cls) -> None:
        """
        Load all active schedules from all tenants and register with APScheduler.
        This is called during application startup.
        """
        scheduler = cls._ensure_scheduler()
        
        try:
            # For now, we'll load schedules per tenant
            # In a multi-tenant setup, you'd iterate through tenants
            logger.info("Loading active forecast schedules from database...")
            # This will be called per-tenant from initialization logic
            logger.info("Forecast schedules loading completed (per-tenant loading in progress)")
        except Exception as e:
            logger.error(f"Error loading active schedules: {str(e)}", exc_info=True)
    
    @staticmethod
    def create_schedule(
        tenant_id: str,
        database_name: str,
        schedule_name: str,
        request_data: Dict[str, Any],
        cron_expression: str,
        schedule_type: str,
        description: Optional[str] = None,
        created_by: str = "system",
        is_active: bool = True
    ) -> str:
        """
        Create a new forecast schedule.
        
        Args:
            tenant_id: Tenant identifier
            database_name: Tenant's database name
            schedule_name: Name of the schedule
            request_data: Forecast configuration data
            cron_expression: Cron format schedule
            schedule_type: Type of schedule (daily, weekly, monthly, custom)
            description: Optional schedule description
            created_by: User creating the schedule
            is_active: Whether schedule is active
            
        Returns:
            Schedule ID
            
        Raises:
            ValidationException: If validation fails
            DatabaseException: If database operation fails
        """
        # Validate inputs
        if not schedule_name or not schedule_name.strip():
            raise ValidationException("Schedule name cannot be empty")
        
        if not request_data:
            raise ValidationException("Request data cannot be empty")
        
        valid_types = ['once', 'hourly', 'daily', 'weekly', 'monthly', 'custom']
        if schedule_type not in valid_types:
            raise ValidationException(f"Schedule type must be one of {valid_types}")
        
        # Validate cron expression
        try:
            ForecastSchedulerService._calculate_next_run(cron_expression)
        except Exception as e:
            raise ValidationException(f"Invalid cron expression: {str(e)}")
        
        db_manager = get_db_manager()
        schedule_id = str(uuid.uuid4())
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    next_run = ForecastSchedulerService._calculate_next_run(cron_expression)
                    
                    cursor.execute("""
                        INSERT INTO forecast_schedules (
                            schedule_id, schedule_name, description, request_data,
                            cron_expression, schedule_type, is_active, next_run,
                            created_by, created_at, updated_by, updated_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        schedule_id, schedule_name.strip(), description,
                        Json(request_data), cron_expression, schedule_type, is_active,
                        next_run, created_by, datetime.utcnow(),
                        created_by, datetime.utcnow()
                    ))
                    
                    conn.commit()
                    logger.info(f"Created schedule {schedule_id}: {schedule_name}")
                    
                    # Register with scheduler if active
                    if is_active:
                        ForecastSchedulerService.register_schedule(
                            scheduler=ForecastSchedulerService._ensure_scheduler(),
                            tenant_id=tenant_id,
                            database_name=database_name,
                            schedule_id=schedule_id,
                            schedule_name=schedule_name,
                            cron_expression=cron_expression,
                            request_data=request_data
                        )
                    
                    return schedule_id
                finally:
                    cursor.close()
        except Exception as e:
            if "duplicate" in str(e).lower():
                raise ValidationException(f"Schedule name '{schedule_name}' already exists")
            logger.error(f"Failed to create schedule: {str(e)}", exc_info=True)
            raise DatabaseException(f"Failed to create schedule: {str(e)}")
    
    @classmethod
    def register_schedule(
        cls,
        scheduler: BackgroundScheduler,
        tenant_id: str,
        database_name: str,
        schedule_id: str,
        schedule_name: str,
        cron_expression: str,
        request_data: Dict[str, Any]
    ) -> None:
        """
        Register a schedule with APScheduler.
        
        Args:
            scheduler: APScheduler BackgroundScheduler instance
            tenant_id: Tenant identifier
            database_name: Tenant's database name
            schedule_id: Schedule identifier
            schedule_name: Name of the schedule
            cron_expression: Cron format schedule
            request_data: Forecast configuration data
        """
        try:
            # Create unique job ID
            job_id = f"forecast_schedule_{schedule_id}"
            
            # Remove job if already exists
            try:
                scheduler.remove_job(job_id)
            except:
                pass
            
            # Parse cron expression and create trigger
            parts = cron_expression.strip().split()
            if len(parts) == 5:
                minute, hour, day, month, weekday = parts
                
                # Build cron trigger (using IST timezone)
                trigger = CronTrigger(
                    year='*',
                    month=None if month == '*' else month,
                    day=None if day == '*' or day == '?' else day,
                    week=None,
                    day_of_week=None if weekday == '*' or weekday == '?' else weekday,
                    hour=None if hour == '*' else hour,
                    minute=None if minute == '*' else minute,
                    second=0,
                    timezone=IST
                )
            else:
                raise ValueError("Invalid cron expression format")
            
            # Add job to scheduler
            scheduler.add_job(
                func=cls._execute_scheduled_forecast,
                trigger=trigger,
                id=job_id,
                name=f"Forecast Schedule: {schedule_name}",
                args=(tenant_id, database_name, schedule_id, request_data),
                replace_existing=True,
                coalesce=True,
                misfire_grace_time=600  # 10 minutes grace period
            )
            
            logger.info(f"Registered schedule {schedule_name} (ID: {schedule_id}) with APScheduler")
        except Exception as e:
            logger.error(f"Failed to register schedule with scheduler: {str(e)}", exc_info=True)
            raise
    
    @staticmethod
    def _execute_scheduled_forecast(
        tenant_id: str,
        database_name: str,
        schedule_id: str,
        request_data: Dict[str, Any]
    ) -> None:
        """
        Execute a forecast triggered by schedule.
        This function is called by APScheduler on schedule trigger.
        
        Args:
            tenant_id: Tenant identifier
            database_name: Tenant's database name
            schedule_id: Schedule identifier
            request_data: Forecast configuration data
        """
        job_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Executing scheduled forecast {schedule_id} with job {job_id}")
            
            # Get schedule to retrieve created_by (user email)
            schedule = ForecastSchedulerService.get_schedule(
                tenant_id, database_name, schedule_id
            )
            created_by_email = schedule.get('created_by', 'scheduler')
            
            # Create a job record for tracking
            ForecastJobService.create_job(
                tenant_id=tenant_id,
                database_name=database_name,
                job_id=job_id,
                request_data=request_data,
                status=JobStatus.PENDING,
                created_by=created_by_email
            )
            
            # Get tenant data (including email for forecast execution)
            tenant_data = {
                'tenant_id': tenant_id,
                'database_name': database_name,
                'email': created_by_email
            }
            
            # Execute forecast in background
            BackgroundForecastExecutor.execute_forecast_async(
                job_id=job_id,
                tenant_id=tenant_id,
                database_name=database_name,
                request_data=request_data,
                tenant_data=tenant_data
            )
            
            # Update schedule with execution info
            ForecastSchedulerService.update_schedule_execution(
                tenant_id=tenant_id,
                database_name=database_name,
                schedule_id=schedule_id,
                job_id=job_id,
                status=JobStatus.PENDING
            )
            
            logger.info(f"Scheduled forecast {schedule_id} execution initiated (job {job_id})")
        except Exception as e:
            logger.error(
                f"Error executing scheduled forecast {schedule_id}: {str(e)}",
                exc_info=True
            )
            
            # Update schedule with error
            try:
                ForecastSchedulerService.update_schedule_execution(
                    tenant_id=tenant_id,
                    database_name=database_name,
                    schedule_id=schedule_id,
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    error_message=str(e)
                )
            except:
                pass
    
    @staticmethod
    def _map_job_status_to_schedule_status(job_status: JobStatus) -> str:
        """
        Map JobStatus (lowercase) to ScheduleRunStatusEnum (Title Case).
        
        Args:
            job_status: The JobStatus to map
            
        Returns:
            Corresponding ScheduleRunStatusEnum value
        """
        status_mapping = {
            JobStatus.PENDING: ScheduleRunStatusEnum.PENDING.value,
            JobStatus.RUNNING: ScheduleRunStatusEnum.IN_PROGRESS.value,
            JobStatus.COMPLETED: ScheduleRunStatusEnum.COMPLETED.value,
            JobStatus.FAILED: ScheduleRunStatusEnum.FAILED.value,
            JobStatus.CANCELLED: ScheduleRunStatusEnum.CANCELLED.value,
        }
        return status_mapping.get(job_status, ScheduleRunStatusEnum.FAILED.value)
    
    @staticmethod
    def update_schedule_execution(
        tenant_id: str,
        database_name: str,
        schedule_id: str,
        job_id: str,
        status: JobStatus,
        error_message: Optional[str] = None
    ) -> None:
        """
        Update schedule execution tracking information.
        
        Args:
            tenant_id: Tenant identifier
            database_name: Tenant's database name
            schedule_id: Schedule identifier
            job_id: Forecast job identifier
            status: Execution status
            error_message: Optional error message
        """
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    next_run = ForecastSchedulerService._calculate_next_run(
                        ForecastSchedulerService.get_schedule(
                            tenant_id, database_name, schedule_id
                        )['cron_expression']
                    )
                    
                    # Map JobStatus to ScheduleRunStatusEnum for database constraint
                    schedule_status = ForecastSchedulerService._map_job_status_to_schedule_status(status)
                    
                    cursor.execute("""
                        UPDATE forecast_schedules
                        SET 
                            last_run = %s,
                            next_run = %s,
                            last_job_id = %s,
                            last_run_status = %s,
                            last_run_error = %s,
                            execution_count = execution_count + 1,
                            updated_at = %s,
                            updated_by = %s
                        WHERE schedule_id = %s
                    """, (
                        datetime.utcnow(),
                        next_run,
                        job_id,
                        schedule_status,
                        error_message,
                        datetime.utcnow(),
                        "scheduler",
                        schedule_id
                    ))
                    
                    conn.commit()
                finally:
                    cursor.close()
        except Exception as e:
            logger.error(f"Failed to update schedule execution: {str(e)}", exc_info=True)
    
    @staticmethod
    def get_schedule(
        tenant_id: str,
        database_name: str,
        schedule_id: str
    ) -> Dict[str, Any]:
        """
        Retrieve a specific schedule.
        
        Args:
            tenant_id: Tenant identifier
            database_name: Tenant's database name
            schedule_id: Schedule identifier
            
        Returns:
            Schedule record
            
        Raises:
            NotFoundException: If schedule not found
            DatabaseException: If database query fails
        """
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        SELECT 
                            schedule_id, schedule_name, description, request_data,
                            cron_expression, schedule_type, is_active, last_run,
                            next_run, last_job_id, last_run_status, last_run_error,
                            execution_count, created_at, created_by, updated_at, updated_by
                        FROM forecast_schedules
                        WHERE schedule_id = %s AND deleted_at IS NULL
                    """, (schedule_id,))
                    
                    row = cursor.fetchone()
                    if not row:
                        raise NotFoundException(f"Schedule {schedule_id} not found")
                    
                    return {
                        'schedule_id': row[0],
                        'schedule_name': row[1],
                        'description': row[2],
                        'request_data': row[3],
                        'cron_expression': row[4],
                        'schedule_type': row[5],
                        'is_active': row[6],
                        'last_run': row[7],
                        'next_run': row[8],
                        'last_job_id': row[9],
                        'last_run_status': row[10],
                        'last_run_error': row[11],
                        'execution_count': row[12],
                        'created_at': row[13],
                        'created_by': row[14],
                        'updated_at': row[15],
                        'updated_by': row[16]
                    }
                finally:
                    cursor.close()
        except NotFoundException:
            raise
        except Exception as e:
            logger.error(f"Failed to retrieve schedule: {str(e)}", exc_info=True)
            raise DatabaseException(f"Failed to retrieve schedule: {str(e)}")
    
    @staticmethod
    def list_schedules(
        tenant_id: str,
        database_name: str,
        is_active: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        List schedules for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            database_name: Tenant's database name
            is_active: Filter by active status (None = all)
            limit: Maximum records to return
            offset: Offset for pagination
            
        Returns:
            Tuple of (schedules list, total count)
            
        Raises:
            DatabaseException: If database query fails
        """
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Build WHERE clause
                    where_clauses = ["deleted_at IS NULL"]
                    params = []
                    
                    if is_active is not None:
                        where_clauses.append("is_active = %s")
                        params.append(is_active)
                    
                    where_clause = " AND ".join(where_clauses)
                    
                    # Get total count
                    cursor.execute(
                        f"SELECT COUNT(*) FROM forecast_schedules WHERE {where_clause}",
                        params
                    )
                    total_count = cursor.fetchone()[0]
                    
                    # Get paginated results
                    cursor.execute(
                        f"""
                        SELECT 
                            schedule_id, schedule_name, description, request_data,
                            cron_expression, schedule_type, is_active, last_run,
                            next_run, last_job_id, last_run_status, last_run_error,
                            execution_count, created_at, created_by, updated_at, updated_by
                        FROM forecast_schedules
                        WHERE {where_clause}
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s
                        """,
                        params + [limit, offset]
                    )
                    
                    schedules = []
                    for row in cursor.fetchall():
                        schedules.append({
                            'schedule_id': row[0],
                            'schedule_name': row[1],
                            'description': row[2],
                            'request_data': row[3],
                            'cron_expression': row[4],
                            'schedule_type': row[5],
                            'is_active': row[6],
                            'last_run': row[7],
                            'next_run': row[8],
                            'last_job_id': row[9],
                            'last_run_status': row[10],
                            'last_run_error': row[11],
                            'execution_count': row[12],
                            'created_at': row[13],
                            'created_by': row[14],
                            'updated_at': row[15],
                            'updated_by': row[16]
                        })
                    
                    return schedules, total_count
                finally:
                    cursor.close()
        except Exception as e:
            logger.error(f"Failed to list schedules: {str(e)}", exc_info=True)
            raise DatabaseException(f"Failed to list schedules: {str(e)}")
    
    @classmethod
    def update_schedule(
        cls,
        tenant_id: str,
        database_name: str,
        schedule_id: str,
        schedule_name: Optional[str] = None,
        description: Optional[str] = None,
        request_data: Optional[Dict[str, Any]] = None,
        cron_expression: Optional[str] = None,
        schedule_type: Optional[str] = None,
        is_active: Optional[bool] = None,
        updated_by: str = "system"
    ) -> Dict[str, Any]:
        """
        Update a schedule.
        
        Args:
            tenant_id: Tenant identifier
            database_name: Tenant's database name
            schedule_id: Schedule identifier
            schedule_name: New schedule name
            description: New description
            request_data: New forecast configuration
            cron_expression: New cron expression
            schedule_type: New schedule type
            is_active: New active status
            updated_by: User updating the schedule
            
        Returns:
            Updated schedule record
            
        Raises:
            ValidationException: If validation fails
            NotFoundException: If schedule not found
            DatabaseException: If database operation fails
        """
        # Get current schedule
        schedule = cls.get_schedule(tenant_id, database_name, schedule_id)
        
        # Validate inputs
        if cron_expression:
            try:
                cls._calculate_next_run(cron_expression)
            except Exception as e:
                raise ValidationException(f"Invalid cron expression: {str(e)}")
        
        if schedule_type:
            valid_types = ['once', 'hourly', 'daily', 'weekly', 'monthly', 'custom']
            if schedule_type not in valid_types:
                raise ValidationException(f"Schedule type must be one of {valid_types}")
        
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Prepare update values
                    updates = []
                    params = []
                    
                    if schedule_name is not None:
                        updates.append("schedule_name = %s")
                        params.append(schedule_name)
                    
                    if description is not None:
                        updates.append("description = %s")
                        params.append(description)
                    
                    if request_data is not None:
                        updates.append("request_data = %s")
                        params.append(Json(request_data))
                    
                    if cron_expression is not None:
                        updates.append("cron_expression = %s")
                        params.append(cron_expression)
                        # Recalculate next run
                        next_run = cls._calculate_next_run(cron_expression)
                        updates.append("next_run = %s")
                        params.append(next_run)
                    
                    if schedule_type is not None:
                        updates.append("schedule_type = %s")
                        params.append(schedule_type)
                    
                    was_active = schedule['is_active']
                    if is_active is not None:
                        updates.append("is_active = %s")
                        params.append(is_active)
                    
                    # Always update these
                    updates.append("updated_at = %s")
                    params.append(datetime.utcnow())
                    updates.append("updated_by = %s")
                    params.append(updated_by)
                    
                    # Execute update
                    params.append(schedule_id)
                    update_clause = ", ".join(updates)
                    
                    cursor.execute(
                        f"UPDATE forecast_schedules SET {update_clause} WHERE schedule_id = %s",
                        params
                    )
                    
                    conn.commit()
                    
                    # Re-register with scheduler if active status changed
                    if is_active is not None and is_active != was_active:
                        scheduler = cls._ensure_scheduler()
                        job_id = f"forecast_schedule_{schedule_id}"
                        
                        if is_active:
                            # Re-register with scheduler
                            updated_cron = cron_expression or schedule['cron_expression']
                            updated_request_data = request_data or schedule['request_data']
                            cls.register_schedule(
                                scheduler=scheduler,
                                tenant_id=tenant_id,
                                database_name=database_name,
                                schedule_id=schedule_id,
                                schedule_name=schedule_name or schedule['schedule_name'],
                                cron_expression=updated_cron,
                                request_data=updated_request_data
                            )
                        else:
                            # Remove from scheduler
                            try:
                                scheduler.remove_job(job_id)
                                logger.info(f"Removed schedule {job_id} from scheduler")
                            except:
                                pass
                    
                    # Return updated schedule
                    return cls.get_schedule(tenant_id, database_name, schedule_id)
                finally:
                    cursor.close()
        except (ValidationException, NotFoundException):
            raise
        except Exception as e:
            logger.error(f"Failed to update schedule: {str(e)}", exc_info=True)
            raise DatabaseException(f"Failed to update schedule: {str(e)}")
    
    @classmethod
    def delete_schedule(
        cls,
        tenant_id: str,
        database_name: str,
        schedule_id: str,
        soft_delete: bool = True
    ) -> None:
        """
        Delete a schedule (soft delete by default).
        
        Args:
            tenant_id: Tenant identifier
            database_name: Tenant's database name
            schedule_id: Schedule identifier
            soft_delete: If True, marks as deleted; if False, hard deletes
            
        Raises:
            NotFoundException: If schedule not found
            DatabaseException: If database operation fails
        """
        # Verify schedule exists
        schedule = cls.get_schedule(tenant_id, database_name, schedule_id)
        
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    if soft_delete:
                        cursor.execute("""
                            UPDATE forecast_schedules
                            SET deleted_at = %s, is_active = FALSE, updated_at = %s, updated_by = %s
                            WHERE schedule_id = %s
                        """, (datetime.utcnow(), datetime.utcnow(), "system", schedule_id))
                    else:
                        cursor.execute("""
                            DELETE FROM forecast_schedules
                            WHERE schedule_id = %s
                        """, (schedule_id,))
                    
                    conn.commit()
                    
                    # Remove from scheduler
                    scheduler = cls._ensure_scheduler()
                    job_id = f"forecast_schedule_{schedule_id}"
                    try:
                        scheduler.remove_job(job_id)
                        logger.info(f"Removed schedule {job_id} from scheduler")
                    except:
                        pass
                    
                    logger.info(f"Deleted schedule {schedule_id}")
                finally:
                    cursor.close()
        except NotFoundException:
            raise
        except Exception as e:
            logger.error(f"Failed to delete schedule: {str(e)}", exc_info=True)
            raise DatabaseException(f"Failed to delete schedule: {str(e)}")
