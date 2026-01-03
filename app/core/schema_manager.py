"""
Schema management utilities for tenant isolation with FULL AUDIT TRAILS.
Handles tenant database creation and table setup with proper audit fields.
"""

from typing import List, Dict, Any
from app.models.database_models import FieldDefinition
from app.core.database import get_db_manager
from app.core.exceptions import DatabaseException, ValidationException
import logging

logger = logging.getLogger(__name__)

class SchemaManager:
    """Manages database schemas and tenant isolation with audit trails."""
    
    @staticmethod
    def create_tenant_database(tenant_id: str, database_name: str) -> bool:
        """
        Create a new database for tenant and initialize tables with audit trails.
        
        Args:
            tenant_id: Unique tenant identifier
            database_name: Name for the tenant database
            
        Returns:
            True if database created successfully
            
        Raises:
            DatabaseException: If database creation fails
        """
        db_manager = get_db_manager()
        
        try:
            # Create the database
            db_manager.create_tenant_database(database_name)
            
            # Initialize tables in the new database
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # ========================================================================
                    # Create field_catalogue table with FULL AUDIT TRAIL
                    # ========================================================================
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS field_catalogue (
                            catalogue_id UUID PRIMARY KEY,
                            version INT DEFAULT 1,
                            status VARCHAR(50) DEFAULT 'DRAFT',
                            fields_json JSONB NOT NULL,
                            
                            -- Audit fields
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            created_by VARCHAR(255) NOT NULL,
                            updated_at TIMESTAMP,
                            updated_by VARCHAR(255)
                        )
                    """)
                    logger.info(" Created field_catalogue table with audit trail")
                    
                    # ========================================================================
                    # Create master_data table (modified when catalogue finalized)
                    # Will have FULL AUDIT TRAIL added later
                    # ========================================================================
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS master_data (
                            master_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            
                            -- Audit fields (will be customized based on field catalogue)
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            created_by VARCHAR(255) NOT NULL,
                            updated_at TIMESTAMP,
                            updated_by VARCHAR(255),
                            deleted_at TIMESTAMP
                        )
                    """)
                    logger.info(" Created master_data table with audit trail")
                    
                    # ========================================================================
                    # Create sales_data table with PARTIAL AUDIT (created only - immutable)
                    # ========================================================================
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS sales_data (
                            sales_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            master_id UUID NOT NULL REFERENCES master_data(master_id),
                            date DATE NOT NULL,
                            quantity DECIMAL(18, 2) NOT NULL,
                            uom VARCHAR(20) NOT NULL,
                            unit_price DECIMAL(18, 2),
                            
                            -- Audit fields (created only - transactions are immutable)
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            created_by VARCHAR(255) NOT NULL
                        )
                    """)
                    logger.info(" Created sales_data table with partial audit trail")
                    
                    # ========================================================================
                    # Create users table with FULL AUDIT TRAIL
                    # ========================================================================
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS users (
                            user_id UUID PRIMARY KEY,
                            tenant_id UUID NOT NULL,
                            email VARCHAR(255) NOT NULL UNIQUE,
                            password_hash VARCHAR(255) NOT NULL,
                            first_name VARCHAR(100),
                            last_name VARCHAR(100),
                            role VARCHAR(50) NOT NULL DEFAULT 'user',
                            status VARCHAR(50) NOT NULL DEFAULT 'ACTIVE',
                            
                            -- Audit fields
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            created_by VARCHAR(255) DEFAULT 'system',
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_by VARCHAR(255),
                            last_login TIMESTAMP,
                            
                            CONSTRAINT check_user_role CHECK (role IN ('admin', 'user', 'manager')),
                            CONSTRAINT check_user_status CHECK (status IN ('ACTIVE', 'INACTIVE', 'SUSPENDED'))
                        )
                    """)
                    logger.info(" Created users table with full audit trail")

                    # ========================================================================
                    # Create upload_history table (created only - immutable)
                    # ========================================================================
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS upload_history (
                            upload_id UUID PRIMARY KEY,
                            upload_type VARCHAR(50) NOT NULL,
                            file_name VARCHAR(255) NOT NULL,
                            total_rows INT DEFAULT 0,
                            success_count INT DEFAULT 0,
                            failed_count INT DEFAULT 0,
                            status VARCHAR(50) NOT NULL,
                            
                            -- Audit fields (created only - upload records are immutable)
                            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            uploaded_by VARCHAR(255) NOT NULL
                        )
                    """)
                    logger.info(" Created upload_history table with partial audit trail")
                    
                    # Create indexes
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sales_data_date ON sales_data(date)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sales_data_master_id ON sales_data(master_id)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_master_data_deleted ON master_data(deleted_at) WHERE deleted_at IS NULL")
                    
                    # ========================================================================
                    # Create triggers for auto-updating updated_at
                    # ========================================================================
                    cursor.execute("""
                        CREATE OR REPLACE FUNCTION update_updated_at_column()
                        RETURNS TRIGGER AS $$
                        BEGIN
                            NEW.updated_at = CURRENT_TIMESTAMP;
                            RETURN NEW;
                        END;
                        $$ language 'plpgsql';
                    """)
                    
                    cursor.execute("""
                        DROP TRIGGER IF EXISTS update_users_updated_at ON users;
                    """)
                    cursor.execute("""
                        CREATE TRIGGER update_users_updated_at 
                            BEFORE UPDATE ON users
                            FOR EACH ROW
                            EXECUTE FUNCTION update_updated_at_column();
                    """)
                    logger.info(" Created auto-update triggers")
                    
                    conn.commit()
                    logger.info(f" Database initialized successfully for tenant: {tenant_id}")

                    # Initialize forecasting tables and seed default data
                    SchemaManager.initialize_forecasting_tables(tenant_id, database_name)
                    SchemaManager.seed_default_algorithms(tenant_id, database_name)
                    SchemaManager.seed_default_versions(tenant_id, database_name, created_by="system")

                    return True
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"Failed to create tenant database {database_name}: {str(e)}")
            raise DatabaseException(f"Failed to create tenant database: {str(e)}")
    
    @staticmethod
    def create_master_data_table(
        tenant_id: str,
        database_name: str,
        fields: List[FieldDefinition]
    ) -> bool:
        """
        Create dynamic master data table based on field catalogue with FULL AUDIT TRAIL.
        Also creates sales_data table with dynamic target and date columns.
        
        Args:
            tenant_id: Tenant identifier
            database_name: Tenant's database name
            fields: List of field definitions
        Returns:
            True if table created successfully
        Raises:
            DatabaseException: If table creation fails
        """
        db_manager = get_db_manager()
        
        try:
            # Find target variable and date field
            target_field = None
            date_field = None
            master_fields = []
            
            for field in fields:
                if field.is_target_variable:
                    if target_field is not None:
                        raise ValidationException("Only one field can be marked as target variable")
                    target_field = field
                elif field.is_date_field:
                    if date_field is not None:
                        raise ValidationException("Only one field can be marked as date field")
                    date_field = field
                else:
                    master_fields.append(field)
            
            # Validate required fields
            if target_field is None:
                raise ValidationException("Field catalogue must have one target variable field")
            if date_field is None:
                raise ValidationException("Field catalogue must have one date field")
            
            # Validate data types
            if target_field.data_type.upper() not in ['DECIMAL']:
                raise ValidationException("Target variable must be DECIMAL type")
            if date_field.data_type.upper() not in ['DATE', 'TIMESTAMP']:
                raise ValidationException("Date field must be DATE or TIMESTAMP type")
            
            logger.info(f"Creating tables with target variable: {target_field.field_name}, date field: {date_field.field_name}")
            
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Drop existing tables
                    cursor.execute("DROP TABLE IF EXISTS sales_data CASCADE")
                    cursor.execute("DROP TABLE IF EXISTS master_data CASCADE")
                    
                    # ====================================================================
                    # Build master_data table columns with FULL AUDIT TRAIL
                    # ====================================================================
                    columns = []
                    columns.append("master_id UUID PRIMARY KEY DEFAULT gen_random_uuid()")
                    
                    field_names = []
                    for field in master_fields:
                        sql_type = field.get_sql_type()
                        columns.append(f'"{field.field_name}" {sql_type}')
                        field_names.append(f'"{field.field_name}"')
                    
                    # Add FULL AUDIT columns
                    columns.extend([
                        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                        "created_by VARCHAR(255) NOT NULL",
                        "updated_at TIMESTAMP",
                        "updated_by VARCHAR(255)",
                        "deleted_at TIMESTAMP"
                    ])

                    # Create master_data table
                    create_table_sql = f"CREATE TABLE master_data ({', '.join(columns)});"
                    cursor.execute(create_table_sql)

                    # Create composite unique constraint for ON CONFLICT support
                    composite_fields = field_names
                    constraint_columns = ', '.join(composite_fields)
                    cursor.execute(f'ALTER TABLE master_data ADD CONSTRAINT master_data_composite_unique UNIQUE ({constraint_columns})')
                    cursor.execute(f"CREATE INDEX idx_master_data_deleted ON master_data(deleted_at) WHERE deleted_at IS NULL")
                    
                    # Create trigger for auto-updating updated_at
                    cursor.execute("""
                        DROP TRIGGER IF EXISTS update_master_data_updated_at ON master_data;
                    """)
                    cursor.execute("""
                        CREATE TRIGGER update_master_data_updated_at 
                            BEFORE UPDATE ON master_data
                            FOR EACH ROW
                            EXECUTE FUNCTION update_updated_at_column();
                    """)
                    
                    # ====================================================================
                    # Create sales_data table with dynamic columns (PARTIAL AUDIT)
                    # ====================================================================
                    target_sql_type = target_field.get_sql_type()
                    date_sql_type = date_field.get_sql_type()
                    
                    cursor.execute(f"""
                        CREATE TABLE sales_data (
                            sales_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            master_id UUID NOT NULL REFERENCES master_data(master_id),
                            "{date_field.field_name}" {date_sql_type} NOT NULL,
                            "{target_field.field_name}" {target_sql_type} NOT NULL,
                            uom VARCHAR(20) NOT NULL,
                            unit_price DECIMAL(18, 2),

                            -- Audit fields (created only - transactions are immutable)
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            created_by VARCHAR(255) NOT NULL
                        )
                    """)

                    # ====================================================================
                    # Create forecast_data table with dynamic columns (SIMILAR TO sales_data)
                    # ====================================================================
                    cursor.execute(f"""
                        CREATE TABLE IF NOT EXISTS forecast_data (
                            forecast_data_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            master_id UUID NOT NULL REFERENCES master_data(master_id),
                            forecast_run_id UUID NOT NULL REFERENCES forecast_runs(forecast_run_id) ON DELETE CASCADE,
                            "{date_field.field_name}" {date_sql_type} NOT NULL,
                            "{target_field.field_name}" {target_sql_type} NOT NULL,
                            uom VARCHAR(20) NOT NULL,
                            unit_price DECIMAL(18, 2),

                            -- Audit fields (created only - transactions are immutable)
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            created_by VARCHAR(255) NOT NULL
                        )
                    """)

                    # Add unique constraint for batch upsert functionality
                    cursor.execute(f'ALTER TABLE sales_data ADD CONSTRAINT sales_data_master_date_unique UNIQUE (master_id, "{date_field.field_name}")')
                    cursor.execute(f'ALTER TABLE forecast_data ADD CONSTRAINT forecast_data_master_date_run_unique UNIQUE (master_id, "{date_field.field_name}", forecast_run_id)')

                    # Create indexes on sales_data and forecast_data
                    cursor.execute(f'CREATE INDEX idx_sales_data_date ON sales_data("{date_field.field_name}")')
                    cursor.execute(f'CREATE INDEX idx_sales_data_master_id ON sales_data(master_id)')
                    
                    cursor.execute(f'CREATE INDEX idx_forecast_data_date ON forecast_data("{date_field.field_name}")')
                    cursor.execute(f'CREATE INDEX idx_forecast_data_master_id ON forecast_data(master_id)')
                    cursor.execute(f'CREATE INDEX idx_forecast_data_run_id ON forecast_data(forecast_run_id)')
                    
                    # Store metadata about target and date fields
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS field_catalogue_metadata (
                            target_field_name VARCHAR(255) NOT NULL,
                            date_field_name VARCHAR(255) NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    # Check if metadata exists
                    cursor.execute("SELECT 1 FROM field_catalogue_metadata LIMIT 1")
                    if cursor.fetchone():
                        cursor.execute("""
                            UPDATE field_catalogue_metadata 
                            SET target_field_name = %s,
                                date_field_name = %s,
                                created_at = CURRENT_TIMESTAMP
                        """, (target_field.field_name, date_field.field_name))
                    else:
                        cursor.execute("""
                            INSERT INTO field_catalogue_metadata (target_field_name, date_field_name)
                            VALUES (%s, %s)
                        """, (target_field.field_name, date_field.field_name))
                    
                    conn.commit()
                    logger.info(
                        f" Created master_data and sales_data tables in {database_name} with "
                        f"target: {target_field.field_name}, date: {date_field.field_name}"
                    )
                    return True
                finally:
                    cursor.close()
        except ValidationException:
            raise
        except Exception as e:
            logger.error(f"Failed to create tables in {database_name}: {str(e)}")
            raise DatabaseException(f"Failed to create tables: {str(e)}")
        
    @staticmethod
    def add_upload_history_table(tenant_id: str, database_name: str) -> bool:
        """
        Ensure upload_history table exists in tenant database.
        
        Args:
            tenant_id: Tenant identifier
            database_name: Tenant's database name
            
        Returns:
            True if successful
        """
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS upload_history (
                            upload_id UUID PRIMARY KEY,
                            upload_type VARCHAR(50) NOT NULL,
                            file_name VARCHAR(255) NOT NULL,
                            total_rows INT DEFAULT 0,
                            success_count INT DEFAULT 0,
                            failed_count INT DEFAULT 0,
                            status VARCHAR(50) NOT NULL,
                            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            uploaded_by VARCHAR(255) NOT NULL
                        )
                    """)
                    return True
                finally:
                    cursor.close()
        except Exception as e:
            logger.error(f"Failed to ensure upload_history table in {database_name}: {str(e)}")
            raise DatabaseException(f"Failed to ensure upload history table: {str(e)}")
    
    @staticmethod
    def table_exists(tenant_id: str, database_name: str, table_name: str) -> bool:
        """
        Check if a table exists in tenant database.
        
        Args:
            tenant_id: Tenant identifier
            database_name: Tenant's database name
            table_name: Table name to check
            
        Returns:
            True if table exists
        """
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = 'public' 
                            AND table_name = %s
                        )
                    """, (table_name,))
                    return cursor.fetchone()[0]
                finally:
                    cursor.close()
        except Exception as e:
            logger.error(f"Failed to check if table exists: {str(e)}")
            return False
    
    @staticmethod
    def validate_tenant_database(database_name: str) -> Dict[str, Any]:
        """
        Validate tenant database structure and report any issues.
        
        Args:
            database_name: Tenant's database name
            
        Returns:
            Dictionary with validation results
            
        Raises:
            DatabaseException: If validation fails
        """
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Get all tables in the database
                    cursor.execute("""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = 'public'
                    """)
                    
                    tables = {row[0] for row in cursor.fetchall()}
                    
                    # Check required tables
                    required_tables = {'field_catalogue', 'master_data', 'sales_data', 'upload_history'}
                    missing_tables = required_tables - tables
                    
                    # Get row counts
                    row_counts = {}
                    for table in required_tables:
                        if table in tables:
                            cursor.execute(f"SELECT COUNT(*) FROM {table}")
                            row_counts[table] = cursor.fetchone()[0]
                    
                    return {
                        "database_name": database_name,
                        "all_tables_exist": len(missing_tables) == 0,
                        "missing_tables": list(missing_tables),
                        "row_counts": row_counts,
                        "valid": len(missing_tables) == 0
                    }
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"Failed to validate database {database_name}: {str(e)}")
            raise DatabaseException(f"Database validation failed: {str(e)}")

    @staticmethod
    def drop_tenant_database(tenant_id: str, database_name: str) -> bool:
        """
        Drop tenant's database (use with caution!).
        
        Args:
            tenant_id: Tenant identifier
            database_name: Tenant's database name
            
        Returns:
            True if successful
        """
        db_manager = get_db_manager()
        
        try:
            db_manager.drop_tenant_database(database_name)
            logger.info(f"Dropped database for tenant {tenant_id}: {database_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to drop tenant database: {str(e)}")
            raise DatabaseException(f"Failed to drop tenant database: {str(e)}")

    @staticmethod
    def initialize_forecasting_tables(tenant_id: str, database_name: str) -> bool:
        """
        Initialize all forecasting tables in tenant database with PROPER AUDIT TRAILS.
        
        Args:
            tenant_id: Tenant identifier
            database_name: Tenant's database name
            
        Returns:
            True if initialization successful
            
        Raises:
            DatabaseException: If initialization fails
        """
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Read and execute forecasting schema with audit trails
                    forecasting_schema = """
                    -- ========================================================================
                    -- Algorithms table (MINIMAL AUDIT - system config)
                    -- ========================================================================
                    CREATE TABLE IF NOT EXISTS algorithms (
                        algorithm_id SERIAL PRIMARY KEY,
                        algorithm_name VARCHAR(255) NOT NULL UNIQUE,
                        default_parameters JSONB NOT NULL,
                        algorithm_type VARCHAR(50) NOT NULL,
                        description TEXT,
                        
                        -- Audit fields (minimal - system configuration)
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        
                        CONSTRAINT check_algorithm_type CHECK (algorithm_type IN ('ML', 'Statistic', 'Hybrid'))
                    );

                    -- ========================================================================
                    -- Forecast Versions table (FULL AUDIT TRAIL)
                    -- ========================================================================
                    CREATE TABLE IF NOT EXISTS forecast_versions (
                        version_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        version_name VARCHAR(255) NOT NULL,
                        version_type VARCHAR(50) NOT NULL,
                        is_active BOOLEAN DEFAULT FALSE,
                        
                        -- Audit fields (full trail - user configuration)
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_by VARCHAR(255),
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_by VARCHAR(255),
                        
                        CONSTRAINT check_version_type CHECK (version_type IN ('Baseline', 'Simulation', 'Final')),
                        CONSTRAINT unique_version_name UNIQUE(version_name)
                    );

                    CREATE INDEX IF NOT EXISTS idx_forecast_versions_active ON forecast_versions(is_active);
                    CREATE INDEX IF NOT EXISTS idx_forecast_versions_type ON forecast_versions(version_type);

                    -- ========================================================================
                    -- External Factors table (FULL AUDIT TRAIL + SOFT DELETE)
                    -- ========================================================================
                    CREATE TABLE IF NOT EXISTS external_factors (
                        factor_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        date DATE NOT NULL,
                        factor_name VARCHAR(255) NOT NULL,
                        factor_value DECIMAL(18, 4) NOT NULL,
                        unit VARCHAR(50),
                        source VARCHAR(255),
                        
                        -- Audit fields (full trail with soft delete)
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_by VARCHAR(255),
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_by VARCHAR(255),
                        deleted_at TIMESTAMP,
                        
                        CONSTRAINT unique_factor_per_date UNIQUE (factor_name, date)
                    );

                    CREATE INDEX IF NOT EXISTS idx_external_factors_date ON external_factors(date);
                    CREATE INDEX IF NOT EXISTS idx_external_factors_name ON external_factors(factor_name);
                    CREATE INDEX IF NOT EXISTS idx_external_factors_composite ON external_factors(factor_name, date);
                    CREATE INDEX IF NOT EXISTS idx_external_factors_active ON external_factors(factor_name, date) WHERE deleted_at IS NULL;

                    -- ========================================================================
                    -- Forecast Runs table (FULL AUDIT TRAIL)
                    -- ========================================================================
                    CREATE TABLE IF NOT EXISTS forecast_runs (
                        forecast_run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        version_id UUID NOT NULL REFERENCES forecast_versions(version_id) ON DELETE CASCADE,
                        forecast_filters JSONB,
                        forecast_start DATE NOT NULL,
                        forecast_end DATE NOT NULL,
                        history_start DATE,
                        history_end DATE,
                        run_status VARCHAR(50) NOT NULL DEFAULT 'Pending',
                        run_progress INTEGER DEFAULT 0,
                        total_records INTEGER DEFAULT 0,
                        processed_records INTEGER DEFAULT 0,
                        failed_records INTEGER DEFAULT 0,
                        error_message TEXT,
                        
                        -- Audit fields (full trail)
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_by VARCHAR(255),
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_by VARCHAR(255),
                        started_at TIMESTAMP,
                        completed_at TIMESTAMP,
                        
                        CONSTRAINT check_run_status CHECK (run_status IN ('Pending', 'In-Progress', 'Completed', 'Completed with Errors', 'Failed', 'Cancelled')),
                        CONSTRAINT check_run_progress CHECK (run_progress >= 0 AND run_progress <= 100),
                        CONSTRAINT check_forecast_dates CHECK (forecast_end >= forecast_start)
                    );

                    CREATE INDEX IF NOT EXISTS idx_forecast_runs_status ON forecast_runs(run_status);
                    CREATE INDEX IF NOT EXISTS idx_forecast_runs_version ON forecast_runs(version_id);
                    CREATE INDEX IF NOT EXISTS idx_forecast_runs_created ON forecast_runs(created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_forecast_runs_composite ON forecast_runs(run_status, created_at DESC);

                    -- ========================================================================
                    -- Forecast Algorithms Mapping table (PARTIAL AUDIT)
                    -- ========================================================================
                    CREATE TABLE IF NOT EXISTS forecast_algorithms_mapping (
                        mapping_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        forecast_run_id UUID NOT NULL REFERENCES forecast_runs(forecast_run_id) ON DELETE CASCADE,
                        algorithm_id INTEGER NOT NULL REFERENCES algorithms(algorithm_id),
                        algorithm_name VARCHAR(255) NOT NULL,
                        custom_parameters JSONB,
                        execution_order INTEGER NOT NULL DEFAULT 1,
                        execution_status VARCHAR(50) DEFAULT 'Pending',
                        started_at TIMESTAMP,
                        completed_at TIMESTAMP,
                        error_message TEXT,
                        
                        -- Audit fields (created only)
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_by VARCHAR(255),
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        
                        CONSTRAINT check_execution_status CHECK (execution_status IN ('Pending', 'Running', 'Completed', 'Failed')),
                        CONSTRAINT unique_algo_per_run UNIQUE(forecast_run_id, algorithm_id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_forecast_algo_mapping_run ON forecast_algorithms_mapping(forecast_run_id);
                    CREATE INDEX IF NOT EXISTS idx_forecast_algo_mapping_algo ON forecast_algorithms_mapping(algorithm_id);
                    CREATE INDEX IF NOT EXISTS idx_forecast_algo_mapping_status ON forecast_algorithms_mapping(forecast_run_id, execution_status);

                    -- ========================================================================
                    -- Forecast Results table (CREATED ONLY - immutable forecast outputs)
                    -- ========================================================================
                    CREATE TABLE IF NOT EXISTS forecast_results (
                        result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        forecast_run_id UUID NOT NULL REFERENCES forecast_runs(forecast_run_id) ON DELETE CASCADE,
                        version_id UUID NOT NULL REFERENCES forecast_versions(version_id) ON DELETE CASCADE,
                        mapping_id UUID NOT NULL REFERENCES forecast_algorithms_mapping(mapping_id) ON DELETE CASCADE,
                        algorithm_id INTEGER NOT NULL REFERENCES algorithms(algorithm_id),
                        
                        -- Core forecast data (UPDATED SCHEMA)
                        date DATE NOT NULL,
                        value DECIMAL(18, 4) NOT NULL,
                        type VARCHAR(50) NOT NULL,
                        
                        -- Confidence intervals (only for future_forecast)
                        confidence_interval_lower DECIMAL(18, 4),
                        confidence_interval_upper DECIMAL(18, 4),
                        confidence_level VARCHAR(20),
                        
                        -- Accuracy metrics (only for testing_forecast)
                        accuracy_metric DECIMAL(5, 2),
                        metric_type VARCHAR(50),
                        
                        metadata JSONB,
                        
                        -- Audit fields (created only - results are immutable)
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_by VARCHAR(255),
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        
                        CONSTRAINT check_result_type CHECK (type IN ('testing_actual', 'testing_forecast', 'future_forecast'))
                    );

                    CREATE INDEX IF NOT EXISTS idx_forecast_results_run ON forecast_results(forecast_run_id);
                    CREATE INDEX IF NOT EXISTS idx_forecast_results_date ON forecast_results(date);
                    CREATE INDEX IF NOT EXISTS idx_forecast_results_algo ON forecast_results(algorithm_id);
                    CREATE INDEX IF NOT EXISTS idx_forecast_results_composite ON forecast_results(forecast_run_id, date);
                    CREATE INDEX IF NOT EXISTS idx_forecast_results_version ON forecast_results(version_id);
                    CREATE INDEX IF NOT EXISTS idx_forecast_results_type ON forecast_results(type);
                    CREATE INDEX IF NOT EXISTS idx_forecast_results_type_date ON forecast_results(type, date);
                    -- ========================================================================
                    -- Forecast Audit Log table (CREATED ONLY - immutable log entries)
                    -- ========================================================================
                    CREATE TABLE IF NOT EXISTS forecast_audit_log (
                        audit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        forecast_run_id UUID NOT NULL REFERENCES forecast_runs(forecast_run_id) ON DELETE CASCADE,
                        action VARCHAR(50) NOT NULL,
                        entity_type VARCHAR(100),
                        entity_id UUID,
                        details JSONB,
                        
                        -- Audit fields (created only - logs are immutable)
                        performed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        performed_by VARCHAR(255),
                        
                        CONSTRAINT check_action CHECK (action IN ('Created', 'Updated', 'Deleted', 'Executed', 'Cancelled'))
                    );

                    CREATE INDEX IF NOT EXISTS idx_forecast_audit_run ON forecast_audit_log(forecast_run_id);
                    CREATE INDEX IF NOT EXISTS idx_forecast_audit_timestamp ON forecast_audit_log(performed_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_forecast_audit_action ON forecast_audit_log(action);
                    
                    -- ========================================================================
                    -- Create triggers for auto-updating updated_at columns
                    -- ========================================================================
                    DROP TRIGGER IF EXISTS update_forecast_versions_updated_at ON forecast_versions;
                    CREATE TRIGGER update_forecast_versions_updated_at 
                        BEFORE UPDATE ON forecast_versions
                        FOR EACH ROW
                        EXECUTE FUNCTION update_updated_at_column();
                    
                    DROP TRIGGER IF EXISTS update_external_factors_updated_at ON external_factors;
                    CREATE TRIGGER update_external_factors_updated_at 
                        BEFORE UPDATE ON external_factors
                        FOR EACH ROW
                        EXECUTE FUNCTION update_updated_at_column();
                    
                    DROP TRIGGER IF EXISTS update_forecast_runs_updated_at ON forecast_runs;
                    CREATE TRIGGER update_forecast_runs_updated_at 
                        BEFORE UPDATE ON forecast_runs
                        FOR EACH ROW
                        EXECUTE FUNCTION update_updated_at_column();
                    """
                    
                    cursor.execute(forecasting_schema)
                    conn.commit()
                    logger.info(f" Forecasting tables initialized successfully for tenant: {tenant_id} with full audit trails")
                    return True
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"Failed to initialize forecasting tables for {database_name}: {str(e)}")
            raise DatabaseException(f"Failed to initialize forecasting tables: {str(e)}")

    @staticmethod
    def seed_default_algorithms(tenant_id: str, database_name: str) -> bool:
        """
        Seed default algorithms in tenant database.
        
        Args:
            tenant_id: Tenant identifier
            database_name: Tenant's database name
            
        Returns:
            True if seeding successful
            
        Raises:
            DatabaseException: If seeding fails
        """
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Insert default algorithms with specific IDs
                    algorithms_data = [
                        (1, 'ARIMA', '{"order": [1, 1, 1]}', 'Statistic', 'AutoRegressive Integrated Moving Average'),
                        (2, 'Linear Regression', '{}', 'Statistic', 'Linear regression with feature engineering'),
                        (3, 'Polynomial Regression', '{"degree": 2}', 'Statistic', 'Polynomial regression'),
                        (4, 'Exponential Smoothing', '{"alpha": 0.3}', 'Statistic', 'Simple exponential smoothing'),
                        (5, 'Enhanced Exponential Smoothing', '{"alphas": [0.1, 0.3, 0.5]}', 'Statistic', 'Multiple alpha exponential smoothing'),
                        (6, 'Holt Winters', '{"season_length": 12}', 'Statistic', 'Triple exponential smoothing'),
                        (7, 'Prophet', '{"window": 3}', 'Statistic', 'Facebook Prophet algorithm'),
                        (8, 'LSTM Neural Network', '{"window": 3}', 'ML', 'Long Short-Term Memory neural network'),
                        (9, 'XGBoost', '{"n_estimators_list": [50, 100]}', 'ML', 'XGBoost gradient boosting'),
                        (10, 'SVR', '{"C_list": [1, 10, 100]}', 'ML', 'Support Vector Regression'),
                        (11, 'KNN', '{"n_neighbors_list": [7, 10]}', 'ML', 'K-Nearest Neighbors regression'),
                        (12, 'Gaussian Process', '{}', 'ML', 'Gaussian Process Regression'),
                        (13, 'Neural Network', '{"hidden_layer_sizes_list": [[10], [20, 10]]}', 'ML', 'Multi-layer Perceptron'),
                        (14, 'Random Forest', '{"n_estimators": 100}', 'ML', 'Random Forest Regression'),
                        (999, 'Best Fit', '{}', 'Hybrid', 'Advanced AI/ML auto model selection')
                    ]

                    for algo_id, algo_name, default_params, algo_type, description in algorithms_data:
                        cursor.execute("""
                            INSERT INTO algorithms (algorithm_id, algorithm_name, default_parameters, algorithm_type, description)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (algorithm_id) DO NOTHING
                        """, (algo_id, algo_name, default_params, algo_type, description))
                    
                    conn.commit()
                    logger.info(f"✅ Default algorithms seeded for tenant: {tenant_id}")
                    return True
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"Failed to seed algorithms for {database_name}: {str(e)}")
            raise DatabaseException(f"Failed to seed algorithms: {str(e)}")

    @staticmethod
    def seed_default_versions(tenant_id: str, database_name: str, created_by: str = "system") -> bool:
        """
        Seed default forecast versions for tenant.
        
        Args:
            tenant_id: Tenant identifier
            database_name: Tenant's database name
            created_by: User creating the versions
            
        Returns:
            True if seeding successful
            
        Raises:
            DatabaseException: If seeding fails
        """
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Insert default versions
                    versions_data = [
                        ('Baseline', 'Baseline', True, created_by),
                        ('Simulation', 'Simulation', False, created_by),
                        ('Final', 'Final', False, created_by)
                    ]
                    
                    for version_name, version_type, is_active, created_by_user in versions_data:
                        cursor.execute("""
                            INSERT INTO forecast_versions (version_name, version_type, is_active, created_by)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (version_name) DO NOTHING
                        """, (version_name, version_type, is_active, created_by_user))
                    
                    conn.commit()
                    logger.info(f"✅ Default forecast versions seeded for tenant: {tenant_id}")
                    return True
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"Failed to seed versions for {database_name}: {str(e)}")
            raise DatabaseException(f"Failed to seed versions: {str(e)}")