"""
Schema management utilities for tenant isolation.
Handles tenant database creation and table setup.
"""

from typing import List, Dict, Any
from app.models.database_models import FieldDefinition
from app.core.database import get_db_manager
from app.core.exceptions import DatabaseException, ValidationException
import logging

logger = logging.getLogger(__name__)

class SchemaManager:
    """Manages database schemas and tenant isolation."""
    
    @staticmethod
    def create_tenant_database(tenant_id: str, database_name: str) -> bool:
        """
        Create a new database for tenant and initialize tables.
        
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
                    # Create tenants table (for tenant's own record)
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS tenants (
                            tenant_id UUID PRIMARY KEY,
                            tenant_name VARCHAR(255) NOT NULL,
                            tenant_identifier VARCHAR(100) NOT NULL,
                            admin_email VARCHAR(255) NOT NULL,
                            admin_password_hash VARCHAR(255) NOT NULL,
                            database_name VARCHAR(100) NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            status VARCHAR(50) DEFAULT 'ACTIVE'
                        )
                    """)
                    
                    # Create field_catalogue table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS field_catalogue (
                            catalogue_id UUID PRIMARY KEY,
                            tenant_id UUID NOT NULL,
                            version INT DEFAULT 1,
                            status VARCHAR(50) DEFAULT 'DRAFT',
                            fields_json JSONB NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            created_by VARCHAR(255) NOT NULL,
                            updated_at TIMESTAMP,
                            updated_by VARCHAR(255)
                        )
                    """)
                    
                    # Create master_data table (will be modified when catalogue finalized)
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS master_data (
                            master_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            tenant_id UUID NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            created_by VARCHAR(255) NOT NULL,
                            updated_at TIMESTAMP,
                            updated_by VARCHAR(255)
                        )
                    """)
                    
                    # Create sales_data table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS sales_data (
                            sales_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            tenant_id UUID NOT NULL,
                            master_id UUID NOT NULL REFERENCES master_data(master_id),
                            date DATE NOT NULL,
                            quantity DECIMAL(18, 2) NOT NULL,
                            uom VARCHAR(20) NOT NULL,
                            unit_price DECIMAL(18, 2),
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            created_by VARCHAR(255) NOT NULL
                        )
                    """)
                    
                    # Create upload_history table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS upload_history (
                            upload_id UUID PRIMARY KEY,
                            tenant_id UUID NOT NULL,
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
                    
                    # Create indexes
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sales_data_date ON sales_data(date)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sales_data_master_id ON sales_data(master_id)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_upload_history_tenant ON upload_history(tenant_id)")
                    
                    conn.commit()
                    logger.info(f"Database initialized successfully for tenant: {tenant_id}")

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
        Create dynamic master data table based on field catalogue.
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
            if target_field.data_type.upper() not in ['NUMERIC', 'DECIMAL']:
                raise ValidationException("Target variable must be numeric type")
            if date_field.data_type.upper() not in ['DATE', 'TIMESTAMP']:
                raise ValidationException("Date field must be DATE or TIMESTAMP type")
            
            logger.info(f"Creating tables with target variable: {target_field.field_name}, date field: {date_field.field_name}")
            
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Drop existing tables
                    cursor.execute("DROP TABLE IF EXISTS sales_data CASCADE")
                    cursor.execute("DROP TABLE IF EXISTS master_data CASCADE")
                    
                    # Build master_data table columns
                    columns = []
                    columns.append("master_id UUID PRIMARY KEY DEFAULT gen_random_uuid()")
                    
                    field_names = []
                    for field in master_fields:
                        sql_type = field.get_sql_type()
                        columns.append(f'"{field.field_name}" {sql_type}')
                        field_names.append(f'"{field.field_name}"')
                    
                    # Add audit columns
                    columns.extend([
                        "tenant_id UUID NOT NULL",
                        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                        "created_by VARCHAR(255) NOT NULL",
                        "updated_at TIMESTAMP",
                        "updated_by VARCHAR(255)"
                    ])
                    
                    # Add composite unique constraint
                    composite_fields = field_names + ['"tenant_id"']
                    unique_constraint = f"UNIQUE NULLS DISTINCT ({', '.join(composite_fields)})"
                    columns.append(unique_constraint)
                    
                    # Create master_data table
                    create_table_sql = f"CREATE TABLE master_data ({', '.join(columns)})"
                    cursor.execute(create_table_sql)
                    
                    # Create index
                    index_columns = ', '.join(composite_fields)
                    cursor.execute(f"CREATE INDEX idx_master_data_composite ON master_data ({index_columns})")
                    
                    # Create sales_data table with dynamic columns
                    target_sql_type = target_field.get_sql_type()
                    date_sql_type = date_field.get_sql_type()
                    
                    cursor.execute(f"""
                        CREATE TABLE sales_data (
                            sales_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            tenant_id UUID NOT NULL,
                            master_id UUID NOT NULL REFERENCES master_data(master_id),
                            "{date_field.field_name}" {date_sql_type} NOT NULL,
                            "{target_field.field_name}" {target_sql_type} NOT NULL,
                            uom VARCHAR(20) NOT NULL,
                            unit_price DECIMAL(18, 2),
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            created_by VARCHAR(255) NOT NULL
                        )
                    """)
                    
                    # Create indexes on sales_data
                    cursor.execute(f'CREATE INDEX idx_sales_data_date ON sales_data("{date_field.field_name}")')
                    cursor.execute(f'CREATE INDEX idx_sales_data_master_id ON sales_data(master_id)')
                    
                    # Store metadata about target and date fields
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS field_catalogue_metadata (
                            tenant_id UUID NOT NULL,
                            target_field_name VARCHAR(255) NOT NULL,
                            date_field_name VARCHAR(255) NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            CONSTRAINT unique_metadata_per_tenant UNIQUE(tenant_id)
                        )
                    """)
                    
                    cursor.execute("""
                        INSERT INTO field_catalogue_metadata (tenant_id, target_field_name, date_field_name)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (tenant_id) DO UPDATE SET
                            target_field_name = EXCLUDED.target_field_name,
                            date_field_name = EXCLUDED.date_field_name,
                            created_at = CURRENT_TIMESTAMP
                    """, (tenant_id, target_field.field_name, date_field.field_name))
                    
                    conn.commit()
                    logger.info(
                        f"Created master_data and sales_data tables in {database_name} with "
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
                            tenant_id UUID NOT NULL,
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
                    required_tables = {'tenants', 'field_catalogue', 'master_data', 'sales_data', 'upload_history'}
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
        Initialize all forecasting tables in tenant database.
        
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
                    # Read and execute forecasting schema
                    forecasting_schema = """
                    -- Algorithms table (already exists)
                    CREATE TABLE IF NOT EXISTS algorithms (
                        algorithm_id SERIAL PRIMARY KEY,
                        algorithm_name VARCHAR(255) NOT NULL UNIQUE,
                        default_parameters JSONB NOT NULL,
                        algorithm_type VARCHAR(50) NOT NULL,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        CONSTRAINT check_algorithm_type CHECK (algorithm_type IN ('ML', 'Statistic', 'Hybrid'))
                    );

                    -- Forecast Versions table
                    CREATE TABLE IF NOT EXISTS forecast_versions (
                        version_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        tenant_id UUID NOT NULL,
                        version_name VARCHAR(255) NOT NULL,
                        version_type VARCHAR(50) NOT NULL,
                        is_active BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_by VARCHAR(255),
                        updated_by VARCHAR(255),
                        CONSTRAINT check_version_type CHECK (version_type IN ('Baseline', 'Simulation', 'Final')),
                        CONSTRAINT unique_version_name_per_tenant UNIQUE(tenant_id, version_name)
                    );

                    CREATE INDEX IF NOT EXISTS idx_forecast_versions_tenant ON forecast_versions(tenant_id);
                    CREATE INDEX IF NOT EXISTS idx_forecast_versions_active ON forecast_versions(tenant_id, is_active);
                    CREATE INDEX IF NOT EXISTS idx_forecast_versions_type ON forecast_versions(tenant_id, version_type);

                    -- ============================================================================
                    -- UPDATED: External Factors table with UNIQUE CONSTRAINT
                    -- ============================================================================
                    CREATE TABLE IF NOT EXISTS external_factors (
                        factor_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        tenant_id UUID NOT NULL,
                        date DATE NOT NULL,
                        factor_name VARCHAR(255) NOT NULL,
                        factor_value DECIMAL(18, 4) NOT NULL,
                        unit VARCHAR(50),
                        source VARCHAR(255),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_by VARCHAR(255),
                        updated_by VARCHAR(255),
                        deleted_at TIMESTAMP,
                        -- NEW: Unique constraint to prevent duplicates
                        CONSTRAINT unique_factor_per_date UNIQUE (tenant_id, factor_name, date)
                    );

                    CREATE INDEX IF NOT EXISTS idx_external_factors_tenant ON external_factors(tenant_id);
                    CREATE INDEX IF NOT EXISTS idx_external_factors_date ON external_factors(tenant_id, date);
                    CREATE INDEX IF NOT EXISTS idx_external_factors_name ON external_factors(tenant_id, factor_name);
                    CREATE INDEX IF NOT EXISTS idx_external_factors_composite ON external_factors(tenant_id, factor_name, date);
                    -- NEW: Index for active (non-deleted) factors
                    CREATE INDEX IF NOT EXISTS idx_external_factors_active ON external_factors(tenant_id, factor_name, date) WHERE deleted_at IS NULL;

                    -- Forecast Runs table
                    CREATE TABLE IF NOT EXISTS forecast_runs (
                        forecast_run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        tenant_id UUID NOT NULL,
                        version_id UUID NOT NULL REFERENCES forecast_versions(version_id) ON DELETE CASCADE,
                        forecast_filters JSONB,
                        forecast_start DATE NOT NULL,
                        forecast_end DATE NOT NULL,
                        run_status VARCHAR(50) NOT NULL DEFAULT 'Pending',
                        run_progress INTEGER DEFAULT 0,
                        run_percentage_frequency INTEGER DEFAULT 10,
                        total_records INTEGER DEFAULT 0,
                        processed_records INTEGER DEFAULT 0,
                        failed_records INTEGER DEFAULT 0,
                        error_message TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        started_at TIMESTAMP,
                        completed_at TIMESTAMP,
                        created_by VARCHAR(255),
                        updated_by VARCHAR(255),
                        CONSTRAINT check_run_status CHECK (run_status IN ('Pending', 'In-Progress', 'Completed', 'Completed with Errors', 'Failed', 'Cancelled')),
                        CONSTRAINT check_run_progress CHECK (run_progress >= 0 AND run_progress <= 100),
                        CONSTRAINT check_forecast_dates CHECK (forecast_end >= forecast_start)
                    );

                    CREATE INDEX IF NOT EXISTS idx_forecast_runs_tenant ON forecast_runs(tenant_id);
                    CREATE INDEX IF NOT EXISTS idx_forecast_runs_status ON forecast_runs(tenant_id, run_status);
                    CREATE INDEX IF NOT EXISTS idx_forecast_runs_version ON forecast_runs(tenant_id, version_id);
                    CREATE INDEX IF NOT EXISTS idx_forecast_runs_created ON forecast_runs(tenant_id, created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_forecast_runs_composite ON forecast_runs(tenant_id, run_status, created_at DESC);

                    -- Forecast Algorithms Mapping table
                    CREATE TABLE IF NOT EXISTS forecast_algorithms_mapping (
                        mapping_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        tenant_id UUID NOT NULL,
                        forecast_run_id UUID NOT NULL REFERENCES forecast_runs(forecast_run_id) ON DELETE CASCADE,
                        algorithm_id INTEGER NOT NULL REFERENCES algorithms(algorithm_id),
                        algorithm_name VARCHAR(255) NOT NULL,
                        custom_parameters JSONB,
                        execution_order INTEGER NOT NULL DEFAULT 1,
                        execution_status VARCHAR(50) DEFAULT 'Pending',
                        started_at TIMESTAMP,
                        completed_at TIMESTAMP,
                        error_message TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_by VARCHAR(255),
                        CONSTRAINT check_execution_status CHECK (execution_status IN ('Pending', 'Running', 'Completed', 'Failed')),
                        CONSTRAINT unique_algo_per_run UNIQUE(forecast_run_id, algorithm_id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_forecast_algo_mapping_tenant ON forecast_algorithms_mapping(tenant_id);
                    CREATE INDEX IF NOT EXISTS idx_forecast_algo_mapping_run ON forecast_algorithms_mapping(forecast_run_id);
                    CREATE INDEX IF NOT EXISTS idx_forecast_algo_mapping_algo ON forecast_algorithms_mapping(algorithm_id);
                    CREATE INDEX IF NOT EXISTS idx_forecast_algo_mapping_status ON forecast_algorithms_mapping(forecast_run_id, execution_status);

                    -- Forecast Results table
                    CREATE TABLE IF NOT EXISTS forecast_results (
                        result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        tenant_id UUID NOT NULL,
                        forecast_run_id UUID NOT NULL REFERENCES forecast_runs(forecast_run_id) ON DELETE CASCADE,
                        version_id UUID NOT NULL REFERENCES forecast_versions(version_id) ON DELETE CASCADE,
                        mapping_id UUID NOT NULL REFERENCES forecast_algorithms_mapping(mapping_id) ON DELETE CASCADE,
                        algorithm_id INTEGER NOT NULL REFERENCES algorithms(algorithm_id),
                        forecast_date DATE NOT NULL,
                        forecast_quantity DECIMAL(18, 4) NOT NULL,
                        confidence_interval_lower DECIMAL(18, 4),
                        confidence_interval_upper DECIMAL(18, 4),
                        confidence_level VARCHAR(20),
                        accuracy_metric DECIMAL(5, 2),
                        metric_type VARCHAR(50),
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_by VARCHAR(255)
                    );

                    CREATE INDEX IF NOT EXISTS idx_forecast_results_tenant ON forecast_results(tenant_id);
                    CREATE INDEX IF NOT EXISTS idx_forecast_results_run ON forecast_results(forecast_run_id);
                    CREATE INDEX IF NOT EXISTS idx_forecast_results_date ON forecast_results(tenant_id, forecast_date);
                    CREATE INDEX IF NOT EXISTS idx_forecast_results_algo ON forecast_results(algorithm_id);
                    CREATE INDEX IF NOT EXISTS idx_forecast_results_composite ON forecast_results(tenant_id, forecast_run_id, forecast_date);
                    CREATE INDEX IF NOT EXISTS idx_forecast_results_version ON forecast_results(version_id);

                    -- Forecast Audit Log table
                    CREATE TABLE IF NOT EXISTS forecast_audit_log (
                        audit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        tenant_id UUID NOT NULL,
                        forecast_run_id UUID NOT NULL REFERENCES forecast_runs(forecast_run_id) ON DELETE CASCADE,
                        action VARCHAR(50) NOT NULL,
                        entity_type VARCHAR(100),
                        entity_id UUID,
                        performed_by VARCHAR(255),
                        performed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        details JSONB,
                        CONSTRAINT check_action CHECK (action IN ('Created', 'Updated', 'Deleted', 'Executed', 'Cancelled'))
                    );

                    CREATE INDEX IF NOT EXISTS idx_forecast_audit_tenant ON forecast_audit_log(tenant_id);
                    CREATE INDEX IF NOT EXISTS idx_forecast_audit_run ON forecast_audit_log(forecast_run_id);
                    CREATE INDEX IF NOT EXISTS idx_forecast_audit_timestamp ON forecast_audit_log(performed_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_forecast_audit_action ON forecast_audit_log(action);
                    """
                    
                    cursor.execute(forecasting_schema)
                    conn.commit()
                    logger.info(f"Forecasting tables initialized successfully for tenant: {tenant_id}")
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
                    # Insert default algorithms with specific IDs to match forecast_execution_service routing
                    algorithms_data = [
                        (1, 'ARIMA', '{"order": [1, 1, 1]}', 'Statistic', 'AutoRegressive Integrated Moving Average - Statistical time series forecasting'),
                        (2, 'Linear Regression', '{}', 'Statistic', 'Linear regression with feature engineering and external factors support'),
                        (3, 'Polynomial Regression', '{"degree": 2}', 'Statistic', 'Polynomial regression with external factors integration'),
                        (4, 'Exponential Smoothing', '{"alpha": 0.3}', 'Statistic', 'Simple exponential smoothing'),
                        (5, 'Enhanced Exponential Smoothing', '{"alphas": [0.1, 0.3, 0.5]}', 'Statistic', 'Multiple alpha values exponential smoothing with external factors'),
                        (6, 'Holt Winters', '{"season_length": 12}', 'Statistic', 'Triple exponential smoothing for seasonal data'),
                        (7, 'Prophet', '{"window": 3}', 'Statistic', 'Facebook Prophet algorithm (placeholder implementation)'),
                        (8, 'LSTM Neural Network', '{"window": 3}', 'ML', 'Long Short-Term Memory neural network (placeholder implementation)'),
                        (9, 'XGBoost', '{"n_estimators_list": [50, 100], "learning_rate_list": [0.05, 0.1, 0.2], "max_depth_list": [3, 4, 5]}', 'ML', 'XGBoost-like gradient boosting with hyperparameter tuning and external factors support'),
                        (10, 'SVR', '{"C_list": [1, 10, 100], "epsilon_list": [0.1, 0.2]}', 'ML', 'Support Vector Regression with hyperparameter tuning and external factors'),
                        (11, 'KNN', '{"n_neighbors_list": [7, 10]}', 'ML', 'K-Nearest Neighbors regression with hyperparameter tuning and external factors'),
                        (12, 'Gaussian Process', '{}', 'ML', 'Gaussian Process Regression with hyperparameter tuning and scaling'),
                        (13, 'Neural Network', '{"hidden_layer_sizes_list": [[10], [20, 10]], "alpha_list": [0.001, 0.01]}', 'ML', 'Multi-layer Perceptron Neural Network with hyperparameter tuning and external factors'),
                        (999, 'Best Fit', '{}', 'Hybrid', 'Advanced AI/ML auto model selection - runs all algorithms and selects the best performing one')
                    ]

                    for algo_id, algo_name, default_params, algo_type, description in algorithms_data:
                        cursor.execute("""
                            INSERT INTO algorithms (algorithm_id, algorithm_name, default_parameters, algorithm_type, description)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (algorithm_id) DO NOTHING
                        """, (algo_id, algo_name, default_params, algo_type, description))
                    
                    conn.commit()
                    logger.info(f"Default algorithms seeded for tenant: {tenant_id}")
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
                            INSERT INTO forecast_versions (tenant_id, version_name, version_type, is_active, created_by)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (tenant_id, version_name) DO NOTHING
                        """, (tenant_id, version_name, version_type, is_active, created_by_user))
                    
                    conn.commit()
                    logger.info(f"Default forecast versions seeded for tenant: {tenant_id}")
                    return True
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"Failed to seed versions for {database_name}: {str(e)}")
            raise DatabaseException(f"Failed to seed versions: {str(e)}")
