-- Smart Demand Planning Tool - Database Initialization Script
-- Creates the public schema tables for tenant management

-- Create public.tenants table
CREATE TABLE IF NOT EXISTS public.tenants (
    tenant_id UUID PRIMARY KEY,
    tenant_name VARCHAR(255) NOT NULL,
    tenant_identifier VARCHAR(100) UNIQUE NOT NULL,
    admin_email VARCHAR(255) NOT NULL,
    admin_password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'ACTIVE',
    CONSTRAINT check_status CHECK (status IN ('ACTIVE', 'INACTIVE', 'SUSPENDED'))
);

-- Create index on tenant_identifier for faster lookups
CREATE INDEX IF NOT EXISTS idx_tenants_identifier ON public.tenants(tenant_identifier);
CREATE INDEX IF NOT EXISTS idx_tenants_email ON public.tenants(admin_email);

-- Create audit log table (optional but recommended)
CREATE TABLE IF NOT EXISTS public.audit_log (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES public.tenants(tenant_id) ON DELETE CASCADE,
    action VARCHAR(50) NOT NULL,
    entity_type VARCHAR(100),
    entity_id UUID,
    performed_by VARCHAR(255),
    performed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    details JSONB
);

CREATE INDEX IF NOT EXISTS idx_audit_tenant ON public.audit_log(tenant_id);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON public.audit_log(performed_at DESC);

-- Comments for documentation
COMMENT ON TABLE public.tenants IS 'Master table storing all tenant information';
COMMENT ON TABLE public.audit_log IS 'Audit trail for all tenant operations';

-- Smart Demand Planning Tool - Forecasting Schema
-- Creates forecasting tables for tenant databases

-- 1. Algorithms Table (static lookup table)
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

-- Insert default algorithms
INSERT INTO algorithms (algorithm_id, algorithm_name, default_parameters, algorithm_type, description) VALUES
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
ON CONFLICT (algorithm_id) DO NOTHING;

-- 2. Forecast Versions Table
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

CREATE INDEX idx_forecast_versions_tenant ON forecast_versions(tenant_id);
CREATE INDEX idx_forecast_versions_active ON forecast_versions(tenant_id, is_active);
CREATE INDEX idx_forecast_versions_type ON forecast_versions(tenant_id, version_type);

-- 3. External Factors Table (tenant-specific)
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
    deleted_at TIMESTAMP
);

CREATE INDEX idx_external_factors_tenant ON external_factors(tenant_id);
CREATE INDEX idx_external_factors_date ON external_factors(tenant_id, date);
CREATE INDEX idx_external_factors_name ON external_factors(tenant_id, factor_name);
CREATE INDEX idx_external_factors_composite ON external_factors(tenant_id, factor_name, date);

-- 4. Forecast Runs Table
CREATE TABLE IF NOT EXISTS forecast_runs (
    forecast_run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
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

CREATE INDEX idx_forecast_runs_tenant ON forecast_runs(tenant_id);
CREATE INDEX idx_forecast_runs_status ON forecast_runs(tenant_id, run_status);
CREATE INDEX idx_forecast_runs_version ON forecast_runs(tenant_id, version_id);
CREATE INDEX idx_forecast_runs_created ON forecast_runs(tenant_id, created_at DESC);
CREATE INDEX idx_forecast_runs_composite ON forecast_runs(tenant_id, run_status, created_at DESC);

-- 5. Forecast Algorithms Mapping Table
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

CREATE INDEX idx_forecast_algo_mapping_tenant ON forecast_algorithms_mapping(tenant_id);
CREATE INDEX idx_forecast_algo_mapping_run ON forecast_algorithms_mapping(forecast_run_id);
CREATE INDEX idx_forecast_algo_mapping_algo ON forecast_algorithms_mapping(algorithm_id);
CREATE INDEX idx_forecast_algo_mapping_status ON forecast_algorithms_mapping(forecast_run_id, execution_status);

-- 6. Forecast Results Table
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

CREATE INDEX idx_forecast_results_tenant ON forecast_results(tenant_id);
CREATE INDEX idx_forecast_results_run ON forecast_results(forecast_run_id);
CREATE INDEX idx_forecast_results_date ON forecast_results(tenant_id, forecast_date);
CREATE INDEX idx_forecast_results_algo ON forecast_results(algorithm_id);
CREATE INDEX idx_forecast_results_composite ON forecast_results(tenant_id, forecast_run_id, forecast_date);
CREATE INDEX idx_forecast_results_version ON forecast_results(version_id);

-- 7. Forecast Audit Log Table
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

CREATE INDEX idx_forecast_audit_tenant ON forecast_audit_log(tenant_id);
CREATE INDEX idx_forecast_audit_run ON forecast_audit_log(forecast_run_id);
CREATE INDEX idx_forecast_audit_timestamp ON forecast_audit_log(performed_at DESC);
CREATE INDEX idx_forecast_audit_action ON forecast_audit_log(action);
