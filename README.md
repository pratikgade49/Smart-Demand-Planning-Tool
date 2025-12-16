# Smart Demand Planning Tool

A multi-tenant demand planning and forecasting application built with FastAPI, PostgreSQL, and advanced machine learning algorithms. This tool provides dynamic field catalogues, Excel data uploads, and multi-algorithm forecast comparisons.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [Database Structure](#database-structure)
- [API Documentation](#api-documentation)
- [Usage Examples](#usage-examples)
- [Logging & Monitoring](#logging--monitoring)

---

## Overview

### Key Features

- **Multi-Tenant Architecture**: Each tenant gets isolated PostgreSQL databases
- **Dynamic Field Catalogue**: Define custom master data fields, relationships, and constraints
- **Flexible Data Upload**: Import data from Excel with automatic validation and distribution
- **Advanced Forecasting**: Multiple algorithms (Linear Regression, XGBoost, Exponential Smoothing, ARIMA)
- **Scenario Comparison**: Compare multiple forecast scenarios with statistical metrics
- **External Factors Integration**: FRED API integration for economic indicators
- **Comprehensive Logging**: Structured logging with performance metrics and audit trails
- **Role-Based Access**: Admin and user authentication with JWT tokens

### Technology Stack

- **Framework**: FastAPI 0.115.0
- **Database**: PostgreSQL (master + tenant-specific databases)
- **ORM**: SQLAlchemy 2.0
- **Auth**: JWT with python-jose and passlib
- **Data Processing**: pandas, numpy
- **ML/Forecasting**: scikit-learn, XGBoost, statsmodels
- **API Server**: Uvicorn

---

## Architecture

### Multi-Database Design

```
Master Database (smart_demand_master)
├── tenants (tenant metadata)
├── audit_log (system-wide audit)

Tenant Database (tenant_acme_corp)
├── field_catalogue (custom fields)
├── master_data (entity dimension data)
├── sales_data (transactional data)
├── external_factors (economic indicators)
├── forecast_versions (model configurations)
├── forecast_runs (execution records)
├── forecast_algorithms_mapping (algorithm configs)
├── forecast_results (predictions)
└── forecast_audit_log (execution audit)
```

### Directory Structure

```
Smart-Demand-Planning-Tool/
├── app/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── auth_routes.py
│   │   │   ├── field_catalogue_routes.py
│   │   │   ├── upload_routes.py
│   │   │   ├── forecasting_routes.py
│   │   │   ├── external_factors_routes.py
│   │   │   └── forecast_comparison_routes.py
│   │   └── dependencies.py
│   ├── core/
│   │   ├── database.py (connection management)
│   │   ├── auth_service.py (authentication logic)
│   │   ├── field_catalogue_service.py
│   │   ├── excel_upload_service.py
│   │   ├── forecasting_service.py
│   │   ├── forecast_execution_service.py
│   │   ├── forecast_comparison_service.py
│   │   ├── external_factors_service.py
│   │   ├── schema_manager.py (DDL operations)
│   │   ├── logging_config.py
│   │   └── exceptions.py
│   ├── models/
│   │   └── database_models.py
│   ├── schemas/
│   │   ├── auth.py
│   │   ├── field_catalogue.py
│   │   ├── forecasting.py
│   │   ├── upload.py
│   │   └── master_data.py
│   └── config.py
├── logs/
│   ├── app.log
│   └── error.log
├── main.py
├── requirements.txt
└── README.md
```

---

## Installation & Setup

### Prerequisites

- Python 3.9+
- PostgreSQL 12+
- pip (Python package manager)

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd Smart-Demand-Planning-Tool
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment

Create `.env` file in project root:

```env
# Database
DB_HOST=localhost
DB_PORT=5433
DB_USER=postgres
DB_PASSWORD=root
MASTER_DATABASE_URL=postgresql://postgres:root@localhost:5433/smart_demand_master

# Application
APP_NAME=Smart Demand Planning Tool
APP_VERSION=1.0.0
ENVIRONMENT=development
DEBUG=True

# Authentication
SECRET_KEY=your-secret-key-here-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# FRED API (Optional, for economic indicators)
FRED_API_KEY=your-fred-api-key

# Logging
LOG_LEVEL=INFO
LOG_FILE_PATH=logs
```

### Step 5: Start Application

**Note**: Database tables are automatically created when the application starts. No manual database initialization is required.

```bash
# Development
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

Access API docs at: `http://localhost:8000/api/docs`

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_HOST` | PostgreSQL hostname | localhost |
| `DB_PORT` | PostgreSQL port | 5433 |
| `DB_USER` | Database user | postgres |
| `DB_PASSWORD` | Database password | root |
| `MASTER_DATABASE_URL` | Master database connection string | - |
| `SECRET_KEY` | JWT secret key | - |
| `ALGORITHM` | JWT algorithm | HS256 |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Token expiration time | 1440 |
| `FRED_API_KEY` | FRED API key for economic data | - |
| `DEBUG` | Enable debug mode | False |
| `ENVIRONMENT` | Deployment environment | production |

### Database Connection Pool

- **Min Connections**: 1
- **Max Connections**: 10 (master), configurable per tenant
- **Auto-recycling**: Enabled for stale connections

---

## Database Structure

> **Note**: All tables are automatically created when the application starts. The master database tables are initialized on first run, and tenant-specific tables are created dynamically when each tenant is registered or uses specific features.

### Master Database Tables

#### `tenants`
Stores tenant registration and metadata.

```sql
CREATE TABLE tenants (
    tenant_id UUID PRIMARY KEY,
    tenant_name VARCHAR(255) NOT NULL,
    tenant_identifier VARCHAR(100) UNIQUE NOT NULL,
    admin_email VARCHAR(255) UNIQUE NOT NULL,
    admin_password_hash VARCHAR(255) NOT NULL,
    database_name VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) CHECK (status IN ('ACTIVE', 'INACTIVE', 'SUSPENDED'))
);
```

**Indexes**:
- `idx_tenants_identifier` on `tenant_identifier`
- `idx_tenants_email` on `admin_email`
- `idx_tenants_database_name` on `database_name`

#### `audit_log`
System-wide audit trail for all operations.

```sql
CREATE TABLE audit_log (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    action VARCHAR(50) NOT NULL,
    entity_type VARCHAR(100),
    entity_id UUID,
    performed_by VARCHAR(255),
    performed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    details JSONB
);
```

### Tenant Database Tables

#### `field_catalogue`
Dynamic field definitions for master data.

```sql
CREATE TABLE field_catalogue (
    catalogue_id UUID PRIMARY KEY,
    version INT NOT NULL,
    status VARCHAR(50) NOT NULL,  -- DRAFT, FINALIZED
    fields_json JSONB NOT NULL,   -- JSON array of field definitions
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255) NOT NULL,
    updated_at TIMESTAMP,
    updated_by VARCHAR(255)
);
```

**Field Definition Structure** (inside `fields_json`):
```json
{
    "field_name": "product_id",
    "data_type": "Char",
    "field_length": 50,
    "is_characteristic": true,
    "is_unique_key": true,
    "parent_field_name": null,
    "is_target_variable": false,
    "is_date_field": false,
    "default_value": null
}
```

#### `master_data`
Dynamic entity dimension table (columns based on field catalogue).

```sql
CREATE TABLE master_data (
    master_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Dynamic columns based on field catalogue
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255) NOT NULL,
    updated_at TIMESTAMP,
    updated_by VARCHAR(255)
);
```

#### `sales_data`
Transactional data with dynamic target and date columns.

```sql
CREATE TABLE sales_data (
    sales_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    master_id UUID NOT NULL REFERENCES master_data(master_id),
    -- Dynamic date column (e.g., "sales_date" TIMESTAMP)
    -- Dynamic target column (e.g., "quantity" NUMERIC)
    uom VARCHAR(20) NOT NULL,
    unit_price DECIMAL(18, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255) NOT NULL
);

CREATE INDEX idx_sales_data_date ON sales_data(<date_field>);
CREATE INDEX idx_sales_data_master_id ON sales_data(master_id);
```

#### `upload_history`
Records of all file uploads and data imports.

```sql
CREATE TABLE upload_history (
    upload_id UUID PRIMARY KEY,
    upload_type VARCHAR(50) NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    total_rows INT DEFAULT 0,
    success_count INT DEFAULT 0,
    failed_count INT DEFAULT 0,
    status VARCHAR(50) NOT NULL,  -- completed, failed
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    uploaded_by VARCHAR(255) NOT NULL
);
```

#### `forecast_versions`
Version management for forecast models.

```sql
CREATE TABLE forecast_versions (
    version_id UUID PRIMARY KEY,
    version_name VARCHAR(255) NOT NULL,
    version_type VARCHAR(50) NOT NULL,  -- BASELINE, OPTIMIZED, SCENARIO
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255) NOT NULL,
    updated_at TIMESTAMP,
    updated_by VARCHAR(255),
    CONSTRAINT unique_version_name UNIQUE(version_name)
);

CREATE INDEX idx_forecast_versions_active ON forecast_versions(is_active);
CREATE INDEX idx_forecast_versions_type ON forecast_versions(version_type);
```

#### `forecast_runs`
Individual forecast execution records.

```sql
CREATE TABLE forecast_runs (
    forecast_run_id UUID PRIMARY KEY,
    version_id UUID NOT NULL REFERENCES forecast_versions(version_id),
    forecast_filters JSONB NOT NULL,  -- entity, aggregation, interval
    forecast_start DATE NOT NULL,
    forecast_end DATE NOT NULL,
    run_status VARCHAR(50) NOT NULL,   -- Pending, Running, Completed, Failed
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255) NOT NULL,
    updated_at TIMESTAMP,
    updated_by VARCHAR(255)
);

CREATE INDEX idx_forecast_runs_status ON forecast_runs(run_status);
CREATE INDEX idx_forecast_runs_version ON forecast_runs(version_id);
```

#### `forecast_algorithms_mapping`
Algorithm assignment and parameters for forecast runs.

```sql
CREATE TABLE forecast_algorithms_mapping (
    mapping_id UUID PRIMARY KEY,
    forecast_run_id UUID NOT NULL REFERENCES forecast_runs(forecast_run_id),
    algorithm_id VARCHAR(50) NOT NULL,
    algorithm_name VARCHAR(100) NOT NULL,
    custom_parameters JSONB,
    execution_status VARCHAR(50),  -- Pending, Completed, Failed
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_forecast_algo_mapping_run ON forecast_algorithms_mapping(forecast_run_id);
```

#### `forecast_results`
Forecast predictions and accuracy metrics.

```sql
CREATE TABLE forecast_results (
    result_id UUID PRIMARY KEY,
    mapping_id UUID NOT NULL REFERENCES forecast_algorithms_mapping(mapping_id),
    forecast_run_id UUID NOT NULL REFERENCES forecast_runs(forecast_run_id),
    forecast_date DATE NOT NULL,
    forecast_quantity NUMERIC NOT NULL,
    accuracy_metric NUMERIC,
    confidence_interval_lower NUMERIC,
    confidence_interval_upper NUMERIC,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_forecast_results_date ON forecast_results(forecast_date);
CREATE INDEX idx_forecast_results_composite ON forecast_results(forecast_run_id, forecast_date);
```

#### `external_factors`
Economic indicators and external variables.

```sql
CREATE TABLE external_factors (
    factor_id UUID PRIMARY KEY,
    date DATE NOT NULL,
    factor_name VARCHAR(255) NOT NULL,
    factor_value NUMERIC NOT NULL,
    unit VARCHAR(50),
    source VARCHAR(255),
    deleted_at TIMESTAMP,
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP,
    updated_by VARCHAR(255),
    CONSTRAINT unique_factor UNIQUE NULLS DISTINCT(factor_name, date)
);

CREATE INDEX idx_external_factors_date ON external_factors(date);
CREATE INDEX idx_external_factors_name ON external_factors(factor_name);
```

---

## API Documentation

### Authentication Endpoints

#### Register Tenant
```
POST /api/v1/auth/register
Content-Type: application/json

{
    "tenant_name": "Acme Corp",
    "tenant_identifier": "acme_corp",
    "email": "admin@acme.com",
    "password": "secure_password"
}

Response: 201 Created
{
    "tenant_id": "uuid",
    "database_name": "tenant_acme_corp",
    "message": "Tenant registered successfully"
}
```

#### Login
```
POST /api/v1/auth/login
Content-Type: application/json

{
    "tenant_identifier": "acme_corp",
    "email": "admin@acme.com",
    "password": "secure_password"
}

Response: 200 OK
{
    "access_token": "eyJhbGc...",
    "token_type": "bearer",
    "tenant_id": "uuid",
    "email": "admin@acme.com"
}
```

### Field Catalogue Endpoints

#### Create Field Catalogue
```
POST /api/v1/field-catalogue
Authorization: Bearer <token>
Content-Type: application/json

{
    "fields": [
        {
            "field_name": "product_id",
            "data_type": "Char",
            "field_length": 50,
            "is_characteristic": true,
            "is_unique_key": true,
            "is_target_variable": false,
            "is_date_field": false
        },
        {
            "field_name": "sales_date",
            "data_type": "Date",
            "is_characteristic": false,
            "is_unique_key": false,
            "is_target_variable": false,
            "is_date_field": true
        },
        {
            "field_name": "quantity",
            "data_type": "Numeric",
            "is_characteristic": false,
            "is_unique_key": false,
            "is_target_variable": true,
            "is_date_field": false
        }
    ]
}

Response: 201 Created
{
    "catalogue_id": "uuid",
    "fields": [...],
    "version": 1,
    "status": "DRAFT"
}
```

#### Finalize Field Catalogue
```
POST /api/v1/field-catalogue/{catalogue_id}/finalize
Authorization: Bearer <token>

Response: 200 OK
{
    "catalogue_id": "uuid",
    "status": "FINALIZED",
    "fields": [...]
}
```

#### Get Field Catalogue
```
GET /api/v1/field-catalogue/{catalogue_id}
Authorization: Bearer <token>

Response: 200 OK
{
    "catalogue_id": "uuid",
    "fields": [...],
    "version": 1,
    "status": "FINALIZED"
}
```

#### List Field Catalogues
```
GET /api/v1/field-catalogue?page=1&page_size=50
Authorization: Bearer <token>

Response: 200 OK
{
    "data": [...],
    "page": 1,
    "page_size": 50,
    "total_count": 10
}
```

### Data Upload Endpoints

#### Upload Excel File
```
POST /api/v1/upload/excel
Authorization: Bearer <token>
Content-Type: multipart/form-data

Parameters:
  file: <Excel file>
  upload_type: "mixed_data"
  catalogue_id: "uuid"

Response: 201 Created
{
    "upload_id": "uuid",
    "upload_type": "mixed_data",
    "file_name": "sales_data.xlsx",
    "total_rows": 1000,
    "success_count": 999,
    "failed_count": 1,
    "status": "completed",
    "errors": [
        {"row": 15, "error": "Invalid date format"}
    ]
}
```

#### Get Upload History
```
GET /api/v1/upload/history?page=1&page_size=50
Authorization: Bearer <token>

Response: 200 OK
{
    "data": [
        {
            "upload_id": "uuid",
            "upload_type": "mixed_data",
            "file_name": "sales_data.xlsx",
            "total_rows": 1000,
            "success_count": 999,
            "failed_count": 1,
            "status": "completed",
            "uploaded_at": "2025-12-16T14:00:00",
            "uploaded_by": "admin@acme.com"
        }
    ],
    "page": 1,
    "page_size": 50,
    "total_count": 5
}
```

### Forecasting Endpoints

#### Create Forecast Version
```
POST /api/v1/forecasting/versions
Authorization: Bearer <token>
Content-Type: application/json

{
    "version_name": "Q1 2025 Baseline",
    "version_type": "BASELINE",
    "description": "Baseline forecast for Q1 2025"
}

Response: 201 Created
{
    "version_id": "uuid",
    "version_name": "Q1 2025 Baseline",
    "version_type": "BASELINE",
    "is_active": true
}
```

#### Create Forecast Run
```
POST /api/v1/forecasting/runs
Authorization: Bearer <token>
Content-Type: application/json

{
    "version_id": "uuid",
    "forecast_filters": {
        "aggregation_level": "product-location",
        "interval": "MONTHLY",
        "entity_identifier": "1001-Loc1"
    },
    "forecast_start": "2025-01-01",
    "forecast_end": "2025-03-31",
    "algorithms": [
        {
            "algorithm_id": "linear_regression",
            "algorithm_name": "Linear Regression",
            "custom_parameters": {}
        },
        {
            "algorithm_id": "xgboost",
            "algorithm_name": "XGBoost",
            "custom_parameters": {"max_depth": 5}
        }
    ]
}

Response: 202 Accepted
{
    "forecast_run_id": "uuid",
    "version_id": "uuid",
    "run_status": "Running",
    "created_at": "2025-12-16T14:00:00"
}
```

#### Get Forecast Run
```
GET /api/v1/forecasting/runs/{forecast_run_id}
Authorization: Bearer <token>

Response: 200 OK
{
    "forecast_run_id": "uuid",
    "version_id": "uuid",
    "forecast_filters": {...},
    "forecast_start": "2025-01-01",
    "forecast_end": "2025-03-31",
    "run_status": "Completed",
    "created_at": "2025-12-16T14:00:00"
}
```

### Forecast Comparison Endpoints

#### Compare Forecasts
```
POST /api/v1/forecasting/compare
Authorization: Bearer <token>
Content-Type: application/json

{
    "entity_identifier": "1001-Loc1",
    "aggregation_level": "product-location",
    "interval": "MONTHLY",
    "forecast_run_ids": ["uuid1", "uuid2", "uuid3"]
}

Response: 200 OK
{
    "entity": {
        "identifier": "1001-Loc1",
        "aggregation_level": "product-location",
        "field_values": {"product": "1001", "location": "Loc1"}
    },
    "historical_data": [
        {"date": "2024-01-31", "actual_quantity": 1000, "transaction_count": 45},
        ...
    ],
    "available_forecasts": [
        {
            "forecast_run_id": "uuid",
            "forecast_name": "Q1 2025 Baseline - XGBoost",
            "algorithm_name": "XGBoost",
            "external_factors": ["GDP", "Inflation"],
            "accuracy_metrics": {"accuracy": 0.92}
        }
    ],
    "forecast_data": {
        "uuid": [
            {"date": "2025-01-31", "forecast_quantity": 1050, "accuracy_metric": 0.92},
            ...
        ]
    },
    "comparison_matrix": {
        "period_overlap": {
            "start": "2025-01-31",
            "end": "2025-03-31",
            "total_periods": 3
        },
        "pairwise_differences": [
            {
                "forecast_1": "uuid1",
                "forecast_2": "uuid2",
                "average_difference": -25.50,
                "percentage_difference": -2.43,
                "correlation": 0.9876,
                "rmse": 48.23,
                "comparison_points": 3
            }
        ]
    }
}
```

### External Factors Endpoints

#### Create External Factor
```
POST /api/v1/external-factors
Authorization: Bearer <token>
Content-Type: application/json

{
    "factor_name": "GDP",
    "values": [
        {"date": "2025-01-01", "value": 28500.50, "unit": "Billions USD"},
        {"date": "2025-02-01", "value": 28650.75, "unit": "Billions USD"}
    ]
}

Response: 201 Created
{
    "factor_id": "uuid",
    "factor_name": "GDP",
    "values_imported": 2,
    "created_at": "2025-12-16T14:00:00"
}
```

#### Get Available Factors
```
GET /api/v1/external-factors/available
Authorization: Bearer <token>

Response: 200 OK
{
    "factors": [
        {
            "factor_name": "GDP",
            "earliest_date": "2024-01-01",
            "latest_date": "2025-12-31",
            "data_points": 730,
            "unit": "Billions USD",
            "source": "FRED",
            "avg_value": 28500.50,
            "min_value": 27000.00,
            "max_value": 30000.00
        }
    ]
}
```

---

## Usage Examples

### Example 1: Complete Workflow

```bash
# 1. Register tenant
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_name": "Acme Corp",
    "tenant_identifier": "acme_corp",
    "email": "admin@acme.com",
    "password": "securepass123"
  }'

# Store returned token
TOKEN="eyJhbGc..."

# 2. Create field catalogue
curl -X POST http://localhost:8000/api/v1/field-catalogue \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "fields": [
      {
        "field_name": "product_id",
        "data_type": "Char",
        "field_length": 50,
        "is_characteristic": true,
        "is_unique_key": true,
        "is_target_variable": false,
        "is_date_field": false
      },
      {
        "field_name": "sales_quantity",
        "data_type": "Numeric",
        "is_characteristic": false,
        "is_unique_key": false,
        "is_target_variable": true,
        "is_date_field": false
      },
      {
        "field_name": "sales_date",
        "data_type": "Date",
        "is_characteristic": false,
        "is_unique_key": false,
        "is_target_variable": false,
        "is_date_field": true
      }
    ]
  }'

# Store catalogue_id
CATALOGUE_ID="uuid"

# 3. Finalize catalogue
curl -X POST http://localhost:8000/api/v1/field-catalogue/$CATALOGUE_ID/finalize \
  -H "Authorization: Bearer $TOKEN"

# 4. Upload Excel data
curl -X POST http://localhost:8000/api/v1/upload/excel \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@sales_data.xlsx" \
  -F "upload_type=mixed_data" \
  -F "catalogue_id=$CATALOGUE_ID"

# 5. Create forecast version
curl -X POST http://localhost:8000/api/v1/forecasting/versions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "version_name": "Q1 2025 Forecast",
    "version_type": "BASELINE"
  }'

# Store version_id
VERSION_ID="uuid"

# 6. Run forecast
curl -X POST http://localhost:8000/api/v1/forecasting/runs \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "version_id": "'$VERSION_ID'",
    "forecast_filters": {
      "aggregation_level": "product",
      "interval": "MONTHLY",
      "entity_identifier": "PROD001"
    },
    "forecast_start": "2025-01-01",
    "forecast_end": "2025-03-31",
    "algorithms": [
      {
        "algorithm_id": "linear_regression",
        "algorithm_name": "Linear Regression"
      }
    ]
  }'
```

### Example 2: Python Client

```python
import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

class DemandPlanningClient:
    def __init__(self, tenant_id, base_url=BASE_URL):
        self.base_url = base_url
        self.tenant_id = tenant_id
        self.token = None
    
    def login(self, email, password):
        """Authenticate and get access token"""
        response = requests.post(
            f"{self.base_url}/auth/login",
            json={
                "tenant_identifier": self.tenant_id,
                "email": email,
                "password": password
            }
        )
        self.token = response.json()["access_token"]
        return self.token
    
    def create_forecast_run(self, version_id, entity, interval):
        """Create and execute forecast"""
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.post(
            f"{self.base_url}/forecasting/runs",
            headers=headers,
            json={
                "version_id": version_id,
                "forecast_filters": {
                    "entity_identifier": entity,
                    "aggregation_level": "product",
                    "interval": interval
                },
                "forecast_start": "2025-01-01",
                "forecast_end": "2025-03-31",
                "algorithms": [
                    {"algorithm_id": "xgboost", "algorithm_name": "XGBoost"}
                ]
            }
        )
        return response.json()
    
    def get_comparison(self, entities, interval):
        """Compare multiple forecasts"""
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.post(
            f"{self.base_url}/forecasting/compare",
            headers=headers,
            json={
                "entity_identifier": entities[0],
                "aggregation_level": "product",
                "interval": interval
            }
        )
        return response.json()

# Usage
client = DemandPlanningClient("acme_corp")
client.login("admin@acme.com", "password")
result = client.create_forecast_run("version-uuid", "PROD001", "MONTHLY")
```

---

## Logging & Monitoring

### Log Files

- **Application Log**: `logs/app.log` - All operations, info level
- **Error Log**: `logs/error.log` - Errors and critical issues only

### Log Format

```
2025-12-16 14:00:00 - app.core.forecasting_service - [INFO] - Forecast execution started: forecast_run_id=uuid
2025-12-16 14:00:00 - app.core.forecasting_service - [DEBUG] - Algorithm: xgboost, Parameters: {}
2025-12-16 14:00:15 - app.core.forecasting_service - [INFO] - Forecast completed: accuracy=0.92, rmse=48.23
```

### Performance Metrics

Enable performance logging in `config.py`:

```python
ENABLE_PERFORMANCE_LOGGING = True
```

Logged metrics:
- Operation start/end times
- Memory usage (MB, %)
- CPU usage (%)
- Query execution times
- Cache hit rates

### Monitoring Best Practices

1. **Health Check**: Monitor `logs/error.log` for error spikes
2. **Performance**: Check operation durations in `logs/app.log`
3. **Database**: Monitor connection pool status via `GET /api/v1/health`
4. **Forecasts**: Track forecast run completion via `run_status`

---

## Troubleshooting

### Common Issues

#### Issue: "Database connection failed"
```
Solution: 
1. Verify PostgreSQL is running: psql -U postgres
2. Check DB_HOST, DB_PORT in .env
3. Verify DATABASE_URL format
```

#### Issue: "Field catalogue must be finalized"
```
Solution:
1. Call POST /field-catalogue/{id}/finalize first
2. This creates the master_data table with dynamic columns
```

#### Issue: "Upload failed: column tenant_id does not exist"
```
Solution:
This is expected - tenant_id is removed from tenant-scoped tables
since each tenant has its own isolated database.
Use database_name instead of tenant_id for connections.
```

#### Issue: "Forecast execution timeout"
```
Solution:
1. Increase forecast_end date range
2. Reduce number of algorithms
3. Check external_factors availability
4. Monitor logs for specific errors
```

---

## Security Considerations

- **JWT Tokens**: 24-hour expiration by default, change `SECRET_KEY` in production
- **Password Hashing**: bcrypt with 12 salt rounds
- **Database Isolation**: Each tenant in separate database for security
- **CORS**: Configure allowed origins in `main.py` for production
- **SQL Injection**: Parameterized queries used throughout
- **Logging**: Sensitive data (passwords) excluded from logs

---

## Contributing

1. Create feature branch: `git checkout -b feature/name`
2. Make changes and test locally
3. Commit with clear messages: `git commit -m "feat: description"`
4. Push and create pull request

---

## Support

For issues, questions, or feature requests:
- Check logs in `logs/` directory
- Review API docs at `http://localhost:8000/api/docs`
- Contact: support@demandplanning.example.com

---

## License

Proprietary - Smart Demand Planning Tool © 2025

---

**Last Updated**: December 16, 2025  
**Version**: 1.0.0
