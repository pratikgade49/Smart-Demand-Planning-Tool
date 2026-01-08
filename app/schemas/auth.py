"""
Pydantic schemas for authentication endpoints.
"""

from pydantic import BaseModel, Field, EmailStr, field_validator, ConfigDict
from typing import Optional

class TenantRegisterRequest(BaseModel):
    """Request schema for tenant registration."""
    
    tenant_name: str = Field(..., min_length=1, max_length=255, description="Tenant name")
    tenant_identifier: str = Field(..., min_length=1, max_length=100, description="Unique tenant identifier")
    email: EmailStr = Field(..., description="Admin email")
    password: str = Field(..., min_length=8, description="Admin password")
    
    @field_validator("tenant_identifier")
    @classmethod
    def validate_tenant_identifier(cls, v: str) -> str:
        """Validate tenant identifier format."""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Tenant identifier can only contain alphanumeric characters, hyphens, and underscores")
        return v.lower()

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tenant_name": "Acme Corporation",
                "tenant_identifier": "acme_corp",
                "email": "admin@acme.com",
                "password": "securepassword123"
            }
        }
    )

class TenantLoginRequest(BaseModel):
    """Request schema for tenant login."""

    tenant_identifier: str = Field(..., description="Tenant identifier")
    email: EmailStr = Field(..., description="Email")
    password: str = Field(..., description="Password")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tenant_identifier": "acme_corp",
                "email": "admin@acme.com",
                "password": "securepassword123"
            }
        }
    )

class TenantLoginResponse(BaseModel):
    """Response schema for successful tenant login."""

    tenant_id: str
    tenant_name: str
    access_token: str
    token_type: str = "bearer"

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
                "tenant_name": "Acme Corporation",
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer"
            }
        }
    )

class TenantRegisterResponse(BaseModel):
    """Response schema for successful tenant registration."""

    tenant_id: str
    tenant_name: str
    message: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
                "tenant_name": "Acme Corporation",
                "message": "Tenant registered successfully. Please login to continue."
            }
        }
    )

class TenantOnboardRequest(BaseModel):
    """Request schema for tenant onboarding."""

    tenant_name: str = Field(..., min_length=1, max_length=255, description="Tenant name")
    tenant_identifier: str = Field(..., min_length=1, max_length=100, description="Unique tenant identifier")
    subscription: str = Field(..., description="Subscription type")
    cpu_size: str = Field(..., description="CPU Size")
    network_load_gb_month: float = Field(..., ge=0, description="Network load in GB per month")
    db_space: str = Field(..., description="Database space")
    no_of_users: int = Field(..., ge=1, description="Number of users")
    subscription_start_date: str = Field(..., description="Subscription start date (YYYY-MM-DD)")
    subscription_end_date: str = Field(..., description="Subscription end date (YYYY-MM-DD)")

    @field_validator("tenant_identifier")
    @classmethod
    def validate_tenant_identifier(cls, v: str) -> str:
        """Validate tenant identifier format."""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Tenant identifier can only contain alphanumeric characters, hyphens, and underscores")
        return v.lower()

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tenant_name": "Acme Corporation",
                "tenant_identifier": "acme_corp",
                "subscription": "Enterprise",
                "cpu_size": "4-core",
                "network_load_gb_month": 100.5,
                "db_space": "100GB",
                "no_of_users": 50,
                "subscription_start_date": "2024-01-01",
                "subscription_end_date": "2024-12-31"
            }
        }
    )

class TenantOnboardResponse(BaseModel):
    """Response schema for successful tenant onboarding."""

    tenant_id: str
    tenant_name: str
    tenant_identifier: str
    database_name: str
    tenant_url: str
    message: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
                "tenant_name": "Acme Corporation",
                "tenant_identifier": "acme_corp",
                "database_name": "tenant_acme_corp_db",
                "tenant_url": "http://localhost:8000/acme_corp",
                "message": "Tenant onboarded successfully."
            }
        }
    )

class UserRegisterRequest(BaseModel):
    """Request schema for user registration."""

    tenant_id: str = Field(..., description="Tenant ID to register under")
    email: EmailStr = Field(..., description="User email")
    password: str = Field(..., min_length=8, description="User password")
    first_name: Optional[str] = Field(None, max_length=100, description="User first name")
    last_name: Optional[str] = Field(None, max_length=100, description="User last name")
    role: str = Field("user", description="User role (admin, user, etc.)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
                "email": "user@acme.com",
                "password": "securepassword123",
                "first_name": "John",
                "last_name": "Doe",
                "role": "user"
            }
        }
    )

class UserRegisterResponse(BaseModel):
    """Response schema for successful user registration."""

    user_id: str
    tenant_id: str
    email: str
    message: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "660e8400-e29b-41d4-a716-446655440001",
                "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
                "email": "user@acme.com",
                "message": "User registered successfully. Please login to continue."
            }
        }
    )

class UserLoginRequest(BaseModel):
    """Request schema for user login."""

    tenant_identifier: str = Field(..., description="Tenant identifier")
    email: EmailStr = Field(..., description="User email")
    password: str = Field(..., description="User password")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tenant_identifier": "acme_corp",
                "email": "user@acme.com",
                "password": "securepassword123"
            }
        }
    )

class UserLoginResponse(BaseModel):
    """Response schema for successful user login."""

    user_id: str
    tenant_id: str
    tenant_name: str
    email: str
    access_token: str
    token_type: str = "bearer"

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "660e8400-e29b-41d4-a716-446655440001",
                "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
                "tenant_name": "Acme Corporation",
                "email": "user@acme.com",
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer"
            }
        }
    )

class TenantListResponse(BaseModel):
    """Response schema for listing tenants."""

    tenants: list[dict]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tenants": [
                    {
                        "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
                        "tenant_name": "Acme Corporation",
                        "tenant_identifier": "acme_corp"
                    }
                ]
            }
        }
    )
