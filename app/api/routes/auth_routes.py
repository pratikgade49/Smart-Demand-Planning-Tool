"""
Authentication API Routes.
Tenant registration and login endpoints.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from typing import Dict, Any, Optional
from app.schemas.auth import (
    TenantLoginRequest,
    TenantLoginResponse,
    TenantOnboardRequest,
    TenantOnboardResponse,
    UserRegisterRequest,
    UserRegisterResponse,
    UserLoginRequest,
    UserLoginResponse,
    TenantListResponse
)
from app.core.auth_service import AuthService
from app.core.rbac_service import RBACService
from app.core.responses import ResponseHandler
from app.core.exceptions import AppException
from app.api.dependencies import get_current_tenant, get_current_user
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=Dict[str, Any])
async def login_tenant(request: TenantLoginRequest):
    """
    Login tenant and get JWT access token.
    """
    try:
        result = AuthService.login_tenant(request)
        
        return ResponseHandler.success(data=result)
        
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in login_tenant: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/verify-token", response_model=Dict[str, Any])
async def verify_token(token: str):
    """
    Verify JWT token validity.
    """
    try:
        payload = AuthService.verify_tenant_token(token)
        
        return ResponseHandler.success(data={
            "valid": True,
            "tenant_id": payload.get("tenant_id"),
            "email": payload.get("email")
        })
        
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in verify_token: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    
@router.get("/tenant/{tenant_id}", response_model=Dict[str, Any], tags=["Tenant Management"])
async def get_tenant_info(
    tenant_id: str,
    current_tenant: Dict = Depends(get_current_tenant)
):
    """
    Get tenant information.
    Only tenant can view their own information.
    """
    try:
        # Validate access
        if current_tenant["tenant_id"] != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot access other tenant's information"
            )
        
        result = AuthService.get_tenant_by_id(tenant_id)
        return ResponseHandler.success(data=result)
        
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Error getting tenant info: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/tenant/{tenant_id}/status", response_model=Dict[str, Any], tags=["Tenant Management"])
async def update_tenant_status(
    tenant_id: str,
    status: str,
    current_tenant: Dict = Depends(get_current_tenant)
):
    """
    Update tenant status (ACTIVE, INACTIVE, SUSPENDED).
    Admin operation - currently restricted to the tenant itself.
    """
    try:
        # Validate access
        if current_tenant["tenant_id"] != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot modify other tenant's status"
            )
        
        if status not in ["ACTIVE", "INACTIVE", "SUSPENDED"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid status. Must be ACTIVE, INACTIVE, or SUSPENDED"
            )
        
        AuthService.update_tenant_status(tenant_id, status)
        
        return ResponseHandler.success(data={
            "tenant_id": tenant_id,
            "status": status,
            "message": f"Tenant status updated to {status}"
        })
        
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Error updating tenant status: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/onboard-tenant", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def onboard_tenant(request: TenantOnboardRequest):
    """
    Onboard a new tenant.
    Creates tenant database without creating any users.
    """
    try:
        result = AuthService.onboard_tenant(request)

        response_data = {
            "tenant_id": result["tenant_id"],
            "tenant_name": result["tenant_name"],
            "tenant_identifier": result["tenant_identifier"],
            "database_name": result["database_name"],
            "tenant_url": result["tenant_url"],
            "message": "Tenant onboarded successfully."
        }

        return ResponseHandler.success(data=response_data, status_code=201)

    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in onboard_tenant: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/register-user", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def register_user(request: UserRegisterRequest):
    """
    Register a new user under an existing tenant.
    """
    try:
        result = AuthService.register_user(request)

        response_data = {
            "user_id": result["user_id"],
            "tenant_id": result["tenant_id"],
            "email": result["email"],
            "message": "User registered successfully. Please login to continue."
        }

        return ResponseHandler.success(data=response_data, status_code=201)

    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in register_user: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/user-login", response_model=Dict[str, Any])
async def login_user(request: UserLoginRequest):
    """
    Login user and get JWT access token.
    """
    try:
        result = AuthService.login_user(request)

        return ResponseHandler.success(data=result)

    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in login_user: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/tenants", response_model=Dict[str, Any])
async def list_tenants():
    """
    List all active tenants for user registration dropdown.
    """
    try:
        result = AuthService.list_tenants()

        return ResponseHandler.success(data={"tenants": result})

    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in list_tenants: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/verify-user-token", response_model=Dict[str, Any])
async def verify_user_token(token: str):
    """
    Verify JWT user token validity.
    """
    try:
        payload = AuthService.verify_user_token(token)

        return ResponseHandler.success(data={
            "valid": True,
            "user_id": payload.get("user_id"),
            "tenant_id": payload.get("tenant_id"),
            "email": payload.get("email")
        })

    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in verify_user_token: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/user/accessible-objects", response_model=Dict[str, Any])
async def get_user_accessible_objects(current_user: Dict = Depends(get_current_user)):
    """
    Get list of objects the current user has access to.
    Used by Flutter UI to show/hide menu items.
    """
    try:
        accessible_objects = RBACService.get_user_accessible_objects(
            current_user["user_id"],
            current_user["database_name"]
        )

        return ResponseHandler.success(data={
            "accessible_objects": accessible_objects
        })

    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in get_user_accessible_objects: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/rbac/roles", response_model=Dict[str, Any], tags=["RBAC Management"])
async def get_all_roles(current_user: Dict = Depends(get_current_user)):
    """
    Get all available roles for assignment.
    """
    try:
        roles = RBACService.get_all_roles(current_user["database_name"])

        return ResponseHandler.success(data={"roles": roles})

    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in get_all_roles: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/rbac/objects", response_model=Dict[str, Any], tags=["RBAC Management"])
async def get_all_objects(current_user: Dict = Depends(get_current_user)):
    """
    Get all available objects for assignment.
    """
    try:
        objects = RBACService.get_all_objects(current_user["database_name"])

        return ResponseHandler.success(data={"objects": objects})

    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in get_all_objects: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/rbac/assign", response_model=Dict[str, Any], tags=["RBAC Management"])
async def assign_role_to_user(
    user_id: str,
    role_id: int,
    object_id: int,
    reporting_to: Optional[str] = None,
    current_user: Dict = Depends(get_current_user)
):
    """
    Assign a role-object combination to a user.
    Admin operation.
    """
    try:
        result = RBACService.assign_role_to_user(
            user_id=user_id,
            role_id=role_id,
            object_id=object_id,
            assigned_by=current_user["user_id"],
            database_name=current_user["database_name"],
            reporting_to=reporting_to
        )

        return ResponseHandler.success(data=result, status_code=201)

    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in assign_role_to_user: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/rbac/assignment/{assignment_id}", response_model=Dict[str, Any], tags=["RBAC Management"])
async def revoke_assignment(
    assignment_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Revoke a role assignment.
    Admin operation.
    """
    try:
        success = RBACService.revoke_assignment(
            assignment_id=assignment_id,
            revoked_by=current_user["user_id"],
            database_name=current_user["database_name"]
        )

        return ResponseHandler.success(data={
            "assignment_id": assignment_id,
            "message": "Assignment revoked successfully"
        })

    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in revoke_assignment: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/rbac/assignments", response_model=Dict[str, Any], tags=["RBAC Management"])
async def get_assignments_for_admin(current_user: Dict = Depends(get_current_user)):
    """
    Get all assignments for admin management.
    """
    try:
        assignments = RBACService.get_assignments_for_admin(current_user["database_name"])

        return ResponseHandler.success(data={"assignments": assignments})

    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in get_assignments_for_admin: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
