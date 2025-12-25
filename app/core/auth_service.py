"""
Authentication Service with Dual-Write to Master and Tenant Databases.
UPDATED: Users are now stored in both master (lookup) and tenant (source of truth) databases.
"""

import bcrypt
import jwt
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging

from app.core.database import get_db_manager
from app.core.schema_manager import SchemaManager
from app.core.user_sync_service import UserSyncService
from app.core.exceptions import (
    AuthenticationException,
    ValidationException,
    ConflictException,
    DatabaseException
)
from app.config import settings

logger = logging.getLogger(__name__)


class AuthService:
    """Authentication and authorization service with dual-write support."""
    
    # JWT Configuration
    SECRET_KEY = settings.JWT_SECRET_KEY
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_HOURS = 24

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt."""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')

    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(
            password.encode('utf-8'),
            hashed_password.encode('utf-8')
        )

    @staticmethod
    def generate_token(payload: Dict[str, Any]) -> str:
        """Generate JWT token."""
        expiration = datetime.utcnow() + timedelta(hours=AuthService.ACCESS_TOKEN_EXPIRE_HOURS)
        payload_with_exp = {**payload, "exp": expiration}
        token = jwt.encode(payload_with_exp, AuthService.SECRET_KEY, algorithm=AuthService.ALGORITHM)
        return token

    @staticmethod
    def verify_tenant_token(token: str) -> Dict[str, Any]:
        """Verify and decode tenant JWT token."""
        try:
            payload = jwt.decode(token, AuthService.SECRET_KEY, algorithms=[AuthService.ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationException("Token has expired")
        except jwt.InvalidTokenError:
            raise AuthenticationException("Invalid token")

    @staticmethod
    def verify_user_token(token: str) -> Dict[str, Any]:
        """Verify and decode user JWT token."""
        return AuthService.verify_tenant_token(token)

    @staticmethod
    def register_tenant(request) -> Dict[str, Any]:
        """
        Register a new tenant (backward compatible).
        Creates tenant database and admin user in tenants table.
        """
        db_manager = get_db_manager()
        tenant_id = str(uuid.uuid4())
        
        # Generate database name
        database_name = f"tenant_{request.tenant_identifier.lower()}_db"
        
        # Hash admin password
        password_hash = AuthService.hash_password(request.admin_password)
        
        try:
            # Insert tenant into master database
            with db_manager.get_master_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        INSERT INTO public.tenants
                        (tenant_id, tenant_name, tenant_identifier, admin_email,
                         admin_password_hash, database_name, status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        tenant_id,
                        request.tenant_name,
                        request.tenant_identifier,
                        request.admin_email,
                        password_hash,
                        database_name,
                        'ACTIVE'
                    ))
                    conn.commit()
                finally:
                    cursor.close()
            
            # Create tenant database
            SchemaManager.create_tenant_database(tenant_id, database_name)
            
            logger.info(f"Tenant registered successfully: {request.tenant_identifier}")
            
            return {
                "tenant_id": tenant_id,
                "tenant_name": request.tenant_name,
                "tenant_identifier": request.tenant_identifier,
                "database_name": database_name
            }
            
        except Exception as e:
            logger.error(f"Failed to register tenant: {str(e)}")
            raise DatabaseException(f"Failed to register tenant: {str(e)}")

    @staticmethod
    def login_tenant(request) -> Dict[str, Any]:
        """
        Login tenant admin (backward compatible).
        """
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_master_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        SELECT tenant_id, tenant_name, tenant_identifier, admin_email,
                               admin_password_hash, database_name, status
                        FROM public.tenants
                        WHERE tenant_identifier = %s
                    """, (request.tenant_identifier,))
                    
                    result = cursor.fetchone()
                    
                    if not result:
                        raise AuthenticationException("Invalid credentials")
                    
                    (tenant_id, tenant_name, tenant_identifier, admin_email,
                     password_hash, database_name, status) = result
                    
                    if status != 'ACTIVE':
                        raise AuthenticationException(f"Tenant account is {status}")
                    
                    if not AuthService.verify_password(request.password, password_hash):
                        raise AuthenticationException("Invalid credentials")
                    
                    # Generate JWT token
                    token_payload = {
                        "tenant_id": str(tenant_id),
                        "tenant_identifier": tenant_identifier,
                        "email": admin_email,
                        "database_name": database_name
                    }
                    
                    token = AuthService.generate_token(token_payload)
                    
                    return {
                        "access_token": token,
                        "token_type": "bearer",
                        "tenant_id": str(tenant_id),
                        "tenant_name": tenant_name,
                        "email": admin_email
                    }
                    
                finally:
                    cursor.close()
                    
        except AuthenticationException:
            raise
        except Exception as e:
            logger.error(f"Failed to login tenant: {str(e)}")
            raise DatabaseException(f"Failed to login tenant: {str(e)}")

    @staticmethod
    def onboard_tenant(request) -> Dict[str, Any]:
        """
        Onboard a new tenant without creating users.
        Just creates tenant database.
        """
        db_manager = get_db_manager()
        tenant_id = str(uuid.uuid4())
        
        # Generate database name
        database_name = f"tenant_{request.tenant_identifier.lower()}_db"
        
        try:
            # Insert tenant into master database (no admin credentials)
            with db_manager.get_master_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        INSERT INTO public.tenants
                        (tenant_id, tenant_name, tenant_identifier, database_name, status)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        tenant_id,
                        request.tenant_name,
                        request.tenant_identifier,
                        database_name,
                        'ACTIVE'
                    ))
                    conn.commit()
                finally:
                    cursor.close()
            
            # Create tenant database
            SchemaManager.create_tenant_database(tenant_id, database_name)
            
            logger.info(f"Tenant onboarded successfully: {request.tenant_identifier}")
            
            return {
                "tenant_id": tenant_id,
                "tenant_name": request.tenant_name,
                "tenant_identifier": request.tenant_identifier,
                "database_name": database_name
            }
            
        except Exception as e:
            logger.error(f"Failed to onboard tenant: {str(e)}")
            raise DatabaseException(f"Failed to onboard tenant: {str(e)}")

    @staticmethod
    def register_user(request) -> Dict[str, Any]:
        """
        Register a new user with DUAL-WRITE to master and tenant databases.
        
        Flow:
        1. Validate tenant exists and get database_name
        2. Check email uniqueness in master (per tenant)
        3. Write to TENANT database first (SOURCE OF TRUTH)
        4. Sync to MASTER database (LOOKUP CACHE)
        5. Handle sync failures gracefully
        
        Args:
            request: User registration request with tenant_id, email, password, etc.
            
        Returns:
            Created user information
        """
        db_manager = get_db_manager()
        user_id = str(uuid.uuid4())
        
        try:
            # Step 1: Validate tenant exists and get database_name
            with db_manager.get_master_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        SELECT tenant_id, database_name, status
                        FROM public.tenants
                        WHERE tenant_id = %s
                    """, (request.tenant_id,))
                    
                    tenant_result = cursor.fetchone()
                    
                    if not tenant_result:
                        raise ValidationException(f"Tenant not found: {request.tenant_id}")
                    
                    tenant_id, database_name, tenant_status = tenant_result
                    
                    if tenant_status != 'ACTIVE':
                        raise ValidationException(f"Tenant is {tenant_status}")
                    
                    # Step 2: Check email uniqueness in master (per tenant)
                    cursor.execute("""
                        SELECT user_id
                        FROM public.users
                        WHERE tenant_id = %s AND email = %s
                    """, (tenant_id, request.email))
                    
                    if cursor.fetchone():
                        raise ConflictException(f"Email {request.email} is already registered for this tenant")
                    
                finally:
                    cursor.close()
            
            # Step 3: Write to TENANT database first (SOURCE OF TRUTH)
            password_hash = AuthService.hash_password(request.password)
            
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Check email uniqueness in tenant database as well
                    cursor.execute("""
                        SELECT user_id
                        FROM users
                        WHERE email = %s
                    """, (request.email,))
                    
                    if cursor.fetchone():
                        raise ConflictException(f"Email {request.email} is already registered")
                    
                    # Insert user in tenant database
                    cursor.execute("""
                        INSERT INTO users
                        (user_id, tenant_id, email, password_hash, first_name, last_name,
                         role, status, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        user_id,
                        tenant_id,
                        request.email,
                        password_hash,
                        request.first_name,
                        request.last_name,
                        request.role,
                        'ACTIVE',
                        datetime.utcnow(),
                        datetime.utcnow()
                    ))
                    
                    conn.commit()
                    logger.info(f"✅ User {request.email} created in tenant database: {database_name}")
                    
                finally:
                    cursor.close()
            
            # Step 4: Sync to MASTER database (LOOKUP CACHE)
            sync_success = UserSyncService.sync_user_to_master(
                user_id=user_id,
                tenant_id=str(tenant_id),
                database_name=database_name,
                email=request.email,
                role=request.role,
                is_active=True
            )
            
            if not sync_success:
                # Log warning but don't fail - user exists in tenant DB (source of truth)
                logger.warning(
                    f"⚠️  User {request.email} created in tenant DB but master sync failed. "
                    f"Background job will retry sync."
                )
            else:
                logger.info(f"✅ User {request.email} synced to master database")
            
            return {
                "user_id": user_id,
                "tenant_id": str(tenant_id),
                "email": request.email,
                "first_name": request.first_name,
                "last_name": request.last_name,
                "role": request.role,
                "status": "ACTIVE"
            }
            
        except (ValidationException, ConflictException):
            raise
        except Exception as e:
            logger.error(f"Failed to register user: {str(e)}")
            raise DatabaseException(f"Failed to register user: {str(e)}")

    @staticmethod
    def login_user(request) -> Dict[str, Any]:
        """
        Login user with FAST LOOKUP using master database.
        
        NEW Flow (OPTIMIZED):
        1. Fast lookup in master.users JOIN tenants (get tenant_id + database_name)
        2. Verify password in tenant database (source of truth)
        3. Update last_login timestamp
        4. Generate JWT token
        
        OLD Flow (SLOW - commented out):
        - Loop through all tenant databases searching for email
        - Very slow with many tenants
        
        Args:
            request: Login request with email, password, and tenant_identifier
            
        Returns:
            JWT token and user information
        """
        db_manager = get_db_manager()
        
        try:
            # Step 1: Fast lookup in master database (single query with JOIN)
            with db_manager.get_master_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        SELECT 
                            u.user_id,
                            u.tenant_id,
                            u.email,
                            u.database_name,
                            u.role,
                            u.is_active,
                            t.tenant_name,
                            t.tenant_identifier,
                            t.status as tenant_status
                        FROM public.users u
                        JOIN public.tenants t ON u.tenant_id = t.tenant_id
                        WHERE t.tenant_identifier = %s
                          AND u.email = %s
                          AND u.is_active = TRUE
                    """, (request.tenant_identifier, request.email))
                    
                    result = cursor.fetchone()
                    
                    if not result:
                        raise AuthenticationException("Invalid credentials")
                    
                    (user_id, tenant_id, email, database_name, role, is_active,
                     tenant_name, tenant_identifier, tenant_status) = result
                    
                    # Check tenant status
                    if tenant_status != 'ACTIVE':
                        raise AuthenticationException(f"Tenant account is {tenant_status}")
                    
                finally:
                    cursor.close()
            
            logger.info(f"✅ Fast lookup: Found user {email} in tenant {tenant_identifier} (master query)")
            
            # Step 2: Verify password in TENANT database (source of truth)
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        SELECT user_id, password_hash, first_name, last_name, role, status
                        FROM users
                        WHERE user_id = %s
                    """, (user_id,))
                    
                    tenant_result = cursor.fetchone()
                    
                    if not tenant_result:
                        # User exists in master but not in tenant - sync issue
                        logger.error(
                            f"❌ SYNC ERROR: User {email} exists in master but not in tenant {database_name}"
                        )
                        raise AuthenticationException("Account sync error. Please contact support.")
                    
                    (tenant_user_id, password_hash, first_name, last_name, 
                     tenant_role, tenant_status) = tenant_result
                    
                    # Check user status in tenant database
                    if tenant_status != 'ACTIVE':
                        raise AuthenticationException(f"User account is {tenant_status}")
                    
                    # Verify password
                    if not AuthService.verify_password(request.password, password_hash):
                        raise AuthenticationException("Invalid credentials")
                    
                    # Step 3: Update last_login timestamp in tenant database
                    cursor.execute("""
                        UPDATE users
                        SET last_login = %s,
                            updated_at = %s
                        WHERE user_id = %s
                    """, (datetime.utcnow(), datetime.utcnow(), user_id))
                    
                    conn.commit()
                    
                finally:
                    cursor.close()
            
            logger.info(f"✅ Password verified for user {email} in tenant database")
            
            # Step 4: Generate JWT token
            token_payload = {
                "user_id": str(user_id),
                "tenant_id": str(tenant_id),
                "tenant_identifier": tenant_identifier,
                "email": email,
                "database_name": database_name,
                "role": tenant_role  # Use role from tenant DB (source of truth)
            }
            
            token = AuthService.generate_token(token_payload)
            
            logger.info(f"✅ User {email} logged in successfully")
            
            return {
                "access_token": token,
                "token_type": "bearer",
                "user_id": str(user_id),
                "tenant_id": str(tenant_id),
                "tenant_name": tenant_name,
                "email": email,
                "first_name": first_name,
                "last_name": last_name,
                "role": tenant_role
            }
            
        except AuthenticationException:
            raise
        except Exception as e:
            logger.error(f"Failed to login user: {str(e)}", exc_info=True)
            raise DatabaseException(f"Failed to login user: {str(e)}")

    @staticmethod
    def get_tenant_by_id(tenant_id: str) -> Dict[str, Any]:
        """Get tenant information by ID."""
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_master_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        SELECT tenant_id, tenant_name, tenant_identifier,
                               admin_email, database_name, created_at, status
                        FROM public.tenants
                        WHERE tenant_id = %s
                    """, (tenant_id,))
                    
                    result = cursor.fetchone()
                    
                    if not result:
                        raise ValidationException(f"Tenant not found: {tenant_id}")
                    
                    (tid, name, identifier, admin_email, db_name, created_at, status) = result
                    
                    return {
                        "tenant_id": str(tid),
                        "tenant_name": name,
                        "tenant_identifier": identifier,
                        "admin_email": admin_email,
                        "database_name": db_name,
                        "created_at": created_at.isoformat() if created_at else None,
                        "status": status
                    }
                    
                finally:
                    cursor.close()
                    
        except ValidationException:
            raise
        except Exception as e:
            logger.error(f"Failed to get tenant: {str(e)}")
            raise DatabaseException(f"Failed to get tenant: {str(e)}")

    @staticmethod
    def update_tenant_status(tenant_id: str, status: str) -> None:
        """Update tenant status."""
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_master_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        UPDATE public.tenants
                        SET status = %s
                        WHERE tenant_id = %s
                    """, (status, tenant_id))
                    
                    if cursor.rowcount == 0:
                        raise ValidationException(f"Tenant not found: {tenant_id}")
                    
                    conn.commit()
                    
                finally:
                    cursor.close()
                    
        except ValidationException:
            raise
        except Exception as e:
            logger.error(f"Failed to update tenant status: {str(e)}")
            raise DatabaseException(f"Failed to update tenant status: {str(e)}")

    @staticmethod
    def list_tenants() -> list:
        """List all active tenants for user registration dropdown."""
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_master_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        SELECT tenant_id, tenant_name, tenant_identifier
                        FROM public.tenants
                        WHERE status = 'ACTIVE'
                        ORDER BY tenant_name
                    """)
                    
                    tenants = []
                    for row in cursor.fetchall():
                        tenants.append({
                            "tenant_id": str(row[0]),
                            "tenant_name": row[1],
                            "tenant_identifier": row[2]
                        })
                    
                    return tenants
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"Failed to list tenants: {str(e)}")
            raise DatabaseException(f"Failed to list tenants: {str(e)}")