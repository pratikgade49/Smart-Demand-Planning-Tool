"""
User Synchronization Service.
Handles syncing users between tenant databases (source of truth) and master database (lookup cache).
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from app.core.database import get_db_manager
from app.core.exceptions import DatabaseException

logger = logging.getLogger(__name__)


class UserSyncService:
    """Service for synchronizing users between tenant and master databases."""

    @staticmethod
    def sync_user_to_master(
        user_id: str,
        tenant_id: str,
        database_name: str,
        email: str,
        role: str,
        is_active: bool
    ) -> bool:
        """
        Sync user metadata from tenant database to master database.
        
        Args:
            user_id: User identifier
            tenant_id: Tenant identifier
            database_name: Tenant's database name
            email: User email
            role: User role
            is_active: Whether user is active
            
        Returns:
            True if sync successful
        """
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_master_connection() as conn:
                cursor = conn.cursor()
                try:
                    # Upsert user in master database
                    cursor.execute("""
                        INSERT INTO public.users
                        (user_id, tenant_id, email, database_name, role, is_active, last_synced_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (tenant_id, email) 
                        DO UPDATE SET
                            user_id = EXCLUDED.user_id,
                            database_name = EXCLUDED.database_name,
                            role = EXCLUDED.role,
                            is_active = EXCLUDED.is_active,
                            updated_at = CURRENT_TIMESTAMP,
                            last_synced_at = EXCLUDED.last_synced_at
                    """, (
                        user_id,
                        tenant_id,
                        email,
                        database_name,
                        role,
                        is_active,
                        datetime.utcnow()
                    ))
                    
                    conn.commit()
                    logger.info(f"  Synced user {email} to master database")
                    return True
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"Failed to sync user to master: {str(e)}")
            # Don't raise - user still exists in tenant database (source of truth)
            return False

    @staticmethod
    def sync_user_deletion_to_master(
        user_id: str,
        tenant_id: str
    ) -> bool:
        """
        Sync user soft deletion from tenant to master database.
        
        Args:
            user_id: User identifier
            tenant_id: Tenant identifier
            
        Returns:
            True if sync successful
        """
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_master_connection() as conn:
                cursor = conn.cursor()
                try:
                    # Mark user as inactive in master
                    cursor.execute("""
                        UPDATE public.users
                        SET is_active = FALSE,
                            updated_at = CURRENT_TIMESTAMP,
                            last_synced_at = CURRENT_TIMESTAMP
                        WHERE user_id = %s AND tenant_id = %s
                    """, (user_id, tenant_id))
                    
                    conn.commit()
                    logger.info(f"  Synced user deletion to master database")
                    return True
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"Failed to sync user deletion to master: {str(e)}")
            return False

    @staticmethod
    def sync_user_update_to_master(
        user_id: str,
        tenant_id: str,
        email: Optional[str] = None,
        role: Optional[str] = None,
        is_active: Optional[bool] = None
    ) -> bool:
        """
        Sync user updates from tenant to master database.
        
        Args:
            user_id: User identifier
            tenant_id: Tenant identifier
            email: New email (if changed)
            role: New role (if changed)
            is_active: New active status (if changed)
            
        Returns:
            True if sync successful
        """
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_master_connection() as conn:
                cursor = conn.cursor()
                try:
                    # Build update query dynamically
                    updates = []
                    params = []
                    
                    if email is not None:
                        updates.append("email = %s")
                        params.append(email)
                    
                    if role is not None:
                        updates.append("role = %s")
                        params.append(role)
                    
                    if is_active is not None:
                        updates.append("is_active = %s")
                        params.append(is_active)
                    
                    if not updates:
                        return True  # Nothing to update
                    
                    # Add standard fields
                    updates.extend(["updated_at = CURRENT_TIMESTAMP", "last_synced_at = CURRENT_TIMESTAMP"])
                    params.extend([user_id, tenant_id])
                    
                    query = f"""
                        UPDATE public.users
                        SET {', '.join(updates)}
                        WHERE user_id = %s AND tenant_id = %s
                    """
                    
                    cursor.execute(query, params)
                    conn.commit()
                    
                    logger.info(f"  Synced user update to master database")
                    return True
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"Failed to sync user update to master: {str(e)}")
            return False

    @staticmethod
    def validate_sync_status(tenant_id: str, database_name: str) -> Dict[str, Any]:
        """
        Validate sync status between tenant and master databases.
        Identifies users that are out of sync.
        
        Args:
            tenant_id: Tenant identifier
            database_name: Tenant's database name
            
        Returns:
            Dictionary with sync status information
        """
        db_manager = get_db_manager()
        
        try:
            # Get users from tenant database (source of truth)
            with db_manager.get_tenant_connection(database_name) as tenant_conn:
                tenant_cursor = tenant_conn.cursor()
                try:
                    tenant_cursor.execute("""
                        SELECT user_id, email, role, status
                        FROM users
                        WHERE tenant_id = %s
                    """, (tenant_id,))
                    
                    tenant_users = {}
                    for row in tenant_cursor.fetchall():
                        user_id, email, role, status = row
                        tenant_users[str(user_id)] = {
                            'email': email,
                            'role': role,
                            'is_active': (status == 'ACTIVE')
                        }
                        
                finally:
                    tenant_cursor.close()
            
            # Get users from master database
            with db_manager.get_master_connection() as master_conn:
                master_cursor = master_conn.cursor()
                try:
                    master_cursor.execute("""
                        SELECT user_id, email, role, is_active
                        FROM public.users
                        WHERE tenant_id = %s
                    """, (tenant_id,))
                    
                    master_users = {}
                    for row in master_cursor.fetchall():
                        user_id, email, role, is_active = row
                        master_users[str(user_id)] = {
                            'email': email,
                            'role': role,
                            'is_active': is_active
                        }
                        
                finally:
                    master_cursor.close()
            
            # Compare and find discrepancies
            missing_in_master = []
            mismatched = []
            extra_in_master = []
            
            for user_id, tenant_data in tenant_users.items():
                if user_id not in master_users:
                    missing_in_master.append({
                        'user_id': user_id,
                        'email': tenant_data['email']
                    })
                else:
                    master_data = master_users[user_id]
                    if (tenant_data['email'] != master_data['email'] or
                        tenant_data['role'] != master_data['role'] or
                        tenant_data['is_active'] != master_data['is_active']):
                        mismatched.append({
                            'user_id': user_id,
                            'tenant_data': tenant_data,
                            'master_data': master_data
                        })
            
            for user_id in master_users:
                if user_id not in tenant_users:
                    extra_in_master.append({
                        'user_id': user_id,
                        'email': master_users[user_id]['email']
                    })
            
            sync_status = {
                'tenant_id': tenant_id,
                'database_name': database_name,
                'total_tenant_users': len(tenant_users),
                'total_master_users': len(master_users),
                'in_sync': len(missing_in_master) == 0 and len(mismatched) == 0 and len(extra_in_master) == 0,
                'missing_in_master': missing_in_master,
                'mismatched': mismatched,
                'extra_in_master': extra_in_master
            }
            
            return sync_status
            
        except Exception as e:
            logger.error(f"Failed to validate sync status: {str(e)}")
            raise DatabaseException(f"Failed to validate sync status: {str(e)}")

    @staticmethod
    def repair_sync(tenant_id: str, database_name: str) -> Dict[str, Any]:
        """
        Repair sync issues by syncing tenant database to master.
        Tenant database is the source of truth.
        
        Args:
            tenant_id: Tenant identifier
            database_name: Tenant's database name
            
        Returns:
            Dictionary with repair results
        """
        db_manager = get_db_manager()
        
        try:
            # Get current sync status
            sync_status = UserSyncService.validate_sync_status(tenant_id, database_name)
            
            repaired_count = 0
            failed_count = 0
            
            # Sync missing users
            for user in sync_status['missing_in_master']:
                # Get full user data from tenant
                with db_manager.get_tenant_connection(database_name) as conn:
                    cursor = conn.cursor()
                    try:
                        cursor.execute("""
                            SELECT user_id, email, role, status
                            FROM users
                            WHERE user_id = %s
                        """, (user['user_id'],))
                        
                        row = cursor.fetchone()
                        if row:
                            user_id, email, role, status = row
                            success = UserSyncService.sync_user_to_master(
                                str(user_id),
                                tenant_id,
                                database_name,
                                email,
                                role,
                                status == 'ACTIVE'
                            )
                            if success:
                                repaired_count += 1
                            else:
                                failed_count += 1
                    finally:
                        cursor.close()
            
            # Fix mismatched users (tenant wins)
            for mismatch in sync_status['mismatched']:
                tenant_data = mismatch['tenant_data']
                success = UserSyncService.sync_user_update_to_master(
                    mismatch['user_id'],
                    tenant_id,
                    email=tenant_data['email'],
                    role=tenant_data['role'],
                    is_active=tenant_data['is_active']
                )
                if success:
                    repaired_count += 1
                else:
                    failed_count += 1
            
            # Remove extra users in master (they don't exist in tenant)
            for user in sync_status['extra_in_master']:
                success = UserSyncService.sync_user_deletion_to_master(
                    user['user_id'],
                    tenant_id
                )
                if success:
                    repaired_count += 1
                else:
                    failed_count += 1
            
            return {
                'tenant_id': tenant_id,
                'repaired_count': repaired_count,
                'failed_count': failed_count,
                'total_issues': len(sync_status['missing_in_master']) + 
                               len(sync_status['mismatched']) + 
                               len(sync_status['extra_in_master'])
            }
            
        except Exception as e:
            logger.error(f"Failed to repair sync: {str(e)}")
            raise DatabaseException(f"Failed to repair sync: {str(e)}")