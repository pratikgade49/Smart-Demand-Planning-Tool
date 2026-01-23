"""
RBAC (Role-Based Access Control) Service.
Manages user role assignments, object permissions, and access control checks.
"""

from typing import List, Dict, Any, Optional
from app.core.database import get_db_manager
from app.core.exceptions import DatabaseException, ValidationException
import logging

logger = logging.getLogger(__name__)


class RBACService:
    """Service for managing RBAC operations."""

    @staticmethod
    def get_user_assignments(user_id: str, database_name: str) -> List[Dict[str, Any]]:
        """
        Get all active role-object assignments for a user.

        Args:
            user_id: User ID
            database_name: Tenant database name

        Returns:
            List of assignment dictionaries
        """
        db_manager = get_db_manager()

        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        SELECT ra.assignment_id, ra.user_id, ra.role_id, r.role_name,
                               ra.object_id, o.object_name, o.object_type,
                               ra.reporting_to, ra.assigned_by, ra.assigned_at
                        FROM roles_assignment ra
                        JOIN roles r ON ra.role_id = r.role_id
                        JOIN objects o ON ra.object_id = o.object_id
                        WHERE ra.user_id = %s AND ra.is_active = TRUE
                        ORDER BY ra.assigned_at DESC
                    """, (user_id,))

                    assignments = []
                    for row in cursor.fetchall():
                        assignments.append({
                            "assignment_id": str(row[0]),
                            "user_id": str(row[1]),
                            "role_id": row[2],
                            "role_name": row[3],
                            "object_id": row[4],
                            "object_name": row[5],
                            "object_type": row[6],
                            "reporting_to": str(row[7]) if row[7] else None,
                            "assigned_by": str(row[8]),
                            "assigned_at": row[9].isoformat() if row[9] else None
                        })

                    return assignments

                finally:
                    cursor.close()

        except Exception as e:
            logger.error(f"Failed to get user assignments for {user_id}: {str(e)}")
            raise DatabaseException(f"Failed to get user assignments: {str(e)}")

    @staticmethod
    def get_user_accessible_objects(user_id: str, database_name: str) -> List[str]:
        """
        Get list of object names the user has access to.

        Args:
            user_id: User ID
            database_name: Tenant database name

        Returns:
            List of accessible object names
        """
        assignments = RBACService.get_user_assignments(user_id, database_name)
        return list(set(assignment["object_name"] for assignment in assignments))

    @staticmethod
    def check_user_access(user_id: str, object_name: str, database_name: str) -> bool:
        """
        Check if user has access to a specific object.

        Args:
            user_id: User ID
            object_name: Object name to check
            database_name: Tenant database name

        Returns:
            True if access granted
        """
        accessible_objects = RBACService.get_user_accessible_objects(user_id, database_name)
        return object_name in accessible_objects

    @staticmethod
    def check_user_role_access(user_id: str, object_name: str, min_role_id: int, database_name: str) -> bool:
        """
        Check if user has access to a specific object with role level >= min_role_id.

        Args:
            user_id: User ID
            object_name: Object name to check
            min_role_id: Minimum required role_id (1=View, 2=Edit, 3=Delete)
            database_name: Tenant database name

        Returns:
            True if access granted with sufficient role level
        """
        logger.info(f"Checking role access for user {user_id} to object '{object_name}' with min_role_id {min_role_id}")

        db_manager = get_db_manager()

        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # First, get all user's assignments for debugging
                    cursor.execute("""
                        SELECT ra.role_id, o.object_name, ra.is_active
                        FROM roles_assignment ra
                        JOIN objects o ON ra.object_id = o.object_id
                        WHERE ra.user_id = %s
                        ORDER BY ra.object_id
                    """, (user_id,))
                    user_assignments = cursor.fetchall()
                    logger.info(f"User {user_id} assignments: {[(row[0], row[1], row[2]) for row in user_assignments]}")

                    # Now check specific access
                    cursor.execute("""
                        SELECT ra.role_id
                        FROM roles_assignment ra
                        JOIN objects o ON ra.object_id = o.object_id
                        WHERE ra.user_id = %s AND o.object_name = %s AND ra.role_id >= %s AND ra.is_active = TRUE
                        LIMIT 1
                    """, (user_id, object_name, min_role_id))

                    result = cursor.fetchone()
                    access_granted = result is not None
                    logger.info(f"Access check for user {user_id} to '{object_name}' (min_role {min_role_id}): {access_granted}")
                    return access_granted

                finally:
                    cursor.close()

        except Exception as e:
            logger.error(f"Failed to check user role access for {user_id}: {str(e)}")
            raise DatabaseException(f"Failed to check user role access: {str(e)}")

    @staticmethod
    def assign_role_to_user(
        user_id: str,
        role_id: int,
        object_id: int,
        assigned_by: str,
        database_name: str,
        reporting_to: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Assign a role-object combination to a user.

        Args:
            user_id: User to assign to
            role_id: Role ID
            object_id: Object ID
            assigned_by: User performing assignment
            database_name: Tenant database name
            reporting_to: Optional reporting user ID

        Returns:
            Assignment details
        """
        db_manager = get_db_manager()

        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Check if assignment already exists
                    cursor.execute("""
                        SELECT assignment_id FROM roles_assignment
                        WHERE user_id = %s AND role_id = %s AND object_id = %s AND is_active = TRUE
                    """, (user_id, role_id, object_id))

                    if cursor.fetchone():
                        raise ValidationException(f"Assignment already exists for user {user_id}, role {role_id}, object {object_id}")

                    # Insert new assignment
                    cursor.execute("""
                        INSERT INTO roles_assignment
                        (user_id, role_id, object_id, reporting_to, assigned_by)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING assignment_id, assigned_at
                    """, (user_id, role_id, object_id, reporting_to, assigned_by))

                    result = cursor.fetchone()
                    assignment_id, assigned_at = result

                    conn.commit()

                    logger.info(f"Assigned role {role_id} object {object_id} to user {user_id}")

                    return {
                        "assignment_id": str(assignment_id),
                        "user_id": user_id,
                        "role_id": role_id,
                        "object_id": object_id,
                        "reporting_to": reporting_to,
                        "assigned_by": assigned_by,
                        "assigned_at": assigned_at.isoformat()
                    }

                finally:
                    cursor.close()

        except ValidationException:
            raise
        except Exception as e:
            logger.error(f"Failed to assign role to user: {str(e)}")
            raise DatabaseException(f"Failed to assign role: {str(e)}")

    @staticmethod
    def revoke_assignment(assignment_id: str, revoked_by: str, database_name: str) -> bool:
        """
        Revoke a role assignment by marking it inactive.

        Args:
            assignment_id: Assignment ID to revoke
            revoked_by: User revoking the assignment
            database_name: Tenant database name

        Returns:
            True if revoked successfully
        """
        db_manager = get_db_manager()

        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        UPDATE roles_assignment
                        SET is_active = FALSE
                        WHERE assignment_id = %s
                    """, (assignment_id,))

                    if cursor.rowcount == 0:
                        raise ValidationException(f"Assignment {assignment_id} not found")

                    conn.commit()

                    logger.info(f"Revoked assignment {assignment_id} by {revoked_by}")

                    return True

                finally:
                    cursor.close()

        except ValidationException:
            raise
        except Exception as e:
            logger.error(f"Failed to revoke assignment: {str(e)}")
            raise DatabaseException(f"Failed to revoke assignment: {str(e)}")

    @staticmethod
    def get_all_roles(database_name: str) -> List[Dict[str, Any]]:
        """
        Get all available roles.

        Args:
            database_name: Tenant database name

        Returns:
            List of roles
        """
        db_manager = get_db_manager()

        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        SELECT role_id, role_name, description, created_at
                        FROM roles
                        ORDER BY role_id
                    """)

                    roles = []
                    for row in cursor.fetchall():
                        roles.append({
                            "role_id": row[0],
                            "role_name": row[1],
                            "description": row[2],
                            "created_at": row[3].isoformat() if row[3] else None
                        })

                    return roles

                finally:
                    cursor.close()

        except Exception as e:
            logger.error(f"Failed to get roles: {str(e)}")
            raise DatabaseException(f"Failed to get roles: {str(e)}")

    @staticmethod
    def get_all_objects(database_name: str) -> List[Dict[str, Any]]:
        """
        Get all available objects.

        Args:
            database_name: Tenant database name

        Returns:
            List of objects
        """
        db_manager = get_db_manager()

        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        SELECT object_id, object_type, object_name, description, created_at
                        FROM objects
                        ORDER BY object_id
                    """)

                    objects = []
                    for row in cursor.fetchall():
                        objects.append({
                            "object_id": row[0],
                            "object_type": row[1],
                            "object_name": row[2],
                            "description": row[3],
                            "created_at": row[4].isoformat() if row[4] else None
                        })

                    return objects

                finally:
                    cursor.close()

        except Exception as e:
            logger.error(f"Failed to get objects: {str(e)}")
            raise DatabaseException(f"Failed to get objects: {str(e)}")

    @staticmethod
    def get_assignments_for_admin(database_name: str) -> List[Dict[str, Any]]:
        """
        Get all assignments for admin management.

        Args:
            database_name: Tenant database name

        Returns:
            List of assignments with assignment_id, user_id, role_name, object_id, reporting_to
        """
        db_manager = get_db_manager()

        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        SELECT ra.assignment_id, ra.user_id, r.role_name, ra.object_id, ra.reporting_to
                        FROM roles_assignment ra
                        JOIN roles r ON ra.role_id = r.role_id
                        WHERE ra.is_active = TRUE
                        ORDER BY ra.user_id, ra.assigned_at DESC
                    """)

                    assignments = []
                    for row in cursor.fetchall():
                        assignments.append({
                            "assignment_id": str(row[0]),
                            "user_id": str(row[1]),
                            "role_name": row[2],
                            "object_id": row[3],
                            "reporting_to": str(row[4]) if row[4] else None
                        })

                    return assignments

                finally:
                    cursor.close()

        except Exception as e:
            logger.error(f"Failed to get assignments for admin: {str(e)}")
            raise DatabaseException(f"Failed to get assignments: {str(e)}")