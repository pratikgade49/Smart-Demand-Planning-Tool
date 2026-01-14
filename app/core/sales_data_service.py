"""
Sales Data Service.
Handles retrieval of sales data with flexible filtering, pagination, and summary statistics.
Supports dynamic date and target columns based on field catalogue.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import date
from app.core.database import get_db_manager
from app.core.exceptions import ValidationException, NotFoundException, DatabaseException
from app.schemas.sales_data import (
    SalesDataQueryRequest,
    SalesDataResponse,
    SalesDataRecord,
    SalesDataSummary,
    SalesDataFilter
)

logger = logging.getLogger(__name__)


class SalesDataService:
    """Service for sales data retrieval operations."""

    @staticmethod
    def _get_field_names(cursor) -> Tuple[str, str]:
        """
        Retrieve the dynamic date and target field names from field_catalogue_metadata.
        
        Args:
            cursor: Database cursor
            
        Returns:
            Tuple of (date_field_name, target_field_name)
        """
        try:
            cursor.execute("""
                SELECT date_field_name, target_field_name 
                FROM field_catalogue_metadata 
                LIMIT 1
            """)
            result = cursor.fetchone()
            if not result:
                raise NotFoundException("Field catalogue metadata not found. Please create a field catalogue first.")
            return result[0], result[1]
        except Exception as e:
            logger.error(f"Error retrieving field names from metadata: {str(e)}")
            raise DatabaseException(f"Failed to retrieve field catalogue metadata: {str(e)}")

    @staticmethod
    def get_sales_data(
        tenant_id: str,
        database_name: str,
        request: SalesDataQueryRequest
    ) -> SalesDataResponse:
        """
        Retrieve sales data records with flexible filtering and pagination.

        Args:
            tenant_id: The tenant identifier
            database_name: The tenant's database name
            request: SalesDataQueryRequest containing filters, date range, and pagination

        Returns:
            SalesDataResponse with paginated records and metadata
        """
        db_manager = get_db_manager()

        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()

                # Get dynamic field names
                date_field, target_field = SalesDataService._get_field_names(cursor)

                # Build the base query
                base_query = f"""
                    SELECT 
                        sd.sales_id,
                        sd.master_id,
                        sd."{date_field}",
                        sd."{target_field}",
                        sd.uom,
                        sd.unit_price,
                        sd.created_at,
                        sd.created_by
                    FROM sales_data sd
                    WHERE 1=1
                """

                params: List[Any] = []

                # Add filter conditions
                if request.filters:
                    for filter_obj in request.filters:
                        if filter_obj.values:
                            placeholders = ",".join(["%s"] * len(filter_obj.values))
                            base_query += f"""
                                AND sd.master_id IN (
                                    SELECT master_id FROM master_data
                                    WHERE "{filter_obj.field_name}" = ANY(ARRAY[{placeholders}])
                                )
                            """
                            params.extend(filter_obj.values)

                # Add date range filters
                if request.from_date:
                    base_query += f' AND sd."{date_field}" >= %s'
                    params.append(request.from_date)

                if request.to_date:
                    base_query += f' AND sd."{date_field}" <= %s'
                    params.append(request.to_date)

                # Add sorting
                if request.sort_by:
                    sort_field = request.sort_by
                    # Validate sort field to prevent SQL injection
                    allowed_fields = [
                        date_field, target_field, "uom", "unit_price",
                        "created_at", "created_by"
                    ]
                    if sort_field not in allowed_fields:
                        sort_field = date_field

                    sort_order = request.sort_order.upper()
                    base_query += f' ORDER BY sd."{sort_field}" {sort_order}'
                else:
                    base_query += f' ORDER BY sd."{date_field}" {request.sort_order.upper()}'

                # Get total count
                count_query = f"SELECT COUNT(*) FROM ({base_query}) as count_table"
                cursor.execute(count_query, params)
                total_count = cursor.fetchone()[0]

                # Calculate pagination
                offset = (request.page - 1) * request.page_size
                total_pages = (total_count + request.page_size - 1) // request.page_size

                # Add pagination
                paginated_query = base_query + f" LIMIT %s OFFSET %s"
                paginated_params = params + [request.page_size, offset]

                # Execute query
                cursor.execute(paginated_query, paginated_params)
                rows = cursor.fetchall()

                # Build records with master data
                records = []
                for row in rows:
                    sales_id, master_id, sales_date, target_value, uom, unit_price, created_at, created_by = row

                    # Get master data for this record
                    master_data = SalesDataService._get_master_data_for_id(
                        cursor, master_id
                    )

                    record = SalesDataRecord(
                        sales_id=sales_id,
                        master_id=master_id,
                        date=sales_date,
                        quantity=target_value,
                        unit_price=unit_price,
                        total_amount=None,  # Calculated from quantity * unit_price if needed
                        created_at=str(created_at),
                        updated_at=str(created_at),  # Using created_at since there's no updated_at in sales_data
                        master_data=master_data
                    )
                    records.append(record)

                return SalesDataResponse(
                    records=records,
                    total_count=total_count,
                    page=request.page,
                    page_size=request.page_size,
                    total_pages=total_pages,
                    has_next=request.page < total_pages,
                    has_previous=request.page > 1
                )

        except (NotFoundException, DatabaseException) as e:
            logger.error(f"Error in get_sales_data: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in get_sales_data: {str(e)}")
            raise DatabaseException(f"Failed to retrieve sales data: {str(e)}")

    @staticmethod
    def get_sales_data_summary(
        tenant_id: str,
        database_name: str,
        filters: Optional[List[Dict[str, Any]]] = None,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None
    ) -> SalesDataSummary:
        """
        Get summary statistics for sales data.

        Args:
            tenant_id: The tenant identifier
            database_name: The tenant's database name
            filters: Optional list of filters with field_name and values
            from_date: Optional start date
            to_date: Optional end date

        Returns:
            SalesDataSummary with aggregated statistics
        """
        db_manager = get_db_manager()

        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()

                # Get dynamic field names
                date_field, target_field = SalesDataService._get_field_names(cursor)

                # Build the query
                query = f"""
                    SELECT 
                        COUNT(*) as total_records,
                        SUM(sd."{target_field}") as total_quantity,
                        NULL as total_amount,
                        AVG(sd."{target_field}") as avg_quantity,
                        AVG(sd.unit_price) as avg_price,
                        MIN(sd."{date_field}") as min_date,
                        MAX(sd."{date_field}") as max_date
                    FROM sales_data sd
                    WHERE 1=1
                """

                params: List[Any] = []

                # Add filter conditions
                if filters:
                    for filter_obj in filters:
                        if filter_obj.get("values"):
                            placeholders = ",".join(["%s"] * len(filter_obj["values"]))
                            query += f"""
                                AND sd.master_id IN (
                                    SELECT master_id FROM master_data
                                    WHERE "{filter_obj['field_name']}" = ANY(ARRAY[{placeholders}])
                                )
                            """
                            params.extend(filter_obj["values"])

                # Add date range filters
                if from_date:
                    query += f' AND sd."{date_field}" >= %s'
                    params.append(from_date)

                if to_date:
                    query += f' AND sd."{date_field}" <= %s'
                    params.append(to_date)

                cursor.execute(query, params)
                result = cursor.fetchone()

                if not result:
                    return SalesDataSummary(
                        total_records=0,
                        total_quantity=0.0,
                        total_amount=0.0,
                        avg_quantity=0.0,
                        avg_price=None,
                        date_range={"min_date": None, "max_date": None},
                        field_summaries={}
                    )

                (total_records, total_quantity, total_amount, avg_quantity,
                 avg_price, min_date, max_date) = result

                # Get field summaries if filters are provided
                field_summaries = {}
                if filters:
                    for filter_obj in filters:
                        field_name = filter_obj["field_name"]
                        field_summary_query = f"""
                            SELECT 
                                md."{field_name}",
                                COUNT(*) as count,
                                SUM(sd."{target_field}") as total_qty
                            FROM sales_data sd
                            JOIN master_data md ON sd.master_id = md.master_id
                            WHERE 1=1
                        """
                        field_params = []

                        # Add date range to field summary if provided
                        if from_date:
                            field_summary_query += f' AND sd."{date_field}" >= %s'
                            field_params.append(from_date)

                        if to_date:
                            field_summary_query += f' AND sd."{date_field}" <= %s'
                            field_params.append(to_date)

                        field_summary_query += f' GROUP BY md."{field_name}"'

                        try:
                            cursor.execute(field_summary_query, field_params)
                            field_results = cursor.fetchall()
                            field_summaries[field_name] = {
                                "values": [
                                    {
                                        "value": row[0],
                                        "count": row[1],
                                        "total_quantity": row[2]
                                    }
                                    for row in field_results
                                ]
                            }
                        except Exception as e:
                            logger.warning(f"Could not get summary for field {field_name}: {str(e)}")
                            field_summaries[field_name] = {"error": str(e)}

                return SalesDataSummary(
                    total_records=total_records or 0,
                    total_quantity=float(total_quantity) if total_quantity else 0.0,
                    total_amount=float(total_amount) if total_amount else None,
                    avg_quantity=float(avg_quantity) if avg_quantity else 0.0,
                    avg_price=float(avg_price) if avg_price else None,
                    date_range={
                        "min_date": min_date,
                        "max_date": max_date
                    },
                    field_summaries=field_summaries
                )

        except (NotFoundException, DatabaseException) as e:
            logger.error(f"Error in get_sales_data_summary: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in get_sales_data_summary: {str(e)}")
            raise DatabaseException(f"Failed to retrieve sales data summary: {str(e)}")

    @staticmethod
    def _get_master_data_for_id(cursor, master_id: str) -> Dict[str, Any]:
        """
        Helper method to retrieve master data fields for a given master_id.

        Args:
            cursor: Database cursor
            master_id: The master data ID

        Returns:
            Dictionary of master data fields and values
        """
        try:
            cursor.execute(
                """
                SELECT * FROM master_data WHERE master_id = %s
                """,
                (master_id,)
            )
            row = cursor.fetchone()

            if not row:
                return {}

            # Get column names
            col_names = [desc[0] for desc in cursor.description]

            # Build dictionary, excluding system fields
            exclude_fields = {"master_id", "tenant_id", "created_at", "created_by", "updated_at", "updated_by","uom","deleted_at"}
            master_data = {
                col_names[i]: row[i]
                for i in range(len(col_names))
                if col_names[i] not in exclude_fields
            }

            return master_data

        except Exception as e:
            logger.warning(f"Could not retrieve master data for master_id {master_id}: {str(e)}")
            return {}

    @staticmethod
    def get_sales_records_ui(
        database_name: str,
        request: SalesDataQueryRequest
    ) -> Dict[str, Any]:
        """
        Specialized method for UI to retrieve sales records with pagination based on master_id.
        Supports filtering on master data and date range on sales data.
        """
        db_manager = get_db_manager()
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()

                # Get dynamic field names from field catalogue metadata
                date_field, target_field = SalesDataService._get_field_names(cursor)

                # Use provided dates or defaults
                start_date = request.from_date if request.from_date else '2025-01-01'
                end_date = request.to_date if request.to_date else '2026-03-31'
                
                # Build master_data query with filters
                master_where = "WHERE 1=1"
                master_params = []
                if request.filters:
                    for f in request.filters:
                        if f.values:
                            placeholders = ",".join(["%s"] * len(f.values))
                            master_where += f' AND "{f.field_name}" IN ({placeholders})'
                            master_params.extend(f.values)

                # Get total master records count for pagination
                count_query = f"SELECT COUNT(*) FROM public.master_data {master_where}"
                cursor.execute(count_query, master_params)
                total_master_count = cursor.fetchone()[0]

                # Calculate offset
                offset = (request.page - 1) * request.page_size

                # Get paginated master_ids, ordered by master_data fields (ASC) instead of master_id
                order_cols: List[str] = []
                order_by = "master_id ASC"
                sales_order_by = f'sd.master_id, sd."{date_field}" ASC'
                try:
                    cursor.execute("""
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_name = 'master_data'
                        ORDER BY ordinal_position
                    """)
                    all_cols = [r[0] for r in cursor.fetchall()]
                    exclude_cols = {
                        "master_id",
                        "tenant_id",
                        "created_at",
                        "created_by",
                        "updated_at",
                        "updated_by",
                        "deleted_at",
                        "uom",
                    }
                    order_cols = [c for c in all_cols if c not in exclude_cols]
                    if order_cols:
                        order_by = ", ".join([f'"{c}" ASC' for c in order_cols])
                        sales_order_by = ", ".join(
                            [f'md."{c}" ASC' for c in order_cols] + [f'sd."{date_field}" ASC']
                        )
                except Exception:
                    pass

                master_id_query = f"""
                    SELECT master_id 
                    FROM public.master_data 
                    {master_where}
                    ORDER BY {order_by}
                    LIMIT %s OFFSET %s
                """
                cursor.execute(master_id_query, master_params + [request.page_size, offset])
                master_ids = [row[0] for row in cursor.fetchall()]

                if not master_ids:
                    return {
                        "records": [],
                        "total_count": total_master_count
                    }

                # Get sales data for these master_ids within date range
                placeholders = ",".join(["%s"] * len(master_ids))
                join_master_data = "JOIN public.master_data md ON sd.master_id = md.master_id" if order_cols else ""
                sales_query = f"""
                    SELECT sd.sales_id, sd.master_id, sd."{date_field}", sd."{target_field}", sd.uom 
                    FROM public.sales_data sd
                    {join_master_data}
                    WHERE sd.master_id IN ({placeholders})
                    AND sd."{date_field}" BETWEEN %s AND %s
                    ORDER BY {sales_order_by}
                """
                cursor.execute(sales_query, master_ids + [start_date, end_date])
                rows = cursor.fetchall()

                # Pre-fetch master data for all master_ids in this page
                master_data_map = {}
                if master_ids:
                    placeholders = ",".join(["%s"] * len(master_ids))
                    cursor.execute(f'SELECT * FROM public.master_data WHERE master_id IN ({placeholders})', master_ids)
                    col_names = [desc[0] for desc in cursor.description]
                    exclude_fields = {"master_id", "tenant_id", "created_at", "created_by", "updated_at", "updated_by"}
                    
                    master_id_idx = col_names.index("master_id")
                    for row in cursor.fetchall():
                        m_id = row[master_id_idx]
                        m_data = {
                            col_names[i]: row[i]
                            for i in range(len(col_names))
                            if col_names[i] not in exclude_fields
                        }
                        master_data_map[m_id] = m_data

                results = []
                for row in rows:
                    sales_id, master_id, sales_date, quantity, uom = row
                    results.append({
                        "sales_id": str(sales_id),
                        "master_data": master_data_map.get(master_id, {}),
                        "date": str(sales_date),
                        "UOM": uom,
                        "Quantity": float(quantity) if quantity is not None else 0.0
                    })

                return {
                    "records": results,
                    "total_count": total_master_count
                }

        except Exception as e:
            logger.error(f"Error in get_sales_records_ui: {str(e)}")
            raise DatabaseException(f"Failed to retrieve sales records for UI: {str(e)}")
