"""
SAP IBP Data Ingestion API routes.
Endpoints for reading data from SAP IBP and ingesting into master_data and sales_data tables.
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import logging

from app.core.sap_ibp_ingestion_service import SapIbpIngestionService, ingest_sap_ibp_data
from app.core.responses import ResponseHandler
from app.core.exceptions import AppException
from app.api.dependencies import get_current_tenant
from app.core.sap_ibp_client import DynamicODataClient, ConnectionConfig, ReadConfig, parse_ibp_date

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sap-ibp", tags=["SAP IBP"])


class ConnectionConfigRequest(BaseModel):
    """SAP IBP connection configuration."""
    base_url: str = Field(..., description="SAP IBP base URL")
    username: str = Field(..., description="SAP IBP username")
    password: str = Field(..., description="SAP IBP password")
    sap_client: Optional[str] = Field(None, description="SAP client")
    timeout: float = Field(60.0, description="Request timeout in seconds")


class FilterConditionRequest(BaseModel):
    """Filter condition for SAP IBP query."""
    field: str = Field(..., description="Field name to filter on")
    operator: str = Field(..., description="Filter operator (eq, ne, gt, ge, lt, le, in)")
    value: Any = Field(..., description="Filter value")


class FilterGroupRequest(BaseModel):
    """Filter group for SAP IBP query."""
    and_conditions: Optional[List[FilterConditionRequest]] = Field(None, alias="and")
    or_conditions: Optional[List[FilterConditionRequest]] = Field(None, alias="or")

    class Config:
        populate_by_name = True


class ReadConfigRequest(BaseModel):
    """SAP IBP read configuration."""
    service_path: str = Field(..., description="SAP IBP service path")
    entity_set: str = Field(..., description="Entity set name")
    select: List[str] = Field(..., description="Fields to select")
    filter: Optional[Dict[str, Any]] = Field(None, description="Filter conditions")
    orderby: Optional[List[str]] = Field(None, description="Order by fields")
    top: Optional[int] = Field(None, description="Maximum number of records")
    skip: Optional[int] = Field(None, description="Number of records to skip")
    format: str = Field("json", description="Response format")


class FieldMappingRequest(BaseModel):
    """Field mapping configuration."""
    sap_field: str = Field(..., description="SAP IBP field name")
    db_field: str = Field(..., description="Database table field name")


class IngestionRequest(BaseModel):
    """SAP IBP data ingestion request."""
    connection: ConnectionConfigRequest
    read: ReadConfigRequest
    master_data_mappings: Optional[List[FieldMappingRequest]] = Field(None, description="Field mappings for master_data table (optional, defaults will be used)")
    sales_data_mappings: Optional[List[FieldMappingRequest]] = Field(None, description="Field mappings for sales_data table (optional, defaults will be used)")


class ReadDataRequest(BaseModel):
    """SAP IBP data read request."""
    connection: ConnectionConfigRequest
    read: ReadConfigRequest


@router.post("/read", response_model=Dict[str, Any])
async def read_sap_ibp_data(
    request: ReadDataRequest,
    tenant_data: Dict = Depends(get_current_tenant)
):
    """
    Read data from SAP IBP using dynamic configuration.

    - **connection**: SAP IBP connection configuration
    - **read**: Read configuration with entity set, select fields, filters, etc.
    """
    try:
        from app.core.sap_ibp_client import DynamicODataClient, ConnectionConfig, ReadConfig

        # Convert request models to dicts for compatibility
        connection_config = request.connection.model_dump()
        read_config = request.read.model_dump()

        # Convert dict configs to Pydantic models
        conn_config = ConnectionConfig(**connection_config)
        read_cfg = ReadConfig(**read_config)

        # Read data from SAP IBP
        async with DynamicODataClient(conn_config) as client:
            logger.info("Reading data from SAP IBP", extra={
                "operation": "sap_ibp_read",
                "entity_set": read_cfg.entity_set,
                "tenant_id": tenant_data["tenant_id"]
            })

            raw_data = await client.read_data(read_cfg)

        result = {
            "count": len(raw_data),
            "data": raw_data,
            "entity_set": read_cfg.entity_set,
            "selected_fields": read_cfg.select
        }

        logger.info(f"Successfully read {len(raw_data)} records from SAP IBP", extra={
            "operation": "sap_ibp_read",
            "record_count": len(raw_data),
            "entity_set": read_cfg.entity_set
        })

        return ResponseHandler.success(data=result)

    except Exception as e:
        logger.error(f"Failed to read SAP IBP data: {str(e)}", extra={
            "operation": "sap_ibp_read",
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=f"Failed to read SAP IBP data: {str(e)}")


@router.post("/ingest", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def ingest_sap_ibp_data_endpoint(
    request: IngestionRequest,
    background_tasks: BackgroundTasks,
    tenant_data: Dict = Depends(get_current_tenant)
):
    """
    Ingest data from SAP IBP into master_data and sales_data tables with field mapping.

    - **connection**: SAP IBP connection configuration
    - **read**: Read configuration with entity set, select fields, filters, etc.
    - **master_data_mappings**: Field mappings for master_data table
    - **sales_data_mappings**: Field mappings for sales_data table
    """
    try:
        # Convert request models to dicts
        connection_config = request.connection.model_dump()
        read_config = request.read.model_dump()

        # Ensure UOMTOID is in the select list (needed for UOM field in sales_data)
        if "UOMTOID" not in read_config.get("select", []):
            read_config["select"] = list(read_config.get("select", [])) + ["UOMTOID"]

        # Create field mapping dictionaries (use None if not provided)
        master_mappings = {mapping.sap_field: mapping.db_field for mapping in request.master_data_mappings} if request.master_data_mappings else None
        sales_mappings = {mapping.sap_field: mapping.db_field for mapping in request.sales_data_mappings} if request.sales_data_mappings else None

        logger.info("Starting SAP IBP data ingestion", extra={
            "operation": "sap_ibp_ingest",
            "entity_set": read_config["entity_set"],
            "tenant_id": tenant_data["tenant_id"],
            "database_name": tenant_data["database_name"],
            "select_fields": read_config["select"]
        })

        # Run ingestion in background for large datasets
        background_tasks.add_task(
            _perform_ingestion,
            connection_config,
            read_config,
            master_mappings,
            sales_mappings,
            tenant_data["database_name"],
            tenant_data["tenant_id"]
        )

        return ResponseHandler.success(
            data={
                "message": "SAP IBP data ingestion started in background",
                "status": "processing",
                "entity_set": read_config["entity_set"]
            },
            status_code=201
        )

    except Exception as e:
        logger.error(f"Failed to start SAP IBP data ingestion: {str(e)}", extra={
            "operation": "sap_ibp_ingest",
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=f"Failed to start ingestion: {str(e)}")


async def _perform_ingestion(
    connection_config: Dict[str, Any],
    read_config: Dict[str, Any],
    master_mappings: Dict[str, str],
    sales_mappings: Dict[str, str],
    database_name: str,
    tenant_id: str
):
    """
    Perform the actual data ingestion in background.

    Args:
        connection_config: SAP IBP connection config
        read_config: Read configuration
        master_mappings: Field mappings for master_data
        sales_mappings: Field mappings for sales_data
        database_name: Target database name
        tenant_id: Tenant identifier
    """
    try:
        

        # Convert dict configs to Pydantic models
        conn_config = ConnectionConfig(**connection_config)
        read_cfg = ReadConfig(**read_config)

        # Read data from SAP IBP
        async with DynamicODataClient(conn_config) as client:
            raw_data = await client.read_data(read_cfg)

        if not raw_data:
            logger.warning("No data returned from SAP IBP for ingestion")
            return

        logger.info(f"Read {len(raw_data)} records from SAP IBP for ingestion")

        # Process and insert data with field mappings
        # Insert master_data and get master_id mappings
        master_inserted, master_id_map = await _insert_master_data_with_mapping(
            raw_data, master_mappings, database_name, tenant_id
        )
        # Insert sales_data with master_id references
        sales_inserted = await _insert_sales_data_with_mapping(
            raw_data, sales_mappings, database_name, tenant_id, master_mappings, master_id_map
        )

        logger.info("SAP IBP data ingestion completed successfully", extra={
            "operation": "sap_ibp_ingest",
            "records_read": len(raw_data),
            "master_data_inserted": master_inserted,
            "sales_data_inserted": sales_inserted,
            "tenant_id": tenant_id
        })

    except Exception as e:
        logger.error(f"SAP IBP data ingestion failed: {str(e)}", extra={
            "operation": "sap_ibp_ingest",
            "error": str(e),
            "tenant_id": tenant_id
        })


async def _insert_master_data_with_mapping(
    raw_data: List[Dict[str, Any]],
    field_mappings: Dict[str, str],
    database_name: str,
    tenant_id: str
) -> tuple[int, Dict[tuple, str]]:
    """
    Insert data into master_data table using field mappings.
    Returns count of inserted records and a mapping of (sap_field_values) -> master_id.

    Args:
        raw_data: Raw data from SAP IBP
        field_mappings: Mapping from SAP fields to DB fields
        database_name: Target database name
        tenant_id: Tenant identifier

    Returns:
        Tuple of (number of records inserted, dict mapping dimension values to master_id)
    """
    if not raw_data or not field_mappings:
        return 0, {}

    from app.core.database import get_db_manager
    from datetime import datetime
    import uuid

    db_manager = get_db_manager()
    inserted_count = 0
    master_id_map = {}  # Maps tuple of dimension values to master_id

    # Build dynamic insert query based on mappings
    db_fields = list(field_mappings.values())
    sap_fields = list(field_mappings.keys())

    # Add required fields for master_data
    db_fields.extend(['created_at', 'created_by'])
    placeholders = ', '.join(['%s'] * len(db_fields))

    insert_query = f"""
        INSERT INTO master_data ({', '.join(db_fields)})
        VALUES ({placeholders})
    """

    with db_manager.get_tenant_connection(database_name) as conn:
        cursor = conn.cursor()

        try:
            now = datetime.now()
            seen_dimensions = set()  # Track unique dimension combinations to avoid duplicates

            for record in raw_data:
                try:
                    # Map SAP fields to DB fields
                    values = []
                    dimension_tuple = []
                    
                    for sap_field in sap_fields:
                        value = record.get(sap_field)

                        # Parse dates if needed
                        if sap_field.lower().endswith('_tstamp') or 'date' in sap_field.lower():
                            if value:
                                parsed_date = parse_ibp_date(value)
                                if parsed_date:
                                    value = parsed_date.isoformat()

                        values.append(value)
                        dimension_tuple.append(value)

                    # Create unique key for this dimension combination
                    dimension_key = tuple(dimension_tuple)

                    # Only insert if we haven't seen this dimension combination before
                    if dimension_key not in seen_dimensions:
                        # Add audit fields
                        values.extend([now, tenant_id or 'system'])

                        cursor.execute(insert_query, values)
                        inserted_count += 1
                        
                        # Store the mapping - will retrieve after commit
                        seen_dimensions.add(dimension_key)

                except Exception as e:
                    logger.warning(f"Failed to insert master data record: {str(e)}", extra={
                        "record": record,
                        "error": str(e),
                        "tenant_id": tenant_id
                    })
                    continue  # Skip this record and continue with next

            conn.commit()
            logger.info(f"Inserted {inserted_count} records into master_data")

            # Now fetch the master_id values for all inserted dimensions
            cursor.execute(f"""
                SELECT {', '.join(db_fields[:-2])}, master_id 
                FROM master_data 
                ORDER BY created_at DESC 
                LIMIT %s
            """, (inserted_count,))
            
            for row in cursor.fetchall():
                # Map dimension values to master_id
                dimension_values = row[:-1]  # All but last column
                master_id = row[-1]  # Last column is master_id
                master_id_map[tuple(dimension_values)] = master_id

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to commit master data batch: {str(e)}")
        finally:
            cursor.close()

    return inserted_count, master_id_map


async def _insert_sales_data_with_mapping(
    raw_data: List[Dict[str, Any]],
    field_mappings: Dict[str, str],
    database_name: str,
    tenant_id: str,
    master_mappings: Dict[str, str],
    master_id_map: Dict[tuple, str]
) -> int:
    """
    Insert data into sales_data table using field mappings and master_id references.

    Args:
        raw_data: Raw data from SAP IBP
        field_mappings: Mapping from SAP fields to DB fields (for sales_data)
        database_name: Target database name
        tenant_id: Tenant identifier
        master_mappings: Mapping from SAP fields to DB fields (for master_data - to identify dimensions)
        master_id_map: Mapping of dimension values to master_id

    Returns:
        Number of records inserted
    """
    if not raw_data or not field_mappings or not master_id_map or not master_mappings:
        return 0

    from app.core.database import get_db_manager
    from datetime import datetime

    db_manager = get_db_manager()
    inserted_count = 0

    # Build dynamic insert query based on mappings
    db_fields = list(field_mappings.values())
    sap_fields = list(field_mappings.keys())
    
    # Get master dimension SAP fields for lookup
    master_sap_fields = list(master_mappings.keys())

    # Add required fields for sales_data (master_id, date, quantity, uom, created_at, created_by)
    # Insert fields include: master_id, mapped fields, uom, created_at, created_by
    insert_fields = ['master_id'] + db_fields + ['uom', 'created_at', 'created_by']
    placeholders = ', '.join(['%s'] * len(insert_fields))

    insert_query = f"""
        INSERT INTO sales_data ({', '.join(insert_fields)})
        VALUES ({placeholders})
    """

    with db_manager.get_tenant_connection(database_name) as conn:
        cursor = conn.cursor()

        try:
            now = datetime.now()

            for record in raw_data:
                try:
                    # Extract dimension values using master_mappings to lookup master_id
                    dimension_values = []
                    for master_sap_field in master_sap_fields:
                        value = record.get(master_sap_field)
                        dimension_values.append(value)
                    
                    dimension_key = tuple(dimension_values)
                    
                    # Look up the master_id for this dimension combination
                    master_id = master_id_map.get(dimension_key)
                    
                    if not master_id:
                        logger.warning(f"No master_id found for dimensions: {dimension_key}", extra={
                            "record": record,
                            "master_sap_fields": master_sap_fields,
                            "tenant_id": tenant_id
                        })
                        continue

                    # Create savepoint for this record to handle individual record errors
                    cursor.execute("SAVEPOINT sp_sales_record")
                    
                    try:
                        # Now insert sales data with the master_id
                        values = [master_id]  # Start with master_id
                        
                        for sap_field in sap_fields:
                            value = record.get(sap_field)

                            # Parse dates if needed
                            if sap_field.lower().endswith('_tstamp') or 'date' in sap_field.lower():
                                if value:
                                    parsed_date = parse_ibp_date(value)
                                    if parsed_date:
                                        value = parsed_date.isoformat()

                            # Convert quantities to float if needed
                            if sap_field.lower().endswith('qty') or 'quantity' in sap_field.lower():
                                if value is not None:
                                    try:
                                        value = float(value)
                                    except (ValueError, TypeError):
                                        value = 0.0

                            values.append(value)

                        # Add UOM (Unit of Measure) - default to 'EA' if not in mappings
                        uom_value = record.get('UOMTOID')
                        if not uom_value:
                            uom_value = 'EA'  # Default UOM if UOMTOID is missing or null
                        values.append(uom_value)
                        
                        # Add audit fields
                        values.extend([now, tenant_id or 'system'])

                        cursor.execute(insert_query, values)
                        cursor.execute("RELEASE SAVEPOINT sp_sales_record")
                        inserted_count += 1
                        
                    except Exception as inner_e:
                        # Rollback savepoint on individual record error
                        cursor.execute("ROLLBACK TO SAVEPOINT sp_sales_record")
                        logger.warning(f"Failed to insert sales data record: {str(inner_e)}", extra={
                            "record": record,
                            "error": str(inner_e),
                            "tenant_id": tenant_id
                        })
                        continue

                except Exception as e:
                    logger.warning(f"Failed to process sales data record: {str(e)}", extra={
                        "record": record,
                        "error": str(e),
                        "tenant_id": tenant_id
                    })
                    continue  # Skip this record and continue with next

            conn.commit()
            logger.info(f"Inserted {inserted_count} records into sales_data")

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to commit sales data batch: {str(e)}")
        finally:
            cursor.close()

    return inserted_count


@router.get("/ingestion-history", response_model=Dict[str, Any])
async def get_ingestion_history(
    tenant_data: Dict = Depends(get_current_tenant),
    page: int = 1,
    page_size: int = 50
):
    """
    Get SAP IBP ingestion history for the tenant.

    - **page**: Page number (default: 1)
    - **page_size**: Records per page (default: 50, max: 100)
    """
    try:
        if page_size > 100:
            page_size = 100

        from app.core.database import get_db_manager

        db_manager = get_db_manager()
        offset = (page - 1) * page_size

        with db_manager.get_tenant_connection(tenant_data["database_name"]) as conn:
            cursor = conn.cursor()
            try:
                # Get total count (assuming we have an ingestion_history table)
                cursor.execute(
                    "SELECT COUNT(*) FROM sap_ibp_ingestion_history"
                )
                total_count = cursor.fetchone()[0]

                # Get paginated results
                cursor.execute("""
                    SELECT ingestion_id, entity_set, records_read, master_data_inserted,
                           sales_data_inserted, status, started_at, completed_at
                    FROM sap_ibp_ingestion_history
                    ORDER BY started_at DESC
                    LIMIT %s OFFSET %s
                """, (page_size, offset))

                ingestions = []
                for row in cursor.fetchall():
                    ingestion_id, entity_set, records_read, master_inserted, sales_inserted, status, started_at, completed_at = row
                    ingestions.append({
                        "ingestion_id": ingestion_id,
                        "entity_set": entity_set,
                        "records_read": records_read,
                        "master_data_inserted": master_inserted,
                        "sales_data_inserted": sales_inserted,
                        "status": status,
                        "started_at": started_at.isoformat() if started_at else None,
                        "completed_at": completed_at.isoformat() if completed_at else None
                    })

                return ResponseHandler.list_response(
                    data=ingestions,
                    page=page,
                    page_size=page_size,
                    total_count=total_count
                )

            except Exception as e:
                # If table doesn't exist yet, return empty result
                logger.debug(f"Ingestion history table may not exist: {str(e)}")
                return ResponseHandler.list_response(
                    data=[],
                    page=page,
                    page_size=page_size,
                    total_count=0
                )
            finally:
                cursor.close()

    except Exception as e:
        logger.error(f"Unexpected error in get_ingestion_history: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
