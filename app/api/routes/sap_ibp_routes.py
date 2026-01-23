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
from app.api.dependencies import get_current_tenant, require_object_access
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
    tenant_data: Dict = Depends(get_current_tenant),
    _: Dict = Depends(require_object_access("Allow Edit", min_role_id=2))
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
    tenant_data: Dict = Depends(get_current_tenant),
    _: Dict = Depends(require_object_access("Allow Edit", min_role_id=2))
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


async def _find_existing_master_data(
    cursor,
    field_mappings: Dict[str, str],
    record: Dict[str, Any]
) -> Optional[str]:
    """
    Find existing master_data record by the mapped fields.

    Args:
        cursor: Database cursor
        field_mappings: Mapping from SAP fields to DB fields
        record: SAP record data

    Returns:
        master_id if found, None otherwise
    """
    try:
        # Build WHERE clause for finding existing record
        where_conditions = []
        params = []

        for sap_field, db_field in field_mappings.items():
            value = record.get(sap_field)

            # Parse dates if needed
            if sap_field.lower().endswith('_tstamp') or 'date' in sap_field.lower():
                if value:
                    from app.core.sap_ibp_client import parse_ibp_date
                    parsed_date = parse_ibp_date(value)
                    if parsed_date:
                        value = parsed_date.isoformat()

            if value is None:
                where_conditions.append(f'"{db_field}" IS NULL')
            else:
                where_conditions.append(f'"{db_field}" = %s')
                params.append(value)

        if not where_conditions:
            return None

        where_clause = ' AND '.join(where_conditions)
        query = f'SELECT master_id FROM master_data WHERE {where_clause}'

        cursor.execute(query, params)
        result = cursor.fetchone()

        return result[0] if result else None

    except Exception as e:
        logger.warning(f"Failed to find existing master data: {str(e)}")
        return None


async def _insert_master_data_with_mapping(
    raw_data: List[Dict[str, Any]],
    field_mappings: Dict[str, str],
    database_name: str,
    tenant_id: str
) -> tuple[int, Dict[tuple, str]]:
    """
    Insert data into master_data table using field mappings with UPSERT logic.
    Returns count of inserted/updated records and a mapping of (sap_field_values) -> master_id.

    Args:
        raw_data: Raw data from SAP IBP
        field_mappings: Mapping from SAP fields to DB fields
        database_name: Target database name
        tenant_id: Tenant identifier

    Returns:
        Tuple of (number of records inserted/updated, dict mapping dimension values to master_id)
    """
    if not raw_data or not field_mappings:
        return 0, {}

    from app.core.database import get_db_manager
    from datetime import datetime
    import uuid

    db_manager = get_db_manager()
    upserted_count = 0
    master_id_map = {}  # Maps tuple of dimension values to master_id

    # Build dynamic insert query based on mappings
    db_fields = list(field_mappings.values())
    sap_fields = list(field_mappings.keys())

    # Add required fields for master_data
    db_fields.extend(['created_at', 'created_by', 'updated_at', 'updated_by'])
    placeholders = ', '.join(['%s'] * len(db_fields))

    insert_query = f"""
        INSERT INTO master_data ({', '.join(db_fields)})
        VALUES ({placeholders})
    """

    with db_manager.get_tenant_connection(database_name) as conn:
        cursor = conn.cursor()

        try:
            now = datetime.now()

            for record in raw_data:
                try:
                    # Check if record already exists
                    existing_master_id = await _find_existing_master_data(cursor, field_mappings, record)

                    if existing_master_id:
                        # Update existing record
                        update_fields = []
                        update_values = []

                        for sap_field, db_field in field_mappings.items():
                            value = record.get(sap_field)

                            # Parse dates if needed
                            if sap_field.lower().endswith('_tstamp') or 'date' in sap_field.lower():
                                if value:
                                    parsed_date = parse_ibp_date(value)
                                    if parsed_date:
                                        value = parsed_date.isoformat()

                            update_fields.append(f'"{db_field}" = %s')
                            update_values.append(value)

                        # Add audit fields for update
                        update_fields.extend(['updated_at = %s', 'updated_by = %s'])
                        update_values.extend([now, tenant_id or 'system'])

                        # Build WHERE clause for update
                        where_conditions = []
                        where_values = []

                        for sap_field, db_field in field_mappings.items():
                            value = record.get(sap_field)

                            # Parse dates if needed
                            if sap_field.lower().endswith('_tstamp') or 'date' in sap_field.lower():
                                if value:
                                    parsed_date = parse_ibp_date(value)
                                    if parsed_date:
                                        value = parsed_date.isoformat()

                            if value is None:
                                where_conditions.append(f'"{db_field}" IS NULL')
                            else:
                                where_conditions.append(f'"{db_field}" = %s')
                                where_values.append(value)

                        where_clause = ' AND '.join(where_conditions)
                        update_query = f'UPDATE master_data SET {", ".join(update_fields)} WHERE {where_clause}'

                        cursor.execute(update_query, update_values + where_values)

                        # Use existing master_id for mapping
                        master_id = existing_master_id
                    else:
                        # Insert new record
                        values = []

                        for sap_field in sap_fields:
                            value = record.get(sap_field)

                            # Parse dates if needed
                            if sap_field.lower().endswith('_tstamp') or 'date' in sap_field.lower():
                                if value:
                                    parsed_date = parse_ibp_date(value)
                                    if parsed_date:
                                        value = parsed_date.isoformat()

                            values.append(value)

                        # Add audit fields
                        values.extend([now, tenant_id or 'system', now, tenant_id or 'system'])

                        cursor.execute(insert_query, values)

                        # Get the master_id of the newly inserted record
                        master_id = await _find_existing_master_data(cursor, field_mappings, record)

                    upserted_count += 1

                    # Build dimension tuple for mapping
                    dimension_tuple = []
                    for sap_field in sap_fields:
                        value = record.get(sap_field)
                        if sap_field.lower().endswith('_tstamp') or 'date' in sap_field.lower():
                            if value:
                                parsed_date = parse_ibp_date(value)
                                if parsed_date:
                                    value = parsed_date.isoformat()
                        dimension_tuple.append(value)

                    master_id_map[tuple(dimension_tuple)] = master_id

                except Exception as e:
                    logger.warning(f"Failed to upsert master data record: {str(e)}", extra={
                        "record": record,
                        "error": str(e),
                        "tenant_id": tenant_id
                    })
                    continue  # Skip this record and continue with next

            conn.commit()
            logger.info(f"Upserted {upserted_count} records into master_data")

            # Now fetch all master_id values for the processed dimensions
            # Build WHERE clause to get master_ids for all processed records
            where_conditions = []
            all_dimension_values = []

            for record in raw_data:
                dimension_values = []
                for sap_field in sap_fields:
                    value = record.get(sap_field)
                    if sap_field.lower().endswith('_tstamp') or 'date' in sap_field.lower():
                        if value:
                            parsed_date = parse_ibp_date(value)
                            if parsed_date:
                                value = parsed_date.isoformat()
                    dimension_values.append(value)

                # Build WHERE condition for this record
                record_conditions = []
                record_params = []
                for i, field in enumerate(field_mappings.values()):
                    if dimension_values[i] is None:
                        record_conditions.append(f'"{field}" IS NULL')
                    else:
                        record_conditions.append(f'"{field}" = %s')
                        record_params.append(dimension_values[i])

                where_conditions.append(f"({' AND '.join(record_conditions)})")
                all_dimension_values.extend(record_params)

            if where_conditions:
                where_clause = ' OR '.join(where_conditions)
                select_fields = ', '.join(f'"{field}"' for field in field_mappings.values())
                query = f'SELECT {select_fields}, master_id FROM master_data WHERE {where_clause}'

                cursor.execute(query, all_dimension_values)

                for row in cursor.fetchall():
                    dimension_values = row[:-1]  # All but last column
                    master_id = row[-1]  # Last column is master_id
                    master_id_map[tuple(dimension_values)] = master_id

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to commit master data batch: {str(e)}")
        finally:
            cursor.close()

    return upserted_count, master_id_map


async def _insert_sales_data_with_mapping(
    raw_data: List[Dict[str, Any]],
    field_mappings: Dict[str, str],
    database_name: str,
    tenant_id: str,
    master_mappings: Dict[str, str],
    master_id_map: Dict[tuple, str]
) -> int:
    """
    Insert data into sales_data table using field mappings and master_id references with batch upsert.

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

    # Identify date and quantity field names for UPSERT conflict resolution
    date_field = None
    quantity_field = None
    for sap_field, db_field in field_mappings.items():
        if sap_field.lower().endswith('_tstamp') or 'date' in sap_field.lower():
            date_field = db_field
        elif sap_field.lower().endswith('qty') or 'quantity' in sap_field.lower():
            quantity_field = db_field

    # Build UPSERT query to prevent duplicates
    conflict_target = f"(master_id, {date_field})" if date_field else "(master_id)"
    update_fields = []
    if quantity_field:
        update_fields.append(f'"{quantity_field}" = EXCLUDED."{quantity_field}"')
    update_fields.extend([
        'uom = EXCLUDED.uom',
        'unit_price = EXCLUDED.unit_price',
        'created_at = EXCLUDED.created_at',
        'created_by = EXCLUDED.created_by'
    ])

    with db_manager.get_tenant_connection(database_name) as conn:
        cursor = conn.cursor()

        try:
            now = datetime.now()

            # Process in batches for better performance
            batch_size = 1000
            total_records = len(raw_data)

            for batch_start in range(0, total_records, batch_size):
                batch_end = min(batch_start + batch_size, total_records)
                batch_data = raw_data[batch_start:batch_end]

                logger.info(f"Processing SAP IBP sales data batch: {batch_start}-{batch_end-1} of {total_records}")

                # Prepare batch data
                values_list = []

                for record in batch_data:
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

                        # Now prepare sales data values
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

                        values_list.append(values)

                    except Exception as e:
                        logger.warning(f"Failed to prepare sales data record: {str(e)}", extra={
                            "record": record,
                            "error": str(e),
                            "tenant_id": tenant_id
                        })
                        continue

                # Execute batch upsert if we have data
                if values_list:
                    try:
                        # Create batch upsert query
                        placeholders = ', '.join(['%s'] * len(insert_fields))
                        value_placeholders = ', '.join([f"({placeholders})"] * len(values_list))

                        batch_upsert_query = f"""
                            INSERT INTO sales_data ({', '.join(insert_fields)})
                            VALUES {value_placeholders}
                            ON CONFLICT {conflict_target}
                            DO UPDATE SET {', '.join(update_fields)}
                        """

                        # Flatten the values list
                        flattened_values = [item for sublist in values_list for item in sublist]

                        cursor.execute(batch_upsert_query, flattened_values)
                        inserted_count += len(values_list)

                        logger.debug(f"Batch upsert successful: {len(values_list)} records")

                    except Exception as batch_e:
                        logger.warning(f"Batch upsert failed, falling back to individual inserts: {str(batch_e)}")

                        # Fall back to individual upserts
                        for values in values_list:
                            try:
                                cursor.execute("SAVEPOINT sp_sales_record")

                                single_query = f"""
                                    INSERT INTO sales_data ({', '.join(insert_fields)})
                                    VALUES ({placeholders})
                                    ON CONFLICT {conflict_target}
                                    DO UPDATE SET {', '.join(update_fields)}
                                """

                                cursor.execute(single_query, values)
                                cursor.execute("RELEASE SAVEPOINT sp_sales_record")
                                inserted_count += 1

                            except Exception as single_e:
                                cursor.execute("ROLLBACK TO SAVEPOINT sp_sales_record")
                                logger.warning(f"Failed to insert individual sales data record: {str(single_e)}", extra={
                                    "error": str(single_e),
                                    "tenant_id": tenant_id
                                })
                                continue

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
    page_size: int = 50,
    _: Dict = Depends(require_object_access("Forecast", min_role_id=1))
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
