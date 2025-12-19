"""
SAP IBP Data Ingestion Service.
Handles reading data from SAP IBP and ingesting into master_data and sales_data tables.
"""

import asyncio
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from app.core.database import get_db_manager
from app.core.logging_config import get_logger
from app.core.exceptions import DatabaseException

from app.core.sap_ibp_client import DynamicODataClient, ConnectionConfig, ReadConfig, parse_ibp_date

logger = get_logger(__name__)


class SapIbpIngestionService:
    """Service for ingesting data from SAP IBP into database tables."""

    def __init__(self):
        self.db_manager = get_db_manager()

    async def ingest_ibp_data(
        self,
        connection_config: Dict[str, Any],
        read_config: Dict[str, Any],
        database_name: str,
        tenant_id: Optional[str] = None,
        master_data_mappings: Optional[Dict[str, str]] = None,
        sales_data_mappings: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Ingest data from SAP IBP into master_data and sales_data tables with proper data separation.

        The service separates combined SAP IBP data into:
        - master_data: Unique combinations of customer, product, location (dimension data)
        - sales_data: Transactional data with quantities, dates, prices (fact data)

        Args:
            connection_config: SAP IBP connection configuration
            read_config: Read configuration for SAP IBP
            database_name: Target database name
            tenant_id: Optional tenant identifier
            master_data_fields: Fields that define master data uniqueness (default: ['CUSTID', 'PRDID', 'LOCID'])
            sales_data_fields: Fields for sales data (default: ['ACTUALSQTY', 'PERIODID3_TSTAMP', 'UNITPRICE'])

        Returns:
            Ingestion results summary
        """
        logger.info("Starting SAP IBP data ingestion with data separation", extra={
            "operation": "sap_ibp_ingestion",
            "database_name": database_name
        })

        # Set default field mappings if not provided
        if master_data_mappings is None:
            master_data_mappings = {
                'CUSTID': 'customer',
                'PRDID': 'product',
                'LOCID': 'location'
            }
        if sales_data_mappings is None:
            sales_data_mappings = {
                'ACTUALSQTY': 'quantity',
                'PERIODID3_TSTAMP': 'date',
                'UNITPRICE': 'unit_price'
            }

        # Validate that required mappings are present for data separation
        required_master_fields = {'customer', 'product', 'location'}
        required_sales_fields = {'quantity', 'date'}

        if not all(field in master_data_mappings.values() for field in required_master_fields):
            raise ValueError(f"Master data mappings must include all required fields: {required_master_fields}")

        if not all(field in sales_data_mappings.values() for field in required_sales_fields):
            raise ValueError(f"Sales data mappings must include all required fields: {required_sales_fields}")

        try:
            # Convert dict configs to Pydantic models
            conn_config = ConnectionConfig(**connection_config)
            read_cfg = ReadConfig(**read_config)

            # Read data from SAP IBP
            async with DynamicODataClient(conn_config) as client:
                logger.info("Reading data from SAP IBP")
                raw_data = await client.read_data(read_cfg)

            if not raw_data:
                logger.warning("No data returned from SAP IBP")
                return {
                    "status": "no_data",
                    "message": "No data returned from SAP IBP",
                    "records_read": 0,
                    "master_data_inserted": 0,
                    "sales_data_inserted": 0
                }

            logger.info(f"Read {len(raw_data)} records from SAP IBP")

            # Process and separate data
            master_inserted, sales_inserted = await self._process_and_separate_data(
                raw_data, database_name, tenant_id, master_data_mappings, sales_data_mappings
            )

            result = {
                "status": "success",
                "records_read": len(raw_data),
                "master_data_inserted": master_inserted,
                "sales_data_inserted": sales_inserted,
                "total_inserted": master_inserted + sales_inserted,
                "data_separation": {
                    "master_data_mappings": master_data_mappings,
                    "sales_data_mappings": sales_data_mappings
                }
            }

            logger.info("SAP IBP data ingestion with separation completed successfully", extra={
                "operation": "sap_ibp_ingestion",
                "records_read": len(raw_data),
                "master_data_inserted": master_inserted,
                "sales_data_inserted": sales_inserted
            })

            return result

        except Exception as e:
            logger.error(f"SAP IBP data ingestion failed: {str(e)}", extra={
                "operation": "sap_ibp_ingestion",
                "error": str(e)
            })
            raise

    async def _insert_master_data(
        self,
        raw_data: List[Dict[str, Any]],
        database_name: str,
        tenant_id: Optional[str] = None
    ) -> int:
        """
        Insert data into master_data table.
        Assumes master_data table has columns: CUSTID, PRDID, UOMTOID, PERIODID3_TSTAMP

        Args:
            raw_data: Raw data from SAP IBP
            database_name: Target database name
            tenant_id: Optional tenant identifier

        Returns:
            Number of records inserted
        """
        if not raw_data:
            return 0

        inserted_count = 0

        with self.db_manager.get_tenant_connection(database_name) as conn:
            cursor = conn.cursor()

            try:
                # Prepare insert statement for master_data
                # No ON CONFLICT clause since no unique constraints exist
                insert_query = """
                    INSERT INTO master_data (CUSTID, PRDID, UOMTOID, PERIODID3_TSTAMP, created_at, created_by)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """

                now = datetime.now()

                for record in raw_data:
                    try:
                        # Extract values from record
                        custid = record.get('CUSTID')
                        prdid = record.get('PRDID')
                        uomtoid = record.get('UOMTOID')
                        periodid3_tstamp = record.get('PERIODID3_TSTAMP')

                        # Parse timestamp if needed
                        if periodid3_tstamp:
                            parsed_date = parse_ibp_date(periodid3_tstamp)
                            if parsed_date:
                                periodid3_tstamp = parsed_date.isoformat()

                        # Insert record
                        cursor.execute(insert_query, (
                            custid, prdid, uomtoid, periodid3_tstamp, now, tenant_id or 'system'
                        ))
                        inserted_count += 1

                    except Exception as e:
                        logger.warning(f"Failed to insert master data record: {str(e)}", extra={
                            "record": record,
                            "error": str(e)
                        })
                        continue  # Skip this record and continue with next

                conn.commit()
                logger.info(f"Inserted {inserted_count} records into master_data")

            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to commit master data batch: {str(e)}")
            finally:
                cursor.close()

        return inserted_count

    async def _insert_sales_data(
        self,
        raw_data: List[Dict[str, Any]],
        database_name: str,
        tenant_id: Optional[str] = None
    ) -> int:
        """
        Insert data into sales_data table.
        Assumes sales_data table has columns: CUSTID, PRDID, ACTUALSQTY, UOMTOID, PERIODID3_TSTAMP

        Args:
            raw_data: Raw data from SAP IBP
            database_name: Target database name
            tenant_id: Optional tenant identifier

        Returns:
            Number of records inserted
        """
        if not raw_data:
            return 0

        inserted_count = 0

        with self.db_manager.get_tenant_connection(database_name) as conn:
            cursor = conn.cursor()

            try:
                # Prepare insert statement for sales_data
                # No ON CONFLICT clause since no unique constraints exist
                # sales_data table has: sales_id, master_id, date, quantity, uom, unit_price, created_at, created_by
                insert_query = """
                    INSERT INTO sales_data (CUSTID, PRDID, ACTUALSQTY, UOMTOID, PERIODID3_TSTAMP, created_at, created_by)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """

                now = datetime.now()

                for record in raw_data:
                    try:
                        # Extract values from record
                        custid = record.get('CUSTID')
                        prdid = record.get('PRDID')
                        actualsqty = record.get('ACTUALSQTY')
                        uomtoid = record.get('UOMTOID')
                        periodid3_tstamp = record.get('PERIODID3_TSTAMP')

                        # Parse timestamp if needed
                        if periodid3_tstamp:
                            parsed_date = parse_ibp_date(periodid3_tstamp)
                            if parsed_date:
                                periodid3_tstamp = parsed_date.isoformat()

                        # Convert quantity to float if needed
                        if actualsqty is not None:
                            try:
                                actualsqty = float(actualsqty)
                            except (ValueError, TypeError):
                                actualsqty = 0.0

                        # Insert record
                        cursor.execute(insert_query, (
                            custid, prdid, actualsqty, uomtoid, periodid3_tstamp, now, tenant_id or 'system'
                        ))
                        inserted_count += 1

                    except Exception as e:
                        logger.warning(f"Failed to insert sales data record: {str(e)}", extra={
                            "record": record,
                            "error": str(e)
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

    async def _process_and_separate_data(
        self,
        raw_data: List[Dict[str, Any]],
        database_name: str,
        tenant_id: str,
        master_data_mappings: Dict[str, str],
        sales_data_mappings: Dict[str, str]
    ) -> tuple[int, int]:
        """
        Process combined SAP IBP data and separate into master_data and sales_data tables.

        This method:
        1. Extracts unique combinations of customer, product, location for master_data
        2. Creates master_data records with auto-generated master_id
        3. Creates sales_data records linked to master_data via master_id

        Args:
            raw_data: Combined data from SAP IBP
            database_name: Target database name
            tenant_id: Tenant identifier

        Returns:
            Tuple of (master_data_inserted, sales_data_inserted)
        """
        if not raw_data:
            return 0, 0

        # Get actual column names from field catalogue metadata
        target_field_name, date_field_name = await self._get_field_catalogue_metadata(database_name)

        # Extract unique master data combinations
        master_data_map = {}  # (customer, product, location) -> master_id
        master_records = []
        sales_records = []

        for record in raw_data:
            # Extract dimension fields for master data using mappings
            customer = None
            product = None
            location = None

            # Map SAP fields to internal field names for master data
            for sap_field, internal_field in master_data_mappings.items():
                if internal_field == 'customer':
                    customer = record.get(sap_field)
                elif internal_field == 'product':
                    product = record.get(sap_field)
                elif internal_field == 'location':
                    location = record.get(sap_field)

            # Create unique key for master data
            master_key = (customer, product, location)

            # If this combination doesn't exist, create master record
            if master_key not in master_data_map:
                master_id = len(master_data_map) + 1  # Simple auto-increment
                master_data_map[master_key] = master_id

                master_records.append({
                    'master_id': master_id,
                    'customer': customer,
                    'product': product,
                    'location': location,
                    'tenant_id': tenant_id
                })

            # Create sales record linked to master data
            master_id = master_data_map[master_key]

            # Extract sales data fields using mappings
            quantity = None
            date = None
            unit_price = None

            for sap_field, internal_field in sales_data_mappings.items():
                if internal_field == 'quantity':
                    quantity = record.get(sap_field)
                elif internal_field == 'date':
                    date = record.get(sap_field)
                elif internal_field == 'unit_price':
                    unit_price = record.get(sap_field)

            sales_records.append({
                'master_id': master_id,
                'quantity': quantity,
                'date': date,
                'unit_price': unit_price,
                'tenant_id': tenant_id,
                'target_field_name': target_field_name,
                'date_field_name': date_field_name
            })

        # Insert master data first
        master_inserted = await self._insert_master_data_normalized(master_records, database_name, master_data_mappings)

        # Insert sales data with master_id references
        sales_inserted = await self._insert_sales_data_normalized(sales_records, database_name, target_field_name, date_field_name)

        return master_inserted, sales_inserted

    async def _insert_master_data_normalized(
        self,
        master_records: List[Dict[str, Any]],
        database_name: str,
        master_data_mappings: Dict[str, str]
    ) -> int:
        """
        Insert normalized master data records.

        Args:
            master_records: List of master data records
            database_name: Target database name
            master_data_mappings: Mapping from SAP field names to internal field names

        Returns:
            Number of records inserted
        """
        if not master_records:
            return 0

        inserted_count = 0

        with self.db_manager.get_tenant_connection(database_name) as conn:
            cursor = conn.cursor()

            try:
                # Build column names from mappings (SAP field names)
                sap_columns = list(master_data_mappings.keys())
                column_names = ['master_id'] + sap_columns + ['tenant_id', 'created_at', 'created_by', 'updated_at', 'updated_by']

                # Build placeholders
                placeholders = ', '.join(['%s'] * len(column_names))

                # Simple insert without ON CONFLICT (no unique constraints on SAP fields)
                insert_query = f"""
                    INSERT INTO master_data ({', '.join(f'"{col}"' for col in column_names)})
                    VALUES ({placeholders})
                """

                now = datetime.now()

                for record in master_records:
                    try:
                        # Map internal field names to values
                        values = [record['master_id']]
                        for sap_field in sap_columns:
                            internal_field = master_data_mappings[sap_field]
                            values.append(record[internal_field])
                        values.extend([
                            now,
                            record.get('tenant_id', 'system')
                        ])

                        cursor.execute(insert_query, values)
                        inserted_count += 1

                    except Exception as e:
                        logger.warning(f"Failed to insert master data record: {str(e)}", extra={
                            "record": record,
                            "error": str(e)
                        })
                        continue  # Skip this record and continue with next

                conn.commit()
                logger.info(f"Inserted {inserted_count} records into master_data")

            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to commit master data batch: {str(e)}")
            finally:
                cursor.close()

        return inserted_count

    async def _get_field_catalogue_metadata(self, database_name: str) -> tuple[str, str]:
        """
        Get target field name and date field name from field catalogue metadata.

        Args:
            database_name: Target database name

        Returns:
            Tuple of (target_field_name, date_field_name)
        """
        with self.db_manager.get_tenant_connection(database_name) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    SELECT target_field_name, date_field_name
                    FROM field_catalogue_metadata
                    LIMIT 1
                """)
                result = cursor.fetchone()
                if result:
                    return result[0], result[1]
                else:
                    # Default fallback if no metadata exists
                    return 'quantity', 'date'
            finally:
                cursor.close()

    async def _insert_sales_data_normalized(
        self,
        sales_records: List[Dict[str, Any]],
        database_name: str,
        target_field_name: str,
        date_field_name: str
    ) -> int:
        """
        Insert normalized sales data records with master_id references.

        Args:
            sales_records: List of sales data records
            database_name: Target database name
            target_field_name: Name of the target field column
            date_field_name: Name of the date field column

        Returns:
            Number of records inserted
        """
        if not sales_records:
            return 0

        inserted_count = 0

        with self.db_manager.get_tenant_connection(database_name) as conn:
            cursor = conn.cursor()

            try:
                # Use dynamic column names for target and date fields
                insert_query = f"""
                    INSERT INTO sales_data (master_id, "{target_field_name}", "{date_field_name}", uom, unit_price, created_at, created_by)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """

                now = datetime.now()

                for record in sales_records:
                    try:
                        # Parse date if needed
                        date_value = record.get('date')
                        if date_value:
                            parsed_date = parse_ibp_date(date_value)
                            if parsed_date:
                                date_value = parsed_date.isoformat()

                        # Convert quantity and unit_price to appropriate types
                        quantity = record.get('quantity')
                        if quantity is not None:
                            try:
                                quantity = float(quantity)
                            except (ValueError, TypeError):
                                quantity = 0.0

                        unit_price = record.get('unit_price')
                        if unit_price is not None:
                            try:
                                unit_price = float(unit_price)
                            except (ValueError, TypeError):
                                unit_price = 0.0

                        cursor.execute(insert_query, (
                            record['master_id'],
                            quantity,  # target field value
                            date_value,  # date field value
                            'EA',  # default UOM
                            unit_price,
                            now,
                            record.get('tenant_id', 'system')
                        ))
                        inserted_count += 1

                    except Exception as e:
                        logger.warning(f"Failed to insert sales data record: {str(e)}", extra={
                            "record": record,
                            "error": str(e)
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


# Convenience function for direct usage
async def ingest_sap_ibp_data(
    connection_config: Dict[str, Any],
    read_config: Dict[str, Any],
    database_name: str,
    tenant_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to ingest SAP IBP data.

    Args:
        connection_config: SAP IBP connection configuration
        read_config: Read configuration for SAP IBP
        database_name: Target database name
        tenant_id: Optional tenant identifier

    Returns:
        Ingestion results summary
    """
    service = SapIbpIngestionService()
    return await service.ingest_ibp_data(connection_config, read_config, database_name, tenant_id)


# Example usage (configurations should be provided dynamically from UI/API)
# All values (entity_set, select fields, filter conditions, etc.) are dynamic
# and should be passed as parameters to the ingest_sap_ibp_data function

# Example of how it would be called with dynamic configurations:
#
# connection_config = {
#     "base_url": "https://your-dynamic-url.com",
#     "username": "DYNAMIC_USER",
#     "password": "DYNAMIC_PASSWORD",
#     "sap_client": "DYNAMIC_CLIENT",
#     "timeout": 60.0
# }
#
# read_config = {
#     "service_path": "/sap/opu/odata/IBP/PLANNING_DATA_API_SRV",
#     "entity_set": "DYNAMIC_ENTITY_SET",  # Selected from UI
#     "select": ["DYNAMIC_FIELD1", "DYNAMIC_FIELD2"],  # Selected from UI
#     "filter": {
#         "and": [
#             {"field": "DYNAMIC_FIELD", "operator": "DYNAMIC_OP", "value": "DYNAMIC_VAL"}
#         ]
#     }
# }
#
# result = await ingest_sap_ibp_data(
#     connection_config=connection_config,
#     read_config=read_config,
#     database_name="tenant_db_name"
# )
