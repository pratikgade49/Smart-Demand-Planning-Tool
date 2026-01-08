"""
Enhanced Excel upload service with comprehensive logging.
"""

import uuid
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from io import BytesIO
import time

from app.core.database import get_db_manager
from app.core.schema_manager import SchemaManager
from app.core.field_catalogue_service import FieldCatalogueService
from app.core.exceptions import ValidationException, DatabaseException, NotFoundException
from app.schemas.upload import ExcelUploadResponse
from app.core.logging_config import get_logger, log_operation_start, log_operation_end

logger = get_logger(__name__)


class ExcelUploadService:
    """Service for handling Excel file uploads with comprehensive logging."""

    SALES_DATA_PATTERNS = {
        'date': ['date', 'sales_date', 'transaction_date', 'period'],
        'quantity': ['quantity', 'qty', 'volume', 'amount'],
        'uom': ['uom', 'unit', 'unit_of_measure', 'measure'],
        'unit_price': ['unit_price', 'price', 'rate', 'cost', 'unitprice']
    }

    @staticmethod
    def get_db_manager():
        """Get database manager instance."""
        return get_db_manager()

    @staticmethod
    def validate_excel_file(file_content: bytes) -> pd.DataFrame:
        """Validate and parse Excel file content with logging."""
        log_operation_start(logger, "validate_excel_file", file_size_bytes=len(file_content))
        start_time = time.time()
        
        try:
            logger.debug(f"Parsing Excel file, size: {len(file_content)} bytes")
            df = pd.read_excel(BytesIO(file_content), engine='openpyxl')
            
            if df.empty:
                logger.warning("Excel file is empty")
                raise ValidationException("Excel file is empty")

            original_rows = len(df)
            df = df.dropna(how='all')
            removed_rows = original_rows - len(df)
            
            if removed_rows > 0:
                logger.info(f"Removed {removed_rows} empty rows from Excel file")

            if df.empty:
                logger.error("Excel file contains no valid data rows after cleanup")
                raise ValidationException("Excel file contains no valid data rows")

            if len(df.columns) < 2:
                logger.error(f"Excel file has insufficient columns: {len(df.columns)}")
                raise ValidationException("Excel file must have at least 2 columns")

            # Clean column names
            df.columns = df.columns.str.strip()
            
            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Excel validation successful: {len(df)} rows, {len(df.columns)} columns, {duration_ms:.2f}ms",
                extra={
                    "rows": len(df),
                    "columns": len(df.columns),
                    "duration_ms": duration_ms
                }
            )
            
            log_operation_end(
                logger, 
                "validate_excel_file", 
                success=True, 
                rows=len(df),
                columns=len(df.columns)
            )
            
            return df

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            if isinstance(e, ValidationException):
                log_operation_end(logger, "validate_excel_file", success=False, error=str(e))
                raise
            logger.error(f"Error parsing Excel file: {str(e)}", exc_info=True)
            log_operation_end(logger, "validate_excel_file", success=False, error=str(e))
            raise ValidationException(f"Invalid Excel file format: {str(e)}")

    @staticmethod
    def identify_column_types(
        df: pd.DataFrame,
        field_catalogue: Dict[str, Any]
    ) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:  # ✅ Now returns 3 values
        """
        Identify column types with detailed logging.
        FIXED: Now returns actual field names for dynamic column creation.
        
        Returns:
            Tuple of (master_data_mapping, sales_data_mapping, field_names_mapping)
            field_names_mapping contains: {'target_field': 'actual_name', 'date_field': 'actual_name'}
        """
        log_operation_start(logger, "identify_column_types")
        
        df_columns_lower = {col.lower().strip().replace(' ', '_'): col for col in df.columns}
        master_data_mapping = {}
        sales_data_mapping = {}

        # Get target and date field names from catalogue (DYNAMIC!)
        target_field_name = None
        date_field_name = None
        
        for field in field_catalogue.get('fields', []):
            if field.get('is_target_variable'):
                target_field_name = field['field_name']
            if field.get('is_date_field'):
                date_field_name = field['field_name']
        
        if not target_field_name:
            raise ValidationException("Field catalogue must have a target variable field")
        if not date_field_name:
            raise ValidationException("Field catalogue must have a date field")
        
        logger.info(f"Dynamic fields - Target: {target_field_name}, Date: {date_field_name}")

        # ✅ NEW: Store actual field names for later use
        field_names_mapping = {
            'target_field': target_field_name,
            'date_field': date_field_name
        }

        # Normalize catalogue field names
        catalogue_fields_lower = {}
        for field in field_catalogue.get('fields', []):
            if field.get('is_target_variable') or field.get('is_date_field'):
                continue
            normalized_key = field['field_name'].lower().strip().replace(' ', '_')
            catalogue_fields_lower[normalized_key] = field['field_name']

        logger.info(f"Excel columns: {list(df.columns)}")
        logger.info(f"Catalogue master fields: {list(catalogue_fields_lower.values())}")

        # Identify sales data columns
        sales_columns_found = set()
        
        # Map target field (DYNAMIC)
        target_normalized = target_field_name.lower().strip().replace(' ', '_')
        if target_normalized in df_columns_lower:
            sales_data_mapping['target'] = df_columns_lower[target_normalized]
            sales_columns_found.add(df_columns_lower[target_normalized])
            logger.debug(f"Mapped target field '{target_field_name}' to column '{df_columns_lower[target_normalized]}'")
        else:
            raise ValidationException(f"Target field '{target_field_name}' not found in Excel")
        
        # Map date field (DYNAMIC)
        date_normalized = date_field_name.lower().strip().replace(' ', '_')
        if date_normalized in df_columns_lower:
            sales_data_mapping['date'] = df_columns_lower[date_normalized]
            sales_columns_found.add(df_columns_lower[date_normalized])
            logger.debug(f"Mapped date field '{date_field_name}' to column '{df_columns_lower[date_normalized]}'")
        else:
            # Fallback to patterns if exact match fails
            date_mapped = False
            for pattern in ExcelUploadService.SALES_DATA_PATTERNS['date']:
                if pattern in df_columns_lower:
                    sales_data_mapping['date'] = df_columns_lower[pattern]
                    sales_columns_found.add(df_columns_lower[pattern])
                    logger.debug(f"Mapped date field '{date_field_name}' to column '{df_columns_lower[pattern]}' using pattern")
                    date_mapped = True
                    break
            if not date_mapped:
                raise ValidationException(f"Date field '{date_field_name}' not found in Excel")

        # Map UOM if exists
        if 'uom' in df_columns_lower:
            sales_data_mapping['uom'] = df_columns_lower['uom']
            sales_columns_found.add(df_columns_lower['uom'])
        else:
            logger.info("UoM column not found, will use default value 'EACH'")
            sales_data_mapping['uom'] = None

        # Map unit price if exists
        for price_pattern in ['unit_price', 'price', 'unitprice']:
            if price_pattern in df_columns_lower:
                sales_data_mapping['unit_price'] = df_columns_lower[price_pattern]
                sales_columns_found.add(df_columns_lower[price_pattern])
                break
        else:
            sales_data_mapping['unit_price'] = None

        # Map remaining columns to master fields
        skipped_columns = []
        for excel_col in df.columns:
            if excel_col not in sales_columns_found:
                excel_col_normalized = excel_col.lower().strip().replace(' ', '_')
                
                if excel_col_normalized in catalogue_fields_lower:
                    catalogue_field_name = catalogue_fields_lower[excel_col_normalized]
                    master_data_mapping[catalogue_field_name] = excel_col
                    logger.debug(f"Mapped master field '{catalogue_field_name}' to Excel column '{excel_col}'")
                else:
                    skipped_columns.append(excel_col)

        if skipped_columns:
            logger.warning(f"Skipping columns not in catalogue: {skipped_columns}")

        logger.info(f"Column mapping complete - Master: {len(master_data_mapping)}, Sales: {len(sales_data_mapping)}")
        
        log_operation_end(logger, "identify_column_types", success=True)

        return master_data_mapping, sales_data_mapping, field_names_mapping

    @staticmethod
    def process_mixed_data_upload(
        tenant_id: str,
        database_name: str,
        df: pd.DataFrame,
        field_catalogue: Dict[str, Any],
        user_email: str
    ) -> Tuple[int, int, List[Dict[str, Any]]]:
        """Process mixed data upload with comprehensive logging and batch upsert."""
        log_operation_start(logger, "process_mixed_data_upload", tenant_id=tenant_id, total_rows=len(df))

        logger.perf.log_performance_snapshot("Before processing upload")

        start_time = time.time()
        db_manager = get_db_manager()
        success_count = 0
        failed_count = 0
        errors = []

        # ✅ FIXED: Now receives field_names_mapping
        try:
            master_data_mapping, sales_mapping, field_names_mapping = ExcelUploadService.identify_column_types(
                df, field_catalogue
            )
        except Exception as e:
            logger.error(f"Failed to identify column types: {str(e)}")
            log_operation_end(logger, "process_mixed_data_upload", success=False, error=str(e))
            raise

        logger.info(f"Starting mixed data processing: {len(df)} rows")

        with db_manager.get_tenant_connection(database_name) as conn:
            cursor = conn.cursor()
            try:
                # Process in batches for better performance
                batch_size = 1000
                total_rows = len(df)

                for batch_start in range(0, total_rows, batch_size):
                    batch_end = min(batch_start + batch_size, total_rows)
                    batch_df = df.iloc[batch_start:batch_end]

                    logger.info(f"Processing batch {batch_start//batch_size + 1}: rows {batch_start}-{batch_end-1}")

                    try:
                        batch_success, batch_errors = ExcelUploadService.process_batch_upsert(
                            cursor=cursor,
                            tenant_id=tenant_id,
                            batch_df=batch_df,
                            master_data_mapping=master_data_mapping,
                            sales_mapping=sales_mapping,
                            field_names_mapping=field_names_mapping,
                            field_catalogue=field_catalogue,
                            user_email=user_email,
                            batch_start=batch_start
                        )

                        success_count += batch_success
                        failed_count += len(batch_errors)
                        errors.extend(batch_errors)

                        if success_count % 5000 == 0:
                            logger.info(f"Progress: {success_count}/{total_rows} rows processed")
                            logger.perf.log_memory_usage(f"After {success_count} rows")

                    except Exception as e:
                        logger.error(f"Batch processing failed for rows {batch_start}-{batch_end-1}: {str(e)}")
                        # Fall back to individual processing for this batch
                        for idx, row in batch_df.iterrows():
                            try:
                                cursor.execute(f"SAVEPOINT row_{idx}")

                                row_dict = row.to_dict()

                                ExcelUploadService.process_single_row(
                                    cursor=cursor,
                                    tenant_id=tenant_id,
                                    row_dict=row_dict,
                                    master_data_mapping=master_data_mapping,
                                    sales_mapping=sales_mapping,
                                    field_names_mapping=field_names_mapping,
                                    field_catalogue=field_catalogue,
                                    user_email=user_email
                                )

                                cursor.execute(f"RELEASE SAVEPOINT row_{idx}")
                                success_count += 1

                            except Exception as row_e:
                                cursor.execute(f"ROLLBACK TO SAVEPOINT row_{idx}")
                                cursor.execute(f"RELEASE SAVEPOINT row_{idx}")

                                failed_count += 1
                                error_msg = str(row_e)
                                errors.append({'row': idx + 2, 'error': error_msg})

                                if failed_count <= 10:
                                    logger.error(f"Error processing row {idx + 2}: {error_msg}")

                conn.commit()

                duration_ms = (time.time() - start_time) * 1000
                logger.info(f"Upload processing complete: {success_count} success, {failed_count} failed, {duration_ms:.2f}ms")
                logger.perf.log_performance_snapshot("After processing upload")

                log_operation_end(logger, "process_mixed_data_upload", success=True,
                                success_count=success_count, failed_count=failed_count)

            except Exception as e:
                conn.rollback()
                logger.error(f"Fatal error during upload: {str(e)}", exc_info=True)
                logger.perf.log_performance_snapshot("After upload error")
                log_operation_end(logger, "process_mixed_data_upload", success=False, error=str(e))
                raise DatabaseException(f"Database error during upload: {str(e)}")
            finally:
                cursor.close()

        return success_count, failed_count, errors

    @staticmethod
    def process_single_row(
        cursor,
        tenant_id: str,
        row_dict: Dict[str, Any],
        master_data_mapping: Dict[str, str],
        sales_mapping: Dict[str, str],
        field_names_mapping: Dict[str, str],  # ✅ NEW PARAMETER
        field_catalogue: Dict[str, Any],
        user_email: str
    ) -> None:
        """Process a single row with dynamic column names."""
        
        # Extract master data
        master_data = ExcelUploadService.extract_master_data(
            row_dict, master_data_mapping, field_catalogue
        )

        # Find or create master data record
        master_id = ExcelUploadService.find_or_create_master_record(
            cursor, tenant_id, master_data, user_email, field_catalogue
        )

        # Extract sales data
        sales_data = ExcelUploadService.extract_sales_data(
            row_dict, sales_mapping
        )

        # ✅ FIXED: Use dynamic column names from field_names_mapping
        target_field = field_names_mapping['target_field']
        date_field = field_names_mapping['date_field']
        
        sales_id = str(uuid.uuid4())
        
        # ✅ FIXED: Build dynamic UPSERT query to replace existing records
        cursor.execute(f"""
            INSERT INTO sales_data
            (sales_id, master_id, "{date_field}", "{target_field}", uom, unit_price, created_at, created_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (master_id, "{date_field}")
            DO UPDATE SET
                "{target_field}" = EXCLUDED."{target_field}",
                uom = EXCLUDED.uom,
                unit_price = EXCLUDED.unit_price,
                created_at = EXCLUDED.created_at,
                created_by = EXCLUDED.created_by
        """, (
            sales_id,
            master_id,
            sales_data['date'],      # ✅ Still uses 'date' key internally
            sales_data['quantity'],   # ✅ Still uses 'quantity' key internally
            sales_data['uom'],
            sales_data.get('unit_price'),
            datetime.utcnow(),
            user_email
        ))

    @staticmethod
    def clean_value(value: Any) -> Optional[str]:
        """Clean and normalize a value from Excel."""
        if pd.isna(value):
            return None
        
        value_str = str(value).strip()
        
        if value_str == '' or value_str.lower() in ('nan', 'null', 'none'):
            return None
            
        return value_str

    @staticmethod
    def extract_master_data(
        row_data: Dict[str, Any],
        master_data_mapping: Dict[str, str],
        field_catalogue: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract and clean master data from a row."""
        master_data = {}

        for catalogue_field_name, excel_column_name in master_data_mapping.items():
            raw_value = row_data.get(excel_column_name)
            cleaned_value = ExcelUploadService.clean_value(raw_value)
            master_data[catalogue_field_name] = cleaned_value

        logger.debug(f"Extracted master data: {master_data}")
        return master_data

    @staticmethod
    def extract_sales_data(
        row_data: Dict[str, Any],
        sales_mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """Extract and validate sales data from a row."""
        sales_data = {}

        # Extract date
        date_col = sales_mapping.get('date')
        if date_col and date_col in row_data:
            try:
                date_value = row_data[date_col]
                if pd.isna(date_value):
                    raise ValidationException("Date cannot be empty")
                sales_data['date'] = pd.to_datetime(date_value).date()
            except Exception as e:
                raise ValidationException(f"Invalid date format: {str(e)}")
        else:
            raise ValidationException("Date is required")

        # Extract quantity
        qty_col = sales_mapping.get('target')
        if qty_col and qty_col in row_data:
            try:
                qty_value = row_data[qty_col]
                if pd.isna(qty_value):
                    quantity = 0.0
                else:
                    quantity = float(qty_value)
                if quantity < 0:
                    raise ValidationException("Quantity cannot be negative")
                sales_data['quantity'] = quantity
            except (ValueError, TypeError) as e:
                raise ValidationException(f"Invalid quantity: {str(e)}")
        else:
            raise ValidationException("Quantity is required")

        # Extract UoM
        uom_col = sales_mapping.get('uom')
        if uom_col and uom_col in row_data and not pd.isna(row_data[uom_col]):
            uom_value = str(row_data[uom_col]).strip()
            if len(uom_value) > 20:
                uom_value = uom_value[:20]
            sales_data['uom'] = uom_value
        else:
            sales_data['uom'] = 'EACH'

        # Extract unit price
        price_col = sales_mapping.get('unit_price')
        if price_col and price_col in row_data and not pd.isna(row_data[price_col]):
            try:
                sales_data['unit_price'] = float(row_data[price_col])
            except (ValueError, TypeError):
                logger.debug(f"Invalid unit price, skipping: {row_data[price_col]}")
                sales_data['unit_price'] = None
        else:
            sales_data['unit_price'] = None

        return sales_data

    @staticmethod
    def find_or_create_master_record(
        cursor,
        tenant_id: str,
        master_data: Dict[str, Any],
        user_email: str,
        field_catalogue: Dict[str, Any]
    ) -> str:
        """Find existing master record or create new one based on unique key fields only."""
        
        # Identify unique key fields from catalogue
        unique_key_fields = set()
        fields_list = field_catalogue.get('fields', [])
        
        logger.debug(f"Field catalogue has {len(fields_list)} fields")
        
        for field in fields_list:
            field_name = field.get('field_name')
            is_unique = field.get('is_unique_key')
            logger.debug(f"Field: {field_name}, is_unique_key: {is_unique}, field_data: {field}")
            if is_unique:
                unique_key_fields.add(field_name)
        
        # Build WHERE clause using ONLY unique key fields
        where_conditions = []
        values = []
        unique_key_count = 0
        
        # FIXED: Iterate through unique_key_fields to ensure ALL are included
        for field in unique_key_fields:
            value = master_data.get(field)  # Get value from master_data or None
            if value is None or value == '':
                where_conditions.append(f'"{field}" IS NULL')
            else:
                where_conditions.append(f'"{field}" = %s')
                values.append(value)
            unique_key_count += 1
        

        
        # If no unique key fields configured, log warning and create new record
        if unique_key_count == 0:
            logger.warning("No unique key fields configured in field catalogue")
            master_id = str(uuid.uuid4())
            columns = ['master_id', 'created_at', 'created_by']
            insert_values = [master_id, datetime.utcnow(), user_email]
            
            for field, value in master_data.items():
                columns.append(f'"{field}"')
                insert_values.append(value)
            
            columns_str = ', '.join(columns)
            placeholders = ', '.join(['%s'] * len(insert_values))
            insert_query = f"INSERT INTO master_data ({columns_str}) VALUES ({placeholders}) RETURNING master_id"
            cursor.execute(insert_query, insert_values)
            new_master_id = cursor.fetchone()[0]
            logger.debug(f"Created master record (no unique keys configured): {new_master_id}")
            return new_master_id
        
        # Search for existing master record using unique key fields only
        where_clause = ' AND '.join(where_conditions)
        query = f"SELECT master_id FROM master_data WHERE {where_clause}"
        
        logger.debug(f"Searching for master record with unique keys: {list(unique_key_fields)}")
        cursor.execute(query, values)
        result = cursor.fetchone()
        
        if result:
            logger.debug(f"Found existing master record: {result[0]}")
            return result[0]
        
        # Create new master record with all master_data (including characteristics)
        master_id = str(uuid.uuid4())
        columns = ['master_id', 'created_at', 'created_by']
        insert_values = [master_id, datetime.utcnow(), user_email]
        
        for field, value in master_data.items():
            columns.append(f'"{field}"')
            insert_values.append(value)
        
        columns_str = ', '.join(columns)
        placeholders = ', '.join(['%s'] * len(insert_values))
        insert_query = f"INSERT INTO master_data ({columns_str}) VALUES ({placeholders}) RETURNING master_id"
        
        cursor.execute(insert_query, insert_values)
        new_master_id = cursor.fetchone()[0]
        logger.debug(f"Created new master record: {new_master_id} with {len(master_data)} fields")
        return new_master_id

    @staticmethod
    def process_batch_upsert(
        cursor,
        tenant_id: str,
        batch_df: pd.DataFrame,
        master_data_mapping: Dict[str, str],
        sales_mapping: Dict[str, str],
        field_names_mapping: Dict[str, str],
        field_catalogue: Dict[str, Any],
        user_email: str,
        batch_start: int
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """Process a batch of rows using batch upsert operations for better performance."""
        success_count = 0
        errors = []

        # Get dynamic field names
        target_field = field_names_mapping['target_field']
        date_field = field_names_mapping['date_field']

        # Prepare batch data structures
        master_records = []
        sales_records = []
        master_id_map = {}  # Maps dimension tuples to master_ids

        # Process each row in the batch
        for idx, row in batch_df.iterrows():
            try:
                row_dict = row.to_dict()

                # Extract master data
                master_data = ExcelUploadService.extract_master_data(
                    row_dict, master_data_mapping, field_catalogue
                )

                # Extract sales data
                sales_data = ExcelUploadService.extract_sales_data(
                    row_dict, sales_mapping
                )

                # Create dimension key for master record lookup
                unique_key_fields = set()
                for field in field_catalogue.get('fields', []):
                    if field.get('is_unique_key'):
                        unique_key_fields.add(field['field_name'])

                dimension_values = tuple(master_data.get(field) for field in sorted(unique_key_fields))
                master_records.append((dimension_values, master_data))
                sales_records.append((dimension_values, sales_data, idx))

            except Exception as e:
                errors.append({'row': batch_start + idx + 2, 'error': str(e)})
                continue

        # Batch process master data first
        for dimension_key, master_data in master_records:
            if dimension_key not in master_id_map:
                master_id = ExcelUploadService.find_or_create_master_record(
                    cursor, tenant_id, master_data, user_email, field_catalogue
                )
                master_id_map[dimension_key] = master_id

        # Now batch upsert sales data
        if sales_records:
            # Build batch upsert query for sales_data
            values_list = []
            now = datetime.utcnow()

            for dimension_key, sales_data, idx in sales_records:
                master_id = master_id_map.get(dimension_key)
                if not master_id:
                    errors.append({'row': batch_start + idx + 2, 'error': 'Master record not found'})
                    continue

                sales_id = str(uuid.uuid4())
                values_list.append((
                    sales_id,
                    master_id,
                    sales_data['date'],
                    sales_data['quantity'],
                    sales_data['uom'],
                    sales_data.get('unit_price'),
                    now,
                    user_email
                ))

            if values_list:
                # Deduplicate values_list based on (master_id, date) to avoid ON CONFLICT issues in batch
                unique_values = {}
                for values in values_list:
                    key = (values[1], values[2])  # master_id, date
                    unique_values[key] = values

                deduped_values_list = list(unique_values.values())

                # Create batch upsert query
                placeholders = ', '.join(['%s'] * 8)
                value_placeholders = ', '.join([f"({placeholders})"] * len(deduped_values_list))

                batch_upsert_query = f"""
                    INSERT INTO sales_data
                    (sales_id, master_id, "{date_field}", "{target_field}", uom, unit_price, created_at, created_by)
                    VALUES {value_placeholders}
                    ON CONFLICT (master_id, "{date_field}")
                    DO UPDATE SET
                        "{target_field}" = EXCLUDED."{target_field}",
                        uom = EXCLUDED.uom,
                        unit_price = EXCLUDED.unit_price,
                        created_at = EXCLUDED.created_at,
                        created_by = EXCLUDED.created_by
                """

                # Flatten the values list
                flattened_values = [item for sublist in deduped_values_list for item in sublist]

                try:
                    cursor.execute(batch_upsert_query, flattened_values)
                    success_count = len(values_list)
                except Exception as e:
                    logger.error(f"Batch upsert failed: {str(e)}")
                    # Fall back to individual upserts for this batch
                    for values in values_list:
                        try:
                            single_query = f"""
                                INSERT INTO sales_data
                                (sales_id, master_id, "{date_field}", "{target_field}", uom, unit_price, created_at, created_by)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (master_id, "{date_field}")
                                DO UPDATE SET
                                    "{target_field}" = EXCLUDED."{target_field}",
                                    uom = EXCLUDED.uom,
                                    unit_price = EXCLUDED.unit_price,
                                    created_at = EXCLUDED.created_at,
                                    created_by = EXCLUDED.created_by
                            """
                            cursor.execute(single_query, values)
                            success_count += 1
                        except Exception as single_e:
                            errors.append({'row': batch_start + idx + 2, 'error': str(single_e)})

        return success_count, errors

    @staticmethod
    def upload_excel_file(
        tenant_id: str,
        database_name: str,
        file_content: bytes,
        file_name: str,
        upload_type: str,
        catalogue_id: Optional[str],
        user_email: str
    ) -> ExcelUploadResponse:
        """Main upload method with comprehensive logging."""
        upload_id = str(uuid.uuid4())
        
        log_operation_start(
            logger,
            "upload_excel_file",
            upload_id=upload_id,
            tenant_id=tenant_id,
            file_name=file_name,
            upload_type=upload_type,
            file_size=len(file_content)
        )

        try:
            SchemaManager.add_upload_history_table(tenant_id, database_name)
        except Exception as e:
            logger.warning(f"Could not ensure upload_history table: {str(e)}")

        try:
            df = ExcelUploadService.validate_excel_file(file_content)
            total_rows = len(df)

            logger.info(f"Starting upload processing: {file_name} ({total_rows} rows)")

            if upload_type == "mixed_data":
                if not catalogue_id:
                    raise ValidationException("catalogue_id is required")

                field_catalogue = FieldCatalogueService.get_field_catalogue(
                    tenant_id, database_name, catalogue_id
                )
                
                if field_catalogue['status'] != 'FINALIZED':
                    raise ValidationException("Field catalogue must be finalized")

                success_count, failed_count, errors = ExcelUploadService.process_mixed_data_upload(
                    tenant_id, database_name, df, field_catalogue, user_email
                )
            else:
                raise ValidationException(f"Unsupported upload type: {upload_type}")

            # Log upload to database
            db_manager = get_db_manager()
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        INSERT INTO upload_history
                        (upload_id, upload_type, file_name, total_rows, 
                         success_count, failed_count, status, uploaded_at, uploaded_by)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        upload_id, upload_type, file_name, total_rows,
                        success_count, failed_count, 'completed',
                        datetime.utcnow(), user_email
                    ))
                    conn.commit()
                finally:
                    cursor.close()

            logger.info(
                f"Upload completed successfully: {upload_id}",
                extra={
                    "upload_id": upload_id,
                    "total_rows": total_rows,
                    "success_count": success_count,
                    "failed_count": failed_count,
                    "success_rate": round((success_count / total_rows * 100), 2) if total_rows > 0 else 0
                }
            )
            
            log_operation_end(
                logger,
                "upload_excel_file",
                success=True,
                upload_id=upload_id,
                success_count=success_count,
                failed_count=failed_count
            )

            return ExcelUploadResponse(
                upload_id=upload_id,
                upload_type=upload_type,
                file_name=file_name,
                total_rows=total_rows,
                success_count=success_count,
                failed_count=failed_count,
                status='completed',
                errors=errors[:100],
                uploaded_at=datetime.utcnow(),
                uploaded_by=user_email
            )

        except Exception as e:
            logger.error(f"Upload failed: {str(e)}", exc_info=True)
            log_operation_end(logger, "upload_excel_file", success=False, error=str(e))
            
            # Log failed upload
            try:
                db_manager = get_db_manager()
                with db_manager.get_tenant_connection(database_name) as conn:
                    cursor = conn.cursor()
                    try:
                        cursor.execute(
                            "SELECT 1 FROM upload_history WHERE upload_id = %s",
                            (upload_id,)
                        )
                        if not cursor.fetchone():
                            cursor.execute("""
                                INSERT INTO upload_history
                                (upload_id, upload_type, file_name, total_rows,
                                 success_count, failed_count, status, uploaded_at, uploaded_by)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                upload_id, upload_type, file_name, 0,
                                0, 0, 'failed', datetime.utcnow(), user_email
                            ))
                            conn.commit()
                    finally:
                        cursor.close()
            except Exception:
                pass

            if isinstance(e, (ValidationException, DatabaseException)):
                raise
            raise ValidationException(f"Upload failed: {str(e)}")