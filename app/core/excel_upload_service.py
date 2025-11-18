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
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Identify column types with detailed logging."""
        log_operation_start(logger, "identify_column_types")
        
        df_columns_lower = {col.lower().strip(): col for col in df.columns}
        master_data_mapping = {}
        sales_data_mapping = {}

        # Normalize catalogue field names: convert spaces to underscores and lowercase
        catalogue_fields_lower = {}
        for field in field_catalogue.get('fields', []):
            # Normalize: lowercase, replace spaces with underscores
            normalized_key = field['field_name'].lower().strip().replace(' ', '_')
            catalogue_fields_lower[normalized_key] = field['field_name']

        logger.info(f"Excel columns: {list(df.columns)}")
        logger.info(f"Catalogue fields: {list(catalogue_fields_lower.values())}")

        # Identify sales data columns
        sales_columns_found = set()
        for sales_field, patterns in ExcelUploadService.SALES_DATA_PATTERNS.items():
            for pattern in patterns:
                if pattern in df_columns_lower:
                    sales_data_mapping[sales_field] = df_columns_lower[pattern]
                    sales_columns_found.add(df_columns_lower[pattern])
                    logger.debug(f"Mapped sales field '{sales_field}' to column '{df_columns_lower[pattern]}'")
                    break

        # Validate required sales fields
        if 'date' not in sales_data_mapping:
            logger.error(f"Missing required 'date' column")
            raise ValidationException(
                f"Missing required 'date' column. Expected one of: {ExcelUploadService.SALES_DATA_PATTERNS['date']}"
            )
        
        if 'quantity' not in sales_data_mapping:
            logger.error(f"Missing required 'quantity' column")
            raise ValidationException(
                f"Missing required 'quantity' column. Expected one of: {ExcelUploadService.SALES_DATA_PATTERNS['quantity']}"
            )

        if 'uom' not in sales_data_mapping:
            logger.info("UoM column not found, will use default value 'EACH'")
            sales_data_mapping['uom'] = None

        # Map remaining columns to catalogue fields
        skipped_columns = []
        for excel_col in df.columns:
            if excel_col not in sales_columns_found:
                # Normalize Excel column: lowercase, replace spaces with underscores
                excel_col_normalized = excel_col.lower().strip().replace(' ', '_')
                
                if excel_col_normalized in catalogue_fields_lower:
                    catalogue_field_name = catalogue_fields_lower[excel_col_normalized]
                    master_data_mapping[catalogue_field_name] = excel_col
                    logger.debug(f"Mapped master field '{catalogue_field_name}' to Excel column '{excel_col}'")
                else:
                    skipped_columns.append(excel_col)

        if skipped_columns:
            logger.warning(f"Skipping columns not in catalogue: {skipped_columns}")

        logger.info(
            f"Column mapping complete - Master: {len(master_data_mapping)}, Sales: {len(sales_data_mapping)}",
            extra={
                "master_columns": len(master_data_mapping),
                "sales_columns": len(sales_data_mapping),
                "skipped_columns": len(skipped_columns)
            }
        )
        
        log_operation_end(
            logger,
            "identify_column_types",
            success=True,
            master_columns=len(master_data_mapping),
            sales_columns=len(sales_data_mapping)
        )

        return master_data_mapping, sales_data_mapping

    @staticmethod
    def process_mixed_data_upload(
        tenant_id: str,
        database_name: str,
        df: pd.DataFrame,
        field_catalogue: Dict[str, Any],
        user_email: str
    ) -> Tuple[int, int, List[Dict[str, Any]]]:
        """Process mixed data upload with comprehensive logging."""
        log_operation_start(
            logger,
            "process_mixed_data_upload",
            tenant_id=tenant_id,
            total_rows=len(df)
        )
        
        # Log initial performance
        logger.perf.log_performance_snapshot("Before processing upload")
        
        start_time = time.time()
        db_manager = get_db_manager()
        success_count = 0
        failed_count = 0
        errors = []

        # Identify column types
        try:
            master_data_mapping, sales_mapping = ExcelUploadService.identify_column_types(
                df, field_catalogue
            )
        except Exception as e:
            logger.error(f"Failed to identify column types: {str(e)}")
            log_operation_end(logger, "process_mixed_data_upload", success=False, error=str(e))
            raise

        logger.info(f"Starting mixed data processing: {len(df)} rows")

        with db_manager.get_connection(tenant_id) as conn:
            cursor = conn.cursor()
            try:
                # Process rows with progress logging
                for idx, row in df.iterrows():
                    savepoint_name = f"row_{idx}"
                    
                    try:
                        cursor.execute(f"SAVEPOINT {savepoint_name}")
                        
                        row_dict = row.to_dict()
                        
                        ExcelUploadService.process_single_row(
                            cursor=cursor,
                            tenant_id=tenant_id,
                            row_dict=row_dict,
                            master_data_mapping=master_data_mapping,
                            sales_mapping=sales_mapping,
                            field_catalogue=field_catalogue,
                            user_email=user_email
                        )
                        
                        cursor.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                        success_count += 1
                        
                        # Progress logging
                        if success_count % 100 == 0:
                            logger.info(
                                f"Progress: {success_count}/{len(df)} rows processed",
                                extra={
                                    "processed": success_count,
                                    "total": len(df),
                                    "percent": round((success_count / len(df)) * 100, 2)
                                }
                            )
                            # Log performance every 100 rows
                            logger.perf.log_memory_usage(f"After {success_count} rows")

                    except Exception as e:
                        cursor.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                        cursor.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                        
                        failed_count += 1
                        error_msg = str(e)
                        errors.append({
                            'row': idx + 2,
                            'error': error_msg
                        })
                        
                        if failed_count <= 10:  # Log first 10 errors in detail
                            logger.error(f"Error processing row {idx + 2}: {error_msg}")
                        elif failed_count == 11:
                            logger.warning("More than 10 errors encountered, suppressing detailed error logs")

                conn.commit()
                
                duration_ms = (time.time() - start_time) * 1000
                
                logger.info(
                    f"Upload processing complete: {success_count} success, {failed_count} failed, {duration_ms:.2f}ms",
                    extra={
                        "success_count": success_count,
                        "failed_count": failed_count,
                        "duration_ms": duration_ms,
                        "rows_per_second": round(len(df) / (duration_ms / 1000), 2) if duration_ms > 0 else 0
                    }
                )
                
                # Log final performance
                logger.perf.log_performance_snapshot("After processing upload")
                
                log_operation_end(
                    logger,
                    "process_mixed_data_upload",
                    success=True,
                    success_count=success_count,
                    failed_count=failed_count
                )

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
        field_catalogue: Dict[str, Any],
        user_email: str
    ) -> None:
        """Process a single row with debug logging."""
        # Extract master data
        master_data = ExcelUploadService.extract_master_data(
            row_dict, master_data_mapping, field_catalogue
        )

        # Find or create master data record (using only unique key fields)
        master_id = ExcelUploadService.find_or_create_master_record(
            cursor, tenant_id, master_data, user_email, field_catalogue
        )

        # Extract sales data
        sales_data = ExcelUploadService.extract_sales_data(
            row_dict, sales_mapping
        )

        # Insert sales data
        sales_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO sales_data 
            (sales_id, tenant_id, master_id, date, quantity, uom, unit_price, created_at, created_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            sales_id,
            tenant_id,
            master_id,
            sales_data['date'],
            sales_data['quantity'],
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
        qty_col = sales_mapping.get('quantity')
        if qty_col and qty_col in row_data:
            try:
                qty_value = row_data[qty_col]
                if pd.isna(qty_value):
                    raise ValidationException("Quantity cannot be empty")
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
        
        # Add tenant_id condition
        where_conditions.append('tenant_id = %s')
        values.append(tenant_id)
        
        # If no unique key fields configured, log warning and create new record
        if unique_key_count == 0:
            logger.warning("No unique key fields configured in field catalogue")
            master_id = str(uuid.uuid4())
            columns = ['master_id', 'tenant_id', 'created_at', 'created_by']
            insert_values = [master_id, tenant_id, datetime.utcnow(), user_email]
            
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
        columns = ['master_id', 'tenant_id', 'created_at', 'created_by']
        insert_values = [master_id, tenant_id, datetime.utcnow(), user_email]
        
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
            with db_manager.get_connection(tenant_id) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        INSERT INTO upload_history
                        (upload_id, tenant_id, upload_type, file_name, total_rows, 
                         success_count, failed_count, status, uploaded_at, uploaded_by)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        upload_id, tenant_id, upload_type, file_name, total_rows,
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
                tenant_id=tenant_id,
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
                with db_manager.get_connection(tenant_id) as conn:
                    cursor = conn.cursor()
                    try:
                        cursor.execute(
                            "SELECT 1 FROM upload_history WHERE upload_id = %s",
                            (upload_id,)
                        )
                        if not cursor.fetchone():
                            cursor.execute("""
                                INSERT INTO upload_history
                                (upload_id, tenant_id, upload_type, file_name, total_rows,
                                 success_count, failed_count, status, uploaded_at, uploaded_by)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                upload_id, tenant_id, upload_type, file_name, 0,
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