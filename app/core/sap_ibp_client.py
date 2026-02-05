"""
SAP IBP OData Client.
Handles communication with SAP IBP using OData protocol with CSRF token support.
"""

from __future__ import annotations

import httpx
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator, ConfigDict
from datetime import date, datetime, timedelta
import uuid


class ConnectionConfig(BaseModel):
    """SAP IBP connection configuration."""
    base_url: str
    username: str
    password: str
    sap_client: Optional[str] = None
    timeout: float = 60.0


class FilterCondition(BaseModel):
    """Filter condition for OData queries."""
    field: str
    operator: str  # eq, ne, gt, ge, lt, le, in
    value: Union[str, int, float, List[str]]

    @validator("operator")
    def validate_operator(cls, v: str) -> str:
        allowed = {"eq", "ne", "gt", "ge", "lt", "le", "in"}
        if v.lower() not in allowed:
            raise ValueError(f"Operator must be one of: {', '.join(sorted(allowed))}")
        return v.lower()


class FilterGroup(BaseModel):
    """Filter group for complex OData queries."""
    model_config = ConfigDict(populate_by_name=True, extra='forbid')
    and_conditions: Optional[List[FilterCondition]] = Field(default=None, alias="and")
    or_conditions: Optional[List[FilterCondition]] = Field(default=None, alias="or")


class ReadConfig(BaseModel):
    """SAP IBP read configuration."""
    service_path: str  # e.g., "/sap/opu/odata/IBP/PLANNING_DATA_API_SRV"
    entity_set: str  # e.g., "YSAPIBP1"
    select: List[str]  # Fields to select
    filter: Optional[Union[FilterGroup, Dict[str, Any]]] = None
    orderby: Optional[List[str]] = None
    top: Optional[int] = None
    skip: Optional[int] = None
    format: str = "json"
    inlinecount: Optional[str] = None  # "allpages"


class WriteConfig(BaseModel):
    """SAP IBP write configuration."""
    service_path: str  # e.g., "/sap/opu/odata/IBP/MASTER_DATA_API_SRV"
    entity_set_prefix: str  # e.g., "Y1B"
    entity_type: str  # e.g., "LOCATION"
    entity_set_suffix: str = "Trans"
    navigation_prefix: str = "Nav"
    write_fields: List[str]  # Fields to write (e.g., ["LOCID", "PRDID"])
    segment_field_name: str  # Field name for XYZ segment
    requested_attributes: List[str]  # All attributes for RequestedAttributes
    transaction_id: Optional[str] = None
    do_commit: bool = True


class DynamicODataClient:
    """Dynamic OData client for SAP IBP communication with CSRF token support."""

    def __init__(self, connection: ConnectionConfig):
        self.connection = connection
        self._client: Optional[httpx.AsyncClient] = None
        self._csrf_token: Optional[str] = None

    async def __aenter__(self):
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def open(self):
        """Open HTTP client."""
        if self._client:
            return
        self._client = httpx.AsyncClient(
            base_url=self.connection.base_url.rstrip("/"),
            auth=(self.connection.username, self.connection.password),
            timeout=self.connection.timeout,
            follow_redirects=True,
        )

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._csrf_token = None

    def _build_filter_string(self, filter_config: Union[Dict, FilterGroup]) -> str:
        """Build OData $filter string from filter configuration."""
        if isinstance(filter_config, dict):
            # Simple dict format: {"field": {"operator": "value"}}
            conditions = []
            for field, condition_data in filter_config.items():
                if isinstance(condition_data, dict):
                    for operator, value in condition_data.items():
                        condition = FilterCondition(field=field, operator=operator, value=value)
                        conditions.append(self._format_condition(condition))
                elif isinstance(condition_data, FilterCondition):
                    conditions.append(self._format_condition(condition_data))
            return " and ".join(conditions) if conditions else ""

        # FilterGroup format with and/or
        parts = []

        if filter_config.and_conditions:
            and_parts = [self._format_condition(c) for c in filter_config.and_conditions]
            if len(and_parts) > 1:
                parts.append(f"({' and '.join(and_parts)})")
            elif and_parts:
                parts.append(and_parts[0])

        if filter_config.or_conditions:
            or_parts = [self._format_condition(c) for c in filter_config.or_conditions]
            if len(or_parts) > 1:
                parts.append(f"({' or '.join(or_parts)})")
            elif or_parts:
                parts.append(or_parts[0])

        return " and ".join(parts) if parts else ""

    def _format_condition(self, condition: FilterCondition) -> str:
        """Format single filter condition to OData syntax."""
        field = condition.field
        operator = condition.operator
        value = condition.value

        if operator == "in":
            # Handle IN operator as multiple OR conditions
            if isinstance(value, list):
                or_conditions = [f"{field} eq '{v}'" for v in value]
                return f"({' or '.join(or_conditions)})"
            else:
                return f"{field} eq '{value}'"

        # Handle datetime format
        if isinstance(value, str) and value.startswith("datetime'"):
            return f"{field} {operator} {value}"

        # Auto-detect and format datetime
        if isinstance(value, str):
            try:
                # Try parsing as date
                date.fromisoformat(value[:10])
                if "T" not in value:
                    value = f"{value}T00:00:00"
                return f"{field} {operator} datetime'{value}'"
            except:
                pass

        # String values
        if isinstance(value, str):
            return f"{field} {operator} '{value}'"

        # Numeric values
        return f"{field} {operator} {value}"

    async def read_data(self, read_config: ReadConfig) -> List[Dict[str, Any]]:
        """Read data from IBP using dynamic configuration with CSRF token support."""
        if not self._client:
            raise RuntimeError("Client not opened. Use async context manager.")

        # Build URL
        url = f"{read_config.service_path.rstrip('/')}/{read_config.entity_set}"

        # Build query parameters
        params: Dict[str, str] = {}

        # $select
        if read_config.select:
            params["$select"] = ",".join(read_config.select)

        # $filter
        if read_config.filter:
            filter_str = self._build_filter_string(read_config.filter)
            if filter_str:
                params["$filter"] = filter_str

        # $orderby
        if read_config.orderby:
            params["$orderby"] = ",".join(read_config.orderby)

        # $top
        if read_config.top is not None:
            params["$top"] = str(read_config.top)

        # $skip
        if read_config.skip is not None:
            params["$skip"] = str(read_config.skip)

        # $format
        params["$format"] = read_config.format

        # $inlinecount
        if read_config.inlinecount:
            params["$inlinecount"] = read_config.inlinecount

        # sap-client
        if self.connection.sap_client:
            params["sap-client"] = self.connection.sap_client

        # Execute request with CSRF token support
        headers = {"Accept": "application/json"}

        # Try to get CSRF token for read operations (some SAP systems require it)
        try:
            csrf_token = await self._ensure_csrf_token()
            headers["x-csrf-token"] = csrf_token
        except Exception as e:
            # CSRF token might not be required for read operations in some systems
            # Log warning but continue without token
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"CSRF token not available for read operation: {str(e)}")

        response = await self._client.get(url, params=params, headers=headers)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise httpx.HTTPStatusError(
                f"IBP read failed: {exc.response.status_code} - {exc.response.text[:500]}",
                request=exc.request,
                response=exc.response
            ) from exc

        # Parse response
        try:
            body = response.json()
        except ValueError as exc:
            raise ValueError("IBP returned non-JSON response") from exc

        # Extract results
        if "d" in body and isinstance(body["d"], dict):
            return body["d"].get("results", [])
        return body.get("value", []) or body.get("results", [])

    async def _ensure_csrf_token(self) -> str:
        """Fetch CSRF token from IBP."""
        if self._csrf_token:
            return self._csrf_token

        if not self._client:
            raise RuntimeError("Client not opened.")

        # Try to fetch from $metadata
        metadata_url = "/sap/opu/odata/IBP/MASTER_DATA_API_SRV/$metadata"
        headers = {
            "x-csrf-token": "Fetch",
            "Accept": "*/*"
        }
        params = {}
        if self.connection.sap_client:
            params["sap-client"] = self.connection.sap_client

        response = await self._client.head(metadata_url, headers=headers, params=params)

        if response.status_code == 405:
            # Try GET if HEAD not allowed
            response = await self._client.get(metadata_url, headers=headers, params=params)

        token = response.headers.get("x-csrf-token")
        if not token:
            raise ValueError("Failed to fetch CSRF token from IBP")

        self._csrf_token = token
        return token

    async def write_master_data(
        self,
        write_config: WriteConfig,
        records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Write master data using transaction pattern."""
        if not self._client:
            raise RuntimeError("Client not opened.")

        # Construct entity set name: {prefix}{type}{suffix}
        entity_set = f"{write_config.entity_set_prefix}{write_config.entity_type}{write_config.entity_set_suffix}"

        # Construct navigation property: {nav_prefix}{prefix}{type}
        nav_property = f"{write_config.navigation_prefix}{write_config.entity_set_prefix}{write_config.entity_type}"

        # Build URL
        url = f"{write_config.service_path.rstrip('/')}/{entity_set}"

        # Generate transaction ID if not provided
        trans_id = write_config.transaction_id or str(uuid.uuid4()).replace("-", "")

        # Build payload
        payload = {
            "TransactionID": trans_id,
            "RequestedAttributes": ",".join(write_config.requested_attributes),
            "DoCommit": write_config.do_commit,
            nav_property: records
        }

        # Get CSRF token
        token = await self._ensure_csrf_token()

        # Execute request
        headers = {
            "x-csrf-token": token,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        params = {}
        if self.connection.sap_client:
            params["sap-client"] = self.connection.sap_client

        response = await self._client.post(url, json=payload, headers=headers, params=params)

        # Retry once with fresh token on 403
        if response.status_code == 403:
            self._csrf_token = None
            token = await self._ensure_csrf_token()
            headers["x-csrf-token"] = token
            response = await self._client.post(url, json=payload, headers=headers, params=params)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise httpx.HTTPStatusError(
                f"IBP write failed: {exc.response.status_code} - {exc.response.text[:500]}",
                request=exc.request,
                response=exc.response
            ) from exc

        return response.json() if response.content else {"success": True, "records": len(records)}


def parse_ibp_date(date_value: Any) -> Optional[date]:
    """Parse IBP date formats."""
    if date_value is None:
        return None

    if isinstance(date_value, date):
        return date_value

    text = str(date_value)

    # Handle /Date(milliseconds)/ format
    if "Date(" in text:
        try:
            start = text.find("Date(")
            end = text.find(")", start)
            if start != -1 and end != -1:
                millis = int(text[start + 5:end])
                return datetime.utcfromtimestamp(millis / 1000).date()
        except:
            pass

    # Handle ISO date format
    try:
        return date.fromisoformat(text[:10])
    except:
        return None
