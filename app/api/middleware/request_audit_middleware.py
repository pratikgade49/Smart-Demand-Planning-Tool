"""
Request audit middleware.
Captures request/response details for API performance and debugging.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.auth_service import AuthService
from app.core.database import get_db_manager
from app.core.exceptions import DatabaseException
from psycopg2.extras import Json
import logging

logger = logging.getLogger(__name__)


class RequestAuditMiddleware(BaseHTTPMiddleware):
    """Middleware to store API request/response telemetry in the tenant database."""

    def __init__(
        self,
        app: ASGIApp,
        exclude_paths: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__(app)
        self.exclude_paths = set(exclude_paths or [])

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)

        start_time = datetime.utcnow()
        start_perf = time.perf_counter()

        request_payload = await _get_request_payload(request)
        headers = _redact_headers(dict(request.headers))
        client_ip = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")
        tenant_id, user_id, database_name = _extract_ids_from_token(
            request.headers.get("authorization")
        )
        payload_tenant_id = _extract_payload_value(request_payload, "tenant_id")
        payload_tenant_identifier = _extract_payload_value(
            request_payload,
            "tenant_identifier",
        )
        if tenant_id is None and payload_tenant_id is not None:
            tenant_id = str(payload_tenant_id)
        if database_name is None:
            database_name = _lookup_database_name(
                tenant_id=payload_tenant_id,
                tenant_identifier=payload_tenant_identifier,
            )

        try:
            response = await call_next(request)
        except Exception as exc:
            end_time = datetime.utcnow()
            duration_seconds = time.perf_counter() - start_perf
            _store_request_log(
                request=request,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration_seconds,
                request_payload=request_payload,
                response_status=500,
                success=False,
                message=None,
                error_message=str(exc),
                response_output=None,
                tenant_id=tenant_id,
                user_id=user_id,
                database_name=database_name,
                client_ip=client_ip,
                user_agent=user_agent,
                headers=headers,
            )
            raise

        response_body, response = await _read_response_body(response)
        end_time = datetime.utcnow()
        duration_seconds = time.perf_counter() - start_perf

        response_payload = _safe_json_loads(response_body)
        success = _extract_success(response.status_code, response_payload)
        message = _extract_message(response_payload)
        error_message = _extract_error_message(response.status_code, response_payload)
        response_output = _summarize_output(response_payload)

        _store_request_log(
            request=request,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration_seconds,
            request_payload=request_payload,
            response_status=response.status_code,
            success=success,
            message=message,
            error_message=error_message,
            response_output=response_output,
            tenant_id=tenant_id,
            user_id=user_id,
            database_name=database_name,
            client_ip=client_ip,
            user_agent=user_agent,
            headers=headers,
        )

        return response


async def _get_request_payload(request: Request) -> Optional[Dict[str, Any]]:
    content_type = (request.headers.get("content-type") or "").lower()
    if request.method in {"GET", "DELETE"} and request.query_params:
        return _redact_data(dict(request.query_params))

    body_bytes = await request.body()
    if not body_bytes:
        if request.query_params:
            return _redact_data(dict(request.query_params))
        return None

    if _is_file_payload(content_type, body_bytes):
        return _build_file_payload_summary(content_type, body_bytes)

    body_text = body_bytes.decode("utf-8", errors="ignore")
    parsed = _safe_json_loads(body_text)
    if parsed is None:
        return {"raw": _redact_text(body_text)}
    if isinstance(parsed, dict):
        return _redact_data(parsed)
    if isinstance(parsed, list):
        return _redact_data(parsed)
    return {"raw": _redact_text(body_text)}


def _redact_headers(headers: Dict[str, str]) -> Dict[str, str]:
    redacted = {}
    for key, value in headers.items():
        lower = key.lower()
        if lower in {"authorization", "cookie", "set-cookie"}:
            redacted[key] = "***redacted***"
        else:
            redacted[key] = value
    return redacted


def _extract_payload_value(payload: Any, key: str) -> Optional[Any]:
    if isinstance(payload, dict):
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _lookup_database_name(
    *,
    tenant_id: Optional[Any],
    tenant_identifier: Optional[Any],
) -> Optional[str]:
    if tenant_id is None and tenant_identifier is None:
        return None
    db_manager = get_db_manager()
    try:
        with db_manager.get_master_connection() as conn:
            cursor = conn.cursor()
            if tenant_id is not None:
                cursor.execute(
                    "SELECT database_name FROM public.tenants WHERE tenant_id = %s",
                    (str(tenant_id),),
                )
            else:
                cursor.execute(
                    "SELECT database_name FROM public.tenants WHERE tenant_identifier = %s",
                    (str(tenant_identifier),),
                )
            result = cursor.fetchone()
            return result[0] if result else None
    except Exception:
        logger.exception("Failed to resolve tenant database name")
        return None


def _redact_text(text: str) -> str:
    return text


def _redact_data(payload: Any) -> Any:
    sensitive_keys = {
        "password",
        "password_hash",
        "token",
        "access_token",
        "refresh_token",
        "authorization",
        "api_key",
        "secret",
        "client_secret",
    }
    if isinstance(payload, dict):
        result = {}
        for key, value in payload.items():
            if isinstance(key, str) and key.lower() in sensitive_keys:
                result[key] = "***redacted***"
            else:
                result[key] = _redact_data(value)
        return result
    if isinstance(payload, list):
        return [_redact_data(item) for item in payload]
    return payload


def _extract_ids_from_token(
    auth_header: Optional[str],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    if not auth_header:
        return None, None, None
    if not auth_header.lower().startswith("bearer "):
        return None, None, None
    token = auth_header.split(" ", 1)[1].strip()
    if not token:
        return None, None, None
    try:
        payload = AuthService.verify_user_token(token)
    except Exception:
        return None, None, None
    tenant_id = payload.get("tenant_id")
    user_id = payload.get("user_id")
    database_name = payload.get("database_name")
    return (
        str(tenant_id) if tenant_id is not None else None,
        str(user_id) if user_id is not None else None,
        database_name,
    )


async def _read_response_body(response: Response) -> tuple[bytes, Response]:
    body = b""
    async for chunk in response.body_iterator:
        body += chunk
    new_response = Response(
        content=body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )
    return body, new_response


def _safe_json_loads(payload: Any) -> Any:
    if payload is None:
        return None
    if isinstance(payload, (dict, list)):
        return payload
    try:
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8", errors="ignore")
        return json.loads(payload)
    except Exception:
        return None


def _extract_success(status_code: int, payload: Any) -> bool:
    if isinstance(payload, dict) and isinstance(payload.get("success"), bool):
        return payload["success"]
    return status_code < 400


def _extract_message(payload: Any) -> Optional[str]:
    if isinstance(payload, dict):
        for key in ("message", "detail", "error"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        data = payload.get("data")
        if isinstance(data, dict):
            value = data.get("message")
            if isinstance(value, str) and value.strip():
                return value
    return None


def _extract_error_message(status_code: int, payload: Any) -> Optional[str]:
    if status_code < 400:
        return None
    if isinstance(payload, dict):
        for key in ("error", "detail", "message"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def _summarize_output(payload: Any) -> Any:
    if payload is None:
        return None
    if isinstance(payload, list):
        return {"length": len(payload)}
    if isinstance(payload, dict):
        summarized = {}
        for key, value in payload.items():
            if isinstance(value, list):
                summarized[key] = {"length": len(value)}
            else:
                summarized[key] = _summarize_output(value)
        return summarized
    return payload


def _store_request_log(
    *,
    request: Request,
    start_time: datetime,
    end_time: datetime,
    duration_seconds: float,
    request_payload: Optional[Dict[str, Any]],
    response_status: int,
    success: bool,
    message: Optional[str],
    error_message: Optional[str],
    response_output: Any,
    tenant_id: Optional[str],
    user_id: Optional[str],
    database_name: Optional[str],
    client_ip: Optional[str],
    user_agent: Optional[str],
    headers: Dict[str, str],
) -> None:
    if not database_name:
        return
    db_manager = get_db_manager()
    insert_sql = """
        INSERT INTO public.api_request_logs (
            request_path,
            request_method,
            start_time,
            end_time,
            duration_seconds,
            request_payload,
            response_status,
            success,
            message,
            error_message,
            response_output,
            tenant_id,
            user_id,
            client_ip,
            user_agent,
            headers
        )
        VALUES (
            %(request_path)s,
            %(request_method)s,
            %(start_time)s,
            %(end_time)s,
            %(duration_seconds)s,
            %(request_payload)s,
            %(response_status)s,
            %(success)s,
            %(message)s,
            %(error_message)s,
            %(response_output)s,
            %(tenant_id)s,
            %(user_id)s,
            %(client_ip)s,
            %(user_agent)s,
            %(headers)s
        )
    """
    payload = {
        "request_path": request.url.path,
        "request_method": request.method,
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": round(duration_seconds, 6),
        "request_payload": Json(request_payload) if request_payload is not None else None,
        "response_status": response_status,
        "success": success,
        "message": message,
        "error_message": error_message,
        "response_output": Json(response_output) if response_output is not None else None,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "client_ip": client_ip,
        "user_agent": user_agent,
        "headers": Json(headers),
    }

    try:
        with db_manager.get_tenant_connection(database_name) as conn:
            cursor = conn.cursor()
            cursor.execute(insert_sql, payload)
            conn.commit()
    except DatabaseException:
        logger.exception("Failed to store API request log")
    except Exception:
        logger.exception("Unexpected error while storing API request log")
def _is_file_payload(content_type: str, body_bytes: bytes) -> bool:
    if "multipart/form-data" in content_type:
        return True
    if content_type.startswith("application/octet-stream"):
        return True
    if "application/vnd.openxmlformats-officedocument" in content_type:
        return True
    if "application/vnd.ms-excel" in content_type:
        return True
    if "application/pdf" in content_type:
        return True
    if body_bytes.find(b"\x00") != -1:
        return True
    return False


def _build_file_payload_summary(content_type: str, body_bytes: bytes) -> Dict[str, Any]:
    snippet = body_bytes[:8192].decode("utf-8", errors="ignore")
    filename = _extract_filename(snippet)
    return {
        "file_name": filename,
        "content_type": content_type or None,
        "content_length": len(body_bytes),
    }


def _extract_filename(body_text: str) -> Optional[str]:
    marker = 'filename="'
    start = body_text.find(marker)
    if start == -1:
        return None
    start += len(marker)
    end = body_text.find('"', start)
    if end == -1:
        return None
    name = body_text[start:end].strip()
    return name or None
