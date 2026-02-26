"""
Schema for table-level disaggregation feature.
File: app/schemas/disaggregation_schema.py

Supports three input modes:
  1. periods    – explicit list of {date, quantity} pairs  ← recommended
  2. single     – single date + quantity
  3. range      – date_from + date_to (reads totals from DB automatically)
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Dict, Any
from datetime import date as DateType


class DisaggregationPeriod(BaseModel):
    """One date-quantity pair for multi-period disaggregation."""
    date: DateType = Field(..., description="Period date (YYYY-MM-DD)")
    quantity: float = Field(..., description="Aggregated quantity to disaggregate. Zero is allowed.")


class DisaggregateTableRequest(BaseModel):
    """
    Request to disaggregate aggregated value(s) into lower-level member records.

    Three input modes (exactly one must be provided):
      • periods   – explicit list of {date, quantity} — use this when the
                    frontend has the aggregated values ready (most common case)
      • single    – single date + quantity
      • range     – date_from + date_to; service reads existing totals from
                    the source table automatically per period
    """

    # ── Target ──────────────────────────────────────────────────────────────
    target_table: str = Field(
        ...,
        description=(
            "Table to write disaggregated data into "
            "(e.g. 'final_plan', 'product_manager'). "
            "Must be registered in dynamic_tables."
        )
    )

    # ── Source group identification ─────────────────────────────────────────
    master_data: Dict[str, Any] = Field(
        ...,
        description=(
            "Dimension fields that identify the aggregated group. "
            "Supply only the fields you aggregated on "
            "(e.g. {'product': 'P1'} to disaggregate across all customers of P1)."
        )
    )

    # ── Method ──────────────────────────────────────────────────────────────
    method: str = Field(
        ...,
        description=(
            "Disaggregation method:\n"
            "  • 'own_ratio'   – ratios from target table's own existing data\n"
            "  • 'key_figure'  – ratios from a different reference table\n"
            "  • 'equal'       – equal share for every lower-level member"
        )
    )

    # ── Key-figure table (only for method='key_figure') ─────────────────────
    key_figure_table: Optional[str] = Field(
        default=None,
        description=(
            "Reference table for ratio calculation. "
            "Required when method='key_figure'. "
            "Can be any dynamic table, 'sales_data', or 'forecast_data'."
        )
    )

    # ── Mode 1: periods array ────────────────────────────────────────────────
    periods: Optional[List[DisaggregationPeriod]] = Field(
        default=None,
        description=(
            "Explicit list of {date, quantity} pairs to disaggregate. "
            "Use this when the frontend has the aggregated values ready. "
            "Mutually exclusive with date/quantity and date_from/date_to."
        )
    )

    # ── Mode 2: single date ──────────────────────────────────────────────────
    date: Optional[DateType] = Field(
        default=None,
        description=(
            "Single date to disaggregate (YYYY-MM-DD). "
            "Must be used together with quantity. "
            "Mutually exclusive with periods and date_from/date_to."
        )
    )
    quantity: Optional[float] = Field(
        default=None,
        description=(
            "Aggregated quantity for single-date mode. "
            "Required when date is provided. Zero is allowed."
        )
    )

    # ── Mode 3: date range (reads from DB) ───────────────────────────────────
    date_from: Optional[DateType] = Field(
        default=None,
        description=(
            "Start of range to disaggregate (inclusive, YYYY-MM-DD). "
            "Service reads existing period totals from the source table. "
            "Mutually exclusive with periods and date/quantity."
        )
    )
    date_to: Optional[DateType] = Field(
        default=None,
        description=(
            "End of range to disaggregate (inclusive, YYYY-MM-DD). "
            "Must be >= date_from."
        )
    )

    # ── Ratio look-back window (optional, applies to own_ratio & key_figure) ─
    ratio_date_from: Optional[DateType] = Field(
        default=None,
        description=(
            "Start of look-back window for ratio calculation. "
            "Defaults to all available history when omitted."
        )
    )
    ratio_date_to: Optional[DateType] = Field(
        default=None,
        description=(
            "End of look-back window for ratio calculation. "
            "Defaults to all available history when omitted."
        )
    )

    # ── Validators ──────────────────────────────────────────────────────────
    @field_validator("method")
    @classmethod
    def validate_method(cls, v):
        allowed = {"own_ratio", "key_figure", "equal"}
        if v not in allowed:
            raise ValueError(f"method must be one of {allowed}, got '{v}'")
        return v

    @model_validator(mode="after")
    def validate_request(self):
        # ── key_figure_table required check ───────────────────────────────
        if self.method == "key_figure" and not self.key_figure_table:
            raise ValueError("key_figure_table is required when method='key_figure'")

        # ── Determine which modes are active ──────────────────────────────
        has_periods = self.periods is not None and len(self.periods) > 0
        has_single  = self.date is not None
        has_range   = self.date_from is not None or self.date_to is not None

        active_modes = sum([has_periods, has_single, has_range])

        if active_modes == 0:
            raise ValueError(
                "Provide one of: "
                "(1) periods array, "
                "(2) date + quantity, "
                "(3) date_from + date_to."
            )
        if active_modes > 1:
            raise ValueError(
                "Only one input mode allowed at a time: "
                "periods, OR date+quantity, OR date_from+date_to."
            )

        # ── Single-date mode: quantity required ───────────────────────────
        if has_single and self.quantity is None:
            raise ValueError("quantity is required when date is provided.")

        # ── Range mode: both bounds required + order check ─────────────────
        if has_range:
            if self.date_from is None or self.date_to is None:
                raise ValueError("Both date_from and date_to are required for range mode.")
            if self.date_to < self.date_from:
                raise ValueError("date_to must be >= date_from.")

        # ── Periods mode: must not be empty ───────────────────────────────
        if has_periods and len(self.periods) == 0:
            raise ValueError("periods list must contain at least one entry.")

        return self

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "summary": "Periods mode – key figure (most common use case)",
                    "value": {
                        "target_table": "final_plan",
                        "master_data": {"product": "P1"},
                        "method": "key_figure",
                        "key_figure_table": "sales_data",
                        "periods": [
                            {"date": "2025-01-01", "quantity": 2400},
                            {"date": "2025-02-01", "quantity": 4240},
                            {"date": "2025-03-01", "quantity": 3500}
                        ],
                        "ratio_date_from": "2025-01-01",
                        "ratio_date_to": "2025-03-31"
                    }
                },
                {
                    "summary": "Periods mode – own ratio",
                    "value": {
                        "target_table": "final_plan",
                        "master_data": {"product": "P1"},
                        "method": "own_ratio",
                        "periods": [
                            {"date": "2025-01-01", "quantity": 2400},
                            {"date": "2025-02-01", "quantity": 4240},
                            {"date": "2025-03-01", "quantity": 3500}
                        ]
                    }
                },
                {
                    "summary": "Periods mode – equal distribution",
                    "value": {
                        "target_table": "final_plan",
                        "master_data": {"product": "P1"},
                        "method": "equal",
                        "periods": [
                            {"date": "2025-01-01", "quantity": 2400},
                            {"date": "2025-02-01", "quantity": 4240},
                            {"date": "2025-03-01", "quantity": 3500}
                        ]
                    }
                },
                {
                    "summary": "Single date – own ratio",
                    "value": {
                        "target_table": "final_plan",
                        "master_data": {"product": "P1"},
                        "method": "own_ratio",
                        "date": "2025-01-01",
                        "quantity": 2400
                    }
                },
                {
                    "summary": "Range mode – reads totals from DB automatically",
                    "value": {
                        "target_table": "final_plan",
                        "master_data": {"product": "P1"},
                        "method": "own_ratio",
                        "date_from": "2025-01-01",
                        "date_to": "2025-03-31"
                    }
                }
            ]
        }