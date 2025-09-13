

"""factors.py — utilities for loading and working with factor tables

This module is intentionally UI‑free. It provides small, predictable helpers
that other scenario modules can import.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# -------------------------------
# Core loading & normalization
# -------------------------------

def load_factors(source: str | pd.DataFrame) -> pd.DataFrame:
    """Load a factors table from a CSV path or accept a preloaded DataFrame.

    Expectations (best-effort, tolerant):
    - Column set typically includes a date-ish column like "Date" or "begin month"
    - Allocation columns are labeled like "100E", "90E", ..., "0E" (sometimes with prefixes
      like "SPX60E" or "Global60E"). We detect them by a trailing pattern /(?i)(\d{1,3})E$/.
    - All allocation columns are coerced to numeric (errors become NaN).
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = pd.read_csv(source)

    # Strip column whitespace
    df.columns = [str(c).strip() for c in df.columns]

    # Parse a date-like column if present
    for cand in ("Date", "date", "Begin Date", "begin date", "begin month", "Begin Month"):
        if cand in df.columns:
            try:
                df[cand] = pd.to_datetime(df[cand], errors="coerce")
            except Exception:
                pass
            break

    # Coerce allocation columns to numeric
    for col in allocation_columns(df):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# -------------------------------
# Allocation discovery utilities
# -------------------------------

_ALLOC_TAIL_RE = re.compile(r"(?i)(\d{1,3})E$")  # captures the numeric part before trailing 'E'


def extract_alloc_tail(label: str) -> Optional[str]:
    """Return the canonical trailing alloc label like '60E' from a column name.

    Examples:
      '60E' -> '60E'
      'SPX60E' -> '60E'
      'Global 100E' -> '100E'
      'Equity_0E' -> '0E'
    """
    m = _ALLOC_TAIL_RE.search(str(label).strip())
    if not m:
        return None
    num = m.group(1)
    # guard 0..100 range but keep flexible if user has 110E test cols
    try:
        n = int(num)
        if n < 0:
            return None
    except Exception:
        return None
    return f"{num}E"


def allocation_columns(df: pd.DataFrame) -> List[str]:
    """All columns that look like allocation return series (e.g., '100E','90E',...)."""
    cols: List[str] = []
    for c in df.columns:
        if extract_alloc_tail(c):
            cols.append(c)
    return cols


def list_allocations(df: pd.DataFrame) -> List[str]:
    """Canonical list of allocations (e.g., ['100E','90E',...,'0E']) discovered in a DF.

    Order is descending by the numeric part, so UI can show 100E → 0E.
    """
    tails = {extract_alloc_tail(c) for c in allocation_columns(df)}
    tails.discard(None)
    def _key(x: str) -> int:
        try:
            return -int(x[:-1])  # sort descending numerically
        except Exception:
            return 0
    return sorted(tails, key=_key)


def common_allocs(df_a: pd.DataFrame, df_b: pd.DataFrame) -> List[str]:
    """Intersection of canonical allocations present in both dataframes, 100E→0E.
    Returns an empty list if none.
    """
    a = set(list_allocations(df_a))
    b = set(list_allocations(df_b))
    common = list(a & b)
    def _key(x: str) -> int:
        try:
            return -int(x[:-1])
        except Exception:
            return 0
    return sorted(common, key=_key)

# -------------------------------
# Fee helpers
# -------------------------------

def fee_multiplier_per_step(annual_expense_rate: float, steps_per_year: int = 12) -> float:
    """Compound an annual expense rate into a per-step multiplier.

    Example: 20 bps (0.0020) annual expense, monthly steps → (1-0.0020)**(12/12).
    """
    return (1.0 - float(annual_expense_rate)) ** (steps_per_year / 12)


__all__ = [
    "load_factors",
    "allocation_columns",
    "list_allocations",
    "common_allocs",
    "fee_multiplier_per_step",
    "extract_alloc_tail",
]

# -------------------------------------------------
# TEMP: shared types (will move to core/types.py)
# -------------------------------------------------
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class AppConfig:
    years_to_retire: int
    retirement_years: int
    spend_thinking: float
    spend_frugal: float
    habits: list[dict]            # [{daily, frugal, dpw, wpy, annual}, ...]
    auto: Dict[str, Any]          # car prices, rates, terms, replace, residual flag
    housing: Dict[str, Any]       # prices, down %, overrides, APR, term, tax rate

@dataclass
class ScenarioResult:
    table: pd.DataFrame | None = None        # formatted table for UI
    raw: pd.DataFrame | None = None          # numeric table for CSV/aggregations
    extras: Dict[str, pd.DataFrame] | None = None   # extra tables (schedules, residuals)
    median_withdrawal_df: pd.DataFrame | None = None
    callouts: list[str] | None = None        # bold sentences for UI