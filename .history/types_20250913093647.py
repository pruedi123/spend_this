from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import pandas as pd

@dataclass
class AppConfig:
    years_to_retire: int
    retirement_years: int
    spend_thinking: float
    spend_frugal: float
    habits: List[Dict[str, float]]   # [{daily, frugal, dpw, wpy, annual}, ...]
    auto: Dict[str, Any]             # car prices, rates, terms, replace, residual flag
    housing: Dict[str, Any]          # prices, down %, overrides, APR, term, tax rate

@dataclass
class ScenarioResult:
    table: Optional[pd.DataFrame] = None      # formatted table for UI
    raw: Optional[pd.DataFrame] = None        # numeric table for CSV/aggregations
    extras: Dict[str, pd.DataFrame] = None    # any extra tables (schedules, residuals)
    median_withdrawal_df: Optional[pd.DataFrame] = None
    callouts: List[str] = None                # bold sentences for UI