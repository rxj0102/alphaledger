"""Point-in-time store.

Every observation carries two dates:

* ``period``: the date the value describes (for example a fiscal quarter end
  or a GDP reference month).
* ``available_at``: the first moment the value could have been known
  (filing date, release date, vintage date).

Queries are always made *as of* a decision time. A value is visible only if
``available_at <= as_of``. When a value is revised, the newer vintage is
stored as another row with a later ``available_at``; as-of queries return the
latest vintage known at the decision time, never a later revision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

COLUMNS = ["entity", "field", "period", "available_at", "value", "source"]


class LookAheadError(ValueError):
    """Raised when a query would expose information from the future."""


def _ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


@dataclass
class PITStore:
    """In-memory point-in-time store backed by a pandas DataFrame."""

    _df: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=COLUMNS)
    )
    allow_early_availability: bool = False

    # ------------------------------------------------------------------ write
    def add(self, df: pd.DataFrame) -> None:
        """Append observations. ``df`` must contain the COLUMNS (source optional)."""
        df = df.copy()
        if "source" not in df.columns:
            df["source"] = "unknown"
        missing = set(COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")
        df = df[COLUMNS]
        df["period"] = pd.to_datetime(df["period"])
        df["available_at"] = pd.to_datetime(df["available_at"])
        if df[["entity", "field", "period", "available_at"]].isna().any().any():
            raise ValueError("entity, field, period and available_at may not be null")
        if (df["available_at"] < df["period"]).any() and not self.allow_early_availability:
            # A value about period P is normally known on or after P. Forecasts
            # would break this, so the check can be disabled explicitly.
            bad = df[df["available_at"] < df["period"]].head(3)
            raise ValueError(
                "available_at earlier than period for some rows; set "
                "allow_early_availability=True if these are forecasts:\n"
                f"{bad}"
            )
        frames = [f for f in (self._df, df) if not f.empty]
        self._df = pd.concat(frames, ignore_index=True) if frames else df
        self._df = self._df.drop_duplicates(
            subset=["entity", "field", "period", "available_at"], keep="last"
        ).reset_index(drop=True)

    # ------------------------------------------------------------------- read
    def __len__(self) -> int:
        return len(self._df)

    def fields(self) -> list[str]:
        return sorted(self._df["field"].unique().tolist())

    def as_of(self, as_of, field: str, entities=None) -> pd.DataFrame:
        """All periods of ``field`` as known at ``as_of`` (latest vintage per period).

        Returns columns: entity, period, value, available_at.
        """
        t = _ts(as_of)
        d = self._df[(self._df["field"] == field) & (self._df["available_at"] <= t)]
        if entities is not None:
            d = d[d["entity"].isin(list(entities))]
        if d.empty:
            return pd.DataFrame(columns=["entity", "period", "value", "available_at"])
        d = d.sort_values("available_at")
        latest = d.groupby(["entity", "period"], as_index=False).last()
        return latest[["entity", "period", "value", "available_at"]].sort_values(
            ["entity", "period"]
        ).reset_index(drop=True)

    def latest(self, as_of, field: str, entities=None) -> pd.Series:
        """Most recent period's value per entity, as known at ``as_of``."""
        d = self.as_of(as_of, field, entities)
        if d.empty:
            return pd.Series(dtype=float, name=field)
        last = d.sort_values("period").groupby("entity").last()
        s = last["value"].astype(float)
        s.name = field
        return s

    def panel(self, field: str, dates, lag_days: int = 0) -> pd.DataFrame:
        """Dates x entities panel of the latest value known at each date.

        Each row ``t`` uses only information with ``available_at <= t - lag_days``.
        This is the safe way to turn slow-moving data (fundamentals, macro)
        into a daily signal input.
        """
        dates = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values()
        d = self._df[self._df["field"] == field].copy()
        if d.empty:
            return pd.DataFrame(index=dates, dtype=float)
        d["value"] = d["value"].astype(float)
        d = d.sort_values(["available_at", "period"])
        # For each (entity, available_at) keep the observation with the latest
        # period: that is the newest information released at that moment.
        d = d.groupby(["entity", "available_at"], as_index=False).last()
        wide = d.pivot(index="available_at", columns="entity", values="value")
        query = dates - pd.Timedelta(days=lag_days)
        out = wide.reindex(wide.index.union(query)).sort_index().ffill()
        out = out.reindex(query)
        out.index = dates
        out.index.name = None
        out.columns.name = None
        return out

    def assert_no_lookahead(self, decision_time, used: pd.DataFrame) -> None:
        """Raise LookAheadError if any row of ``used`` was unavailable at decision_time.

        ``used`` must have an ``available_at`` column (as returned by as_of).
        """
        t = _ts(decision_time)
        if "available_at" not in used.columns:
            raise ValueError("used must contain available_at")
        late = used[pd.to_datetime(used["available_at"]) > t]
        if not late.empty:
            raise LookAheadError(
                f"{len(late)} observation(s) became available after {t.date()}"
            )

    def to_frame(self) -> pd.DataFrame:
        return self._df.copy()
