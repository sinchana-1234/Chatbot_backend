"""
Feature Engineering for CGM Glucose Data
Minute → Hourly → Weekly lag ML-ready pipeline
"""

import pandas as pd
import numpy as np

DAY_NAMES = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]


# ─────────────────────────────────────────────────────────────
# 1️⃣ DATA CLEANING
# ─────────────────────────────────────────────────────────────
def clean_cgm_data(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Remove invalid timestamps
    df = df.dropna(subset=["timestamp"])

    # Remove impossible glucose values
    df = df[(df["glucose"] >= 40) & (df["glucose"] <= 400)]

    # Remove duplicates
    df = df.drop_duplicates(subset=["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# ─────────────────────────────────────────────────────────────
# 2️⃣ MINUTE → HOURLY AGGREGATION
# ─────────────────────────────────────────────────────────────
def build_hourly_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex CGM readings to one row per hour and interpolate ONLY small gaps.

    Small gaps (≤3 hours, e.g. brief sensor disconnects) are linearly
    interpolated so they don't break lag features. Larger gaps stay NaN —
    they're not real readings and we must not pretend they are.

    `count` column tracks how many raw readings contributed to each hour:
      count > 0 → real hour with actual measurements
      count == 0 → reindexed slot. May or may not have an interpolated value
                   in glucose_mean depending on whether the gap was ≤3 hours.

    Downstream consumers:
      - compute_week_summary filters to `count > 0` to avoid reporting
        interpolated values as real.
      - engineer_features drops rows missing required lags, which handles
        the wider NaN gaps left behind by the limit=3 interpolation.
    """
    df = df.copy()

    df["hour_timestamp"] = df["timestamp"].dt.floor("h")

    hourly = (
        df.groupby("hour_timestamp")["glucose"]
        .agg(
            glucose_mean="mean",
            glucose_max="max",
            glucose_min="min",
            count="count"
        )
        .reset_index()
    )

    hourly = hourly.set_index("hour_timestamp")

    all_hours = pd.date_range(
        start=hourly.index.min(),
        end=hourly.index.max(),
        freq="h"
    )

    hourly = hourly.reindex(all_hours)

    hourly.index.name = "hour_timestamp"

    # Interpolate gaps of up to 3 consecutive hours only. Bigger gaps stay
    # NaN — the previous version did `.ffill().bfill()` afterward which
    # silently fabricated values across arbitrarily long gaps (e.g. a 71-day
    # gap was forward-filled with the value from before the gap).
    hourly[["glucose_mean","glucose_max","glucose_min"]] = (
        hourly[["glucose_mean","glucose_max","glucose_min"]]
        .interpolate(method="linear", limit=3, limit_area="inside")
    )

    hourly = hourly.reset_index()

    hourly["count"] = hourly["count"].fillna(0)

    return hourly


# ─────────────────────────────────────────────────────────────
# 3️⃣ ADD TIME FEATURES
# ─────────────────────────────────────────────────────────────
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["hour"] = df["hour_timestamp"].dt.hour
    df["dayofweek"] = df["hour_timestamp"].dt.dayofweek
    df["day_name"] = df["dayofweek"].map(lambda x: DAY_NAMES[x])

    iso = df["hour_timestamp"].dt.isocalendar()

    df["iso_year"] = iso.year
    df["iso_week"] = iso.week

    df["week_number"] = (
        df["iso_year"].astype(str)
        + "-"
        + df["iso_week"].astype(str).str.zfill(2)
    )

    # Cyclic encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    return df


# ─────────────────────────────────────────────────────────────
# 4️⃣ LAG FEATURES
#
# Short lags are the primary signal for the prediction model. They work
# from day 3 onward (lag_48 needs 2 days of history). The training pipeline
# requires 5 days minimum to have enough rows after lag NaN removal.
#
# Weekly lags (168, 336) are still computed because the historical
# `compute_week_summary` path uses them. The training feature list in
# prediction_service.py explicitly uses ONLY the short lags + cyclic
# encodings, so the weekly lags are computed but ignored at train/predict
# time. This lets the same engineered dataframe serve both purposes.
# ─────────────────────────────────────────────────────────────

SHORT_LAG_HOURS = [1, 2, 3, 6, 12, 24, 48]


def add_short_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Add short-range hourly lags used by the prediction model."""
    df = df.copy()
    df = df.sort_values("hour_timestamp").set_index("hour_timestamp")

    for h in SHORT_LAG_HOURS:
        df[f"lag_{h}"] = df["glucose_mean"].shift(h)

    df = df.reset_index()
    return df


def add_weekly_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Add weekly lags (used only by historical week summaries, not the model)."""
    df = df.copy()
    df = df.sort_values("hour_timestamp").set_index("hour_timestamp")

    df["lag_168"] = df["glucose_mean"].shift(168)
    df["lag_336"] = df["glucose_mean"].shift(336)

    df = df.reset_index()
    return df


# ─────────────────────────────────────────────────────────────
# 4️⃣c RATE OF CHANGE FEATURES
#
# Why: The model currently knows WHAT glucose was at each lag but not
# WHICH DIRECTION it is moving. Two patients with identical lag_1=130
# are treated identically even if one is rising (was 110 an hour ago)
# and the other is falling (was 150 an hour ago). These rate-of-change
# features make the direction and speed of glucose movement explicit.
#
# roc_1h  = change over last 1 hour  → immediate direction
# roc_3h  = change over last 3 hours → short-term trend
# roc_6h  = change over last 6 hours → medium-term trend
# roc_24h = change over last 24 hours → daily trend vs yesterday
#
# acceleration = whether the rate itself is speeding up or slowing down.
# Positive = glucose rising faster. Negative = rise slowing / reversing.
#
# These are computed AFTER short lags so lag values already exist.
# NaN handling: rows where lags are NaN will produce NaN roc values,
# but those rows are already dropped downstream by engineer_features()
# via the required lag columns check — so no extra dropna needed here.
# ─────────────────────────────────────────────────────────────

def add_rate_of_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rate-of-change features derived from existing lag columns.

    Must be called AFTER add_short_lags() since it reads lag_1..lag_6.

    New columns added:
      roc_1h  : glucose_mean - lag_1   (change in last hour)
      roc_3h  : glucose_mean - lag_3   (change over 3 hours)
      roc_6h  : glucose_mean - lag_6   (change over 6 hours)
      roc_24h : glucose_mean - lag_24  (change vs same hour yesterday)
      acceleration: roc_1h - (lag_1 - lag_2)
                    positive = rate of change is increasing (rising faster)
                    negative = rate of change is decreasing (slowing down)
    """
    df = df.copy()

    # Direction and speed over different time horizons
    # Positive = glucose rising, Negative = glucose falling
    df["roc_1h"]  = df["glucose_mean"] - df["lag_1"]   # change in last 1h
    df["roc_3h"]  = df["glucose_mean"] - df["lag_3"]   # change over 3h
    df["roc_6h"]  = df["glucose_mean"] - df["lag_6"]   # change over 6h
    df["roc_24h"] = df["glucose_mean"] - df["lag_24"]  # vs same hour yesterday

    # Acceleration: is the rate of change itself speeding up or slowing down?
    # roc_1h = current 1-hour change
    # lag_1 - lag_2 = previous 1-hour change
    # acceleration = current change - previous change
    df["acceleration"] = df["roc_1h"] - (df["lag_1"] - df["lag_2"])

    return df


# ─────────────────────────────────────────────────────────────
# 4️⃣b PER-PATIENT HOUR-OF-DAY BASELINE
# ─────────────────────────────────────────────────────────────
def add_hour_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a per-hour glucose baseline as a feature.

    For each row, hour_baseline = the mean glucose this patient typically has
    at that hour-of-day, computed from the rows in this dataframe.

    Why: RF tends to smooth predictions toward the training mean. A patient
    whose recent week is flat will get flat predictions even if their full
    history shows clear meal-time spikes. Providing the per-hour mean as an
    explicit feature gives the model a strong direct signal of "what's
    normal for this hour for this patient."

    The baseline is computed BEFORE lag NaN rows are dropped, so it uses
    every available hour of data (not just rows that survive lag filtering).
    """
    df = df.copy()
    if "hour" not in df.columns:
        df["hour"] = df["hour_timestamp"].dt.hour

    hour_means = df.groupby("hour")["glucose_mean"].mean()
    df["hour_baseline"] = df["hour"].map(hour_means).astype(float)

    # If a particular hour has no observations at all (shouldn't happen with
    # reindexed hourly data but defensive), fill with overall mean.
    if df["hour_baseline"].isna().any():
        df["hour_baseline"] = df["hour_baseline"].fillna(df["glucose_mean"].mean())

    return df


# ─────────────────────────────────────────────────────────────
# 5️⃣ FINAL FEATURE PIPELINE
# ─────────────────────────────────────────────────────────────
def engineer_features(df_raw: pd.DataFrame, include_weekly_lag: bool = False) -> pd.DataFrame:
    """
    Engineer features for the prediction model.

    include_weekly_lag controls whether rows missing lag_168 (1-week lag)
    are dropped from training. Set True for established patients (≥14 days
    of data); False for new patients where lag_168 isn't reliably available.
    The model in prediction_service.py reads its own feature column list and
    will only consume what it was trained on, but the training dataframe
    must contain non-NaN values for whichever lags the model uses.
    """

    df = clean_cgm_data(df_raw)
    df = build_hourly_dataframe(df)
    df = add_time_features(df)
    df = add_hour_baseline(df)
    df = add_short_lags(df)
    df = add_weekly_lags(df)
    df = add_rate_of_change(df)   # ← new: added after short lags

    # Always drop rows missing short lags (they are core features).
    short_lag_cols = [f"lag_{h}" for h in SHORT_LAG_HOURS]
    required = list(short_lag_cols)

    # For established patients, also require lag_168 so the model can use it.
    # lag_336 is intentionally NOT required — we don't add it to FEATURE_COLS
    # anywhere, so requiring it would just throw away training rows.
    if include_weekly_lag:
        required.append("lag_168")

    # Also drop rows where the targets themselves are NaN. After fix #5
    # removed forward-fill, hours with no real data can have NaN in
    # glucose_mean/max/min (only gaps ≤3 hours get interpolated). Such a
    # row's LAG features can still be valid (computed from other hours that
    # did have data), so without this it would slip past the lag-dropna and
    # crash sklearn's model.fit() with "Input y contains NaN." There is no
    # target to learn from, so the row must be dropped.
    required += ["glucose_mean", "glucose_max", "glucose_min"]

    df = df.dropna(subset=required).reset_index(drop=True)

    return df


# ─────────────────────────────────────────────────────────────
# 6️⃣ WEEK SUMMARY
# ─────────────────────────────────────────────────────────────
def compute_week_summary(df: pd.DataFrame, week: str):
    """
    Compute weekly stats from REAL CGM readings only.

    `build_hourly_dataframe` reindexes to a continuous hourly range and fills
    gaps via interpolate+ffill+bfill. Those filled rows have `count == 0` —
    they're padding, not measurements. Counting them as real readings would
    fabricate data for weeks where the patient wore no sensor.

    For a 71-day data gap (e.g. Feb 1 → Apr 13), the previous version would
    forward-fill 168 hours per week with the same value, then report each
    week as "168 readings, mean=92.18, std=0, TIR=100%" — none of which is
    true. This filters to `count > 0` first so the stats reflect what was
    actually measured.
    """
    w = df[df["week_number"] == week]

    if w.empty:
        return None

    year, week_no = week.split("-")

    week_start = pd.Timestamp.fromisocalendar(
        int(year), int(week_no), 1
    ).date()

    week_end = pd.Timestamp.fromisocalendar(
        int(year), int(week_no), 7
    ).date()

    # ─── KEY FIX: only count rows with real readings ───
    w_real = w[w["count"] > 0]
    total_real = int(w_real["count"].sum())  # total raw CGM readings, not rows

    # If the patient had no real readings this week, return a summary that
    # honestly reflects that — don't fabricate stats from interpolated padding.
    if w_real.empty:
        return {
            "week_number": week,
            "week_start": week_start,
            "week_end": week_end,
            "total_readings": 0,
            "overall_mean": 0.0,
            "overall_max": 0.0,
            "overall_min": 0.0,
            "overall_std": 0.0,
            "hypo_events": 0,
            "hyper_events": 0,
            "time_in_range_pct": 0.0,
        }

    mean_val = w_real["glucose_mean"].mean()
    std_val = w_real["glucose_mean"].std()

    if pd.isna(mean_val):
        mean_val = 0

    if pd.isna(std_val):
        std_val = 0

    hours_with_data = len(w_real)
    tir = 0
    if hours_with_data > 0:
        tir = (w_real["glucose_mean"].between(70, 180).sum() / hours_with_data) * 100

    return {
        "week_number": week,
        "week_start": week_start,
        "week_end": week_end,
        "total_readings": total_real,

        "overall_mean": float(round(mean_val, 2)),
        "overall_max": float(w_real["glucose_max"].max()),
        "overall_min": float(w_real["glucose_min"].min()),
        "overall_std": float(round(std_val, 2)),

        "hypo_events": int((w_real["glucose_mean"] < 70).sum()),
        "hyper_events": int((w_real["glucose_mean"] > 180).sum()),

        "time_in_range_pct": float(round(tir, 2))
    }