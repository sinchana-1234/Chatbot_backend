"""
Glucose prediction service.

Model: per-patient Random Forest using short-range hourly lags + cyclic
time encodings + rate-of-change features. Designed to work from 5 days
of CGM data upward; accuracy improves as more history accumulates.

Caching: trained models are cached per patient in-process. A patient's
model is reused until their data grows by at least RETRAIN_GROWTH_DAYS,
at which point it's retrained on the new history.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import date
from typing import Optional
import logging

from config import settings
from feature_engineering import (
    engineer_features,
    clean_cgm_data,
    build_hourly_dataframe,
    add_time_features,
    compute_week_summary,
    SHORT_LAG_HOURS,
    DAY_NAMES,
)

logger = logging.getLogger(__name__)


# Feature columns for the model. We use two configurations based on how
# much history a patient has:
#   - WEEKLY_LAG_THRESHOLD_DAYS+ days of data: include lag_168 (1 week)
#     so weekly patterns get captured. Better accuracy for established patients.
#   - 5 to (threshold-1) days of data: short lags only. Works for new patients.
#
# Both configurations include:
#   - hour_baseline: patient's average glucose at this hour-of-day. Strong
#     direct signal of diurnal pattern. Prevents RF from collapsing to flat
#     predictions when the most recent training window is unusually stable.
#   - roc_*: rate-of-change features. Tell the model the DIRECTION and SPEED
#     of glucose movement, not just the values. Key for improving/worsening
#     patients where glucose today differs significantly from the baseline.
#
#     roc_1h  = current - lag_1   (change in last 1 hour)
#     roc_3h  = current - lag_3   (change over 3 hours, captures meal spikes)
#     roc_6h  = current - lag_6   (change over 6 hours, captures activity)
#     roc_24h = current - lag_24  (vs same hour yesterday — KEY for trend)
#     acceleration = roc_1h - (lag_1 - lag_2)
#                    positive = glucose rising faster
#                    negative = rate of rise slowing / glucose reversing
#
# lag_336 (2-week lag) is deliberately NOT used. Even at 30 days of data,
# lag_336 eats a huge chunk of training rows for marginal benefit.

SHORT_FEATURE_COLS = (
    [f"lag_{h}" for h in SHORT_LAG_HOURS]
    + ["hour_baseline", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    + ["roc_1h", "roc_3h", "roc_6h", "roc_24h", "acceleration"]
)
FULL_FEATURE_COLS = SHORT_FEATURE_COLS + ["lag_168"]

WEEKLY_LAG_THRESHOLD_DAYS = 14  # patients with ≥14 days of data use FULL_FEATURE_COLS

# Data sufficiency thresholds
MIN_HISTORY_DAYS = 5         # absolute floor: refuse below this
MIN_RECENT_COVERAGE = 0.50   # ≥50% of recent hours must be real readings

# Iterative prediction is bounded to avoid compounding-smoothness. For
# days_ahead beyond this, we predict the target day directly without
# feeding earlier predictions forward as lag inputs.
MAX_ITERATIVE_DAYS = 3

# Cache behavior
MAX_CACHED_PATIENTS = 100    # LRU eviction beyond this
RETRAIN_GROWTH_DAYS = 1      # retrain when a patient gains ≥1 day of new data

# TIR (Time in Range) clinical bounds: standard for general diabetes care
TIR_LOWER_BOUND = 70   # mg/dL
TIR_UPPER_BOUND = 180  # mg/dL


class GlucosePredictionService:

    def __init__(self):
        self._cache = {}


    # ─────────────────────────────────────────────
    # LOAD DATA
    # ─────────────────────────────────────────────
    def load_patient_data(
        self,
        db_session,
        patient_id: int,
        limit_days: int = 30,
    ) -> pd.DataFrame:

        from database import CGMReading
        from datetime import datetime, timedelta

        query = db_session.query(CGMReading).filter(
            CGMReading.patient_id == patient_id
        )

        if limit_days is not None:
            cutoff = datetime.utcnow() - timedelta(days=limit_days)
            query = query.filter(CGMReading.local_event_time >= cutoff)

        readings = query.order_by(CGMReading.local_event_time).all()

        if not readings:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "timestamp": r.local_event_time,
                "glucose": r.glucose,
                "patient_id": r.patient_id,
            }
            for r in readings
        ])


    def _choose_feature_set(self, df_raw: pd.DataFrame) -> tuple[list, bool]:
        if df_raw.empty:
            return SHORT_FEATURE_COLS, False
        ts = pd.to_datetime(df_raw["timestamp"], errors="coerce").dropna()
        if ts.empty:
            return SHORT_FEATURE_COLS, False
        real_days = ts.dt.date.nunique()
        if real_days >= WEEKLY_LAG_THRESHOLD_DAYS:
            return FULL_FEATURE_COLS, True
        return SHORT_FEATURE_COLS, False

    # ─────────────────────────────────────────────
    # TRAIN MODEL
    # ─────────────────────────────────────────────
    def train(self, df_raw: pd.DataFrame, patient_id: str, last_data_date: date):

        feature_cols, include_weekly = self._choose_feature_set(df_raw)
        df = engineer_features(df_raw, include_weekly_lag=include_weekly)

        if df.empty:
            raise ValueError("Not enough data after feature engineering.")

        hour_col = df["hour_timestamp"].dt.hour
        mean_baseline_map = df.groupby(hour_col)["glucose_mean"].mean().to_dict()
        max_baseline_map  = df.groupby(hour_col)["glucose_max"].mean().to_dict()
        min_baseline_map  = df.groupby(hour_col)["glucose_min"].mean().to_dict()

        overall_mean = float(df["glucose_mean"].mean())
        overall_max  = float(df["glucose_max"].mean())
        overall_min  = float(df["glucose_min"].mean())

        mean_baselines_per_row = hour_col.map(mean_baseline_map).fillna(overall_mean).to_numpy()
        max_baselines_per_row  = hour_col.map(max_baseline_map).fillna(overall_max).to_numpy()
        min_baselines_per_row  = hour_col.map(min_baseline_map).fillna(overall_min).to_numpy()

        X = df[feature_cols].values
        y_abs = df[["glucose_mean", "glucose_max", "glucose_min"]].values
        y_dev = y_abs - np.column_stack([
            mean_baselines_per_row, max_baselines_per_row, min_baselines_per_row
        ])

        # ─── EVALUATION (DEBUG only) ───────────────────────────────────────
        split_idx = int(len(df) * 0.8)
        metrics = {}
        if settings.DEBUG and split_idx > 0 and len(df) - split_idx >= 1:
            eval_model = RandomForestRegressor(
                n_estimators=settings.RF_N_ESTIMATORS,
                random_state=settings.RF_RANDOM_STATE,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features="sqrt",
                n_jobs=1,
            )
            eval_model.fit(X[:split_idx], y_dev[:split_idx])
            y_pred_dev = eval_model.predict(X[split_idx:])
            test_baselines = np.column_stack([
                mean_baselines_per_row[split_idx:],
                max_baselines_per_row[split_idx:],
                min_baselines_per_row[split_idx:],
            ])
            y_pred_abs = y_pred_dev + test_baselines
            mae = mean_absolute_error(y_abs[split_idx:], y_pred_abs)
            rmse = np.sqrt(mean_squared_error(y_abs[split_idx:], y_pred_abs))
            metrics = {"mae": float(round(mae, 4)), "rmse": float(round(rmse, 4))}
            logger.info(f"Eval for patient {patient_id}: MAE={mae:.2f}, RMSE={rmse:.2f} "
                        f"({'full' if include_weekly else 'short'} features)")

        # ─── FINAL MODEL ───────────────────────────────────────────────────
        model = RandomForestRegressor(
            n_estimators=settings.RF_N_ESTIMATORS,
            random_state=settings.RF_RANDOM_STATE,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=1,
        )
        model.fit(X, y_dev)

        # LRU eviction
        if len(self._cache) >= MAX_CACHED_PATIENTS and patient_id not in self._cache:
            oldest_patient = next(iter(self._cache))
            del self._cache[oldest_patient]
            logger.info(f"Evicted patient {oldest_patient} from model cache")

        self._cache.pop(patient_id, None)
        self._cache[patient_id] = {
            "model": model,
            "trained_on_last_date": last_data_date,
            "feature_cols": feature_cols,
            "include_weekly_lag": include_weekly,
            "hour_baseline_map": mean_baseline_map,
            "max_baseline_map": max_baseline_map,
            "min_baseline_map": min_baseline_map,
            "overall_mean": overall_mean,
            "overall_max": overall_max,
            "overall_min": overall_min,
            "metrics": metrics,
        }
        logger.info(f"Model trained for patient {patient_id} "
                    f"({'full' if include_weekly else 'short'} features, {len(df)} rows, "
                    f"deviation targets)")

    def _get_or_train_model(self, df_raw: pd.DataFrame, patient_id: str, last_data_date: date):
        cached = self._cache.get(patient_id)
        if cached is not None:
            days_since_train = (last_data_date - cached["trained_on_last_date"]).days
            _, should_use_weekly = self._choose_feature_set(df_raw)
            feature_set_changed = (cached["include_weekly_lag"] != should_use_weekly)
            if days_since_train < RETRAIN_GROWTH_DAYS and not feature_set_changed:
                self._cache.pop(patient_id)
                self._cache[patient_id] = cached
                return cached

        self.train(df_raw, patient_id, last_data_date)
        return self._cache[patient_id]


    # ─────────────────────────────────────────────
    # PREDICT
    # ─────────────────────────────────────────────
    def predict(
        self,
        df_raw: pd.DataFrame,
        patient_id: str,
        target_date: date,
        target_hour: Optional[int] = None,
    ) -> dict:

        if df_raw.empty:
            raise ValueError("No data available.")

        df_clean = clean_cgm_data(df_raw)
        df_hourly = build_hourly_dataframe(df_clean)
        df_time = add_time_features(df_hourly)

        if df_hourly.empty:
            raise ValueError("No valid CGM data available.")

        last_data_ts = df_hourly["hour_timestamp"].max()
        first_data_ts = df_hourly["hour_timestamp"].min()
        last_data_date = last_data_ts.date()

        target_ts = pd.Timestamp(target_date)
        days_ahead = (target_ts.normalize() - pd.Timestamp(last_data_date)).days

        # ── Historical ──
        if days_ahead <= 0:
            iso = target_date.isocalendar()
            target_week_number = f"{iso.year}-{str(iso.week).zfill(2)}"
            summary = compute_week_summary(df_time, target_week_number)
            return {
                "type": "historical_summary",
                "patient_id": patient_id,
                "requested_date": target_date.isoformat(),
                "requested_week": target_week_number,
                "week_summary": summary,
            }

        # ── Too far ahead ──
        if days_ahead > 14:
            raise ValueError(
                f"Prediction window is limited to 14 days from the last "
                f"available reading ({last_data_date.isoformat()}). "
                f"Requested date {target_date.isoformat()} is {days_ahead} days ahead."
            )

        # ── Data sufficiency checks ──
        real_data_dates = (
            df_hourly[df_hourly["count"] > 0]["hour_timestamp"]
            .dt.date.unique()
        )
        history_days = len(real_data_dates)
        if history_days < MIN_HISTORY_DAYS:
            raise ValueError(
                f"Need at least {MIN_HISTORY_DAYS} days of CGM history to predict. "
                f"This patient has {history_days} real day(s) of readings."
            )

        coverage_window_days = min(14, history_days)
        coverage_window_hours = coverage_window_days * 24
        window_start = last_data_ts - pd.Timedelta(days=coverage_window_days)
        recent = df_hourly[
            (df_hourly["hour_timestamp"] > window_start)
            & (df_hourly["hour_timestamp"] <= last_data_ts)
        ]
        observed_hours = (recent["count"] > 0).sum()
        coverage = observed_hours / coverage_window_hours if coverage_window_hours else 0
        if coverage < MIN_RECENT_COVERAGE:
            raise ValueError(
                f"Insufficient recent CGM coverage: {coverage:.0%} of the "
                f"last {coverage_window_days} days has real readings "
                f"(need at least {int(MIN_RECENT_COVERAGE * 100)}%)."
            )

        # ── Get model ──
        cached = self._get_or_train_model(df_raw, patient_id, last_data_date)
        model = cached["model"]
        eval_metrics = cached["metrics"]
        feature_cols = cached["feature_cols"]
        include_weekly = cached["include_weekly_lag"]
        hour_baseline_map = cached["hour_baseline_map"]
        max_baseline_map = cached["max_baseline_map"]
        min_baseline_map = cached["min_baseline_map"]
        overall_mean = cached["overall_mean"]
        overall_max = cached["overall_max"]
        overall_min = cached["overall_min"]

        df = engineer_features(df_raw, include_weekly_lag=include_weekly)
        if df.empty:
            raise ValueError("Not enough data after feature engineering.")

        series = df.set_index("hour_timestamp")["glucose_mean"].sort_index()

        lag_offsets = list(SHORT_LAG_HOURS)
        if include_weekly:
            lag_offsets.append(168)

        def build_feature_row(target_ts_local: pd.Timestamp, source_series: pd.Series) -> dict:
            """
            Build the feature dict for one target hour.

            Includes lag features, time features, and rate-of-change features.

            ROC features are computed here at prediction time by looking up
            the same lag values already fetched and computing differences:
              roc_1h  = glucose_at_target - glucose_1h_ago
              roc_3h  = glucose_at_target - glucose_3h_ago
              roc_6h  = glucose_at_target - glucose_6h_ago
              roc_24h = glucose_at_target - glucose_24h_ago (same hour yesterday)
              acceleration = roc_1h - (lag_1 - lag_2)

            At prediction time we don't know the actual glucose at the target
            hour yet — so we use the MOST RECENT known glucose as a proxy
            for the current value when computing ROC. This is the best
            approximation available without real data.
            """
            hour = target_ts_local.hour
            dow = target_ts_local.weekday()

            # ── Lag features ──────────────────────────────────────────────
            row = {}
            lag_values = {}  # store for ROC computation below
            for h in lag_offsets:
                lookup_ts = target_ts_local - pd.Timedelta(hours=h)
                if lookup_ts in source_series.index:
                    val = float(source_series.loc[lookup_ts])
                else:
                    val = source_series.asof(lookup_ts)
                    if pd.isna(val):
                        val = source_series.iloc[-1]
                    val = float(val)
                row[f"lag_{h}"] = val
                lag_values[h] = val

            # ── Time features ─────────────────────────────────────────────
            row["hour_baseline"] = float(hour_baseline_map.get(hour, overall_mean))
            row["hour_sin"] = np.sin(2 * np.pi * hour / 24)
            row["hour_cos"] = np.cos(2 * np.pi * hour / 24)
            row["dow_sin"] = np.sin(2 * np.pi * dow / 7)
            row["dow_cos"] = np.cos(2 * np.pi * dow / 7)

            # ── Rate of change features ───────────────────────────────────
            # At prediction time we use lag_1 as a proxy for "current glucose"
            # since the actual glucose at the target hour is unknown.
            # This means:
            #   roc_1h  = lag_1 - lag_2   (change from 2h ago to 1h ago)
            #   roc_3h  = lag_1 - lag_3   (change from 3h ago to 1h ago)
            #   roc_6h  = lag_1 - lag_6   (change from 6h ago to 1h ago)
            #   roc_24h = lag_1 - lag_24  (vs same period yesterday)
            #   acceleration = roc_1h - (lag_2 - lag_3)
            #
            # During training, the actual current glucose_mean was used.
            # At inference, lag_1 (most recent known value) is the best
            # available proxy — consistent with how lag features are used.
            lag_1  = lag_values.get(1,  source_series.iloc[-1])
            lag_2  = lag_values.get(2,  source_series.iloc[-1])
            lag_3  = lag_values.get(3,  source_series.iloc[-1])
            lag_6  = lag_values.get(6,  source_series.iloc[-1])
            lag_24 = lag_values.get(24, source_series.iloc[-1])

            roc_1h  = lag_1 - lag_2
            roc_3h  = lag_1 - lag_3
            roc_6h  = lag_1 - lag_6
            roc_24h = lag_1 - lag_24
            acceleration = roc_1h - (lag_2 - lag_3)

            row["roc_1h"]       = float(roc_1h)
            row["roc_3h"]       = float(roc_3h)
            row["roc_6h"]       = float(roc_6h)
            row["roc_24h"]      = float(roc_24h)
            row["acceleration"] = float(acceleration)

            return row

        def predict_one_day(day_ts: pd.Timestamp, source_series: pd.Series) -> list:
            day_preds = []
            for hour in range(24):
                target_ts_local = day_ts + pd.Timedelta(hours=hour)
                row = build_feature_row(target_ts_local, source_series)

                X_input = np.array([[row[c] for c in feature_cols]])
                dev_pred = model.predict(X_input)[0]
                mean_abs = dev_pred[0] + hour_baseline_map.get(hour, overall_mean)
                max_abs  = dev_pred[1] + max_baseline_map.get(hour, overall_max)
                min_abs  = dev_pred[2] + min_baseline_map.get(hour, overall_min)

                mean_abs = float(np.clip(mean_abs, 40, 400))
                max_abs  = float(np.clip(max_abs,  40, 400))
                min_abs  = float(np.clip(min_abs,  40, 400))

                day_preds.append({
                    "hour": hour,
                    "predicted_mean": round(mean_abs, 2),
                    "predicted_max":  round(max_abs, 2),
                    "predicted_min":  round(min_abs, 2),
                    "_ts": target_ts_local,
                    "_mean_raw": mean_abs,
                })
            return day_preds

        target_day_ts = pd.Timestamp(target_date)
        final_predictions = None

        if days_ahead <= MAX_ITERATIVE_DAYS:
            current_day = pd.Timestamp(last_data_date) + pd.Timedelta(days=1)
            while current_day <= target_day_ts:
                day_preds = predict_one_day(current_day, series)
                new_entries = pd.Series(
                    {p["_ts"]: p["_mean_raw"] for p in day_preds}
                )
                series = pd.concat([series, new_entries]).sort_index()
                if current_day == target_day_ts:
                    final_predictions = day_preds
                current_day += pd.Timedelta(days=1)
        else:
            final_predictions = predict_one_day(target_day_ts, series)

        predictions = [
            {k: v for k, v in p.items() if not k.startswith("_")}
            for p in (final_predictions or [])
        ]

        if target_hour is not None:
            predictions = [p for p in predictions if p["hour"] == target_hour]

        if predictions:
            in_range = sum(
                1 for p in predictions
                if TIR_LOWER_BOUND <= p["predicted_mean"] <= TIR_UPPER_BOUND
            )
            tir_pct = float(round((in_range / len(predictions)) * 100, 2))
        else:
            tir_pct = 0.0

        lag_1w = (target_ts - pd.Timedelta(days=7)).date().isoformat()
        lag_2w = (target_ts - pd.Timedelta(days=14)).date().isoformat()

        return {
            "patient_id": patient_id,
            "target_date": target_date.isoformat(),
            "target_day_name": DAY_NAMES[target_date.weekday()],
            "days_ahead": days_ahead,
            "input_lag_dates": [lag_2w, lag_1w],
            "predictions": predictions,
            "time_in_range_pct": tir_pct,
            "tir_range_mg_dl": [TIR_LOWER_BOUND, TIR_UPPER_BOUND],
            "model_metrics": eval_metrics,
        }


    # ─────────────────────────────
    # HELPER FUNCTIONS
    # ─────────────────────────────
    def _previous_weeks(self, week_str):
        year, week = week_str.split("-")
        base = pd.Timestamp.fromisocalendar(int(year), int(week), 1)
        prev1 = base - pd.Timedelta(days=7)
        prev2 = base - pd.Timedelta(days=14)
        return [
            f"{prev2.year}-{str(prev2.isocalendar().week).zfill(2)}",
            f"{prev1.year}-{str(prev1.isocalendar().week).zfill(2)}",
        ]

    def _next_iso_week(self, week_str):
        year, week = week_str.split("-")
        base = pd.Timestamp.fromisocalendar(int(year), int(week), 1)
        next_week = base + pd.Timedelta(days=7)
        return f"{next_week.year}-{str(next_week.isocalendar().week).zfill(2)}"


# Singleton
prediction_service = GlucosePredictionService()