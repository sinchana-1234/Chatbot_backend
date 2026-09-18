"""
Diagnostic script: load patient 489's data, train a model, and inspect
what the model actually sees and how it weights its features.
Run from project root: python diagnose.py
"""
import os
from datetime import date
from dotenv import load_dotenv

load_dotenv()

from database import SessionLocal
from prediction_service import prediction_service

PATIENT_ID = "489"
TARGET_DATE = date(2026, 6, 20)

# Load data and trigger a prediction (this populates the cache)
db = SessionLocal()
try:
    df_raw = prediction_service.load_patient_data(db, int(PATIENT_ID))
    print(f"Loaded {len(df_raw)} raw rows for patient {PATIENT_ID}")
    print(f"Date range: {df_raw['timestamp'].min()} to {df_raw['timestamp'].max()}")
    print()

    # Trigger a prediction to populate cache
    result = prediction_service.predict(df_raw, PATIENT_ID, TARGET_DATE)
    print(f"Prediction completed. days_ahead = {result['days_ahead']}")
    print()
finally:
    db.close()

# Now inspect the cached model
cached = prediction_service._cache.get(PATIENT_ID)
if cached is None:
    print("ERROR: no cache entry found for", PATIENT_ID)
    raise SystemExit(1)

print("=" * 60)
print("FEATURE COLUMNS USED:")
print("=" * 60)
for c in cached["feature_cols"]:
    print(f"  {c}")
print()

print("=" * 60)
print("INCLUDE WEEKLY LAG:", cached["include_weekly_lag"])
print("=" * 60)
print()

print("=" * 60)
print("HOUR BASELINE MAP (should vary across hours):")
print("=" * 60)
hb = cached["hour_baseline_map"]
for h in sorted(hb.keys()):
    print(f"  hour {h:2d}: {hb[h]:.2f}")
print()
print(f"  range: {min(hb.values()):.2f} - {max(hb.values()):.2f}")
print(f"  span:  {max(hb.values()) - min(hb.values()):.2f}")
print()

print("=" * 60)
print("FEATURE IMPORTANCES (RandomForest internal weights):")
print("=" * 60)
model = cached["model"]
imps = list(zip(cached["feature_cols"], model.feature_importances_))
imps.sort(key=lambda x: x[1], reverse=True)
for name, imp in imps:
    bar = "#" * int(imp * 80)
    print(f"  {name:18s}: {imp:.4f}  {bar}")
print()

print("=" * 60)
print("TRAINING METRICS:", cached["metrics"])
print("=" * 60)