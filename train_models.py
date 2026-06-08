# -*- coding: utf-8 -*-
"""
MATRIX AI - Model Training Script
Trains all ML models for material property prediction, MQI calculation,
commodity price forecasting, and inverse design.
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

# Optional: XGBoost
try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not available, using GradientBoosting instead")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

DATA = {
    "ds1": os.path.join(BASE_DIR, "DS1_material_properties_5500.csv"),
    "ds2": os.path.join(BASE_DIR, "DS2_commodity_prices_10yr.csv"),
    "ds3": os.path.join(BASE_DIR, "DS3_crossdomain_features_daily.csv"),
    "ds4": os.path.join(BASE_DIR, "DS4_ mqi_weights.csv"),
    "ds5": os.path.join(BASE_DIR, "DS5_element_prices_monthly.csv"),
}

def mae_rmse(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

# ─────────────────────────────────────────────
# TASK 1: MATERIAL PROPERTY MODELING
# ─────────────────────────────────────────────
def train_material_models():
    print("\n" + "="*60)
    print("TASK 1: Material Property Modeling")
    print("="*60)

    ds1 = pd.read_csv(DATA["ds1"])
    ds4 = pd.read_csv(DATA["ds4"])

    # ---- Encoders ----
    le_crystal = LabelEncoder()
    le_category = LabelEncoder()
    ds1["crystal_enc"] = le_crystal.fit_transform(ds1["crystal_system"])
    ds1["category_enc"] = le_category.fit_transform(ds1["category"])

    joblib.dump(le_crystal, os.path.join(MODEL_DIR, "le_crystal.pkl"))
    joblib.dump(le_category, os.path.join(MODEL_DIR, "le_category.pkl"))
    joblib.dump(le_crystal.classes_.tolist(), os.path.join(MODEL_DIR, "crystal_classes.pkl"))
    joblib.dump(le_category.classes_.tolist(), os.path.join(MODEL_DIR, "category_classes.pkl"))

    # ---- Feature sets ----
    STRUCT_FEATURES = ["n_elements", "crystal_enc", "category_enc", "spacegroup_number", "nsites", "volume_A3"]
    TARGETS = {
        "bulk_modulus_GPa": "Bulk Modulus (GPa)",
        "shear_modulus_GPa": "Shear Modulus (GPa)",
        "band_gap_eV": "Band Gap (eV)",
        "density_g_cm3": "Density (g/cm³)",
        "formation_energy_per_atom_eV": "Formation Energy (eV/atom)",
        "melting_point_K": "Melting Point (K)",
        "poisson_ratio": "Poisson Ratio",
        "energy_above_hull_eV": "Energy Above Hull (eV)",
    }

    property_models = {}
    metrics = {}

    X = ds1[STRUCT_FEATURES]
    for col, label in TARGETS.items():
        y = ds1[col]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        if HAS_XGB:
            model = XGBRegressor(n_estimators=300, max_depth=7, learning_rate=0.05,
                                  subsample=0.85, colsample_bytree=0.85, random_state=42)
        else:
            model = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        mae, rmse, r2 = mae_rmse(y_te, pred)
        print(f"  {label:40s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
        property_models[col] = model
        metrics[col] = {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}

    joblib.dump(property_models, os.path.join(MODEL_DIR, "property_models.pkl"))
    joblib.dump(STRUCT_FEATURES, os.path.join(MODEL_DIR, "struct_features.pkl"))

    # ---- Stability classifier ----
    ds1["stability_score"] = (
        (ds1["formation_energy_per_atom_eV"] < 0).astype(int) +
        (ds1["energy_above_hull_eV"] < 0.1).astype(int) +
        (ds1["bulk_modulus_GPa"] > 0).astype(int) +
        (ds1["shear_modulus_GPa"] > 0).astype(int)
    )
    y_stab = ds1["is_stable"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_stab, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    print(f"  {'Stability Classifier':40s}  Accuracy={acc:.4f}")
    joblib.dump(clf, os.path.join(MODEL_DIR, "stability_classifier.pkl"))

    # ---- MQI weights ----
    mqi_weights = dict(zip(ds4["Property"], ds4["Weights"]))
    MQI_MAP = {
        "Bulk Modulus (K)": "bulk_modulus_GPa",
        "Shear Modulus (G)": "shear_modulus_GPa",
        "Formation Energy": "formation_energy_per_atom_eV",
        "Density": "density_g_cm3",
        "Melting Point": "melting_point_K",
        "Band Gap": "band_gap_eV",
    }
    joblib.dump({"weights": mqi_weights, "mapping": MQI_MAP}, os.path.join(MODEL_DIR, "mqi_config.pkl"))

    # Compute DS1 MQI scores for normalisation
    scaler_mqi = MinMaxScaler()
    mqi_cols = ["bulk_modulus_GPa","shear_modulus_GPa","formation_energy_per_atom_eV","density_g_cm3","melting_point_K","band_gap_eV"]
    ds1_mqi = ds1[mqi_cols].copy()
    # For formation energy: lower (more negative) is better → invert
    ds1_mqi["formation_energy_per_atom_eV"] = -ds1_mqi["formation_energy_per_atom_eV"]
    scaler_mqi.fit(ds1_mqi)
    joblib.dump(scaler_mqi, os.path.join(MODEL_DIR, "mqi_scaler.pkl"))
    joblib.dump(mqi_cols, os.path.join(MODEL_DIR, "mqi_cols.pkl"))

    joblib.dump(metrics, os.path.join(MODEL_DIR, "property_metrics.pkl"))
    print("  ✓ Material models saved.")
    return metrics

# ─────────────────────────────────────────────
# TASK 2: COMMODITY PRICE FORECASTING
# ─────────────────────────────────────────────
def train_commodity_models():
    print("\n" + "="*60)
    print("TASK 2: Commodity Price Forecasting")
    print("="*60)

    ds2 = pd.read_csv(DATA["ds2"], parse_dates=["date"])
    ds3 = pd.read_csv(DATA["ds3"], parse_dates=["date"])

    ds2 = ds2.sort_values(["commodity", "date"]).reset_index(drop=True)
    ds3 = ds3.sort_values(["commodity", "date"]).reset_index(drop=True)
    merged = pd.merge(ds2, ds3, on=["date", "commodity"], how="left")

    commodities = merged["commodity"].unique().tolist()
    joblib.dump(commodities, os.path.join(MODEL_DIR, "commodities.pkl"))

    FEATURE_COLS = [
        "open","high","low","volume","daily_return","return_5d","return_21d",
        "volatility_5d_ann","volatility_21d_ann","sma_21","sma_63",
        "bollinger_z","rsi_14","macd","macd_signal","momentum_10d","momentum_21d",
        "mqi","supply_disruption_prob","substitution_elasticity","green_premium_per_kg",
        "herfindahl_index","mqi_5d_trend","mqi_21d_trend","mqi_63d_trend"
    ]

    HORIZONS = [1, 7, 30]  # Forecast horizons in days
    commodity_models = {}
    commodity_scalers = {}
    commodity_metrics = {}
    commodity_history = {}

    for comm in commodities:
        df = merged[merged["commodity"] == comm].copy()
        df = df.dropna(subset=["close"])
        df = df.fillna(method="ffill").fillna(0)

        # Save last N rows for history chart
        hist = df[["date","close","volume","rsi_14","macd"]].tail(252).copy()
        hist["date"] = hist["date"].dt.strftime("%Y-%m-%d")
        commodity_history[comm] = hist.to_dict(orient="records")

        feats_avail = [f for f in FEATURE_COLS if f in df.columns]

        scaler = StandardScaler()
        X_all = df[feats_avail].values
        X_all = scaler.fit_transform(X_all)
        commodity_scalers[comm] = (scaler, feats_avail)

        horizon_models = {}
        horizon_metrics = {}

        for h in HORIZONS:
            df[f"target_{h}d"] = df["close"].shift(-h)
            df_h = df.dropna(subset=[f"target_{h}d"])
            X = X_all[:len(df_h)]
            y = df_h[f"target_{h}d"].values
            split = int(len(X) * 0.8)
            X_tr, X_te = X[:split], X[split:]
            y_tr, y_te = y[:split], y[split:]

            if HAS_XGB:
                model = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.03,
                                      subsample=0.8, colsample_bytree=0.8, random_state=42)
            else:
                model = GradientBoostingRegressor(n_estimators=400, max_depth=5,
                                                   learning_rate=0.03, random_state=42)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            mae, rmse, r2 = mae_rmse(y_te, pred)
            print(f"  {comm:20s} horizon={h:2d}d  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")
            horizon_models[h] = model
            horizon_metrics[h] = {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 4)}

        commodity_models[comm] = horizon_models
        commodity_metrics[comm] = horizon_metrics

    joblib.dump(commodity_models, os.path.join(MODEL_DIR, "commodity_models.pkl"))
    joblib.dump(commodity_scalers, os.path.join(MODEL_DIR, "commodity_scalers.pkl"))
    joblib.dump(commodity_metrics, os.path.join(MODEL_DIR, "commodity_metrics.pkl"))
    joblib.dump(commodity_history, os.path.join(MODEL_DIR, "commodity_history.pkl"))
    print("  ✓ Commodity models saved.")
    return commodity_metrics

# ─────────────────────────────────────────────
# TASK 3: CROSS-DOMAIN CORRELATION
# ─────────────────────────────────────────────
def compute_cross_domain():
    print("\n" + "="*60)
    print("TASK 3: Cross-Domain Correlation Analysis")
    print("="*60)

    ds2 = pd.read_csv(DATA["ds2"], parse_dates=["date"])
    ds3 = pd.read_csv(DATA["ds3"], parse_dates=["date"])
    merged = pd.merge(ds2, ds3, on=["date","commodity"], how="left")

    correlations = {}
    for comm in merged["commodity"].unique():
        df = merged[merged["commodity"] == comm].dropna()
        corr_cols = ["mqi","supply_disruption_prob","substitution_elasticity",
                     "green_premium_per_kg","herfindahl_index","rsi_14","macd",
                     "bollinger_z","volatility_21d_ann"]
        corr_cols = [c for c in corr_cols if c in df.columns]
        corr = df[corr_cols + ["close"]].corr()["close"].drop("close")
        correlations[comm] = corr.to_dict()

    joblib.dump(correlations, os.path.join(MODEL_DIR, "correlations.pkl"))
    print("  Correlations (top features with price):")
    for comm, corr_dict in correlations.items():
        top = sorted(corr_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        print(f"    {comm}: " + ", ".join([f"{k}={v:.3f}" for k,v in top]))
    print("  ✓ Cross-domain correlations saved.")
    return correlations

# ─────────────────────────────────────────────
# TASK 4: ELEMENT PRICE FORECASTING (DS5)
# ─────────────────────────────────────────────
def train_element_models():
    print("\n" + "="*60)
    print("TASK 4/Bonus: Element Price Forecasting")
    print("="*60)

    ds5 = pd.read_csv(DATA["ds5"], parse_dates=["date"])
    ds5 = ds5.sort_values(["element","date"]).reset_index(drop=True)
    elements = ds5["element"].unique().tolist()
    joblib.dump(elements, os.path.join(MODEL_DIR, "elements.pkl"))

    element_models = {}
    element_history = {}

    for elem in elements:
        df = ds5[ds5["element"] == elem].copy().fillna(0)
        hist = df[["date","price_usd_per_kg"]].tail(36).copy()
        hist["date"] = hist["date"].dt.strftime("%Y-%m-%d")
        element_history[elem] = hist.to_dict(orient="records")

        if len(df) < 12:
            continue

        # Create lag features
        for lag in [1, 2, 3, 6, 12]:
            df[f"lag_{lag}"] = df["price_usd_per_kg"].shift(lag)
        df = df.dropna()

        X_cols = ["base_price","monthly_return"] + [f"lag_{l}" for l in [1,2,3,6,12]]
        X_cols = [c for c in X_cols if c in df.columns]
        X = df[X_cols].values
        y = df["price_usd_per_kg"].values

        split = int(len(X)*0.8)
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X[:split], y[:split])
        element_models[elem] = (model, X_cols, df[X_cols + ["price_usd_per_kg"]].iloc[-1:].values)
        mae, rmse, r2 = mae_rmse(y[split:], model.predict(X[split:]))
        print(f"  {elem:5s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

    joblib.dump(element_models, os.path.join(MODEL_DIR, "element_models.pkl"))
    joblib.dump(element_history, os.path.join(MODEL_DIR, "element_history.pkl"))
    print("  ✓ Element price models saved.")

# ─────────────────────────────────────────────
# BONUS: INVERSE DESIGN PREP
# ─────────────────────────────────────────────
def prepare_inverse_design():
    print("\n" + "="*60)
    print("BONUS: Inverse Design Preparation")
    print("="*60)

    ds1 = pd.read_csv(DATA["ds1"])
    ds5 = pd.read_csv(DATA["ds5"])

    # Latest element prices
    latest_prices = ds5.sort_values("date").groupby("element").last()["price_usd_per_kg"].to_dict()
    joblib.dump(latest_prices, os.path.join(MODEL_DIR, "element_latest_prices.pkl"))

    # Save DS1 canonical stats for normalisation  
    stats = ds1[["bulk_modulus_GPa","shear_modulus_GPa","band_gap_eV",
                  "density_g_cm3","formation_energy_per_atom_eV","melting_point_K"]].describe()
    joblib.dump(stats, os.path.join(MODEL_DIR, "ds1_stats.pkl"))

    # Save full DS1 for candidate search
    le_crystal = joblib.load(os.path.join(MODEL_DIR, "le_crystal.pkl"))
    le_category = joblib.load(os.path.join(MODEL_DIR, "le_category.pkl"))
    ds1["crystal_enc"] = le_crystal.transform(ds1["crystal_system"])
    ds1["category_enc"] = le_category.transform(ds1["category"])
    joblib.dump(ds1, os.path.join(MODEL_DIR, "ds1_full.pkl"))

    print("  ✓ Inverse design data prepared.")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("       MATRIX AI - Model Training Suite")
    print("=" * 50)
    prop_metrics = train_material_models()
    comm_metrics = train_commodity_models()
    correlations = compute_cross_domain()
    train_element_models()
    prepare_inverse_design()

    # Save combined summary
    summary = {
        "property_metrics": prop_metrics,
        "commodity_metrics": comm_metrics,
    }
    joblib.dump(summary, os.path.join(MODEL_DIR, "training_summary.pkl"))

    print("\n" + "=" * 50)
    print("         ALL MODELS TRAINED SUCCESSFULLY")
    print("=" * 50)
    print(f"Models saved to: {MODEL_DIR}")
