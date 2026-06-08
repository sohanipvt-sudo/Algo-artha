"""
MATRIX AI - FastAPI Backend
Complete REST API for all tasks: material prediction, commodity forecasting,
cross-domain analysis, MQI calculation, element prices, and inverse design.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import joblib
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = FastAPI(
    title="MATRIX AI API",
    description="Predicting Commodity Markets using Material Science Signals",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ─────────────────────────────────────────────
# LOAD MODELS (lazy, on first request)
# ─────────────────────────────────────────────
_cache = {}

def load(name):
    if name not in _cache:
        path = os.path.join(MODEL_DIR, f"{name}.pkl")
        if not os.path.exists(path):
            raise HTTPException(status_code=503, detail=f"Model {name} not found. Run train_models.py first.")
        _cache[name] = joblib.load(path)
    return _cache[name]

# ─────────────────────────────────────────────
# PYDANTIC SCHEMAS
# ─────────────────────────────────────────────
class MaterialInput(BaseModel):
    n_elements: int = Field(3, ge=1, le=10, description="Number of elements (1-10)")
    crystal_system: str = Field("cubic", description="Crystal system")
    category: str = Field("Metal", description="Material category")
    spacegroup_number: int = Field(225, ge=1, le=230)
    nsites: int = Field(4, ge=1, le=20)
    volume_A3: float = Field(64.0, gt=0, description="Unit cell volume in Å³")
    formula: str = Field("Fe2O3", description="Chemical formula")

class CommodityForecastInput(BaseModel):
    commodity: str = Field("Copper", description="Commodity name")
    horizon: int = Field(30, ge=1, le=30, description="Forecast horizon in days")
    mqi_override: Optional[float] = Field(None, ge=0, le=100, description="Optional MQI override")
    disruption_prob: Optional[float] = Field(None, ge=0.0, le=1.0)

class InverseDesignInput(BaseModel):
    target_bulk_modulus: Optional[float] = Field(None, ge=0)
    target_shear_modulus: Optional[float] = Field(None, ge=0)
    target_band_gap: Optional[float] = Field(None, ge=0)
    target_melting_point: Optional[float] = Field(None, ge=0)
    max_cost_per_kg: Optional[float] = Field(None, gt=0)
    crystal_system: Optional[str] = None
    category: Optional[str] = None
    top_n: int = Field(10, ge=1, le=50)

class MQIInput(BaseModel):
    bulk_modulus: float = Field(..., ge=0)
    shear_modulus: float = Field(..., ge=0)
    formation_energy: float
    density: float = Field(..., gt=0)
    melting_point: float = Field(..., gt=0)
    band_gap: float = Field(..., ge=0)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def compute_mqi(bulk, shear, fe, density, mp, bg, scaler, cols, weights_cfg):
    raw = np.array([[bulk, shear, -fe, density, mp, bg]])  # negate fe for "more negative = better"
    df_raw = pd.DataFrame(raw, columns=cols)
    scaled = scaler.transform(df_raw)
    w = weights_cfg["weights"]
    w_vec = np.array([
        w.get("Bulk Modulus (K)", 0.20),
        w.get("Shear Modulus (G)", 0.20),
        w.get("Formation Energy", 0.20),
        w.get("Density", 0.10),
        w.get("Melting Point", 0.15),
        w.get("Band Gap", 0.15),
    ])
    mqi = float(np.clip(np.dot(scaled[0], w_vec) * 100, 0, 100))
    return round(mqi, 2)

def check_stability(bulk, shear, fe, poisson=None):
    flags = []
    stable = True
    if fe >= 0:
        flags.append("Formation energy must be < 0 eV/atom for thermodynamic stability")
        stable = False
    if bulk <= 0:
        flags.append("Bulk Modulus must be > 0 GPa")
        stable = False
    if shear <= 0:
        flags.append("Shear Modulus must be > 0 GPa")
        stable = False
    if poisson is not None and not (-1 <= poisson <= 0.5):
        flags.append("Poisson ratio must be in [-1, 0.5]")
        stable = False
    return {"physically_stable": stable, "flags": flags}

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(
        os.path.join(BASE_DIR, "index.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )

@app.get("/logo.jpeg", include_in_schema=False)
async def get_logo():
    return FileResponse(os.path.join(BASE_DIR, "logo.jpeg"))

# ─── Info ───

@app.get("/api/info", tags=["Meta"])
def get_api_info():
    """Returns metadata: available commodities, crystal systems, categories, etc."""
    comm = load("commodities")
    crystals = load("crystal_classes")
    cats = load("category_classes")
    elements = load("elements")
    return {
        "commodities": comm,
        "crystal_systems": crystals.tolist() if hasattr(crystals, 'tolist') else list(crystals),
        "categories": cats.tolist() if hasattr(cats, 'tolist') else list(cats),
        "elements": elements,
        "forecast_horizons": [1, 7, 30],
    }

@app.get("/api/metrics", tags=["Meta"])
def get_training_metrics():
    """Returns model training metrics (MAE, RMSE, R²)."""
    try:
        summary = load("training_summary")
        return summary
    except:
        return {"error": "Training summary not available"}

# ─── Task 1: Material Property Prediction ───

@app.post("/api/predict/properties", tags=["Task 1 - Material Modeling"])
def predict_properties(data: MaterialInput):
    """
    Predict all material properties from structural inputs.
    Returns predicted properties, MQI, stability assessment, and physical validity.
    """
    le_crystal = load("le_crystal")
    le_category = load("le_category")
    prop_models = load("property_models")
    struct_feats = load("struct_features")
    mqi_cfg = load("mqi_config")
    mqi_scaler = load("mqi_scaler")
    mqi_cols = load("mqi_cols")

    # Encode
    try:
        cryst_enc = le_crystal.transform([data.crystal_system])[0]
    except ValueError:
        raise HTTPException(400, f"Unknown crystal_system. Valid: {list(le_crystal.classes_)}")
    try:
        cat_enc = le_category.transform([data.category])[0]
    except ValueError:
        raise HTTPException(400, f"Unknown category. Valid: {list(le_category.classes_)}")

    X = np.array([[data.n_elements, cryst_enc, cat_enc,
                   data.spacegroup_number, data.nsites, data.volume_A3]])

    predictions = {}
    for col, model in prop_models.items():
        predictions[col] = round(float(model.predict(X)[0]), 4)

    # Stability check
    stability = check_stability(
        predictions.get("bulk_modulus_GPa", 0),
        predictions.get("shear_modulus_GPa", 0),
        predictions.get("formation_energy_per_atom_eV", 0),
        predictions.get("poisson_ratio", None)
    )

    # MQI
    mqi = compute_mqi(
        predictions.get("bulk_modulus_GPa", 100),
        predictions.get("shear_modulus_GPa", 50),
        predictions.get("formation_energy_per_atom_eV", -1),
        predictions.get("density_g_cm3", 5),
        predictions.get("melting_point_K", 1500),
        predictions.get("band_gap_eV", 1),
        mqi_scaler, mqi_cols, mqi_cfg
    )

    # Stability classifier
    try:
        clf = load("stability_classifier")
        is_stable_pred = bool(clf.predict(X)[0])
    except:
        is_stable_pred = stability["physically_stable"]

    return {
        "input": data.dict(),
        "predicted_properties": {
            "bulk_modulus_GPa": predictions.get("bulk_modulus_GPa"),
            "shear_modulus_GPa": predictions.get("shear_modulus_GPa"),
            "band_gap_eV": predictions.get("band_gap_eV"),
            "density_g_cm3": predictions.get("density_g_cm3"),
            "formation_energy_per_atom_eV": predictions.get("formation_energy_per_atom_eV"),
            "melting_point_K": predictions.get("melting_point_K"),
            "poisson_ratio": predictions.get("poisson_ratio"),
            "energy_above_hull_eV": predictions.get("energy_above_hull_eV"),
        },
        "material_quality_index": mqi,
        "mqi_grade": "Excellent" if mqi >= 75 else "Good" if mqi >= 55 else "Fair" if mqi >= 35 else "Poor",
        "stability": {
            **stability,
            "ml_predicted_stable": is_stable_pred,
        }
    }

@app.post("/api/predict/mqi", tags=["Task 1 - Material Modeling"])
def calculate_mqi(data: MQIInput):
    """Calculate MQI directly from known property values."""
    mqi_cfg = load("mqi_config")
    mqi_scaler = load("mqi_scaler")
    mqi_cols = load("mqi_cols")

    mqi = compute_mqi(
        data.bulk_modulus, data.shear_modulus, data.formation_energy,
        data.density, data.melting_point, data.band_gap,
        mqi_scaler, mqi_cols, mqi_cfg
    )
    stability = check_stability(data.bulk_modulus, data.shear_modulus, data.formation_energy)
    return {
        "mqi": mqi,
        "grade": "Excellent" if mqi >= 75 else "Good" if mqi >= 55 else "Fair" if mqi >= 35 else "Poor",
        "stability": stability,
        "weights_used": mqi_cfg["weights"]
    }

# ─── Task 2: Commodity Forecasting ───

@app.post("/api/forecast/commodity", tags=["Task 2 - Commodity Forecasting"])
def forecast_commodity(data: CommodityForecastInput):
    """
    Forecast commodity price for chosen horizon.
    Supports MQI and disruption probability override for scenario analysis.
    """
    comm_models = load("commodity_models")
    comm_scalers = load("commodity_scalers")
    comm_metrics = load("commodity_metrics")
    comm_history = load("commodity_history")

    if data.commodity not in comm_models:
        raise HTTPException(400, f"Unknown commodity. Valid: {list(comm_models.keys())}")

    h = min([1, 7, 30], key=lambda x: abs(x - data.horizon))
    model = comm_models[data.commodity][h]
    scaler, feat_cols = comm_scalers[data.commodity]

    # Build a representative feature vector from recent data
    hist = comm_history.get(data.commodity, [])
    if hist:
        last = hist[-1]
        feat_defaults = {
            "open": last.get("close", 100),
            "high": last.get("close", 100) * 1.01,
            "low": last.get("close", 100) * 0.99,
            "volume": 1000,
            "daily_return": 0,
            "return_5d": 0,
            "return_21d": 0,
            "volatility_5d_ann": 0.2,
            "volatility_21d_ann": 0.2,
            "sma_21": last.get("close", 100),
            "sma_63": last.get("close", 100),
            "bollinger_z": 0,
            "rsi_14": last.get("rsi_14", 50),
            "macd": last.get("macd", 0),
            "macd_signal": 0,
            "momentum_10d": 0,
            "momentum_21d": 0,
            "mqi": data.mqi_override if data.mqi_override else 65,
            "supply_disruption_prob": data.disruption_prob if data.disruption_prob else 0.15,
            "substitution_elasticity": 0.4,
            "green_premium_per_kg": 0.15,
            "herfindahl_index": 0.1,
            "mqi_5d_trend": 0,
            "mqi_21d_trend": 0,
            "mqi_63d_trend": 0,
        }
    else:
        feat_defaults = {c: 0 for c in feat_cols}

    X_row = np.array([[feat_defaults.get(f, 0) for f in feat_cols]])
    X_scaled = scaler.transform(X_row)
    price_pred = float(model.predict(X_scaled)[0])

    # Recent prices for chart
    historical_prices = [{"date": r["date"], "close": r["close"]} for r in hist[-60:]]
    metrics = comm_metrics[data.commodity].get(h, {})

    # Generate 30-day forecast path
    forecast_path = []
    base = historical_prices[-1]["close"] if historical_prices else price_pred
    for day in range(1, 31):
        ratio = day / 30
        noise = np.random.normal(0, metrics.get("rmse", 5) * 0.05)
        val = base + (price_pred - base) * ratio + noise
        forecast_path.append({"day": day, "predicted_price": round(val, 2)})  

    return {
        "commodity": data.commodity,
        "horizon_days": h,
        "predicted_price": round(price_pred, 2),
        "current_price": historical_prices[-1]["close"] if historical_prices else None,
        "price_change_pct": round((price_pred - historical_prices[-1]["close"]) / historical_prices[-1]["close"] * 100, 2) if historical_prices else None,
        "model_metrics": metrics,
        "historical_prices": historical_prices,
        "forecast_path": forecast_path,
    }

@app.get("/api/commodities/history/{commodity}", tags=["Task 2 - Commodity Forecasting"])
def get_commodity_history(commodity: str, days: int = Query(90, ge=7, le=500)):
    """Get historical price data for a commodity."""
    comm_history = load("commodity_history")
    if commodity not in comm_history:
        raise HTTPException(404, f"Commodity {commodity} not found.")
    hist = comm_history[commodity][-days:]
    return {"commodity": commodity, "history": hist, "total_records": len(hist)}

# ─── Task 3: Cross-Domain Correlations ───

@app.get("/api/correlations", tags=["Task 3 - Cross Domain"])
def get_correlations(commodity: Optional[str] = None):
    """Get material-market correlations for a commodity (or all)."""
    correlations = load("correlations")
    clean_corrs = {c: {k: (0.0 if pd.isna(v) else v) for k, v in feats.items()} for c, feats in correlations.items()}
    if commodity:
        if commodity not in clean_corrs:
            raise HTTPException(404, f"Commodity {commodity} not found.")
        return {"commodity": commodity, "correlations": clean_corrs[commodity]}
    return {"all_commodities": clean_corrs}

@app.get("/api/correlations/matrix", tags=["Task 3 - Cross Domain"])
def get_correlation_matrix():
    """Full correlation matrix for heatmap visualization."""
    correlations = load("correlations")
    clean_corrs = {c: {k: (0.0 if pd.isna(v) else v) for k, v in feats.items()} for c, feats in correlations.items()}
    features = ["mqi", "supply_disruption_prob", "substitution_elasticity",
                "green_premium_per_kg", "herfindahl_index", "rsi_14",
                "macd", "bollinger_z", "volatility_21d_ann"]
    commodities = list(clean_corrs.keys())
    matrix = []
    for feat in features:
        row = {"feature": feat}
        for comm in commodities:
            row[comm] = round(clean_corrs[comm].get(feat, 0.0), 4)
        matrix.append(row)
    return {"matrix": matrix, "features": features, "commodities": commodities}

# ─── Task 4: Element Prices ───

@app.get("/api/elements/history/{element}", tags=["Task 4 - Element Prices"])
def get_element_history(element: str):
    """Get historical price data for a specific element."""
    elem_history = load("element_history")
    if element not in elem_history:
        raise HTTPException(404, f"Element {element} not found.")
    return {"element": element, "history": elem_history[element]}

@app.get("/api/elements/prices", tags=["Task 4 - Element Prices"])
def get_current_element_prices():
    """Get latest element prices for all elements."""
    elem_prices = load("element_latest_prices")
    elem_history = load("element_history")
    result = []
    for elem, price in elem_prices.items():
        hist = elem_history.get(elem, [])
        prev = hist[-2]["price_usd_per_kg"] if len(hist) >= 2 else price
        result.append({
            "element": elem,
            "price_usd_per_kg": round(float(price), 4),
            "monthly_change_pct": round((price - prev) / prev * 100, 2) if prev else 0
        })
    return {"elements": sorted(result, key=lambda x: x["element"])}

# ─── Bonus: Inverse Design ───

@app.post("/api/inverse-design", tags=["Bonus - Inverse Design"])
def inverse_design(data: InverseDesignInput):
    """
    Find materials matching target properties at lowest cost.
    Uses DS1 database + DS5 element prices.
    """
    ds1 = load("ds1_full")
    elem_prices = load("element_latest_prices")
    mqi_cfg = load("mqi_config")
    mqi_scaler = load("mqi_scaler")
    mqi_cols = load("mqi_cols")

    df = ds1.copy()

    has_targets = any([
        data.target_bulk_modulus, data.target_shear_modulus,
        data.target_band_gap, data.target_melting_point
    ])

    # Filter by targets (with 40% tolerance for better results)
    if data.target_bulk_modulus:
        tol = max(data.target_bulk_modulus * 0.4, 10)
        df = df[abs(df["bulk_modulus_GPa"] - data.target_bulk_modulus) <= tol]
    if data.target_shear_modulus:
        tol = max(data.target_shear_modulus * 0.4, 5)
        df = df[abs(df["shear_modulus_GPa"] - data.target_shear_modulus) <= tol]
    if data.target_band_gap:
        tol = max(data.target_band_gap * 0.5, 0.8)
        df = df[abs(df["band_gap_eV"] - data.target_band_gap) <= tol]
    if data.target_melting_point:
        tol = max(data.target_melting_point * 0.25, 100)
        df = df[abs(df["melting_point_K"] - data.target_melting_point) <= tol]
    if data.crystal_system:
        df = df[df["crystal_system"] == data.crystal_system]
    if data.category:
        df = df[df["category"] == data.category]

    # Stability: physical constraints (only filter truly invalid)
    df = df[
        (df["formation_energy_per_atom_eV"] < 0) &
        (df["bulk_modulus_GPa"] >= 0) &
        (df["shear_modulus_GPa"] >= 0)
    ]

    if df.empty:
        return {"candidates": [], "message": "No materials match the given constraints. Try relaxing filters (wider tolerances or fewer constraints)."}

    # Fix: correctly parse chemical formulas like Ta3Au3H2, Fe2O3, NaCl
    def estimate_cost(formula_str):
        import re
        cost = 0.0
        formula_str = str(formula_str).strip()
        # Match element symbol + optional count, e.g. 'Ta3', 'Au', 'H2'
        matches = re.findall(r'([A-Z][a-z]?)(\d+\.?\d*)?', formula_str)
        for elem, count_str in matches:
            if not elem:
                continue
            count = float(count_str) if count_str else 1.0
            price = float(elem_prices.get(elem, 50.0))  # default $50/kg
            cost += price * count
        return round(cost, 4)

    df = df.copy()
    df["estimated_cost"] = df["formula"].apply(estimate_cost)

    if data.max_cost_per_kg:
        df = df[df["estimated_cost"] <= data.max_cost_per_kg]

    if df.empty:
        return {"candidates": [], "message": "No materials within cost constraint. Try increasing the max cost."}

    # Score: normalize each property vs target
    df["score"] = 0.0

    if has_targets:
        if data.target_bulk_modulus:
            df["score"] += 1 - abs(df["bulk_modulus_GPa"] - data.target_bulk_modulus) / max(data.target_bulk_modulus, 1)
        if data.target_shear_modulus:
            df["score"] += 1 - abs(df["shear_modulus_GPa"] - data.target_shear_modulus) / max(data.target_shear_modulus, 1)
        if data.target_band_gap:
            df["score"] += 1 - abs(df["band_gap_eV"] - data.target_band_gap) / max(data.target_band_gap, 0.5)
        if data.target_melting_point:
            df["score"] += 1 - abs(df["melting_point_K"] - data.target_melting_point) / max(data.target_melting_point, 1)
    else:
        # No targets: rank by MQI (best overall materials)
        for _, grp in df.groupby(df.index // max(len(df), 1)):
            pass
        try:
            mqi_data = df[mqi_cols].copy()
            mqi_data["formation_energy_per_atom_eV"] = -mqi_data["formation_energy_per_atom_eV"]
            mqi_data = mqi_data.fillna(0)
            scaled = mqi_scaler.transform(mqi_data)
            w = mqi_cfg["weights"]
            w_vec = np.array([
                w.get("Bulk Modulus (K)", 0.20), w.get("Shear Modulus (G)", 0.20),
                w.get("Formation Energy", 0.20), w.get("Density", 0.10),
                w.get("Melting Point", 0.15), w.get("Band Gap", 0.15),
            ])
            df["score"] = np.clip(np.dot(scaled, w_vec) * 100, 0, 100)
        except Exception:
            df["score"] = -df["formation_energy_per_atom_eV"]  # fallback: most stable

    # Penalize cost (normalized)
    max_cost = df["estimated_cost"].max()
    if max_cost > 0:
        df["cost_penalty"] = df["estimated_cost"] / max_cost
        df["score"] -= df["cost_penalty"] * 0.3

    df = df.sort_values("score", ascending=False).head(data.top_n)

    candidates = []
    for _, row in df.iterrows():
        candidates.append({
            "material_id": str(row.get("material_id", "unknown")),
            "formula": str(row.get("formula", "")),
            "crystal_system": str(row.get("crystal_system", "")),
            "category": str(row.get("category", "")),
            "bulk_modulus_GPa": round(float(row["bulk_modulus_GPa"]), 2),
            "shear_modulus_GPa": round(float(row["shear_modulus_GPa"]), 2),
            "band_gap_eV": round(float(row["band_gap_eV"]), 2),
            "melting_point_K": round(float(row["melting_point_K"]), 1),
            "formation_energy_per_atom_eV": round(float(row["formation_energy_per_atom_eV"]), 4),
            "density_g_cm3": round(float(row["density_g_cm3"]), 3),
            "estimated_cost_usd_per_fu": round(float(row["estimated_cost"]), 4),
            "match_score": round(float(row["score"]), 4),
            "physically_stable": bool(row.get("is_stable", 1)),
        })

    return {
        "total_candidates_found": len(candidates),
        "filters_applied": data.dict(exclude_none=True),
        "ranked_by": "target_match_score" if has_targets else "material_quality_index",
        "candidates": candidates,
    }

# ─── Serve static ───
if os.path.exists(os.path.join(BASE_DIR, "static")):
    app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8081, reload=False)
