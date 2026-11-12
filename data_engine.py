"""
data_engine.py
--------------
All data loading, parsing, and physics calculations.
Mirrors the logic in the three Colab notebooks exactly.
"""

import os
import re
import io
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent / "data"
BAT_DIR     = DATA_DIR / "batteries"
MOTOR_DIR   = DATA_DIR / "motors"
THRUST_DIR  = DATA_DIR / "thrust"

# File-name pattern for BioLogic GCPL battery CSVs:
#   <BatteryName>-<Xma>_discharge-<YmA>_charge_..._GCPL_C01.csv
BATT_PATTERN = re.compile(
    r"(?P<battery>.+?)-(?P<discharge_mA>[\d.]+)mA_discharge-(?P<charge_mA>[\d.]+)mA_charge",
    re.IGNORECASE,
)

NOMINAL_V = 3.7      # volts – used to convert discharge current → power
G_EARTH   = 9.81     # m/s²
RPM2RADS  = 2 * np.pi / 60


# ════════════════════════════════════════════════════════════════════════════
# BATTERY
# ════════════════════════════════════════════════════════════════════════════

def load_battery_csv(filepath: str | Path, raw_bytes: bytes | None = None) -> dict | None:
    """
    Parse one BioLogic GCPL CSV.
    Returns a dict with keys: battery, discharge_mA, charge_mA,
    df_charge, df_discharge, df_full, energy_mWh, max_voltage, source_file.
    Returns None if the filename doesn't match the expected pattern.
    """
    fname = Path(filepath).name
    m = BATT_PATTERN.search(fname)
    if not m:
        return None

    battery      = m.group("battery")
    discharge_mA = float(m.group("discharge_mA"))
    charge_mA    = float(m.group("charge_mA"))

    if raw_bytes is not None:
        df = pd.read_csv(io.BytesIO(raw_bytes), encoding="cp1252")
    else:
        df = pd.read_csv(filepath, encoding="cp1252")

    df_charge    = df[df["ox/red"] == 1].copy()
    df_discharge = df[df["ox/red"] == 0].copy()

    energy_series = df["Energy discharge/W.h"].dropna()
    energy_mWh    = float(energy_series.iloc[-1]) * 1000 if len(energy_series) else 0.0
    max_voltage   = float(df_charge["Ewe/V"].max()) if len(df_charge) else 0.0

    return {
        "battery":      battery,
        "discharge_mA": discharge_mA,
        "charge_mA":    charge_mA,
        "df_charge":    df_charge,
        "df_discharge": df_discharge,
        "df_full":      df,
        "energy_mWh":   energy_mWh,
        "max_voltage":  max_voltage,
        "source_file":  fname,
    }


def load_all_batteries(extra_records: list[dict] | None = None) -> dict[str, list[dict]]:
    """
    Scan BAT_DIR for GCPL CSVs and group by battery name.
    extra_records: list of already-parsed dicts from uploaded files.
    Returns {battery_name: [record, ...]} sorted by discharge current.
    """
    records = []
    for f in sorted(BAT_DIR.glob("*.csv")):
        rec = load_battery_csv(f)
        if rec:
            records.append(rec)

    if extra_records:
        records.extend(extra_records)

    batteries: dict[str, list] = {}
    for r in records:
        batteries.setdefault(r["battery"], []).append(r)

    # Sort each battery's runs by discharge current
    for name in batteries:
        batteries[name].sort(key=lambda r: r["discharge_mA"])

    return batteries


def fit_battery_energy_model(runs: list[dict]) -> dict:
    """
    Linear fit: E_mWh = m * P_mW + b
    P_mW = discharge_mA * NOMINAL_V
    Returns {'m': float, 'b': float, 'R2': float, 'eq_str': str}
    """
    powers  = np.array([r["discharge_mA"] * NOMINAL_V for r in runs])
    energies = np.array([r["energy_mWh"] for r in runs])

    if len(powers) < 2:
        return {"m": 0.0, "b": energies[0] if len(energies) else 0.0, "R2": 0.0, "eq_str": "insufficient data"}

    coeffs  = np.polyfit(powers, energies, 1)
    m, b    = coeffs
    E_pred  = np.polyval(coeffs, powers)
    ss_res  = np.sum((energies - E_pred) ** 2)
    ss_tot  = np.sum((energies - energies.mean()) ** 2)
    R2      = 1 - ss_res / ss_tot if ss_tot else 0.0
    eq_str  = f"E = {m:.3f}·P + {b:.1f},  R²={R2:.2f}"

    return {"m": m, "b": b, "R2": R2, "eq_str": eq_str}


# ════════════════════════════════════════════════════════════════════════════
# MOTOR
# ════════════════════════════════════════════════════════════════════════════

def load_motor_csv(filepath: str | Path | None = None, raw_bytes: bytes | None = None) -> pd.DataFrame:
    """
    Load the motor characteristics CSV.
    Compatible with the Google Sheets published CSV format used in the Colabs.
    """
    if raw_bytes is not None:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    else:
        filepath = filepath or (MOTOR_DIR / "motor_characteristics.csv")
        df = pd.read_csv(filepath)

    df = df.dropna(subset=["Motor Name"])

    numeric_cols = [
        "RPMs", "No Load Current (io)", "Nominal Voltage (V)",
        "Motor ESR (Ω)", "Motor Mass (mg)", "K (Nm/A)",
        "Max Efficiency", "Max Current (A)", "Max Power (W)",
        "Max Torque (stall torque - Ts)", "Motor Diameter (mm)",
        "Motor Length (mm)",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def compute_motor_params(df_motors: pd.DataFrame) -> dict[str, dict]:
    """
    For each motor, compute K (Nm/A) via least-squares through the origin:
      K = Σ(ω · back_EMF) / Σ(ω²)   where back_EMF = V − io·R
    Returns {motor_name: {'mass_mg', 'R', 'K'}}
    """
    motor_params: dict[str, dict] = {}

    # If K is already supplied in the sheet, use it directly
    has_K = "K (Nm/A)" in df_motors.columns

    for motor_name, grp in df_motors.groupby("Motor Name", sort=False):
        grp = grp.dropna(subset=["RPMs", "No Load Current (io)", "Nominal Voltage (V)", "Motor ESR (Ω)"])
        if grp.empty:
            # Try using pre-computed K
            full_grp = df_motors[df_motors["Motor Name"] == motor_name]
            if has_K and not full_grp["K (Nm/A)"].dropna().empty:
                K    = float(full_grp["K (Nm/A)"].dropna().iloc[0])
                R    = float(full_grp["Motor ESR (Ω)"].dropna().iloc[0]) if "Motor ESR (Ω)" in full_grp else 0.0
                mass = float(full_grp["Motor Mass (mg)"].dropna().iloc[0]) if "Motor Mass (mg)" in full_grp else 0.0
                motor_params[str(motor_name)] = {"mass_mg": mass, "R": R, "K": K}
            continue

        R    = float(grp["Motor ESR (Ω)"].iloc[0])
        mass = float(grp["Motor Mass (mg)"].iloc[0]) if "Motor Mass (mg)" in grp.columns else 0.0

        omega = grp["RPMs"] * RPM2RADS
        io    = grp["No Load Current (io)"]
        V     = grp["Nominal Voltage (V)"]
        y     = V - io * R

        mask = (omega > 0) & y.notna() & omega.notna()
        if mask.sum() == 0:
            continue

        K = np.dot(omega[mask], y[mask]) / np.dot(omega[mask], omega[mask])
        motor_params[str(motor_name)] = {"mass_mg": mass, "R": R, "K": K}

    return motor_params


# ════════════════════════════════════════════════════════════════════════════
# THRUST
# ════════════════════════════════════════════════════════════════════════════

def load_thrust_csv(filepath: str | Path | None = None, raw_bytes: bytes | None = None) -> pd.DataFrame:
    """
    Load the prop_lift_tests CSV (same format as prop_lift_tests_33p5mm.csv).
    Renames columns to match the Colab conventions.
    """
    if raw_bytes is not None:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    else:
        # Look for any CSV in the thrust dir; use the first one found
        candidates = list(THRUST_DIR.glob("*.csv"))
        if not candidates:
            # Fall back to empty frame
            return pd.DataFrame()
        df = pd.read_csv(candidates[0])

    df.columns = df.columns.str.strip()
    rename_map = {
        "Motor Type":                       "motor_type",
        "Motor Voltage (V)":                "voltage_V",
        "Motor Current (mA)":               "current_mA",
        "Lift over edge (g)":               "lift_g",
        "Power (W)":                        "power_W",
        "Thrust (N)":                       "thrust_N",
        "Newtons per Watt (efficiency)":    "N_per_W",
        "Prop Flexure Angle (°)":           "flexure_angle_deg",
        "Prop Flexure Angle (deg)":         "flexure_angle_deg",
        "Prop Length (mm)":                 "prop_length_mm",
        "Prop Width (mm)":                  "prop_width_mm",
        "Prop Mass (mg)":                   "prop_mass_mg",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    num_cols = ["voltage_V", "current_mA", "power_W", "thrust_N",
                "prop_length_mm", "prop_width_mm", "N_per_W", "flexure_angle_deg"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["thrust_N", "power_W", "prop_length_mm", "prop_width_mm", "N_per_W"])
    df = df[df["thrust_N"] > 0]
    return df


def load_all_thrust(extra_dfs: list[pd.DataFrame] | None = None) -> pd.DataFrame:
    """Merge built-in thrust data with any user-uploaded extra DataFrames."""
    base = load_thrust_csv()
    frames = [base] if not base.empty else []
    if extra_dfs:
        frames.extend(extra_dfs)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fit_thrust_model(df_thrust: pd.DataFrame, flexure_angle: float = 20.0) -> dict:
    """
    Fit N/W = a·length + b·width + c via multiple linear regression.
    Uses only rows at the specified flexure angle (default 20°, optimal from data).
    Returns {'a_L', 'a_W', 'c', 'R2', 'eq_str', 'baseline_length_mm'}
    """
    if df_thrust.empty:
        return {"a_L": 0, "a_W": 0, "c": 0, "R2": 0, "eq_str": "no data", "baseline_length_mm": 33.5}

    if "flexure_angle_deg" in df_thrust.columns:
        df_fit = df_thrust[df_thrust["flexure_angle_deg"] == flexure_angle].copy()
    else:
        df_fit = df_thrust.copy()

    if df_fit.empty:
        df_fit = df_thrust.copy()

    L  = df_fit["prop_length_mm"].values
    W  = df_fit["prop_width_mm"].values
    nw = df_fit["N_per_W"].values

    mask = (nw > 0) & (nw < 0.1)
    L, W, nw = L[mask], W[mask], nw[mask]

    if len(L) < 3:
        return {"a_L": 0, "a_W": 0, "c": 0, "R2": 0, "eq_str": "insufficient data", "baseline_length_mm": 33.5}

    X = np.column_stack([L, W, np.ones_like(L)])
    coeffs, *_ = np.linalg.lstsq(X, nw, rcond=None)
    a_L, a_W, c = coeffs

    nw_pred = X @ coeffs
    ss_res  = np.sum((nw - nw_pred) ** 2)
    ss_tot  = np.sum((nw - nw.mean()) ** 2)
    R2      = 1 - ss_res / ss_tot if ss_tot else 0.0

    baseline = float(df_fit["prop_length_mm"].mode().iloc[0]) if len(df_fit) else 33.5

    return {
        "a_L": a_L, "a_W": a_W, "c": c, "R2": R2,
        "eq_str": f"N/W = {a_L:.5f}·L + {a_W:.5f}·W + {c:.5f}",
        "baseline_length_mm": baseline,
    }


# ════════════════════════════════════════════════════════════════════════════
# FLIGHT TIME OPTIMIZER
# ════════════════════════════════════════════════════════════════════════════

def predict_NW(length_mm: float, width_mm: float, model: dict) -> float:
    """Scale N/W from baseline using actuator-disk D^4/D^5 = 1/D relation."""
    nw_base = model["a_L"] * model["baseline_length_mm"] + model["a_W"] * width_mm + model["c"]
    D_ratio = length_mm / model["baseline_length_mm"]
    return nw_base / D_ratio if D_ratio > 0 else 0.0


def prop_mass_estimate(length_mm: float, width_mm: float, baseline_mm: float = 33.5) -> float:
    """Scale prop mass from baseline geometry (mg)."""
    base_mass = max(15.0 * width_mm - 17.0, 20.0)
    return base_mass * (length_mm / baseline_mm) ** 2


def run_optimizer(
    motor_params: dict[str, dict],
    batteries_model: list[tuple],
    thrust_model: dict,
    prop_lengths_mm: np.ndarray,
    prop_widths_mm: np.ndarray,
    body_mass_mg: float = 450.0,
    lift_margin_mg: float = 100.0,
    selected_motors: list[str] | None = None,
    selected_batteries: list[str] | None = None,
) -> pd.DataFrame:
    """
    Exhaustive search across motor × battery × prop geometry.
    batteries_model: [(name, mass_mg, m, b), ...]  E_mWh = m*P_mW + b
    Returns DataFrame sorted by flight_time_s descending.
    """
    from itertools import product as iproduct

    if selected_motors:
        motor_items = [(n, p) for n, p in motor_params.items() if n in selected_motors]
    else:
        motor_items = list(motor_params.items())

    if selected_batteries:
        batt_items = [b for b in batteries_model if b[0] in selected_batteries]
    else:
        batt_items = batteries_model

    baseline = thrust_model.get("baseline_length_mm", 33.5)
    results  = []

    for (motor_name, mp), (batt_name, batt_mass_mg, bm, bb), length_mm, width_mm in iproduct(
        motor_items, batt_items, prop_lengths_mm, prop_widths_mm
    ):
        prop_mg   = prop_mass_estimate(length_mm, width_mm, baseline)
        total_mg  = mp["mass_mg"] + batt_mass_mg + body_mass_mg + prop_mg
        total_kg  = total_mg * 1e-6
        thrust_N  = (total_kg + lift_margin_mg * 1e-6) * G_EARTH

        nw = predict_NW(length_mm, width_mm, thrust_model)
        if nw <= 0:
            continue

        P_elec_mW = (thrust_N / nw) * 1000
        E_mWh     = bm * P_elec_mW + bb
        if E_mWh <= 0:
            continue

        flight_s = (E_mWh / P_elec_mW) * 3600
        if flight_s <= 0:
            continue

        results.append({
            "motor":           motor_name,
            "battery":         batt_name,
            "prop_length_mm":  round(float(length_mm), 1),
            "prop_width_mm":   round(float(width_mm), 2),
            "total_mass_mg":   round(total_mg),
            "P_elec_mW":       round(P_elec_mW, 1),
            "E_mWh":           round(E_mWh, 2),
            "flight_time_s":   round(flight_s, 1),
            "thrust_N":        round(thrust_N, 5),
            "NW":              round(nw, 5),
            "motor_K":         round(mp["K"], 6),
            "motor_R":         round(mp["R"], 2),
        })

    if not results:
        return pd.DataFrame()

    return (
        pd.DataFrame(results)
        .sort_values("flight_time_s", ascending=False)
        .reset_index(drop=True)
    )


# ════════════════════════════════════════════════════════════════════════════
# SHARED HARDCODED BATTERY MODELS  (from Flight Time Optimizer V3)
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_BATTERY_MODELS: list[tuple] = [
    # (name,                          mass_mg,  m,       b)
    ("Nichicon 140mg",                   140,  -0.001,   0.3),
    ("Nichicon 575mg",                   575,  -0.002,   2.5),
    ("Powerstream 300mg",                300,  -0.035,  22.8),
    ("Powerstream 410mg",                410,  -0.044,  44.3),
    ("Noname 600mg",                     600,  -0.014,  10.2),
    ("Yellow 712mg",                     712,  -0.125, 127.6),
    ("PowerStream-15x10mm-1100mg",      1100,  -0.010,   6.8),
    ("Polybattery 1300mg",              1300,  -0.043,  55.8),
    ("PowerStream-1300mg",              1300,  -0.044,  43.3),
    ("15x10mm-10mAH 1400mg",           1400,  -0.042,  54.0),
]
