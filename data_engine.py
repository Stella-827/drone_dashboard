"""
data_engine.py  —  physics & data loading, verified against:
  - Kyle_&_Jose_Battery_Characterization_V1
  - Kyle_&_Jose_Flight_Time_Optimizer_V3  (Steps 1-5)
  - prop_lift_tests_33p5mm.csv
"""

import io, re
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR   = Path(__file__).parent / "data"
BAT_DIR    = DATA_DIR / "batteries"
MOTOR_DIR  = DATA_DIR / "motors"
THRUST_DIR = DATA_DIR / "thrust"

NOMINAL_V  = 3.7       # volts — used to convert discharge current → power in battery model
G_EARTH    = 9.81      # m/s²
RPM2RADS   = 2 * np.pi / 60

# BioLogic GCPL filename pattern (from Battery Colab Step 2)
BATT_PATTERN = re.compile(
    r"(?P<battery>.+?)-(?P<discharge_mA>[\d.]+)mA_discharge-(?P<charge_mA>[\d.]+)mA_charge",
    re.IGNORECASE,
)


# ════════════════════════════════════════════════════════════════════════════
# BATTERY  (Battery Characterization V1)
# ════════════════════════════════════════════════════════════════════════════

def load_battery_csv(filepath, raw_bytes=None):
    """Parse one BioLogic GCPL CSV. Returns dict or None."""
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

    # Total discharge energy in mWh — last non-NaN value (Colab Step 2)
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


def load_all_batteries():
    """Scan BAT_DIR, return {battery_name: [record, ...]} sorted by discharge_mA."""
    records = []
    for f in sorted(BAT_DIR.glob("*.csv")):
        rec = load_battery_csv(f)
        if rec:
            records.append(rec)
    batteries = {}
    for r in records:
        batteries.setdefault(r["battery"], []).append(r)
    for name in batteries:
        batteries[name].sort(key=lambda r: r["discharge_mA"])
    return batteries


def fit_battery_energy_model(runs):
    """
    Linear fit from Battery Colab Step 4:
      E_mWh = m * P_mW + b   where P_mW = discharge_mA * NOMINAL_V
    Returns dict with m, b, R2, eq_str.
    """
    powers   = np.array([r["discharge_mA"] * NOMINAL_V for r in runs])
    energies = np.array([r["energy_mWh"] for r in runs])

    if len(powers) < 2:
        b = float(energies[0]) if len(energies) else 0.0
        return {"m": 0.0, "b": b, "R2": 0.0, "eq_str": f"E = {b:.1f} mWh (single point)"}

    coeffs  = np.polyfit(powers, energies, 1)
    m, b    = float(coeffs[0]), float(coeffs[1])
    E_pred  = np.polyval(coeffs, powers)
    ss_res  = np.sum((energies - E_pred) ** 2)
    ss_tot  = np.sum((energies - energies.mean()) ** 2)
    R2      = float(1 - ss_res / ss_tot) if ss_tot else 0.0
    eq_str  = f"E = {m:.4f}·P + {b:.2f}  (R²={R2:.3f})"
    return {"m": m, "b": b, "R2": R2, "eq_str": eq_str}


# ════════════════════════════════════════════════════════════════════════════
# MOTOR  (Optimizer V3 Step 2)
# ════════════════════════════════════════════════════════════════════════════

def load_motor_csv(filepath=None, raw_bytes=None):
    """Load motor characteristics CSV."""
    if raw_bytes is not None:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    else:
        filepath = filepath or (MOTOR_DIR / "motor_characteristics.csv")
        df = pd.read_csv(filepath)
    df = df.dropna(subset=["Motor Name"])
    for col in ["RPMs", "No Load Current (io)", "Nominal Voltage (V)",
                "Motor ESR (Ω)", "Motor Mass (mg)", "K (Nm/A)",
                "Max Efficiency", "Max Current (A)", "Max Power (W)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False).str.strip(),
                errors="coerce"
            )
    return df


def compute_motor_params(df_motors):
    """
    Per-motor K via least-squares through the origin (Optimizer V3 Step 2):
      K = Σ(ω · back_EMF) / Σ(ω²)   where back_EMF = V − io·R
    Returns {motor_name: {mass_mg, R, K}}.
    """
    motor_params = {}
    for motor_name, grp in df_motors.groupby("Motor Name", sort=False):
        grp = grp.dropna(subset=["RPMs", "No Load Current (io)",
                                  "Nominal Voltage (V)", "Motor ESR (Ω)"])
        if grp.empty:
            # Fall back to pre-computed K column if present
            full = df_motors[df_motors["Motor Name"] == motor_name]
            if "K (Nm/A)" in full.columns and not full["K (Nm/A)"].dropna().empty:
                K    = float(full["K (Nm/A)"].dropna().iloc[0])
                R    = float(full["Motor ESR (Ω)"].dropna().iloc[0]) if "Motor ESR (Ω)" in full.columns else 0.0
                mass = float(full["Motor Mass (mg)"].dropna().iloc[0]) if "Motor Mass (mg)" in full.columns else 0.0
                motor_params[str(motor_name)] = {"mass_mg": mass, "R": R, "K": K}
            continue

        R    = float(grp["Motor ESR (Ω)"].iloc[0])
        mass = float(grp["Motor Mass (mg)"].iloc[0]) if "Motor Mass (mg)" in grp.columns else 0.0
        omega = grp["RPMs"] * RPM2RADS
        io    = grp["No Load Current (io)"]
        V     = grp["Nominal Voltage (V)"]
        y     = V - io * R
        mask  = (omega > 0) & y.notna() & omega.notna()
        if mask.sum() == 0:
            continue
        K = float(np.dot(omega[mask], y[mask]) / np.dot(omega[mask], omega[mask]))
        motor_params[str(motor_name)] = {"mass_mg": mass, "R": R, "K": K}
    return motor_params


# ════════════════════════════════════════════════════════════════════════════
# THRUST  (Optimizer V3 Steps 1 & 3)
# ════════════════════════════════════════════════════════════════════════════

def load_thrust_csv(filepath=None, raw_bytes=None):
    """Load prop_lift_tests CSV and standardise column names."""
    if raw_bytes is not None:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    else:
        candidates = list(THRUST_DIR.glob("*.csv"))
        if not candidates:
            return pd.DataFrame()
        df = pd.read_csv(candidates[0])

    df.columns = df.columns.str.strip()
    rename_map = {
        "Motor Type":                    "motor_type",
        "Motor Voltage (V)":             "voltage_V",
        "Motor Current (mA)":            "current_mA",
        "Lift over edge (g)":            "lift_g",
        "Power (W)":                     "power_W",
        "Thrust (N)":                    "thrust_N",
        "Newtons per Watt (efficiency)": "N_per_W",
        "Prop Flexure Angle (°)":        "flexure_angle_deg",
        "Prop Flexure Angle (deg)":      "flexure_angle_deg",
        "Prop Length (mm)":              "prop_length_mm",
        "Prop Width (mm)":               "prop_width_mm",
        "Prop Mass (mg)":                "prop_mass_mg",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    for col in ["voltage_V", "current_mA", "power_W", "thrust_N",
                "prop_length_mm", "prop_width_mm", "N_per_W", "flexure_angle_deg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["thrust_N", "power_W", "prop_length_mm", "prop_width_mm", "N_per_W"])
    df = df[df["thrust_N"] > 0]
    return df


def load_all_thrust(extra_dfs=None):
    base = load_thrust_csv()
    frames = [base] if not base.empty else []
    if extra_dfs:
        frames.extend([d for d in extra_dfs if not d.empty])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fit_thrust_model(df_thrust, flexure_angle=20.0):
    """
    Optimizer V3 Step 3: N/W = a·length + b·width + c via least-squares.
    Uses only rows at the optimal flexure angle (20° from your data).
    """
    if df_thrust.empty:
        return {"a_L": 0, "a_W": 0, "c": 0, "R2": 0,
                "eq_str": "no data", "baseline_length_mm": 33.5}

    if "flexure_angle_deg" in df_thrust.columns:
        df_fit = df_thrust[df_thrust["flexure_angle_deg"] == flexure_angle].copy()
        if df_fit.empty:
            df_fit = df_thrust.copy()
    else:
        df_fit = df_thrust.copy()

    L  = df_fit["prop_length_mm"].values
    W  = df_fit["prop_width_mm"].values
    nw = df_fit["N_per_W"].values
    mask = (nw > 0) & (nw < 0.1)
    L, W, nw = L[mask], W[mask], nw[mask]

    if len(L) < 3:
        return {"a_L": 0, "a_W": 0, "c": 0, "R2": 0,
                "eq_str": "insufficient data", "baseline_length_mm": 33.5}

    X = np.column_stack([L, W, np.ones_like(L)])
    coeffs, *_ = np.linalg.lstsq(X, nw, rcond=None)
    a_L, a_W, c = coeffs
    nw_pred = X @ coeffs
    ss_res  = np.sum((nw - nw_pred) ** 2)
    ss_tot  = np.sum((nw - nw.mean()) ** 2)
    R2      = float(1 - ss_res / ss_tot) if ss_tot else 0.0
    baseline = float(df_fit["prop_length_mm"].mode().iloc[0]) if len(df_fit) else 33.5

    return {
        "a_L": float(a_L), "a_W": float(a_W), "c": float(c), "R2": R2,
        "eq_str": f"N/W = {a_L:.5f}·L + {a_W:.5f}·W + {c:.5f}",
        "baseline_length_mm": baseline,
    }


# ════════════════════════════════════════════════════════════════════════════
# FLIGHT TIME OPTIMIZER  (Optimizer V3 Steps 4-5)
# ════════════════════════════════════════════════════════════════════════════

def predict_NW(length_mm, width_mm, model):
    """
    Scale N/W from baseline using actuator-disk 1/D relation (V3 Step 3 note).
    """
    nw_base = (model["a_L"] * model["baseline_length_mm"]
               + model["a_W"] * width_mm + model["c"])
    D_ratio = length_mm / model["baseline_length_mm"]
    return float(nw_base / D_ratio) if D_ratio > 0 else 0.0


def prop_mass_estimate(length_mm, width_mm, baseline_mm=33.5):
    """Scale prop mass from baseline geometry (mg)."""
    base_mass = max(15.0 * width_mm - 17.0, 20.0)
    return base_mass * (length_mm / baseline_mm) ** 2


def run_optimizer(motor_params, batteries_model, thrust_model,
                  prop_lengths_mm, prop_widths_mm,
                  body_mass_mg=450.0, lift_margin_mg=100.0,
                  selected_motors=None, selected_batteries=None):
    """
    Exhaustive search: motor × battery × prop geometry  (V3 Step 5).
    batteries_model: [(name, mass_mg, m, b), ...]  E_mWh = m*P_mW + b
    Returns DataFrame sorted by flight_time_s desc.
    """
    from itertools import product as iproduct

    motor_items = [(n, p) for n, p in motor_params.items()
                   if not selected_motors or n in selected_motors]
    batt_items  = [b for b in batteries_model
                   if not selected_batteries or b[0] in selected_batteries]

    baseline = thrust_model.get("baseline_length_mm", 33.5)
    results  = []

    for (motor_name, mp), (batt_name, batt_mass_mg, bm, bb), length_mm, width_mm in iproduct(
        motor_items, batt_items, prop_lengths_mm, prop_widths_mm
    ):
        prop_mg  = prop_mass_estimate(length_mm, width_mm, baseline)
        total_mg = mp["mass_mg"] + batt_mass_mg + body_mass_mg + prop_mg
        total_kg = total_mg * 1e-6
        thrust_N = (total_kg + lift_margin_mg * 1e-6) * G_EARTH

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
            "motor":          motor_name,
            "battery":        batt_name,
            "prop_length_mm": round(float(length_mm), 1),
            "prop_width_mm":  round(float(width_mm), 2),
            "total_mass_mg":  round(total_mg),
            "P_elec_mW":      round(P_elec_mW, 1),
            "E_mWh":          round(E_mWh, 2),
            "flight_time_s":  round(flight_s, 1),
            "thrust_N":       round(thrust_N, 5),
            "NW":             round(nw, 5),
            "motor_K":        round(mp["K"], 6),
            "motor_R":        round(mp["R"], 2),
        })

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values("flight_time_s", ascending=False).reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════
# DEFAULT BATTERY MODELS  (from Optimizer V3 Step 4 — exact values)
# ════════════════════════════════════════════════════════════════════════════
# Format: (name, mass_mg, m, b)   E_mWh = m * P_mW + b

DEFAULT_BATTERY_MODELS = [
    ("Nichicon 140mg",           140,  -0.001,   0.3),
    ("Nichicon 575mg",           575,  -0.002,   2.5),
    ("Powerstream 300mg",        300,  -0.035,  22.8),
    ("Powerstream 410mg",        410,  -0.044,  44.3),
    ("Noname 600mg",             600,  -0.014,  10.2),
    ("Yellow 712mg",             712,  -0.125, 127.6),
    ("PowerStream-15x10mm-1100mg", 1100, -0.010,  6.8),
    ("Polybattery 1300mg",      1300,  -0.043,  55.8),
    ("PowerStream-1300mg",      1300,  -0.044,  43.3),
    ("15x10mm-10mAH 1400mg",   1400,  -0.042,  54.0),
]
