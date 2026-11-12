# Drone Flight Optimizer Dashboard

Interactive Python dashboard based on:
- **Kyle & Jose Battery Characterization V1**
- **Kyle & Jose Motor Characterization V1**
- **Kyle & Jose Flight Time Optimizer V3**

---

## Project Structure

```
drone_dashboard/
├── app.py               # Dash app entry point
├── data_engine.py       # All physics & data loading (mirrors the Colabs)
├── requirements.txt
├── Procfile             # for Railway / Render deployment
├── pages/
│   ├── home.py
│   ├── battery.py       # Battery characterization
│   ├── motor.py         # Motor characterization
│   ├── thrust.py        # Prop/thrust characterization
│   └── optimizer.py     # Flight time optimizer
└── data/
    ├── batteries/       # Drop BioLogic GCPL CSV files here
    │   └── 15X10-10mAh-1400mg-*.csv
    ├── motors/
    │   └── motor_characteristics.csv
    └── thrust/          # Drop prop_lift_tests CSVs here
```

---

## Local Setup (VS Code)

```bash
# 1. Clone your repo
git clone https://github.com/<you>/drone_dashboard.git
cd drone_dashboard

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python app.py
```

Open http://127.0.0.1:8050 in your browser.

---

## Adding Data

### Battery CSV files
Drop BioLogic GCPL `.csv` files into `data/batteries/`.  
File names **must** follow the pattern:
```
<BatteryName>-<X>mA_discharge-<Y>mA_charge_..._GCPL_C01.csv
```
Example: `15X10-10mAh-1400mg-300mA_discharge-5mA_charge_03_GCPL_C01.csv`

You can also upload files directly in the Battery tab at runtime.

### Motor CSV
Replace or extend `data/motors/motor_characteristics.csv`.  
Columns: Motor Name, Motor ESR (Ω), RPMs, No Load Current (io), Nominal Voltage (V), Motor Mass (mg), K (Nm/A), …

### Thrust CSV
Add `.csv` files to `data/thrust/` or upload in the Thrust tab.  
Format: same as `prop_lift_tests_33p5mm.csv`.

---

## Dashboard Pages

| Page | What it shows |
|------|---------------|
| 🔋 Battery | Charge/discharge curves, V vs time, energy vs power linear fit |
| ⚙️ Motor | K/R parameter table, efficiency curves, torque–speed curves |
| 🌀 Thrust | N/W vs angle, thrust vs power, N/W surface fit |
| 🚁 Optimizer | Exhaustive motor × battery × prop search, top configs, heatmap |

---

## Deploying to the Web (GitHub → Railway)

1. Push this repo to GitHub.
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub.
3. Railway auto-detects the `Procfile` and installs `requirements.txt`.
4. Set any environment variables (none required by default).
5. Your app gets a public URL instantly.

Alternative: [Render.com](https://render.com) works identically — point to the repo,
set **Start Command** to `gunicorn app:server`.

---

## Physics Background

### Battery model
`E_mWh = m · P_mW + b`  
Linear fit to measured discharge energy at multiple C-rates.

### Motor model  
`K = Σ(ω · back_EMF) / Σ(ω²)`  
Least-squares fit through the origin from measured RPM × voltage data.

### Thrust model  
`N/W = a·length + b·width + c`  
Multiple linear regression on prop geometry at the optimal flexure angle.  
Scales to other lengths via actuator-disk theory: N/W ∝ 1/D.

### Flight time  
`flight_time = (E_mWh / P_elec_mW) × 3600 s`  
where `P_elec_mW = (thrust_N / NW) × 1000`
and `thrust_N = (total_mass_kg + lift_margin_kg) × g`
