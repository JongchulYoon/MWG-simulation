# Microwave Graphene Synthesis — Thermal & Arc Discharge Simulation

Simulation code accompanying the paper:

> **Facile Household Microwave Synthesis of High-Quality Graphene via Oxide-Mediated Arc Discharge Suppression**  
> Jong-Chul Yoon, Ji-Hyun Jang*  
> School of Energy and Chemical Engineering, UNIST  
> *Corresponding author: clau@unist.ac.kr

---

## Overview

1D radial transient heat transfer model for a triple-layer crucible system (Al₂O₃ / SiO₂ / Al₂O₃) under microwave irradiation (2.45 GHz, 600 W).

Two regimes are implemented:

- **HV** (High-Vacuum, ~10⁻³ Torr) — arc discharge persists ~45 s; no oxide passivation
- **LV** (Low-Vacuum, ~320 Torr) — in-situ Fe₂O₃ formation suppresses arcing within 9–10 s via dynamic breakdown threshold

Key physics:
- Magnetic loss + dielectric loss + Joule heating
- Stochastic arc discharge: `p(t) = arc_prob₀ × exp(−t/τ) × Δt`
- Shrinking Core Model for Fe oxidation kinetics (LV only)
- FDM solver in cylindrical coordinates with L'Hôpital center node and convection + radiation surface boundary

All parameters correspond exactly to Tables S1–S4 in the Supplementary Information.

---

## Requirements

```
python >= 3.10
numpy  >= 1.26
numba  >= 0.59
```

Install:

```bash
pip install numpy numba
```

---

## Usage

```bash
python mw_heating_sim.py
```

Output:
```
Running HV...
  HV done | arc events: XXXX | T_center(final) = XXXX.X K
Running LV...
  LV done | arc events: XXXX | T_center(final) = XXXX.X K
```

To run a single regime:

```python
from mw_heating_sim import simulate

T_hist, arc_count = simulate(mode="LV")
# T_hist: list of (time, T_array) snapshots
# arc_count: number of timesteps with at least one arc event
```

---

## Reproducibility

Arc discharge events are modeled stochastically; individual run results will vary, but the qualitative trend (HV: sustained arcing ~45 s, LV: suppression within ~10 s) is reproducible across runs.

---

## File Structure

```
├── mw_heating_sim.py   # simulation code (HV + LV)
└── README.md
```

---

## Citation

If you use this code, please cite:

```
Yoon, J.-C.; Jang, J.-H. Facile Household Microwave Synthesis of High-Quality Graphene
via Oxide-Mediated Arc Discharge Suppression. [Journal, Year, DOI — to be updated upon publication]
```
