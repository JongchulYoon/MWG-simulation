"""
Microwave Heating Simulation — Reproducible Minimal Version
============================================================
1-D cylindrical FDM solver for HV and LV regimes.
Tested with: Python 3.10, numpy 1.26, numba 0.59

Geometry (radially outward):
  [0, r_mix]         — reactive mixture (graphite/Fe composite)
  [r_mix, r_mix+d_i] — inner Al2O3 tube
  [+d_p]             — porous insulation
  [+d_o]             — outer Al2O3 tube

Physics:
  HV — magnetic + dielectric + Joule heating; arc discharge (Poisson, tau=60 s)
  LV — above + oxidation (Shrinking Core Model); dynamic arc threshold via oxide layer

Reference: [Author et al., Journal, Year]  ← fill in before submission
"""

import numpy as np
from numba import njit

# ── Geometry (m) ──────────────────────────────────────────────────────────────
GEO = dict(r_mix=0.0165, d_inner=0.0035, d_porous=0.0075, d_outer=0.0035, dr=5e-5)

# ── Material properties ───────────────────────────────────────────────────────
MAT = dict(
    rho_mix=2000.0, cp_mix=1200.0, k_mix=0.7,      # reactive core
    rho_inner=3950.0, cp_inner=880.0, k_inner=20.0, # Al2O3 inner
    rho_por=220.0,   cp_por=1000.0,  k_por=0.3,     # porous insulation
    rho_out=3950.0,  cp_out=880.0,   k_out=20.0,    # Al2O3 outer
)

# ── Electromagnetic parameters ────────────────────────────────────────────────
EM_BASE = dict(
    freq=2.45e9,         # Hz (2.45 GHz ISM band)
    H_app=40.0,          # A/m  applied magnetic field amplitude
    E_app=1300.0,        # V/m  applied electric field amplitude
    vol_fe=0.13,         # Fe volume fraction in reactive core
    curie_T=1043.15,     # K   Curie temperature of Fe
    sigma_min=10.0,      # S/m minimum electrical conductivity
    sigma_max=2500.0,    # S/m maximum electrical conductivity
    sigma_ref=60.0,      # S/m reference for Joule saturation factor
    arc_E_thresh=500.0,  # V/m arc ignition threshold
    arc_Q=5.0e9,         # W/m³ arc energy density per event
    arc_prob0=1.37014,   # initial arc rate constant (s⁻¹)
)

EM_HV = {**EM_BASE, "tau_arc": 60.0}
EM_LV = {**EM_BASE, "tau_arc": 11.23, "arc_T_cut": 2000.0, "breakdown_beta": 5.851}

# ── Oxidation kinetics (LV only, Shrinking Core Model) ────────────────────────
CHEM = dict(
    H_reaction=7.4e6,      # J/kg  reaction enthalpy
    A_arrhenius=5.716e5,   # s⁻¹  pre-exponential factor
    E_activation=80000.0,  # J/mol activation energy
    R_gas=8.314,           # J/(mol·K)
    T_onset=500.0,         # K   lower cutoff for oxidation
    T_cutoff=909.15,       # K   upper cutoff (FeO→Fe3O4 transition)
    D_O2=1.0e-13,          # m²/s O2 diffusivity through oxide shell
    r_particle=10.0e-6,    # m   mean Fe particle radius
)

# ── Solver ────────────────────────────────────────────────────────────────────
SOL = dict(dt=1.5e-5, total_time=300.0, h_conv=15.0, eps=0.85, T_amb=298.15)


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

@njit
def _sigma(T, sigma_min, sigma_max):
    """Arrhenius-type electrical conductivity, clipped at sigma_max."""
    return min(sigma_min * np.exp(0.002 * (T - 298.15)), sigma_max)


@njit
def _em_base_terms(T_i, vol_fe, H_app, E_app, freq,
                   curie_T, sigma_min, sigma_max, sigma_ref):
    """Magnetic, dielectric, and Joule power densities (W/m³)."""
    mu_0  = 4.0e-7 * np.pi
    eps_0 = 8.854e-12
    mu_r  = 5.0 if T_i < curie_T else 1.0
    omega = 2.0 * np.pi * freq
    P_mag  = omega * mu_0 * mu_r * H_app**2 * vol_fe
    P_diel = omega * eps_0 * 0.5 * E_app**2 * (1.0 - vol_fe)
    sig    = _sigma(T_i, sigma_min, sigma_max)
    factor = 1.0 / (1.0 + (sig / sigma_ref)**1.5)
    P_joule = 0.5 * sig * (E_app * factor)**2 * (1.0 - vol_fe)
    return P_mag + P_diel + P_joule


# ─────────────────────────────────────────────────────────────────────────────
# Physics engines
# ─────────────────────────────────────────────────────────────────────────────

@njit
def source_HV(T, mask, freq, H_app, E_app, vol_fe,
              curie_T, sigma_min, sigma_max, sigma_ref,
              arc_E_thresh, arc_prob_step, arc_Q):
    n    = len(T)
    q    = np.zeros(n)
    iarc = np.zeros(n)
    for i in range(n):
        if not mask[i]:
            continue
        q[i] = _em_base_terms(T[i], vol_fe, H_app, E_app, freq,
                               curie_T, sigma_min, sigma_max, sigma_ref)
        if E_app > arc_E_thresh and np.random.random() < arc_prob_step:
            q[i]    += arc_Q
            iarc[i]  = 1.0
    return q, iarc


@njit
def source_LV(T, fe_frac, mask, freq, H_app, E_app, vol_fe_init,
              curie_T, sigma_min, sigma_max, sigma_ref,
              arc_E_thresh, arc_prob_step, arc_Q, arc_T_cut, breakdown_beta,
              chem_H, chem_A, chem_E, chem_R, T_on, T_off, D_O2, r_p):
    n        = len(T)
    q        = np.zeros(n)
    d_alpha  = np.zeros(n)
    iarc     = np.zeros(n)
    rho_fe   = 7870.0  # kg/m³
    for i in range(n):
        if not mask[i]:
            continue
        fe  = fe_frac[i]
        vfe = vol_fe_init * fe
        q[i] = _em_base_terms(T[i], vfe, H_app, E_app, freq,
                               curie_T, sigma_min, sigma_max, sigma_ref)
        # Dynamic arc threshold: oxide layer raises breakdown field
        E_thr = arc_E_thresh * (1.0 + breakdown_beta * (1.0 - fe))
        if (E_app > E_thr and fe > 0.001 and T[i] < arc_T_cut
                and np.random.random() < arc_prob_step):
            q[i]   += arc_Q
            iarc[i] = 1.0
        # Shrinking Core oxidation
        if fe > 0.001 and T_on < T[i] < T_off:
            k_r      = chem_A * np.exp(-chem_E / (chem_R * T[i]))
            alpha    = 1.0 - fe
            k_d      = (D_O2 / r_p**2) * (1.0 - alpha + 1e-9)**(2.0 / 3.0)
            k_eff    = k_r * k_d / (k_r + k_d)
            da       = k_eff * fe
            d_alpha[i] = da
            q[i]      += da * rho_fe * vol_fe_init * chem_H
    return q, d_alpha, iarc


# ─────────────────────────────────────────────────────────────────────────────
# FDM heat solver — explicit, 1-D cylindrical
# ─────────────────────────────────────────────────────────────────────────────

@njit
def heat_step(T, K, Rho, Cp, q, r, dr, dt, h_conv, T_amb, eps=0.85):
    """
    Interior:  finite difference in cylindrical coords
    r = 0:     L'Hôpital limit (dT/dr = 0)
    r = r_max: convection + Stefan–Boltzmann radiation
    """
    n       = len(T)
    T_new   = np.empty_like(T)
    sig_sb  = 5.67e-8
    for i in range(1, n - 1):
        ri      = r[i]
        f_out   = 0.5*(K[i+1]+K[i]) * (ri + 0.5*dr) * (T[i+1]-T[i]) / dr
        f_in    = 0.5*(K[i]+K[i-1]) * (ri - 0.5*dr) * (T[i]-T[i-1]) / dr
        T_new[i] = T[i] + dt/(Rho[i]*Cp[i]) * ((f_out - f_in)/(ri*dr) + q[i])
    # Center node
    T_new[0] = T[0] + dt/(Rho[0]*Cp[0]) * (2.0*K[0]*(T[1]-T[0])/dr**2 + q[0])
    # Surface node
    s  = n - 1
    rs = r[s]
    f_cond  = 0.5*(K[s]+K[s-1]) * (rs - 0.5*dr) * (T[s-1]-T[s]) / dr
    q_loss  = (h_conv*(T[s]-T_amb) + sig_sb*eps*(T[s]**4 - T_amb**4)) * rs
    T_new[s] = T[s] + dt/(Rho[s]*Cp[s]) * ((f_cond - q_loss)/(rs*dr*0.5) + q[s])
    return T_new


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate(mode="HV", seed=42):
    """
    Parameters
    ----------
    mode : 'HV' or 'LV'
    seed : random seed for reproducibility

    Returns
    -------
    T_hist   : list of (time, T_array) snapshots (every ~1 s)
    arc_count: total time steps with at least one arc event
    """
    np.random.seed(seed)
    em  = EM_HV if mode == "HV" else EM_LV
    dr  = GEO["dr"]
    r1  = GEO["r_mix"]
    r2  = r1 + GEO["d_inner"]
    r3  = r2 + GEO["d_porous"]
    r4  = r3 + GEO["d_outer"]
    r   = np.arange(0, r4 + dr/2, dr)
    n   = len(r)

    K   = np.where(r<=r1, MAT["k_mix"],   np.where(r<=r2, MAT["k_inner"],
          np.where(r<=r3, MAT["k_por"],   MAT["k_out"])))
    Rho = np.where(r<=r1, MAT["rho_mix"], np.where(r<=r2, MAT["rho_inner"],
          np.where(r<=r3, MAT["rho_por"], MAT["rho_out"])))
    Cp  = np.where(r<=r1, MAT["cp_mix"],  np.where(r<=r2, MAT["cp_inner"],
          np.where(r<=r3, MAT["cp_por"],  MAT["cp_out"])))

    mask    = r <= r1
    T       = np.full(n, SOL["T_amb"])
    fe_frac = np.ones(n)
    dt      = SOL["dt"]
    steps   = int(SOL["total_time"] / dt)
    snap    = max(1, steps // 300)  # ~300 snapshots over 300 s

    T_hist, arc_count = [], 0

    for step in range(steps):
        t_now          = step * dt
        arc_prob_step  = em["arc_prob0"] * np.exp(-t_now / em["tau_arc"]) * dt

        if mode == "HV":
            q, iarc = source_HV(
                T, mask, em["freq"], em["H_app"], em["E_app"], em["vol_fe"],
                em["curie_T"], em["sigma_min"], em["sigma_max"], em["sigma_ref"],
                em["arc_E_thresh"], arc_prob_step, em["arc_Q"])
        else:
            ch = CHEM
            q, d_alpha, iarc = source_LV(
                T, fe_frac, mask,
                em["freq"], em["H_app"], em["E_app"], em["vol_fe"],
                em["curie_T"], em["sigma_min"], em["sigma_max"], em["sigma_ref"],
                em["arc_E_thresh"], arc_prob_step, em["arc_Q"],
                em["arc_T_cut"], em["breakdown_beta"],
                ch["H_reaction"], ch["A_arrhenius"], ch["E_activation"],
                ch["R_gas"], ch["T_onset"], ch["T_cutoff"],
                ch["D_O2"], ch["r_particle"])
            fe_frac[mask] -= d_alpha[mask] * dt
            fe_frac        = np.clip(fe_frac, 0.0, 1.0)

        T          = heat_step(T, K, Rho, Cp, q, r, dr, dt,
                               SOL["h_conv"], SOL["T_amb"])
        arc_count += int(iarc[mask].sum() > 0)

        if step % snap == 0:
            T_hist.append((t_now, T.copy()))

    return T_hist, arc_count


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for mode in ("HV", "LV"):
        print(f"Running {mode}...", flush=True)
        hist, arcs = simulate(mode)
        T_center_final = hist[-1][1][0]
        print(f"  {mode} done | arc events: {arcs} | "
              f"T_center(final) = {T_center_final:.1f} K")
