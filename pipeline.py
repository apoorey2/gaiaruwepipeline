import numpy as np
import pandas as pd
from astroquery.gaia import Gaia
from astropy.time import Time
from astropy.coordinates import get_body_barycentric, ICRS, SkyCoord
import astropy.units as u
from gaiaunlimited.scanninglaw import GaiaScanningLaw


#correlation columns needed to construct the 5×5 covariance matrix.
_CORR_COLS = [
    "ra_dec_corr",
    "ra_parallax_corr",
    "ra_pmra_corr",
    "ra_pmdec_corr",
    "dec_parallax_corr",
    "dec_pmra_corr",
    "dec_pmdec_corr",
    "parallax_pmra_corr",
    "parallax_pmdec_corr",
    "pmra_pmdec_corr",
]

_ADQL_QUERY = """
SELECT
    source_id,
    ra, ra_error,
    dec, dec_error,
    parallax, parallax_error,
    pmra, pmra_error,
    pmdec, pmdec_error,
    ruwe,
    phot_g_mean_mag,
    astrometric_n_good_obs_al,
    astrometric_chi2_al,
    {corr_cols}
FROM gaiadr3.gaia_source
WHERE source_id = {source_id}
""".format(
    corr_cols=",\n    ".join(_CORR_COLS),
    source_id="{source_id}",
)


def _build_covariance(row: dict) -> np.ndarray:
    """
    Construct the 5×5 astrometric covariance matrix from the individual
    errors and pairwise correlation coefficients stored in the Gaia archive.

    Parameters: (ra, dec, parallax, pmra, pmdec)
    Units: (mas, mas, mas, mas/yr, mas/yr)
    """
    params  = ["ra", "dec", "parallax", "pmra", "pmdec"]
    errors  = np.array([row[f"{p}_error"] for p in params])

    corr_map = {
        (0, 1): row["ra_dec_corr"],
        (0, 2): row["ra_parallax_corr"],
        (0, 3): row["ra_pmra_corr"],
        (0, 4): row["ra_pmdec_corr"],
        (1, 2): row["dec_parallax_corr"],
        (1, 3): row["dec_pmra_corr"],
        (1, 4): row["dec_pmdec_corr"],
        (2, 3): row["parallax_pmra_corr"],
        (2, 4): row["parallax_pmdec_corr"],
        (3, 4): row["pmra_pmdec_corr"],
    }

    C = np.eye(5)
    for (i, j), rho in corr_map.items():
        C[i, j] = rho
        C[j, i] = rho

    cov = C * np.outer(errors, errors)
    return cov


def query_gaia_archive(source_id: int | str) -> dict:
    """
    Query the Gaia DR3 archive for astrometric + photometric parameters.
    """
    adql = _ADQL_QUERY.format(source_id=int(source_id))
    job  = Gaia.launch_job_async(adql)
    tbl  = job.get_results()

    row = {col: tbl[col][0] for col in tbl.colnames}
    cov = _build_covariance(row)

    params = {
        "source_id"                 : int(source_id),
        "ra"                        : float(row["ra"]),
        "dec"                       : float(row["dec"]),
        "parallax"                  : float(row["parallax"]),
        "pmra"                      : float(row["pmra"]),
        "pmdec"                     : float(row["pmdec"]),
        "ra_error"                  : float(row["ra_error"]),
        "dec_error"                 : float(row["dec_error"]),
        "parallax_error"            : float(row["parallax_error"]),
        "pmra_error"                : float(row["pmra_error"]),
        "pmdec_error"               : float(row["pmdec_error"]),
        "ruwe"                      : float(row["ruwe"]),
        "g_mag"                     : float(row["phot_g_mean_mag"]),
        "astrometric_n_good_obs_al" : int(row["astrometric_n_good_obs_al"]),
        "astrometric_chi2_al"       : float(row["astrometric_chi2_al"]),
        "covariance_matrix"         : cov,
    }
    return params


def compute_parallax_factor(t_tcb: np.ndarray, ra_deg: float, dec_deg: float) -> np.ndarray:
    """
    Compute the Gaia along-scan (AL) parallax factor.

        f(t) = cos(δ)·cos(α)·X⊕(t) + cos(δ)·sin(α)·Y⊕(t) + sin(δ)·Z⊕(t)
    """
    GAIA_TCB_ORIGIN_JD = 2455197.5
    jd_tcb = t_tcb + GAIA_TCB_ORIGIN_JD

    times = Time(jd_tcb, format="jd", scale="tcb")

    earth_bary = get_body_barycentric("earth", times)
    X = earth_bary.x.to(u.au).value
    Y = earth_bary.y.to(u.au).value
    Z = earth_bary.z.to(u.au).value

    ra_rad  = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)
    l_x = np.cos(dec_rad) * np.cos(ra_rad)
    l_y = np.cos(dec_rad) * np.sin(ra_rad)
    l_z = np.sin(dec_rad)

    parallax_factor = l_x * X + l_y * Y + l_z * Z
    return parallax_factor


def query_scanning_law(ra_deg: float, dec_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the Gaia DR3 scanning-law time series for a sky position.
    """
    sl = GaiaScanningLaw()
    result = sl.query(ra_deg, dec_deg)

    spin_phase = np.asarray(result[0], dtype=float)
    t_tcb      = np.asarray(result[1], dtype=float)

    n = min(len(spin_phase), len(t_tcb))
    spin_phase = spin_phase[:n]
    t_tcb      = t_tcb[:n]

    scan_angle = spin_phase % (2 * np.pi)
    parallax_factor = compute_parallax_factor(t_tcb, ra_deg, dec_deg)

    order = np.argsort(t_tcb)
    return t_tcb[order], scan_angle[order], parallax_factor[order]

    


def query_companion(source_id: int | str) -> dict:
    """
    Combines the archive query (astrometry, photometry, RUWE, covariance)
    with the scanning-law query (times, scan angles, parallax factors).
    """
    print(f"[1/2] Querying Gaia DR3 archive for source_id = {source_id} ...")
    params = query_gaia_archive(source_id)
    print(f"      RA={params['ra']:.5f}°  Dec={params['dec']:.5f}°  "
          f"parallax={params['parallax']:.4f} mas  RUWE={params['ruwe']:.3f}")

    print("[2/2] Querying Gaia scanning law (gaiaunlimited) ...")
    t_obs, psi, f = query_scanning_law(params["ra"], params["dec"])
    print(f"      {len(t_obs)} CCD observations retrieved.")

    params["t_obs"]           = t_obs
    params["scan_angle"]      = psi
    params["parallax_factor"] = f

    return params


def print_summary(params: dict) -> None:
    """Print a tidy summary of the query results."""
    print("\n" + "=" * 60)
    print(f"  Gaia DR3  source_id = {params['source_id']}")
    print("=" * 60)
    print(f"  RA            = {params['ra']:.6f} deg")
    print(f"  Dec           = {params['dec']:.6f} deg")
    print(f"  Parallax      = {params['parallax']:.4f} ± {params['parallax_error']:.4f} mas")
    print(f"  pmRA          = {params['pmra']:.4f} ± {params['pmra_error']:.4f} mas/yr")
    print(f"  pmDec         = {params['pmdec']:.4f} ± {params['pmdec_error']:.4f} mas/yr")
    print(f"  RUWE          = {params['ruwe']:.4f}")
    print(f"  G mag         = {params['g_mag']:.3f}")
    print(f"  N_obs (AL)    = {params['astrometric_n_good_obs_al']}")
    print()
    print("  Covariance matrix (ra, dec, plx, pmra, pmdec) [mas²]:")
    cov = params["covariance_matrix"]
    for row in cov:
        print("    " + "  ".join(f"{v:+10.5f}" for v in row))
    print()
    n = len(params["t_obs"])
    print(f"  Scanning law: {n} CCD observations")
    print(f"    t_obs[0:3]            = {params['t_obs'][:3]}")
    print(f"    scan_angle[0:3] [rad] = {params['scan_angle'][:3]}")
    print(f"    scan_angle[0:3] [deg] = {np.rad2deg(params['scan_angle'][:3])}")
    print(f"    parallax_factor[0:3]  = {params['parallax_factor'][:3]}")
    print("=" * 60 + "\n")


# =============================================================================
# Part 2: Forward Modeling Gaia Along-Scan Positions
# =============================================================================

import pytensor.tensor as pt

GAIA_EPOCH_OFFSET_DAYS = 2192.0   # TCB days from 2010-01-01 to J2016.0
DAYS_PER_YEAR          = 365.25
TWO_PI                 = 2.0 * np.pi


def tcb_days_to_yr2016(t_tcb_days):
    return (t_tcb_days - GAIA_EPOCH_OFFSET_DAYS) / DAYS_PER_YEAR


def _eccentric_anomaly_pt(t_yr, period_yr, eccentricity, t_p_yr, steps=6):
    """
    Solve Kepler's equation for eccentric anomaly E via Newton–Raphson.
    """
    M = pt.mod((t_yr - t_p_yr) / period_yr * TWO_PI, TWO_PI)
    E = M + 0.0
    for i in range(steps):
        E = E - (E - eccentricity * pt.sin(E) - M) / (1.0 - eccentricity * pt.cos(E))
    return E


def _orbital_xy_pt(t_yr, period_yr, eccentricity, t_p_yr):
    """
    Compute dimensionless Thiele-Innes orbital coordinates X, Y.
    """
    E = _eccentric_anomaly_pt(t_yr, period_yr, eccentricity, t_p_yr)
    X = pt.cos(E) - eccentricity
    Y = pt.sqrt(1.0 - eccentricity ** 2) * pt.sin(E)
    return X, Y


def _thiele_innes_pt(semimajor_mas, inclination_deg, Omega_deg, omega_deg):
    """
    Compute Thiele-Innes coefficients A, B, F, G in mas.
    """
    i = pt.deg2rad(inclination_deg)
    W = pt.deg2rad(Omega_deg)
    w = pt.deg2rad(omega_deg)

    A = semimajor_mas * ( pt.cos(w) * pt.cos(W) - pt.sin(w) * pt.sin(W) * pt.cos(i))
    B = semimajor_mas * ( pt.cos(w) * pt.sin(W) + pt.sin(w) * pt.cos(W) * pt.cos(i))
    F = semimajor_mas * (-pt.sin(w) * pt.cos(W) - pt.cos(w) * pt.sin(W) * pt.cos(i))
    G = semimajor_mas * (-pt.sin(w) * pt.sin(W) + pt.cos(w) * pt.cos(W) * pt.cos(i))
    return A, B, F, G


def _photocenter_offset_pt(
    t_yr, semimajor_mas, period_yr, eccentricity,
    inclination_deg, Omega_deg, omega_deg, t_p_yr,
    mass_ratio, lum_ratio
):
    """
    Photocenter offset (Δα*, Δδ) in mas due to the companion's orbital motion.
    """
    A, B, F, G = _thiele_innes_pt(semimajor_mas, inclination_deg, Omega_deg, omega_deg)
    X, Y       = _orbital_xy_pt(t_yr, period_yr, eccentricity, t_p_yr)

    w_eff = mass_ratio / (1.0 + mass_ratio) - lum_ratio / (1.0 + lum_ratio)

    dra_mas  = (B * X + G * Y) * w_eff
    ddec_mas = (A * X + F * Y) * w_eff
    return dra_mas, ddec_mas


def single_star(
    t_tcb_days, scan_angle, parallax_factor,
    ra_off, dec_off, pmra, pmdec, parallax_mas,
):
    """
    Gaia AL positions for a single-star model.

    AL(t) = (ra_off + pmra·t_yr)·sin(ψ) + (dec_off + pmdec·t_yr)·cos(ψ) + ϖ·f(t)
    """
    t_yr = tcb_days_to_yr2016(t_tcb_days)
    dra  = ra_off  + pmra  * t_yr
    ddec = dec_off + pmdec * t_yr
    AL_star = dra * pt.sin(scan_angle) + ddec * pt.cos(scan_angle) + parallax_mas * parallax_factor
    return AL_star


def planet_model(
    t_tcb_days, scan_angle, parallax_factor,
    ra_off, dec_off, pmra, pmdec, parallax_mas,
    semimajor_au, inclination_deg, eccentricity,
    Omega_deg, omega_deg, t_p_yr,
    Mp, Ms, lum_ratio=0.0,
):
    """
    Gaia AL positions for a star + planet-companion model.
    """
    t_yr = tcb_days_to_yr2016(t_tcb_days)

    period_yr     = pt.sqrt(semimajor_au ** 3 / (Mp + Ms))
    semimajor_mas = semimajor_au * parallax_mas
    q             = Mp / Ms

    dra_planet, ddec_planet = _photocenter_offset_pt(
        t_yr, semimajor_mas, period_yr, eccentricity,
        inclination_deg, Omega_deg, omega_deg, t_p_yr,
        q, lum_ratio
    )

    dra  = ra_off  + pmra  * t_yr + dra_planet
    ddec = dec_off + pmdec * t_yr + ddec_planet

    AL_planet = dra * pt.sin(scan_angle) + ddec * pt.cos(scan_angle) + parallax_mas * parallax_factor
    return AL_planet


# =============================================================================
# Part 3: RUWE Calculation  — FIXED
# =============================================================================

import os
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")


def sigma_al(g_mag: float) -> float:
    """
    Gaia AL single-measurement uncertainty as a function of G magnitude.

    σ_AL [mas] = sqrt( σ_floor² + (10^(0.2*(G - 12.09)))² )

    This matches the normalization function used by Gaia internally to
    compute RUWE (Lindegren et al. 2021, Appendix A).  The photon-noise
    term 10^(0.2*(G-12.09)) is evaluated at the ACTUAL G magnitude —
    NOT clipped to G=13.

    The previous code used max(G, 13), which floored σ at ~1.5 mas for
    all stars brighter than G=13.  For a G~8.5 star the correct σ is
    ~0.19 mas; the clipped version returned 1.52 mas — 8× too large.
    This made chi²_planet ~64× too small, so ruwe_model was always ≈ 1
    regardless of the companion parameters.
    """
    sigma_phot  = 10.0 ** (0.2 * (float(g_mag) - 12.09))
    sigma_floor = 0.029   # mas — calibration noise floor
    return float(np.sqrt(sigma_floor**2 + sigma_phot**2))


def _design_matrix_np(t_tcb_days: np.ndarray,
                       scan_angle: np.ndarray,
                       parallax_factor: np.ndarray) -> np.ndarray:
    """
    Build the N×5 astrometric design matrix.

    Columns encode sensitivity to (Δα₀, Δδ₀, ϖ, μ_α, μ_δ).
    """
    t_yr    = (t_tcb_days - GAIA_EPOCH_OFFSET_DAYS) / DAYS_PER_YEAR
    sin_psi = np.sin(scan_angle)
    cos_psi = np.cos(scan_angle)

    A = np.column_stack([
        sin_psi,
        cos_psi,
        parallax_factor,
        t_yr * sin_psi,
        t_yr * cos_psi,
    ])
    return A


def _hat_matrix_np(A: np.ndarray) -> np.ndarray:
    """
    OLS hat matrix H = A (AᵀA)⁻¹ Aᵀ
    """
    ATA     = A.T @ A
    ATA_inv = np.linalg.inv(ATA)
    H       = A @ ATA_inv @ A.T
    return H


def precompute_projection(t_tcb_days: np.ndarray,
                           scan_angle: np.ndarray,
                           parallax_factor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Precompute the residual projection matrix (I - H) and the
    noise-floor chi² contribution in numpy.

    The noise floor term accounts for the fact that even a perfect
    single-star model has RUWE ≈ 1 due to measurement noise.  For N
    independent AL observations with variance σ², the projected noise
    has expected sum-of-squares σ² * trace(I - H) = σ² * (N - 5).
    Dividing by σ² * (N - 5) gives an expected chi²_reduced = 1, i.e.
    RUWE_floor = 1.

    Returns
    -------
    I_minus_H     : np.ndarray, shape (N, N)
        Residual projection matrix.
    noise_chi2_dof : float
        trace(I - H) / (N - 5).  For a well-conditioned 5-parameter fit
        this equals exactly 1.0, but we compute it explicitly so the
        function generalises to any number of free parameters or
        degenerate scanning-law configurations.
    """
    A         = _design_matrix_np(t_tcb_days, scan_angle, parallax_factor)
    H         = _hat_matrix_np(A)
    I_minus_H = np.eye(len(t_tcb_days)) - H
    N         = len(t_tcb_days)
    dof       = N - 5
    # trace(I - H) should equal dof for full-rank A, but compute it
    # explicitly for robustness.
    noise_chi2_dof = float(np.trace(I_minus_H)) / dof
    return I_minus_H, noise_chi2_dof


def compute_ruwe(
    al_model,
    I_minus_H: np.ndarray,
    noise_chi2_dof: float,
    g_mag: float,
    N: int,
):
    """
    Model-predicted RUWE including the measurement-noise floor.

    Physics
    -------
    The actual Gaia AL residuals are:

        r_i = (AL_observed_i) - (best-fit 5-param model)_i
            = (noise_i + planet_signal_i) - (fit absorbs 5 dof)

    In terms of our forward model:

        r_planet = (I - H) @ al_model      [planet excess, mas]

    The noise contributes independently:

        E[Σ (r_noise_i / σ)²] = trace(I - H) = N - 5

    So the full chi²_reduced has two additive parts:

        chi²_red = [Σ (r_planet_i / σ)² + Σ (r_noise_i / σ)²] / (N - 5)

    Taking the expectation over noise realizations:

        E[chi²_red] = chi²_planet / (N-5) + noise_chi2_dof

    where noise_chi2_dof ≈ 1 for a well-conditioned fit.  Therefore:

        RUWE_model = sqrt( chi²_planet/(N-5) + noise_chi2_dof )

    This recovers RUWE ≈ 1 when there is no planet signal, and
    RUWE > 1 when the planet adds excess AL scatter that cannot be
    absorbed by the 5-parameter fit.

    Parameters
    ----------
    al_model        : PyTensor vector — synthetic AL positions (mas).
    I_minus_H       : np.ndarray (N, N) — precomputed residual projector.
    noise_chi2_dof  : float — trace(I-H)/(N-5), ≈ 1.0 for typical fits.
    g_mag           : float — Gaia G magnitude (sets σ_AL).
    N               : int — number of AL observations.

    Returns
    -------
    ruwe_model : PyTensor scalar
    """
    # Planet-induced AL residuals: everything the 5-param fit cannot absorb.
    # We expand (I-H) @ al_model as a list-comprehension over rows to avoid
    # the PyTensor C-codegen bug with 2D tensor contractions.
    residuals_planet = pt.stack([
        pt.sum(pt.as_tensor_variable(row) * al_model)
        for row in I_minus_H.astype("float64")
    ])

    sigma = sigma_al(g_mag)

    # chi²_reduced from the planet signal alone
    chi2_planet_dof = pt.sum((residuals_planet / sigma) ** 2) / (N - 5)

    # Add the noise-floor contribution (≈ 1 for a well-conditioned fit).
    # This is the key fix: without this term, RUWE_model → 0 when the
    # planet signal is small, making it impossible to match RUWE_obs ~ 1.5.
    ruwe_model = pt.sqrt(chi2_planet_dof + noise_chi2_dof)
    return ruwe_model


# =============================================================================
# Part 4: Bayesian Inference with PyMC
# =============================================================================

import pymc as pm
import arviz as az

MJUP_MSUN = 9.547919e-4
RUWE_ERR  = 0.01


def build_model(params: dict, Ms: float = 1.0, lum_ratio: float = 0.0):
    """
    Build and return the PyMC model for RUWE-based companion inference.

    Parameters
    ----------
    params : dict  — output of query_companion().
    Ms     : float — host star mass in solar masses.
    lum_ratio : float — L_companion / L_star (0 for dark companion).
    """
    t_np   = params["t_obs"].astype("float64")
    psi_np = params["scan_angle"].astype("float64")
    f_np   = params["parallax_factor"].astype("float64")
    N      = len(t_np)

    # Precompute (I - H) and the noise floor contribution.
    # noise_chi2_dof ≈ 1 ensures RUWE_model → 1 with no planet signal.
    I_minus_H, noise_chi2_dof = precompute_projection(t_np, psi_np, f_np)

    t_pt   = pt.as_tensor_variable(t_np)
    psi_pt = pt.as_tensor_variable(psi_np)
    f_pt   = pt.as_tensor_variable(f_np)

    ruwe_obs = float(params["ruwe"])
    g_mag    = float(params["g_mag"])

    with pm.Model() as model:

        # ---- Astrometric priors ----
        ra_off = pm.Normal("ra_off", mu=0.0, sigma=5.0 * params["ra_error"])
        dec_off = pm.Normal("dec_off", mu=0.0, sigma=5.0 * params["dec_error"])
        pmra = pm.Normal("pmra", mu=params["pmra"], sigma=5.0 * params["pmra_error"])
        pmdec = pm.Normal("pmdec", mu=params["pmdec"], sigma=5.0 * params["pmdec_error"])
        parallax = pm.TruncatedNormal(
            "parallax",
            mu=abs(params["parallax"]) if abs(params["parallax"]) > 0.1 else 1.0,
            sigma=5.0 * params["parallax_error"],
            lower=0.1,
        )

        # ---- Orbital priors ----
        log_a = pm.Uniform("log_a", lower=-1.0, upper=2.0)
        a_au  = pm.Deterministic("a_au", 10.0 ** log_a)

        cos_i = pm.Uniform("cos_i", lower=-1.0, upper=1.0)
        inc   = pm.Deterministic("inc_deg", pt.arccos(cos_i) * 180.0 / np.pi)

        ecc   = pm.Beta("ecc", alpha=1.12, beta=3.09)

        Omega = pm.Uniform("Omega_deg", lower=0.0, upper=360.0)
        omega = pm.Uniform("omega_deg", lower=0.0, upper=360.0)

        tp_frac = pm.Uniform("tp_frac", lower=0.0, upper=1.0)

        # ---- Planet mass prior ----
        log_Mp_jup = pm.Uniform("log_Mp_jup", lower=-1.0, upper=np.log10(80.0))
        Mp_jup     = pm.Deterministic("Mp_jup", 10.0 ** log_Mp_jup)
        Mp_msun    = pm.Deterministic("Mp_msun", Mp_jup * MJUP_MSUN)

        # ---- Jitter / noise term ----
        log_sigma0 = pm.Uniform("log_sigma0", lower=-3.0, upper=0.0)
        sigma0     = pm.Deterministic("sigma0", 10.0 ** log_sigma0)

        # ---- Derived quantities ----
        period_yr = pm.Deterministic(
            "period_yr",
            pt.sqrt(a_au ** 3 / (Mp_msun + Ms))
        )
        tp_yr = pm.Deterministic("tp_yr", tp_frac * period_yr)

        # ---- Forward model ----
        al_pred = planet_model(
            t_tcb_days      = t_pt,
            scan_angle      = psi_pt,
            parallax_factor = f_pt,
            ra_off          = ra_off,
            dec_off         = dec_off,
            pmra            = pmra,
            pmdec           = pmdec,
            parallax_mas    = parallax,
            semimajor_au    = a_au,
            inclination_deg = inc,
            eccentricity    = ecc,
            Omega_deg       = Omega,
            omega_deg       = omega,
            t_p_yr          = tp_yr,
            Mp              = Mp_msun,
            Ms              = Ms,
            lum_ratio       = lum_ratio,
        )

        # ---- RUWE (fixed: now includes noise floor) ----
        ruwe_model = pm.Deterministic(
            "ruwe_model",
            compute_ruwe(al_pred, I_minus_H, noise_chi2_dof, g_mag, N)
        )

        # ---- Likelihood ----
        ruwe_sigma = pm.Deterministic(
            "ruwe_sigma",
            pt.sqrt(RUWE_ERR ** 2 + sigma0 ** 2)
        )

        pm.Normal(
            "ruwe_likelihood",
            mu       = ruwe_model,
            sigma    = ruwe_sigma,
            observed = ruwe_obs,
        )

    return model


def run_inference(
    params: dict,
    Ms: float = 1.0,
    lum_ratio: float = 0.0,
    n_draws: int = 1000,
    n_tune: int = 1000,
    n_chains: int = 2,
    target_accept: float = 0.9,
    random_seed: int = 42,
) -> az.InferenceData:
    """
    Run NUTS sampling on the RUWE companion model.
    """
    model = build_model(params, Ms=Ms, lum_ratio=lum_ratio)

    with model:
        print(f"\nSampling {n_chains} chains × {n_draws} draws "
              f"(+ {n_tune} tuning steps each)...")
        idata = pm.sample(
            draws                = n_draws,
            tune                 = n_tune,
            chains               = n_chains,
            target_accept        = target_accept,
            random_seed          = random_seed,
            progressbar          = True,
            return_inferencedata = True,
        )

    return idata


def print_posterior_summary(idata: az.InferenceData) -> None:
    """Print key posterior statistics."""
    var_names = [
        "parallax", "pmra", "pmdec",
        "a_au", "inc_deg", "ecc", "Omega_deg", "omega_deg",
        "Mp_jup", "period_yr", "sigma0", "ruwe_model",
    ]
    present = [v for v in var_names if v in idata.posterior.data_vars]
    summary = az.summary(idata, var_names=present, round_to=4)
    print("\n" + "=" * 70)
    print("  Posterior summary")
    print("=" * 70)
    print(summary.to_string())
    print("=" * 70 + "\n")

    rhat = az.rhat(idata, var_names=present)
    max_rhat = float(max(rhat[v].values.max() for v in present if v in rhat.data_vars))
    print(f"  Max R-hat : {max_rhat:.4f}  (< 1.01 is good)")

    ess = az.ess(idata, var_names=present)
    min_ess = float(min(ess[v].values.min() for v in present if v in ess.data_vars))
    print(f"  Min ESS   : {min_ess:.0f}   (> 400 is good)")
    print()


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    SOURCE_ID = "203551385701760"
    Ms = 0.96

    params = query_companion(SOURCE_ID)

    print("\nRunning prior predictive check (10 samples)...")
    model = build_model(params, Ms=Ms)
    with model:
        prior = pm.sample_prior_predictive(samples=10, random_seed=42)
    ruwe_prior = prior.prior["ruwe_model"].values.flatten()
    print(f"  Prior RUWE range: [{ruwe_prior.min():.3f}, {ruwe_prior.max():.3f}]")
    print(f"  Observed RUWE   : {params['ruwe']:.4f}")

    idata = run_inference(
        params,
        Ms            = Ms,
        lum_ratio     = 0.0,
        n_draws       = 3000,
        n_tune        = 500,
        n_chains      = 2,
        target_accept = 0.9,
        random_seed   = 42,
    )

    print_posterior_summary(idata)

    out_path = f"posterior_{SOURCE_ID}.nc"
    idata.to_netcdf(out_path)
    print(f"Posterior saved to: {out_path}")
    print(f"Load with: az.from_netcdf('{out_path}')")