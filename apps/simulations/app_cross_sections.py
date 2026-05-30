

from __future__ import annotations

import importlib
import inspect
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# =============================================================================
# Constantes y unidades
# =============================================================================

G_F = 1.1663787e-5          # GeV^{-2}
COS_CABIBBO = 0.97420
M_N = 0.939                 # GeV, masa media del nucleón
M_MU = 0.1056583755         # GeV
M_PI = 0.13957              # GeV
G_A0 = -1.267
M_A_STD = 1.032             # GeV
M_A_HIGH = 1.35             # GeV
M_A_MONO_DEFAULT = 0.75     # GeV, escala monopolar representativa

M_A1 = 1.23                 # GeV, escala del mesón axial a1(1260)
ALPHA_SOFTPION = 0.93
GAMMA_SOFTPION = 0.53       # GeV^{-2}
ALPHA_PCAC_2C = 1.01
GAMMA_PCAC_2C = 0.54        # GeV^{-2}

M_A_BBBA07 = 1.014          # GeV
BBBA07_XI_NODES = np.array([0.0, 1.0/6.0, 1.0/3.0, 0.5, 2.0/3.0, 5.0/6.0, 1.0])
BBBA07_AFA_COEFFS = np.array([1.0000, 0.9207, 0.9795, 1.0480, 1.0516, 1.2874, 0.7707])

T_CUT_Z_AXIAL = 9.0 * M_PI**2
T0_Z_DEFAULT = -0.28
Z_PRESET_DIPOLE_LIKE = np.array([-1.69, 0.20, 2.31, -1.15])
Z_PRESET_SLOW_FALL = np.array([-1.66, 0.098, 2.2245, -0.9325])

AXIAL_MODEL_OPTIONS = [
    "Dipolo estándar",
    "Dipolo con M_A variable",
    "Dipolo alto M_A=1.35 GeV",
    "Monopolar",
    "Dos componentes: Soft-Pion",
    "Dos componentes: PCAC",
    "BBBA2007",
    "Expansión z cercana al dipolo",
    "Expansión z con caída más lenta",
]

AXIAL_COMPARISON_DEFAULTS = [
    "Dipolo estándar",
    "Monopolar",
    "Dos componentes: Soft-Pion",
    "BBBA2007",
    "Expansión z con caída más lenta",
]

MU_P = 2.79284734463
MU_N = -1.91304273
M_V = 0.843                 # GeV, escala vectorial dipolar
LAMBDA_N_GALSTER = 5.6

GEV2_TO_FM2 = 0.038937944   # 1 GeV^{-2} = 0.038937944 fm^2
FM2_TO_1E_MINUS_13_FM2 = 1.0e13

DEFAULT_Q2_MIN = 1.0e-4


# =============================================================================
# Utilidades de estilo
# =============================================================================

def setup_page() -> None:
    st.set_page_config(
        page_title="Secciones eficaces CCQE desde el formalismo tensorial",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
        .theory-card {
            border: 1px solid rgba(120,120,120,0.25);
            border-radius: 12px;
            padding: 1.0rem 1.1rem;
            background: rgba(120,120,120,0.06);
            margin-bottom: 1rem;
        }
        .small-note {font-size: 0.92rem; opacity: 0.82;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str | None = None) -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f"<p class='small-note'>{subtitle}</p>", unsafe_allow_html=True)


def apply_common_axes_style(ax: plt.Axes, xscale: str | None = None) -> None:
    if xscale:
        ax.set_xscale(xscale)
    ax.grid(True, which="both", alpha=0.25)
    ax.tick_params(direction="in", top=True, right=True)


def finalize_figure(fig: plt.Figure) -> None:
    """Apply a robust layout adjustment without crashing the app."""
    try:
        fig.tight_layout()
    except Exception:
        # Streamlit can still render the figure; avoid killing the app because of layout.
        fig.subplots_adjust(left=0.10, right=0.98, bottom=0.14, top=0.90, wspace=0.25)




def figure_to_bytes(fig: plt.Figure, file_format: str) -> bytes:
    """Convierte una figura matplotlib a bytes para descarga en Streamlit."""
    buffer = io.BytesIO()
    save_kwargs = {"format": file_format, "bbox_inches": "tight"}
    if file_format == "png":
        save_kwargs["dpi"] = 300
    fig.savefig(buffer, **save_kwargs)
    buffer.seek(0)
    return buffer.getvalue()


def render_figure_download_buttons(fig: plt.Figure, basename: str) -> None:
    """Muestra botones de descarga sin bloquear la interacción de la app.

    En Streamlit, el contenido de los expanders se ejecuta aunque estén cerrados.
    Si se convierten todas las figuras a PDF/PNG en cada rerun, cada pequeño
    cambio de un slider obliga a regenerar muchos archivos y la app parece no
    responder. Por eso los bytes de descarga se preparan solo cuando el usuario
    activa explícitamente la casilla de esta figura.
    """
    safe_name = (
        basename.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )
    prepare_key = f"prepare_download_{safe_name}"
    with st.expander("Descargar esta gráfica", expanded=False):
        prepare = st.checkbox(
            "Preparar archivos de descarga",
            value=False,
            key=prepare_key,
            help="Actívalo solo cuando quieras descargar esta figura. Así los sliders siguen respondiendo rápido.",
        )
        if not prepare:
            st.caption("Activa la casilla para generar el PDF vectorial y el PNG 300 dpi de esta gráfica.")
            return

        with st.spinner("Generando archivos..."):
            pdf_bytes = figure_to_bytes(fig, "pdf")
            png_bytes = figure_to_bytes(fig, "png")

        col_pdf, col_png = st.columns(2)
        with col_pdf:
            st.download_button(
                "PDF vectorial",
                data=pdf_bytes,
                file_name=f"{safe_name}.pdf",
                mime="application/pdf",
                key=f"download_{safe_name}_pdf",
            )
        with col_png:
            st.download_button(
                "PNG 300 dpi",
                data=png_bytes,
                file_name=f"{safe_name}.png",
                mime="image/png",
                key=f"download_{safe_name}_png",
            )


def safe_geomspace(qmin: float, qmax: float, n: int = 700) -> np.ndarray:
    qmin = max(float(qmin), 1.0e-8)
    qmax = max(float(qmax), qmin * 1.001)
    return np.geomspace(qmin, qmax, n)


def to_1e_minus_13_fm2(value_gev_minus_2: np.ndarray | float) -> np.ndarray:
    """Convierte GeV^{-2} a unidades de 10^{-13} fm^2."""
    return np.asarray(value_gev_minus_2, dtype=float) * GEV2_TO_FM2 * FM2_TO_1E_MINUS_13_FM2


# =============================================================================
# Factores de forma electromagnéticos y vectores
# =============================================================================

def dipole_vector(q2: np.ndarray | float) -> np.ndarray:
    q2 = np.asarray(q2, dtype=float)
    return 1.0 / (1.0 + q2 / M_V**2) ** 2


def sachs_galster(q2: np.ndarray | float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parametrización Galster/dipolar para los factores de Sachs EM."""
    q2 = np.asarray(q2, dtype=float)
    tau = q2 / (4.0 * M_N**2)
    gd = dipole_vector(q2)

    g_ep = gd
    g_mp = MU_P * gd
    g_mn = MU_N * gd
    g_en = -MU_N * tau * gd / (1.0 + LAMBDA_N_GALSTER * tau)

    return g_ep, g_mp, g_en, g_mn


def _coerce_gkex_output(out, q2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Acepta varias salidas posibles de una función sachs_gkex del proyecto."""
    if isinstance(out, dict):
        keys = {k.lower().replace("_", ""): k for k in out.keys()}
        gep = out[keys.get("gep", keys.get("gepq2", "GEp"))]
        gmp = out[keys.get("gmp", keys.get("gmpq2", "GMp"))]
        gen = out[keys.get("gen", keys.get("genq2", "GEn"))]
        gmn = out[keys.get("gmn", keys.get("gmnq2", "GMn"))]
        return tuple(np.asarray(x, dtype=float) for x in (gep, gmp, gen, gmn))
    if isinstance(out, (tuple, list)) and len(out) >= 4:
        return tuple(np.asarray(x, dtype=float) for x in out[:4])
    raise ValueError("Formato de salida GKeX no reconocido.")


@st.cache_resource(show_spinner=False)
def load_project_gkex() -> tuple[Callable | None, str]:
    """Intenta cargar la implementación GKeX real del proyecto."""
    root = Path.cwd()
    for candidate in [root, root / "src", root.parent, root.parent / "src"]:
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    module_candidates = [
        "src.form_factors_gkex",
        "form_factors_gkex",
        "src.vector_form_factors",
        "form_factors_vector",
    ]

    function_candidates = [
        "sachs_gkex",
        "compute_sachs_gkex",
        "gkex_sachs",
        "sachs_form_factors_gkex",
    ]

    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        for function_name in function_candidates:
            fn = getattr(module, function_name, None)
            if callable(fn):
                return fn, f"{module_name}.{function_name}"

    return None, "GKeX no encontrado"


def _evaluate_gkex_function(fn: Callable, q2_value: np.ndarray | float):
    """Evalúa una función GKeX aceptando varias firmas habituales."""
    sig = inspect.signature(fn)
    params = sig.parameters

    if "Q2" in params:
        return fn(Q2=q2_value)
    if "q2" in params:
        return fn(q2=q2_value)
    if "Q2_GeV2" in params:
        return fn(Q2_GeV2=q2_value)

    # Firma tipo sachs_gkex(Q2, M=..., p=...). En este caso basta pasar Q2
    # como primer argumento posicional y dejar los parámetros opcionales por defecto.
    return fn(q2_value)


def sachs_gkex_required(q2: np.ndarray | float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Usa exclusivamente la implementación GKeX real del proyecto.

    Si no se encuentra `src/form_factors_gkex.py`, la app se detiene con un
    error explícito para evitar mostrar resultados con un modelo de respaldo.

    La implementación GKeX del proyecto puede estar escrita para valores escalares
    de Q2. Por eso primero se intenta una evaluación vectorial y, si falla, se
    evalúa punto a punto reconstruyendo arrays con la misma forma que la entrada.
    """
    q2_arr = np.asarray(q2, dtype=float)
    scalar_input = q2_arr.ndim == 0
    q2_eval = float(q2_arr) if scalar_input else q2_arr

    fn, source = load_project_gkex()
    if fn is None:
        raise RuntimeError(
            "No se ha encontrado la implementación real de GKeX. "
            "Asegúrate de que existe `src/form_factors_gkex.py` en el repositorio "
            "y de que contiene una función `sachs_gkex` compatible con la app."
        )

    # Intento rápido: algunas implementaciones aceptan arrays directamente.
    try:
        out = _evaluate_gkex_function(fn, q2_eval)
        return _coerce_gkex_output(out, q2_arr)
    except Exception:
        pass

    # Intento robusto: implementación escalar evaluada punto a punto.
    try:
        q2_flat = np.atleast_1d(q2_arr).astype(float).ravel()
        values = []
        for q2_i in q2_flat:
            out_i = _evaluate_gkex_function(fn, float(q2_i))
            values.append(_coerce_gkex_output(out_i, np.asarray(float(q2_i))))

        shape = np.atleast_1d(q2_arr).shape
        stacked = []
        for idx in range(4):
            arr = np.array([item[idx] for item in values], dtype=float).reshape(shape)
            if scalar_input:
                arr = np.asarray(arr[0], dtype=float)
            stacked.append(arr)

        return tuple(stacked)
    except Exception as exc:
        raise RuntimeError(
            f"La implementación GKeX encontrada en {source} no se ha podido evaluar. "
            "No se usará ningún modelo de respaldo."
        ) from exc


def sachs_form_factors(q2: np.ndarray | float, model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if model == "Galster":
        return sachs_galster(q2)
    return sachs_gkex_required(q2)


def vector_dirac_pauli_isovector(q2: np.ndarray | float, model: str) -> tuple[np.ndarray, np.ndarray]:
    """Construye F_1^V y F_2^V desde Sachs isovectoriales, con el factor 1/2 usado en el TFG."""
    q2 = np.asarray(q2, dtype=float)
    tau_val = q2 / (4.0 * M_N**2)

    g_ep, g_mp, g_en, g_mn = sachs_form_factors(q2, model)

    delta_ge = g_ep - g_en
    delta_gm = g_mp - g_mn

    f1v = (delta_ge + tau_val * delta_gm) / (2.0 * (1.0 + tau_val))
    f2v = (delta_gm - delta_ge) / (2.0 * (1.0 + tau_val))

    return f1v, f2v


# =============================================================================
# Factores de forma axiales
# =============================================================================

def ga_dipole(q2: np.ndarray | float, m_a: float = M_A_STD) -> np.ndarray:
    """Factor de forma axial dipolar."""
    q2 = np.asarray(q2, dtype=float)
    return G_A0 / (1.0 + q2 / m_a**2) ** 2


def ga_monopole(q2: np.ndarray | float, m_a_tilde: float = M_A_MONO_DEFAULT) -> np.ndarray:
    """Factor de forma axial monopolar."""
    q2 = np.asarray(q2, dtype=float)
    return G_A0 / (1.0 + q2 / m_a_tilde**2)


def ga_two_component(
    q2: np.ndarray | float,
    alpha: float,
    gamma: float,
    m_a1: float = M_A1,
) -> np.ndarray:
    """Modelo de dos componentes para G_A."""
    q2 = np.asarray(q2, dtype=float)
    intrinsic = 1.0 / (1.0 + gamma * q2) ** 2
    meson_cloud = 1.0 - alpha + alpha * m_a1**2 / (m_a1**2 + q2)
    return G_A0 * intrinsic * meson_cloud


def ga_two_component_softpion(q2: np.ndarray | float) -> np.ndarray:
    return ga_two_component(q2, ALPHA_SOFTPION, GAMMA_SOFTPION)


def ga_two_component_pcac(q2: np.ndarray | float) -> np.ndarray:
    return ga_two_component(q2, ALPHA_PCAC_2C, GAMMA_PCAC_2C)


def xi_nachtmann_elastic(q2: np.ndarray | float) -> np.ndarray:
    """Variable de Nachtmann elástica usada en BBBA2007."""
    q2 = np.asarray(q2, dtype=float)
    tau_n = q2 / (4.0 * M_N**2)
    xi = np.zeros_like(q2, dtype=float)
    mask = tau_n > 0.0
    xi[mask] = 2.0 / (1.0 + np.sqrt(1.0 + 1.0 / tau_n[mask]))
    return xi


def lagrange_interpolator(x: np.ndarray | float, nodes: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Interpolación de Lagrange."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x, dtype=float)
    for j in range(len(nodes)):
        basis = np.ones_like(x, dtype=float)
        for k in range(len(nodes)):
            if k != j:
                basis *= (x - nodes[k]) / (nodes[j] - nodes[k])
        y += values[j] * basis
    return y


def afa_bbba07_correction(q2: np.ndarray | float) -> np.ndarray:
    """Función correctora A_{F_A}^{25}(xi) de BBBA2007."""
    xi = xi_nachtmann_elastic(q2)
    return lagrange_interpolator(xi, BBBA07_XI_NODES, BBBA07_AFA_COEFFS)


def ga_bbba07(q2: np.ndarray | float) -> np.ndarray:
    """Factor de forma axial BBBA2007."""
    q2 = np.asarray(q2, dtype=float)
    return ga_dipole(q2, M_A_BBBA07) * afa_bbba07_correction(q2)


def z_conformal(
    q2: np.ndarray | float,
    t0: float = T0_Z_DEFAULT,
    t_cut: float = T_CUT_Z_AXIAL,
) -> np.ndarray:
    """Variable conforme z(Q^2; t0, t_cut)."""
    q2 = np.asarray(q2, dtype=float)
    sqrt_num = np.sqrt(t_cut + q2)
    sqrt_ref = np.sqrt(t_cut - t0)
    return (sqrt_num - sqrt_ref) / (sqrt_num + sqrt_ref)


def ga_z_expansion(
    q2: np.ndarray | float,
    coeffs: np.ndarray,
    t0: float = T0_Z_DEFAULT,
    t_cut: float = T_CUT_Z_AXIAL,
) -> np.ndarray:
    """Expansión z normalizada de forma que G_A(0)=g_A."""
    q2 = np.asarray(q2, dtype=float)
    coeffs = np.asarray(coeffs, dtype=float)
    z = z_conformal(q2, t0=t0, t_cut=t_cut)
    z0 = float(z_conformal(0.0, t0=t0, t_cut=t_cut))
    series = np.ones_like(q2, dtype=float)
    for k, coeff in enumerate(coeffs, start=1):
        series += coeff * (z**k - z0**k)
    return G_A0 * series


def gp_pcac(q2: np.ndarray | float, ga_values: np.ndarray | float) -> np.ndarray:
    q2 = np.asarray(q2, dtype=float)
    ga_values = np.asarray(ga_values, dtype=float)
    return (4.0 * M_N**2 / (q2 + M_PI**2)) * ga_values


def axial_label(axial_choice: str) -> str:
    """Etiqueta compacta para leyendas."""
    labels = {
        "Dipolo estándar": "Dipolo",
        "Dipolo con M_A variable": r"Dipolo, $M_A$ variable",
        "Dipolo alto M_A=1.35 GeV": r"Dipolo, $M_A=1.35$ GeV",
        "Monopolar": r"Monopolar",
        "Dos componentes: Soft-Pion": r"2 componentes, Soft-Pion",
        "Dos componentes: PCAC": r"2 componentes, PCAC",
        "BBBA2007": "BBBA2007",
        "Expansión z cercana al dipolo": r"$z$ cercana al dipolo",
        "Expansión z con caída más lenta": r"$z$ caída lenta",
    }
    return labels.get(axial_choice, axial_choice)


def select_axial_model(
    q2: np.ndarray | float,
    axial_choice: str,
    m_a: float,
    m_a_tilde: float = M_A_MONO_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve G_A y G_P para el modelo axial seleccionado."""
    if axial_choice == "Dipolo estándar":
        ga = ga_dipole(q2, M_A_STD)
    elif axial_choice == "Dipolo con M_A variable":
        ga = ga_dipole(q2, m_a)
    elif axial_choice == "Dipolo alto M_A=1.35 GeV":
        ga = ga_dipole(q2, M_A_HIGH)
    elif axial_choice == "Monopolar":
        ga = ga_monopole(q2, m_a_tilde)
    elif axial_choice == "Dos componentes: Soft-Pion":
        ga = ga_two_component_softpion(q2)
    elif axial_choice == "Dos componentes: PCAC":
        ga = ga_two_component_pcac(q2)
    elif axial_choice == "BBBA2007":
        ga = ga_bbba07(q2)
    elif axial_choice == "Expansión z cercana al dipolo":
        ga = ga_z_expansion(q2, Z_PRESET_DIPOLE_LIKE)
    elif axial_choice == "Expansión z con caída más lenta":
        ga = ga_z_expansion(q2, Z_PRESET_SLOW_FALL)
    else:
        ga = ga_dipole(q2, M_A_STD)

    gp = gp_pcac(q2, ga)
    return ga, gp


# =============================================================================
# Cinemática elástica libre
# =============================================================================

def lepton_momentum(e_mu: np.ndarray | float, m_l: float = M_MU) -> np.ndarray:
    e_mu = np.asarray(e_mu, dtype=float)
    return np.sqrt(np.maximum(e_mu**2 - m_l**2, 0.0))


def cos_theta0_from_emu(e_nu: float, e_mu: np.ndarray | float, m_l: float = M_MU) -> np.ndarray:
    e_mu = np.asarray(e_mu, dtype=float)
    k_mu = lepton_momentum(e_mu, m_l)
    numerator = 2.0 * (e_nu * e_mu + M_N * e_mu - M_N * e_nu) - m_l**2
    denominator = 2.0 * e_nu * np.maximum(k_mu, 1.0e-15)
    return numerator / denominator


def elastic_condition(theta: float, e_nu: float, e_mu: float, m_l: float = M_MU) -> float:
    return float(cos_theta0_from_emu(e_nu, e_mu, m_l) - np.cos(theta))


def solve_emu_from_theta(e_nu: float, theta: float, m_l: float = M_MU) -> float:
    """Resuelve la energía del muón final para cinemática elástica libre.

    Usa bisección sobre la condición cos(theta0(E_mu)) = cos(theta).
    """
    e_min = m_l * (1.0 + 1.0e-10)
    e_max = e_nu * (1.0 - 1.0e-10)

    if e_max <= e_min:
        return np.nan

    f_min = elastic_condition(theta, e_nu, e_min, m_l)
    f_max = elastic_condition(theta, e_nu, e_max, m_l)

    if not np.isfinite(f_min) or not np.isfinite(f_max):
        return np.nan

    if f_min * f_max > 0:
        return np.nan

    a, b = e_min, e_max
    fa, fb = f_min, f_max
    for _ in range(90):
        c = 0.5 * (a + b)
        fc = elastic_condition(theta, e_nu, c, m_l)
        if abs(fc) < 1.0e-12:
            return c
        if fa * fc <= 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    return 0.5 * (a + b)


def kinematics_from_theta_grid(e_nu: float, theta_deg: np.ndarray, m_l: float = M_MU) -> pd.DataFrame:
    theta_rad = np.deg2rad(theta_deg)

    e_mu = np.array([solve_emu_from_theta(e_nu, th, m_l) for th in theta_rad])
    k_mu = lepton_momentum(e_mu, m_l)
    omega = e_nu - e_mu
    q2_abs = 2.0 * M_N * omega

    valid = np.isfinite(e_mu) & (e_mu > m_l) & (omega > 0.0) & (q2_abs > 0.0)

    return pd.DataFrame(
        {
            "theta_deg": theta_deg,
            "theta_rad": theta_rad,
            "cos_theta": np.cos(theta_rad),
            "E_mu": e_mu,
            "k_mu": k_mu,
            "omega": omega,
            "Q2_abs": q2_abs,
            "valid": valid,
        }
    )


def kinematics_from_energy_grid(e_nu: float, e_mu_grid: np.ndarray, m_l: float = M_MU) -> pd.DataFrame:
    e_mu = np.asarray(e_mu_grid, dtype=float)
    k_mu = lepton_momentum(e_mu, m_l)
    omega = e_nu - e_mu
    q2_abs = 2.0 * M_N * omega
    cos_theta = cos_theta0_from_emu(e_nu, e_mu, m_l)

    valid = (
        np.isfinite(cos_theta)
        & (np.abs(cos_theta) <= 1.0)
        & (e_mu > m_l)
        & (omega > 0.0)
        & (q2_abs > 0.0)
    )

    theta_rad = np.full_like(e_mu, np.nan)
    theta_rad[valid] = np.arccos(cos_theta[valid])

    return pd.DataFrame(
        {
            "E_mu": e_mu,
            "k_mu": k_mu,
            "omega": omega,
            "Q2_abs": q2_abs,
            "cos_theta": cos_theta,
            "theta_rad": theta_rad,
            "theta_deg": np.rad2deg(theta_rad),
            "valid": valid,
        }
    )


# =============================================================================
# Funciones W_i, contracción y secciones eficaces
# =============================================================================

@dataclass(frozen=True)
class CrossSectionConfig:
    vector_model: str = "Galster"
    axial_model: str = "Dipolo estándar"
    m_a: float = M_A_STD
    m_a_tilde: float = M_A_MONO_DEFAULT
    use_cabibbo: bool = False
    include_muon_mass: bool = True


def effective_gf(use_cabibbo: bool) -> float:
    return G_F * (COS_CABIBBO if use_cabibbo else 1.0)


def structure_functions_wi(
    q2_abs: np.ndarray | float,
    vector_model: str,
    axial_model: str,
    m_a: float = M_A_STD,
    m_a_tilde: float = M_A_MONO_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q2_abs = np.asarray(q2_abs, dtype=float)
    tau_val = q2_abs / (4.0 * M_N**2)

    f1v, f2v = vector_dirac_pauli_isovector(q2_abs, vector_model)
    ga, gp = select_axial_model(q2_abs, axial_model, m_a, m_a_tilde)
    fp = gp / (2.0 * M_N)

    w1 = tau_val * (f1v + f2v) ** 2 + (1.0 + tau_val) * ga**2
    w2 = f1v**2 + tau_val * f2v**2 + ga**2
    w3 = 2.0 * ga * (f1v + f2v)

    # Términos proporcionales a m_mu^2 en la contracción. Se conserva la parte real.
    w4 = ((q2_abs - 4.0 * M_N**2) / (4.0 * M_N**2) ** 2) * f2v**2 - (ga * fp / M_N) + tau_val * fp**2 - (f1v * f2v) / (2.0 * M_N**2)
    w5 = w2

    return w1, w2, w3, w4, w5


def contraction_eta_w(
    e_nu: float,
    e_mu: np.ndarray | float,
    cos_theta: np.ndarray | float,
    q2_abs: np.ndarray | float,
    config: CrossSectionConfig,
    beam: str = "neutrino",
) -> np.ndarray:
    e_mu = np.asarray(e_mu, dtype=float)
    cos_theta = np.asarray(cos_theta, dtype=float)
    q2_abs = np.asarray(q2_abs, dtype=float)

    m_l = M_MU if config.include_muon_mass else 0.0
    k_mu = lepton_momentum(e_mu, m_l)

    w1, w2, w3, w4, w5 = structure_functions_wi(q2_abs, config.vector_model, config.axial_model, config.m_a, config.m_a_tilde)

    # Con la convención usada en el TFG y en este código, G_A(0)<0 y
    # W_3 = 2 G_A (F_1^V+F_2^V). Para que el término de interferencia
    # vector--axial aumente la sección eficaz de neutrinos y reduzca la de
    # antineutrinos, el signo efectivo que multiplica a W_3 debe ser - para
    # neutrinos y + para antineutrinos.
    sign = -1.0 if beam == "neutrino" else +1.0

    term_1 = 2.0 * w1 * e_nu * (e_mu - k_mu * cos_theta)
    term_2 = w2 * e_nu * (e_mu + k_mu * cos_theta)
    term_3 = sign * (w3 / M_N) * e_nu * ((e_nu + e_mu) * (e_mu - k_mu * cos_theta) - m_l**2)
    term_4 = (m_l**2) * w4 * e_nu * (e_mu - k_mu * cos_theta)
    term_5 = -(w5 / M_N) * (m_l**2) * e_nu

    return term_1 + term_2 + term_3 + term_4 + term_5


def recoil_factor(e_nu: float, e_mu: np.ndarray | float, cos_theta: np.ndarray | float, config: CrossSectionConfig) -> np.ndarray:
    m_l = M_MU if config.include_muon_mass else 0.0
    e_mu = np.asarray(e_mu, dtype=float)
    cos_theta = np.asarray(cos_theta, dtype=float)
    k_mu = lepton_momentum(e_mu, m_l)

    return 1.0 + e_nu * (k_mu - e_mu * cos_theta) / (M_N * np.maximum(k_mu, 1.0e-15))


def dsigma_domega_from_kinematics(df: pd.DataFrame, e_nu: float, config: CrossSectionConfig, beam: str) -> np.ndarray:
    gf = effective_gf(config.use_cabibbo)
    valid = df["valid"].to_numpy(dtype=bool)

    e_mu = df["E_mu"].to_numpy(dtype=float)
    k_mu = df["k_mu"].to_numpy(dtype=float)
    cos_theta = df["cos_theta"].to_numpy(dtype=float)
    q2_abs = df["Q2_abs"].to_numpy(dtype=float)

    contr = contraction_eta_w(e_nu, e_mu, cos_theta, q2_abs, config, beam=beam)
    frec = recoil_factor(e_nu, e_mu, cos_theta, config)

    ds = (gf**2 / (4.0 * np.pi**2)) * (k_mu / e_nu) * contr / np.maximum(frec, 1.0e-15)
    ds = to_1e_minus_13_fm2(ds)

    ds[~valid] = np.nan
    ds[ds < 0] = np.nan
    return ds


def dsigma_demu_from_energy_grid(df: pd.DataFrame, e_nu: float, config: CrossSectionConfig, beam: str) -> np.ndarray:
    """Sección diferencial dσ/dE_mu según la forma integrada de la subsección 3.8.

    Se usa la normalización con el factor 2M que aparece en la expresión (3.93) del texto.
    """
    gf = effective_gf(config.use_cabibbo)
    valid = df["valid"].to_numpy(dtype=bool)

    e_mu = df["E_mu"].to_numpy(dtype=float)
    cos_theta = df["cos_theta"].to_numpy(dtype=float)
    q2_abs = df["Q2_abs"].to_numpy(dtype=float)

    contr = contraction_eta_w(e_nu, e_mu, cos_theta, q2_abs, config, beam=beam)
    ds = (2.0 * M_N * gf**2 / (4.0 * np.pi * e_nu**2)) * contr
    ds = to_1e_minus_13_fm2(ds)

    ds[~valid] = np.nan
    ds[ds < 0] = np.nan
    return ds


def total_cross_section_from_energy(
    e_nu: float,
    config: CrossSectionConfig,
    beam: str = "neutrino",
    n_grid: int = 600,
) -> float:
    """Sección eficaz total integrada numéricamente sobre E_mu.

    La integral se evalúa a partir de dσ/dE_mu en la región cinemáticamente
    permitida por |cos(theta_0)| <= 1. El resultado queda en unidades de
    10^{-13} fm^2.

    Por debajo del umbral cinemático no hay producción del leptón cargado,
    por lo que se devuelve sigma_tot = 0. Esto evita que la curva empiece
    artificialmente "en el aire" en la representación gráfica.
    """
    if e_nu <= M_MU:
        return 0.0

    e_mu_grid = np.linspace(M_MU * 1.001, e_nu * 0.999, n_grid)
    df = kinematics_from_energy_grid(e_nu, e_mu_grid, M_MU if config.include_muon_mass else 0.0)
    ds = dsigma_demu_from_energy_grid(df, e_nu, config, beam=beam)

    valid = np.isfinite(ds) & df["valid"].to_numpy(dtype=bool)
    if np.count_nonzero(valid) < 2:
        return 0.0

    return float(np.trapezoid(ds[valid], e_mu_grid[valid]))


@st.cache_data(show_spinner=False, max_entries=2048)
def total_cross_section_from_energy_cached(
    e_nu: float,
    vector_model: str,
    axial_model: str,
    m_a: float,
    m_a_tilde: float,
    use_cabibbo: bool,
    include_muon_mass: bool,
    beam: str,
    n_grid: int,
) -> float:
    """Versión cacheada de la integración total para que los sliders respondan."""
    cfg = CrossSectionConfig(
        vector_model=vector_model,
        axial_model=axial_model,
        m_a=m_a,
        m_a_tilde=m_a_tilde,
        use_cabibbo=use_cabibbo,
        include_muon_mass=include_muon_mass,
    )
    return total_cross_section_from_energy(float(e_nu), cfg, beam=beam, n_grid=int(n_grid))


def _relevant_axial_parameters(axial_model: str, m_a: float, m_a_tilde: float) -> tuple[float, float]:
    """Evita invalidar caches cuando un slider no afecta al modelo seleccionado."""
    m_a_eff = float(m_a) if axial_model == "Dipolo con M_A variable" else M_A_STD
    m_a_tilde_eff = float(m_a_tilde) if axial_model == "Monopolar" else M_A_MONO_DEFAULT
    return m_a_eff, m_a_tilde_eff


def total_cross_section_curve(
    e_values: np.ndarray,
    config: CrossSectionConfig,
    beam: str = "neutrino",
    n_grid: int = 450,
) -> np.ndarray:
    """Curva sigma_total(E_nu) en unidades de 10^{-13} fm^2."""
    m_a_eff, m_a_tilde_eff = _relevant_axial_parameters(config.axial_model, config.m_a, config.m_a_tilde)
    return np.array(
        [
            total_cross_section_from_energy_cached(
                round(float(e), 8),
                config.vector_model,
                config.axial_model,
                round(float(m_a_eff), 8),
                round(float(m_a_tilde_eff), 8),
                bool(config.use_cabibbo),
                bool(config.include_muon_mass),
                beam,
                int(n_grid),
            )
            for e in e_values
        ],
        dtype=float,
    )



# =============================================================================
# Componentes Streamlit
# =============================================================================


def _config_identity_for_plot(config: CrossSectionConfig) -> tuple[str, str, float, float, bool, bool]:
    """Identidad física de una configuración para evitar curvas duplicadas."""
    m_a_eff, m_a_tilde_eff = _relevant_axial_parameters(config.axial_model, config.m_a, config.m_a_tilde)
    return (
        config.vector_model,
        config.axial_model,
        round(float(m_a_eff), 6),
        round(float(m_a_tilde_eff), 6),
        bool(config.use_cabibbo),
        bool(config.include_muon_mass),
    )


def selected_config_label(config: CrossSectionConfig) -> str:
    """Etiqueta breve para la curva de la configuración global seleccionada."""
    label = rf"Seleccionado: {config.vector_model}, {axial_label(config.axial_model)}"
    if config.axial_model == "Dipolo con M_A variable":
        label += rf" ($M_A={config.m_a:.3f}$ GeV)"
    elif config.axial_model == "Monopolar":
        label += rf" ($\widetilde{{M}}_A={config.m_a_tilde:.3f}$ GeV)"
    return label


def benchmark_vector_axial_cases(config: CrossSectionConfig) -> list[tuple[str, CrossSectionConfig, str, float, int]]:
    """Configuraciones de referencia más la configuración global seleccionada.

    Las tres primeras curvas reproducen la comparación fija del TFG. Si el usuario
    cambia la parametrización en la barra lateral, se añade una cuarta curva
    ``Seleccionado`` para que los cambios de modelo y de parámetros tengan una
    respuesta visible sin alterar la disposición de la figura.
    """
    common = dict(
        m_a=config.m_a,
        m_a_tilde=config.m_a_tilde,
        use_cabibbo=config.use_cabibbo,
        include_muon_mass=config.include_muon_mass,
    )

    cases: list[tuple[str, CrossSectionConfig, str, float, int]] = [
        (
            r"Galster, $M_A=1.032$ GeV",
            CrossSectionConfig(
                vector_model="Galster",
                axial_model="Dipolo estándar",
                **common,
            ),
            "-",
            2.1,
            2,
        ),
        (
            r"GKeX, $M_A=1.032$ GeV",
            CrossSectionConfig(
                vector_model="GKeX",
                axial_model="Dipolo estándar",
                **common,
            ),
            "--",
            2.5,
            3,
        ),
        (
            r"GKeX, $M_A=1.35$ GeV",
            CrossSectionConfig(
                vector_model="GKeX",
                axial_model="Dipolo alto M_A=1.35 GeV",
                **common,
            ),
            "-",
            2.4,
            4,
        ),
    ]

    selected_id = _config_identity_for_plot(config)
    benchmark_ids = [_config_identity_for_plot(case_cfg) for _, case_cfg, *_ in cases]
    if selected_id not in benchmark_ids:
        cases.append((selected_config_label(config), config, ":", 2.7, 6))

    return cases


def sidebar_config() -> CrossSectionConfig:
    st.sidebar.header("Parámetros globales")

    vector_model = st.sidebar.selectbox("Modelo vectorial EM", ["Galster", "GKeX"], index=0)
    axial_model = st.sidebar.selectbox(
        "Modelo axial",
        AXIAL_MODEL_OPTIONS,
        index=0,
    )

    m_a = st.sidebar.slider(r"$M_A$ dipolar variable (GeV)", 0.80, 1.50, M_A_STD, 0.005)
    m_a_tilde = st.sidebar.slider(r"$\widetilde{M}_A$ monopolar (GeV)", 0.50, 1.00, M_A_MONO_DEFAULT, 0.005)

    include_muon_mass = st.sidebar.checkbox("Incluir masa del muón", value=True)
    use_cabibbo = st.sidebar.checkbox(r"Incluir factor $\cos\theta_C$", value=False)

    fn, source = load_project_gkex()
    if vector_model == "GKeX":
        if fn is None:
            st.sidebar.error("No se ha encontrado el módulo GKeX real del proyecto.")
            st.stop()
        else:
            st.sidebar.success(f"GKeX cargado desde: {source}")

    st.sidebar.caption(
        "Los modelos axiales implementados coinciden con la sección de estructura axial: "
        "dipolo, monopolo, dos componentes, BBBA2007 y expansión z."
    )

    return CrossSectionConfig(
        vector_model=vector_model,
        axial_model=axial_model,
        m_a=m_a,
        m_a_tilde=m_a_tilde,
        use_cabibbo=use_cabibbo,
        include_muon_mass=include_muon_mass,
    )


def render_formalism_tab() -> None:
    section_header(
        "Formalismo teórico",
        "Resumen de las expresiones implementadas. La app parte del formalismo tensorial del TFG, no de la fórmula compacta de Llewellyn-Smith.",
    )

    st.markdown('<div class="theory-card">', unsafe_allow_html=True)
    st.markdown("**Vértice hadrónico débil cargado**")
    st.latex(
        r"\widetilde{\Gamma}^{\mu}="
        r"F_1^V(Q^2)\gamma^\mu+i\frac{F_2^V(Q^2)}{2M}\sigma^{\mu\nu}Q_\nu"
        r"+G_A(Q^2)\gamma^\mu\gamma^5+\frac{G_P(Q^2)}{2M}Q^\mu\gamma^5"
    )
    st.markdown("**Funciones de estructura hadrónica**")
    st.latex(
        r"\widetilde{W}^{\mu\nu}="
        r"-W_1g^{\mu\nu}+W_2\frac{p_i^\mu p_i^\nu}{M^2}"
        r"+iW_3\frac{\epsilon^{\mu\nu\beta\alpha}p_{i\beta}Q_\alpha}{2M^2}"
        r"+W_4\frac{Q^\mu Q^\nu}{M^2}"
        r"+W_5\frac{p_i^\mu Q^\nu+Q^\mu p_i^\nu}{2M^2}"
    )
    st.latex(
        r"W_1=\tau(F_1^V+F_2^V)^2+(1+\tau)G_A^2,\quad "
        r"W_2=(F_1^V)^2+\tau(F_2^V)^2+G_A^2,\quad "
        r"W_3=2G_A(F_1^V+F_2^V)"
    )
    st.latex(
        r"W_4=\frac{|Q^2|-4M^2}{(4M^2)^2}(F_2^V)^2"
        r"-\frac{G_AF_P}{M}+\tau F_P^2,\quad "
        r"W_5=W_2"
    )
    st.latex(r"\tau=\frac{|Q^2|}{4M^2},\qquad F_P(Q^2)=\frac{G_P(Q^2)}{2M}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="theory-card">', unsafe_allow_html=True)
    st.markdown("**Contracción reducida implementada**")
    st.latex(
        r"\widetilde{\eta}_{\mu\nu}\widetilde{W}^{\mu\nu}="
        r"2W_1E_\nu(E_\mu-|\vec{k}_\mu|\cos\theta_\mu)"
        r"+W_2E_\nu(E_\mu+|\vec{k}_\mu|\cos\theta_\mu)"
    )
    st.latex(
        r"\mp\frac{W_3}{M}E_\nu\left[(E_\nu+E_\mu)(E_\mu-|\vec{k}_\mu|\cos\theta_\mu)-m_\mu^2\right]"
        r"+m_\mu^2W_4E_\nu(E_\mu-|\vec{k}_\mu|\cos\theta_\mu)"
        r"-\frac{W_5}{M}m_\mu^2E_\nu"
    )
    st.markdown(
        r"Con la convención numérica \(G_A(0)<0\) y \(W_3=2G_A(F_1^V+F_2^V)\), "
        r"el signo superior \((-)\) corresponde a neutrinos y el inferior \((+)\) "
        "a antineutrinos. Esta elección implementa el cambio de signo del término "
        "de interferencia vector--axial."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="theory-card">', unsafe_allow_html=True)
    st.markdown("**Secciones eficaces integradas usadas en las representaciones**")
    st.latex(
        r"\frac{d\sigma}{d\Omega}="
        r"\frac{G_F^2}{4\pi^2}\frac{|\vec{k}_\mu|}{E_\nu}"
        r"f_{\rm rec}^{-1}\,"
        r"\widetilde{\eta}_{\mu\nu}\widetilde{W}^{\mu\nu}"
    )
    st.latex(
        r"f_{\rm rec}=1+\frac{E_\nu(|\vec{k}_\mu|-E_\mu\cos\theta_\mu)}{M|\vec{k}_\mu|}"
    )
    st.latex(
        r"\frac{d\sigma}{dE_\mu}="
        r"\frac{2MG_F^2}{4\pi E_\nu^2}"
        r"\widetilde{\eta}_{\mu\nu}\widetilde{W}^{\mu\nu}"
        r"\bigg|_{\cos\theta_\mu=\cos\theta_0}"
    )
    st.latex(
        r"\cos\theta_0="
        r"\frac{2(E_\nu E_\mu+ME_\mu-ME_\nu)-m_\mu^2}{2E_\nu|\vec{k}_\mu|}"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.info(
        r"Las unidades mostradas en las gráficas son \(10^{-13}\\,\\mathrm{fm}^2\) "
        "por estereorradián o por GeV, según el observable."
    )


def render_kinematics_tab(config: CrossSectionConfig) -> None:
    section_header("Cinemática elástica libre", "Relación entre ángulo, energía final y transferencia |Q²|.")

    c1, c2 = st.columns([1.0, 1.0])
    with c1:
        e_nu = st.slider(r"Energía incidente $E_\nu$ (GeV)", 0.20, 5.00, 1.00, 0.01, key="kin_E")
    with c2:
        theta_max = st.slider(r"Ángulo máximo mostrado (grados)", 5.0, 180.0, 120.0, 1.0, key="kin_thmax")

    theta_deg = np.linspace(0.1, theta_max, 700)
    df = kinematics_from_theta_grid(e_nu, theta_deg, M_MU if config.include_muon_mass else 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    ax1, ax2 = axes

    ax1.plot(df["theta_deg"], df["E_mu"], linewidth=2.4)
    ax2.plot(df["theta_deg"], df["Q2_abs"], linewidth=2.4)

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.tick_params(direction="in", top=True, right=True)

    ax1.set_xlabel(r"$\theta_\mu$ (º)")
    ax2.set_xlabel(r"$\theta_\mu$ (º)")
    ax1.set_ylabel(r"$E_\mu\;(\mathrm{GeV})$")
    ax2.set_ylabel(r"$|Q^2|\;(\mathrm{GeV}^2)$")
    ax1.set_title("Energía del muón saliente")
    ax2.set_title("Transferencia de cuadrimomento")

    finalize_figure(fig)
    st.pyplot(fig, use_container_width=True)
    render_figure_download_buttons(fig, "ccqe_cinematica_elastica")

    st.dataframe(
        df.loc[df["valid"], ["theta_deg", "E_mu", "k_mu", "omega", "Q2_abs"]].head(20),
        use_container_width=True,
    )


def render_ccqe_angular_tab(config: CrossSectionConfig) -> None:
    section_header(
        "Sección eficaz angular CCQE",
        "Comparación entre Galster, GKeX y una variación axial efectiva con M_A=1.35 GeV.",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        e_nu = st.slider(r"$E_\nu$ (GeV)", 0.25, 5.00, 1.00, 0.01, key="ang_E")
    with c2:
        beam = st.radio("Canal", ["neutrino", "antineutrino"], horizontal=True, key="ang_beam")
    with c3:
        theta_max = st.slider(r"$\theta_{\max}$ (grados)", 10.0, 180.0, 120.0, 1.0, key="ang_thmax")

    theta_deg = np.linspace(0.1, theta_max, 800)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    ax1, ax2 = axes

    for label, cfg, line_style, line_width, zorder in benchmark_vector_axial_cases(config):
        df = kinematics_from_theta_grid(e_nu, theta_deg, M_MU if cfg.include_muon_mass else 0.0)
        ds = dsigma_domega_from_kinematics(df, e_nu, cfg, beam=beam)

        ax1.plot(df["Q2_abs"], ds, linewidth=line_width, linestyle=line_style, label=label, zorder=zorder)
        ax2.plot(df["theta_deg"], ds, linewidth=line_width, linestyle=line_style, label=label, zorder=zorder)

    ax1.set_xlabel(r"$|Q^2|\;(\mathrm{GeV}^2)$")
    ax2.set_xlabel(r"$\theta_\mu$ (º)")
    ax1.set_ylabel(r"$d\sigma/d\Omega\;(10^{-13}\,\mathrm{fm}^2/\mathrm{sr})$")
    ax2.set_ylabel(r"$d\sigma/d\Omega\;(10^{-13}\,\mathrm{fm}^2/\mathrm{sr})$")
    ax1.set_title(r"Dependencia con $|Q^2|$")
    ax2.set_title(r"Dependencia angular")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.tick_params(direction="in", top=True, right=True)
        ax.legend(frameon=False, fontsize=8)

    finalize_figure(fig)
    st.pyplot(fig, use_container_width=True)
    render_figure_download_buttons(fig, "ccqe_dsigma_domega")

    st.markdown(
        "La comparación muestra que Galster y GKeX producen curvas muy próximas en esta región, "
        "mientras que el aumento efectivo de la masa axial modifica de forma visible la sección eficaz."
    )

def render_ccqe_energy_scan_tab(config: CrossSectionConfig) -> None:
    section_header(
        "Barrido en energía incidente para dσ/dΩ",
        "Comparación de Galster, GKeX y GKeX con una masa axial efectiva alta.",
    )

    beam = st.radio("Canal", ["neutrino", "antineutrino"], horizontal=True, key="scan_beam")
    energies = st.multiselect(
        r"Energías \(E_\nu\) (GeV)",
        [0.5, 1.0, 1.5, 2.0, 3.0],
        default=[0.5, 1.0, 1.5],
        key="scan_energies",
    )

    if not energies:
        st.warning("Selecciona al menos una energía.")
        return

    fig, axes = plt.subplots(len(energies), 2, figsize=(12.4, 3.2 * len(energies)), squeeze=False)

    for row, e_nu in enumerate(energies):
        theta_deg = np.linspace(0.1, 180.0, 900)

        for label, cfg, line_style, line_width, zorder in benchmark_vector_axial_cases(config):
            df = kinematics_from_theta_grid(e_nu, theta_deg, M_MU if cfg.include_muon_mass else 0.0)
            ds = dsigma_domega_from_kinematics(df, e_nu, cfg, beam=beam)

            axes[row, 0].plot(df["Q2_abs"], ds, linewidth=line_width, linestyle=line_style, label=label, zorder=zorder)
            axes[row, 1].plot(df["theta_deg"], ds, linewidth=line_width, linestyle=line_style, label=label, zorder=zorder)

        axes[row, 0].set_ylabel(r"$d\sigma/d\Omega$")
        axes[row, 1].set_ylabel(r"$d\sigma/d\Omega$")

        for ax in axes[row]:
            ax.grid(True, alpha=0.25)
            ax.tick_params(direction="in", top=True, right=True)
            ax.legend(frameon=False, fontsize=8)

    axes[-1, 0].set_xlabel(r"$|Q^2|\;(\mathrm{GeV}^2)$")
    axes[-1, 1].set_xlabel(r"$\theta_\mu$ (º)")

    fig.suptitle(rf"$d\sigma/d\Omega$ para canal {beam}", y=1.002)
    finalize_figure(fig)
    st.pyplot(fig, use_container_width=True)
    render_figure_download_buttons(fig, "ccqe_barrido_dsigma_domega")

    st.caption(r"Unidades del eje vertical: \(10^{-13}\,\mathrm{fm}^2/\mathrm{sr}\).")

def render_energy_differential_tab(config: CrossSectionConfig) -> None:
    section_header(
        "Sección eficaz diferencial respecto a Eμ",
        "Representación de dσ/dEμ evaluada en la condición elástica cosθμ = cosθ0.",
    )

    c1, c2 = st.columns([1.0, 1.0])
    with c1:
        e_nu = st.slider(r"$E_\nu$ (GeV)", 0.25, 5.00, 1.00, 0.01, key="emu_E")
    with c2:
        beam = st.radio("Canal", ["neutrino", "antineutrino"], horizontal=True, key="emu_beam")

    e_mu_grid = np.linspace(M_MU * 1.001, e_nu * 0.999, 900)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    ax1, ax2 = axes

    for label, cfg, line_style, line_width, zorder in benchmark_vector_axial_cases(config):
        df = kinematics_from_energy_grid(e_nu, e_mu_grid, M_MU if cfg.include_muon_mass else 0.0)
        ds = dsigma_demu_from_energy_grid(df, e_nu, cfg, beam=beam)

        ax1.plot(df["E_mu"], ds, linewidth=line_width, linestyle=line_style, label=label, zorder=zorder)
        ax2.plot(df["Q2_abs"], ds, linewidth=line_width, linestyle=line_style, label=label, zorder=zorder)

    ax1.set_xlabel(r"$E_\mu\;(\mathrm{GeV})$")
    ax2.set_xlabel(r"$|Q^2|\;(\mathrm{GeV}^2)$")
    ax1.set_ylabel(r"$d\sigma/dE_\mu\;(10^{-13}\,\mathrm{fm}^2/\mathrm{GeV})$")
    ax2.set_ylabel(r"$d\sigma/dE_\mu\;(10^{-13}\,\mathrm{fm}^2/\mathrm{GeV})$")
    ax1.set_title(r"Dependencia con $E_\mu$")
    ax2.set_title(r"Dependencia con $|Q^2|$")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.tick_params(direction="in", top=True, right=True)
        ax.legend(frameon=False, fontsize=8)

    finalize_figure(fig)
    st.pyplot(fig, use_container_width=True)
    render_figure_download_buttons(fig, "ccqe_dsigma_demu")

    st.info(
        "El soporte físico de la curva se limita a los puntos con |cosθ₀| ≤ 1. "
        "Fuera de esa región la energía del muón no es compatible con la cinemática elástica libre."
    )

def render_energy_differential_scan_tab(config: CrossSectionConfig) -> None:
    section_header(
        "Barrido en energía incidente para dσ/dEμ",
        "Comparación de Galster, GKeX y GKeX con una masa axial efectiva alta.",
    )

    beam = st.radio("Canal", ["neutrino", "antineutrino"], horizontal=True, key="emuscan_beam")
    energies = st.multiselect(
        "Energías Eν (GeV)",
        [0.5, 1.0, 1.5, 2.0, 3.0],
        default=[0.5, 1.0, 1.5],
        key="emuscan_energies",
    )

    if not energies:
        st.warning("Selecciona al menos una energía.")
        return

    fig, axes = plt.subplots(len(energies), 2, figsize=(12.4, 3.2 * len(energies)), squeeze=False)

    for row, e_nu in enumerate(energies):
        e_mu_grid = np.linspace(M_MU * 1.001, e_nu * 0.999, 900)

        for label, cfg, line_style, line_width, zorder in benchmark_vector_axial_cases(config):
            df = kinematics_from_energy_grid(e_nu, e_mu_grid, M_MU if cfg.include_muon_mass else 0.0)
            ds = dsigma_demu_from_energy_grid(df, e_nu, cfg, beam=beam)

            axes[row, 0].plot(df["E_mu"], ds, linewidth=line_width, linestyle=line_style, label=label, zorder=zorder)
            axes[row, 1].plot(df["Q2_abs"], ds, linewidth=line_width, linestyle=line_style, label=label, zorder=zorder)

        axes[row, 0].set_ylabel(r"$d\sigma/dE_\mu$")
        axes[row, 1].set_ylabel(r"$d\sigma/dE_\mu$")

        for ax in axes[row]:
            ax.grid(True, alpha=0.25)
            ax.tick_params(direction="in", top=True, right=True)
            ax.legend(frameon=False, fontsize=8)

    axes[-1, 0].set_xlabel(r"$E_\mu\;(\mathrm{GeV})$")
    axes[-1, 1].set_xlabel(r"$|Q^2|\;(\mathrm{GeV}^2)$")

    fig.suptitle(rf"$d\sigma/dE_\mu$ para canal {beam}", y=1.002)
    finalize_figure(fig)
    st.pyplot(fig, use_container_width=True)
    render_figure_download_buttons(fig, "ccqe_barrido_dsigma_demu")

    st.caption(r"Unidades del eje vertical: \(10^{-13}\,\mathrm{fm}^2/\mathrm{GeV}\).")

def render_nu_antinu_tab(config: CrossSectionConfig) -> None:
    section_header(
        "Diferencia neutrino--antineutrino",
        "La diferencia procede del cambio de signo del término vector--axial proporcional a W₃.",
    )

    st.latex(
        r"R_{\nu/\bar{\nu}}="
        r"\frac{(d\sigma/d\Omega)_{\nu}}{(d\sigma/d\Omega)_{\bar{\nu}}}"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        e_nu = st.slider(r"$E_\nu$ (GeV)", 0.25, 5.00, 1.00, 0.01, key="diff_E")
    with c2:
        vector_model = st.selectbox("Modelo vectorial", ["Galster", "GKeX"], index=0, key="diff_vector")
    with c3:
        theta_max = st.slider(r"$\theta_{\max}$ (grados)", 10.0, 180.0, 120.0, 1.0, key="diff_thmax")

    cfg = CrossSectionConfig(
        vector_model=vector_model,
        axial_model=config.axial_model,
        m_a=config.m_a,
        m_a_tilde=config.m_a_tilde,
        use_cabibbo=config.use_cabibbo,
        include_muon_mass=config.include_muon_mass,
    )

    theta_deg = np.linspace(0.1, theta_max, 800)
    df = kinematics_from_theta_grid(e_nu, theta_deg, M_MU if cfg.include_muon_mass else 0.0)

    ds_nu = dsigma_domega_from_kinematics(df, e_nu, cfg, beam="neutrino")
    ds_anti = dsigma_domega_from_kinematics(df, e_nu, cfg, beam="antineutrino")

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = ds_nu / ds_anti
    ratio[~np.isfinite(ratio)] = np.nan

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.4))
    ax_q2, ax_rq2 = axes[0]
    ax_th, ax_rth = axes[1]

    ax_q2.plot(df["Q2_abs"], ds_nu, linewidth=2.4, label=r"$\nu_\mu+n\to\mu^-+p$")
    ax_q2.plot(df["Q2_abs"], ds_anti, linewidth=2.4, linestyle="--", label=r"$\bar{\nu}_\mu+p\to\mu^++n$")

    ax_th.plot(df["theta_deg"], ds_nu, linewidth=2.4, label=r"$\nu_\mu+n\to\mu^-+p$")
    ax_th.plot(df["theta_deg"], ds_anti, linewidth=2.4, linestyle="--", label=r"$\bar{\nu}_\mu+p\to\mu^++n$")

    ax_rq2.plot(df["Q2_abs"], ratio, linewidth=2.4)
    ax_rth.plot(df["theta_deg"], ratio, linewidth=2.4)

    for ax in (ax_rq2, ax_rth):
        ax.axhline(1.0, linewidth=1.0, alpha=0.45)

    ax_q2.set_xlabel(r"$|Q^2|\;(\mathrm{GeV}^2)$")
    ax_rq2.set_xlabel(r"$|Q^2|\;(\mathrm{GeV}^2)$")
    ax_th.set_xlabel(r"$\theta_\mu$ (º)")
    ax_rth.set_xlabel(r"$\theta_\mu$ (º)")

    ax_q2.set_ylabel(r"$d\sigma/d\Omega\;(10^{-13}\,\mathrm{fm}^2/\mathrm{sr})$")
    ax_th.set_ylabel(r"$d\sigma/d\Omega\;(10^{-13}\,\mathrm{fm}^2/\mathrm{sr})$")
    ax_rq2.set_ylabel(r"$R_{\nu/\bar{\nu}}$")
    ax_rth.set_ylabel(r"$R_{\nu/\bar{\nu}}$")

    ax_q2.set_title(r"Comparación en función de $|Q^2|$")
    ax_rq2.set_title(r"$R_{\nu/\bar{\nu}}=\frac{(d\sigma/d\Omega)_\nu}{(d\sigma/d\Omega)_{\bar{\nu}}}$")
    ax_th.set_title(r"Comparación angular")
    ax_rth.set_title(r"$R_{\nu/\bar{\nu}}=\frac{(d\sigma/d\Omega)_\nu}{(d\sigma/d\Omega)_{\bar{\nu}}}$")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
        ax.tick_params(direction="in", top=True, right=True)

    ax_q2.legend(frameon=False)
    ax_th.legend(frameon=False)

    finalize_figure(fig)
    st.pyplot(fig, use_container_width=True)
    render_figure_download_buttons(fig, "ccqe_nu_vs_antinu")

    st.markdown(
        r"En esta representación se mantienen iguales los factores de forma y la cinemática. "
        r"La sección eficaz de neutrinos queda por encima de la de antineutrinos. "
        r"La diferencia aparece al cambiar el signo del término proporcional a \(W_3\), "
        r"que contiene la interferencia entre la corriente vectorial y la axial. "
        r"El cociente mostrado es \(R_{\nu/\bar{\nu}}=(d\sigma_\nu/d\Omega)/(d\sigma_{\bar{\nu}}/d\Omega)\)."
    )




def render_total_cross_section_tab(config: CrossSectionConfig) -> None:
    section_header(
        "Sección eficaz total",
        "Integración numérica de dσ/dEμ sobre la región cinemáticamente permitida del leptón saliente.",
    )

    st.markdown(
        r"""
La sección eficaz total se calcula como
\[
\sigma_{\rm tot}(E_\nu)=
\int_{\mathrm{cin.}} dE_\mu\,
\frac{d\sigma}{dE_\mu},
\]
donde la integral se restringe a la región compatible con \(|\cos\theta_0|\leq1\).
En esta pestaña se separan explícitamente tres efectos distintos: el canal
\(\nu/\bar{\nu}\), la elección de factores de forma vectoriales y la elección de
la parametrización axial.
"""
    )

    c1, c2 = st.columns([1.0, 1.0])
    with c1:
        e_min, e_max = st.slider(
            r"Rango de $E_\nu$ (GeV)",
            min_value=0.00,
            max_value=5.00,
            value=(0.00, 3.00),
            step=0.05,
            key="total_E_range",
        )
    with c2:
        n_points = st.slider("Número de puntos en energía", 20, 120, 70, 5, key="total_npoints")

    e_values = np.linspace(e_min, e_max, n_points)

    # ------------------------------------------------------------------
    # 1) Comparación física básica: neutrino frente a antineutrino
    # ------------------------------------------------------------------
    st.markdown("### Comparación entre neutrinos y antineutrinos")

    cfg_base = CrossSectionConfig(
        vector_model=config.vector_model,
        axial_model=config.axial_model,
        m_a=config.m_a,
        m_a_tilde=config.m_a_tilde,
        use_cabibbo=config.use_cabibbo,
        include_muon_mass=config.include_muon_mass,
    )

    sigma_nu_base = total_cross_section_curve(e_values, cfg_base, beam="neutrino")
    sigma_antinu_base = total_cross_section_curve(e_values, cfg_base, beam="antineutrino")

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_nu_antinu = sigma_nu_base / sigma_antinu_base
    ratio_nu_antinu[~np.isfinite(ratio_nu_antinu)] = np.nan
    ratio_nu_antinu[(sigma_nu_base <= 0.0) | (sigma_antinu_base <= 0.0)] = np.nan

    fig0, axes0 = plt.subplots(1, 2, figsize=(12.4, 4.4))
    ax_sigma, ax_ratio = axes0

    ax_sigma.plot(e_values, sigma_nu_base, linewidth=2.4, label=r"$\nu_\mu+n\to\mu^-+p$")
    ax_sigma.plot(e_values, sigma_antinu_base, linewidth=2.4, linestyle="--", label=r"$\bar{\nu}_\mu+p\to\mu^++n$")

    ax_ratio.plot(e_values, ratio_nu_antinu, linewidth=2.4)
    ax_ratio.axhline(1.0, linewidth=1.0, alpha=0.45)

    ax_sigma.set_xlabel(r"$E_\nu\;(\mathrm{GeV})$")
    ax_ratio.set_xlabel(r"$E_\nu\;(\mathrm{GeV})$")
    ax_sigma.set_ylabel(r"$\sigma_{\rm tot}\;(10^{-13}\,\mathrm{fm}^2)$")
    ax_ratio.set_ylabel(r"$R^{\rm tot}_{\nu/\bar{\nu}}$")
    ax_sigma.set_title(r"Comparación $\nu$ frente a $\bar{\nu}$")
    ax_ratio.set_title(r"$R^{\rm tot}_{\nu/\bar{\nu}}=\sigma^{\nu}_{\rm tot}/\sigma^{\bar{\nu}}_{\rm tot}$")
    ax_sigma.set_xlim(left=0.0)
    ax_sigma.set_ylim(bottom=0.0)
    ax_ratio.set_xlim(left=0.0)

    for ax in axes0:
        ax.grid(True, alpha=0.25)
        ax.tick_params(direction="in", top=True, right=True)

    ax_sigma.legend(frameon=False, fontsize=8)
    finalize_figure(fig0)
    st.pyplot(fig0, use_container_width=True)
    render_figure_download_buttons(fig0, "ccqe_sigma_total_nu_antinu")

    st.markdown(
        r"El cociente mostrado en el panel derecho está definido como "
        r"\(R^{\rm tot}_{\nu/\bar{\nu}}=\sigma^{\nu}_{\rm tot}/\sigma^{\bar{\nu}}_{\rm tot}\)."
    )

    # ------------------------------------------------------------------
    # 2) Parametrizaciones vectoriales/axiales de referencia, separando canales
    # ------------------------------------------------------------------
    st.markdown("### Comparación de parametrizaciones de referencia")

    fig1, axes1 = plt.subplots(1, 2, figsize=(12.4, 4.4))
    ax_nu, ax_antinu = axes1

    for label, cfg, line_style, line_width, zorder in benchmark_vector_axial_cases(config):
        sigma_nu = total_cross_section_curve(e_values, cfg, beam="neutrino")
        sigma_anti = total_cross_section_curve(e_values, cfg, beam="antineutrino")

        ax_nu.plot(e_values, sigma_nu, linewidth=line_width, linestyle=line_style, label=label, zorder=zorder)
        ax_antinu.plot(e_values, sigma_anti, linewidth=line_width, linestyle=line_style, label=label, zorder=zorder)

    ax_nu.set_title(r"$\nu_\mu+n\to\mu^-+p$")
    ax_antinu.set_title(r"$\bar{\nu}_\mu+p\to\mu^++n$")
    ax_nu.set_ylabel(r"$\sigma_{\rm tot}^{\nu}\;(10^{-13}\,\mathrm{fm}^2)$")
    ax_antinu.set_ylabel(r"$\sigma_{\rm tot}^{\bar{\nu}}\;(10^{-13}\,\mathrm{fm}^2)$")

    for ax in axes1:
        ax.set_xlabel(r"$E_\nu\;(\mathrm{GeV})$")
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.25)
        ax.tick_params(direction="in", top=True, right=True)
        ax.legend(frameon=False, fontsize=8)

    finalize_figure(fig1)
    st.pyplot(fig1, use_container_width=True)
    render_figure_download_buttons(fig1, "ccqe_sigma_total_parametrizaciones")

    st.markdown(
        "En esta representación el canal físico no se codifica en el estilo de línea. "
        "Cada panel corresponde a un canal distinto. Así se evita confundir el efecto "
        "neutrino/antineutrino con el efecto de cambiar la parametrización vectorial o axial."
    )

    # ------------------------------------------------------------------
    # 3) Sensibilidad axial, con cocientes separados para neutrinos y antineutrinos
    # ------------------------------------------------------------------
    st.markdown("### Sensibilidad a modelos axiales")

    selected_axial_total = st.multiselect(
        "Modelos axiales para la comparación total",
        AXIAL_MODEL_OPTIONS,
        default=AXIAL_COMPARISON_DEFAULTS,
        key="total_axial_models",
    )

    if selected_axial_total:
        cfg_d = CrossSectionConfig(
            vector_model=config.vector_model,
            axial_model="Dipolo estándar",
            m_a=config.m_a,
            m_a_tilde=config.m_a_tilde,
            use_cabibbo=config.use_cabibbo,
            include_muon_mass=config.include_muon_mass,
        )
        sigma_d_nu = total_cross_section_curve(e_values, cfg_d, beam="neutrino")
        sigma_d_antinu = total_cross_section_curve(e_values, cfg_d, beam="antineutrino")

        fig2, axes2 = plt.subplots(1, 2, figsize=(12.4, 4.4))
        ax_r_nu, ax_r_antinu = axes2

        for axial_model in selected_axial_total:
            if axial_model == "Dipolo estándar":
                continue

            cfg = CrossSectionConfig(
                vector_model=config.vector_model,
                axial_model=axial_model,
                m_a=config.m_a,
                m_a_tilde=config.m_a_tilde,
                use_cabibbo=config.use_cabibbo,
                include_muon_mass=config.include_muon_mass,
            )

            sigma_nu = total_cross_section_curve(e_values, cfg, beam="neutrino")
            sigma_antinu = total_cross_section_curve(e_values, cfg, beam="antineutrino")

            with np.errstate(divide="ignore", invalid="ignore"):
                ratio_nu = sigma_nu / sigma_d_nu
                ratio_antinu = sigma_antinu / sigma_d_antinu

            ratio_nu[~np.isfinite(ratio_nu)] = np.nan
            ratio_antinu[~np.isfinite(ratio_antinu)] = np.nan
            ratio_nu[(sigma_nu <= 0.0) | (sigma_d_nu <= 0.0)] = np.nan
            ratio_antinu[(sigma_antinu <= 0.0) | (sigma_d_antinu <= 0.0)] = np.nan

            ax_r_nu.plot(e_values, ratio_nu, linewidth=2.1, label=axial_label(axial_model))
            ax_r_antinu.plot(e_values, ratio_antinu, linewidth=2.1, label=axial_label(axial_model))

        for ax in axes2:
            ax.axhline(1.0, linewidth=1.0, alpha=0.45)
            ax.set_xlabel(r"$E_\nu\;(\mathrm{GeV})$")
            ax.set_xlim(left=0.0)
            ax.grid(True, alpha=0.25)
            ax.tick_params(direction="in", top=True, right=True)
            ax.legend(frameon=False, fontsize=8)

        ax_r_nu.set_title(r"Sensibilidad axial en $\nu$")
        ax_r_antinu.set_title(r"Sensibilidad axial en $\bar{\nu}$")
        ax_r_nu.set_ylabel(r"$\sigma^{\nu}_{\rm modelo}/\sigma^{\nu}_{\rm dipolo}$")
        ax_r_antinu.set_ylabel(r"$\sigma^{\bar{\nu}}_{\rm modelo}/\sigma^{\bar{\nu}}_{\rm dipolo}$")

        finalize_figure(fig2)
        st.pyplot(fig2, use_container_width=True)
        render_figure_download_buttons(fig2, "ccqe_sigma_total_sensibilidad_axial")

    st.info(
        "Para la memoria, conviene no mezclar en una única leyenda el canal "
        "neutrino/antineutrino y el modelo de factores de forma. Es más claro "
        "definir primero el cociente total y después comparar modelos en paneles separados."
    )


# =============================================================================
# MINERvA hidrógeno: datos, plegado en flujo y comparación experimental
# =============================================================================

def find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """Busca columnas de forma robusta, ignorando espacios y mayúsculas."""
    cols = list(df.columns)
    strip_map = {c.strip(): c for c in cols}

    for cand in candidates:
        if cand in strip_map:
            return strip_map[cand]

    lower_exact = {c.strip().lower(): c for c in cols}
    for cand in candidates:
        key = cand.strip().lower()
        if key in lower_exact:
            return lower_exact[key]

    cols_low = [(c, c.strip().lower()) for c in cols]
    for cand in candidates:
        key = cand.strip().lower()
        for orig, low in cols_low:
            if key in low:
                return orig

    raise KeyError(f"No encuentro columnas {candidates}. Columnas disponibles: {list(df.columns)}")


def pick_flux_col(df: pd.DataFrame) -> str:
    """Selecciona la columna de flujo del CSV de MINERvA."""
    for c in df.columns:
        if c.strip().lower().startswith("flux("):
            return c
    return find_col(df, ["flux", "phi"])


def resolve_project_root_for_minerva() -> Path:
    """Localiza la raíz del proyecto buscando refs/minerva_hydrogen."""
    candidates: list[Path] = []

    try:
        here = Path(__file__).resolve()
        candidates.extend([here.parent, *here.parents])
    except Exception:
        pass

    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])

    for cand in candidates:
        if (cand / "refs" / "minerva_hydrogen" / "hydrogen_xsec.csv").exists():
            return cand

    # Si no se encuentra, se devuelve cwd para que el mensaje de error sea claro.
    return cwd


def read_cov_matrix_minerva(path: Path, n: int) -> np.ndarray:
    """Lee la matriz de covarianza aunque el CSV venga sin cabecera real."""
    V = pd.read_csv(path, header=None).to_numpy(dtype=float)

    if V.shape == (n, n + 1):
        V = V[:, 1:]
    elif V.shape == (n + 1, n):
        V = V[1:, :]
    elif V.shape == (n + 1, n + 1):
        V = V[1:, 1:]

    if V.shape != (n, n):
        raise ValueError(f"Covarianza con forma {V.shape}; esperaba {(n, n)} en {path.name}.")

    return V


@st.cache_data(show_spinner=False)
def load_minerva_hydrogen_inputs() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Carga datos, flujo y covarianza total de MINERvA-H."""
    project_root = resolve_project_root_for_minerva()
    refs_dir = project_root / "refs" / "minerva_hydrogen"

    need = [
        refs_dir / "hydrogen_xsec.csv",
        refs_dir / "cov_tot.csv",
        refs_dir / "flux_rhc_numubar_nueconstrained.csv",
    ]
    missing = [p for p in need if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Faltan archivos en refs/minerva_hydrogen:\n" + "\n".join(str(p) for p in missing)
        )

    xsec = pd.read_csv(refs_dir / "hydrogen_xsec.csv")
    xsec.columns = [c.strip() for c in xsec.columns]

    col_q2lo = find_col(xsec, ["Q2low", "Q2Low", "Q2_low"])
    col_q2hi = find_col(xsec, ["Q2High", "Q2high", "Q2_hi", "Q2_high"])
    col_q2c = find_col(xsec, ["Q2center", "Q2_center", "Q2cent"])
    col_xsec = find_col(xsec, ["xsec", "XSec", "dsigma", "dsigdq2", "dsigma_dQ2"])
    col_stat = find_col(xsec, ["stat", "err_stat", "sigma_stat"])
    col_sys = find_col(xsec, ["sys", "err_sys", "sigma_sys"])

    bins_df = pd.DataFrame(
        {
            "q2_low": xsec[col_q2lo].to_numpy(float),
            "q2_high": xsec[col_q2hi].to_numpy(float),
            "q2_center": xsec[col_q2c].to_numpy(float),
            "xsec": xsec[col_xsec].to_numpy(float),
            "err_stat_table": xsec[col_stat].to_numpy(float),
            "err_sys_table": xsec[col_sys].to_numpy(float),
        }
    )
    bins_df["dq2"] = bins_df["q2_high"] - bins_df["q2_low"]
    bins_df["err_tot_table"] = np.sqrt(bins_df["err_stat_table"]**2 + bins_df["err_sys_table"]**2)

    flux = pd.read_csv(refs_dir / "flux_rhc_numubar_nueconstrained.csv")
    flux.columns = [c.strip() for c in flux.columns]
    col_E = find_col(flux, ["Energy(GeV)", "Energy", "E", "enu"])
    col_phi = pick_flux_col(flux)

    flux_df = pd.DataFrame(
        {
            "E": flux[col_E].to_numpy(float),
            "phi": flux[col_phi].to_numpy(float),
        }
    )
    flux_df = flux_df.replace([np.inf, -np.inf], np.nan).dropna()
    flux_df = flux_df.sort_values("E").drop_duplicates(subset=["E"]).reset_index(drop=True)

    n = len(bins_df)
    Vtot_raw = read_cov_matrix_minerva(refs_dir / "cov_tot.csv", n)

    # Las tablas suplementarias están en unidades x 10^-80, mientras que los puntos
    # están expresados en 10^-38 cm^2/GeV^2. Por tanto, la covarianza se reescala a
    # las unidades numéricas del CSV de secciones eficaces.
    Vtot = Vtot_raw * 1.0e-4
    bins_df["err_tot_cov"] = np.sqrt(np.clip(np.diag(Vtot), 0.0, None))

    return bins_df, flux_df, Vtot


@st.cache_data(show_spinner=False)
def load_minerva_hydrogen_stats_inputs() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Carga datos, flujo, covarianza total y covarianza estadística de MINERvA-H."""
    bins_df, flux_df, Vtot = load_minerva_hydrogen_inputs()

    project_root = resolve_project_root_for_minerva()
    refs_dir = project_root / "refs" / "minerva_hydrogen"
    cov_stat_path = refs_dir / "cov_stat.csv"

    if not cov_stat_path.exists():
        raise FileNotFoundError(f"Falta {cov_stat_path}")

    n = len(bins_df)
    Vstat_raw = read_cov_matrix_minerva(cov_stat_path, n)

    # Conversión desde unidades 10^{-80} a las unidades numéricas de los puntos,
    # expresados en 10^{-38} cm^2/GeV^2. Por tanto, se divide por 10^{-76}.
    Vstat = Vstat_raw * 1.0e-4

    bins_df = bins_df.copy()
    bins_df["err_stat_cov"] = np.sqrt(np.clip(np.diag(Vstat), 0.0, None))

    return bins_df, flux_df, Vtot, Vstat


def correlation_from_covariance(V: np.ndarray) -> np.ndarray:
    """Construye la matriz de correlación a partir de una covarianza."""
    diag = np.sqrt(np.clip(np.diag(V), 0.0, None))
    denom = np.outer(diag, diag)

    with np.errstate(divide="ignore", invalid="ignore"):
        rho = V / denom
    rho[~np.isfinite(rho)] = 0.0
    return np.clip(rho, -1.0, 1.0)



def passes_minerva_hydrogen_cuts(E_mu: np.ndarray, cos_theta: np.ndarray) -> np.ndarray:
    """Cortes cinemáticos usados para la comparación MINERvA-H."""
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return (E_mu > 1.5) & (E_mu < 20.0) & (theta < 20.0 * np.pi / 180.0)


def dsigma_dQ2_tensor_minerva_units(
    E_nu: float,
    Q2: np.ndarray | float,
    config: CrossSectionConfig,
    beam: str = "antineutrino",
    apply_cuts: bool = True,
) -> np.ndarray:
    """dσ/dQ² en unidades de MINERvA: 10^-38 cm²/GeV².

    Se obtiene desde el formalismo tensorial de esta app. Primero se evalúa
    dσ/dE_mu y después se usa dE_mu/dQ² = -1/(2M) en cinemática elástica libre.
    No se usa la fórmula compacta de Llewellyn-Smith.
    """
    Q2 = np.asarray(Q2, dtype=float)
    scalar_input = (Q2.ndim == 0)
    Q2_arr = np.atleast_1d(Q2)

    E_mu = E_nu - Q2_arr / (2.0 * M_N)
    k_mu = lepton_momentum(E_mu, M_MU)

    cos_theta = np.full_like(Q2_arr, np.nan, dtype=float)
    valid = (E_mu > M_MU) & (k_mu > 0.0) & (Q2_arr > 0.0) & np.isfinite(Q2_arr)

    cos_theta[valid] = (
        E_nu * E_mu[valid] - 0.5 * (Q2_arr[valid] + M_MU**2)
    ) / (E_nu * np.maximum(k_mu[valid], 1.0e-15))

    valid &= np.isfinite(cos_theta) & (np.abs(cos_theta) <= 1.0)

    if apply_cuts:
        valid &= passes_minerva_hydrogen_cuts(E_mu, cos_theta)

    contr = contraction_eta_w(E_nu, E_mu, cos_theta, Q2_arr, config, beam=beam)
    gf = effective_gf(config.use_cabibbo)
    ds_dE = (2.0 * M_N * gf**2 / (4.0 * np.pi * E_nu**2)) * contr
    ds_dE = to_1e_minus_13_fm2(ds_dE)

    # dσ/dQ² = dσ/dE_mu * |dE_mu/dQ²| = dσ/dE_mu /(2M)
    ds_dQ2_1e_minus_13_fm2 = ds_dE / (2.0 * M_N)

    # 1e-13 fm² = 1e-39 cm² = 0.1 x 1e-38 cm²
    ds_dQ2_minerva = 0.1 * ds_dQ2_1e_minus_13_fm2

    ds_dQ2_minerva[~valid] = 0.0
    ds_dQ2_minerva[ds_dQ2_minerva < 0.0] = 0.0

    if scalar_input:
        return np.asarray(ds_dQ2_minerva[0])
    return ds_dQ2_minerva


def flux_folded_minerva_tensor_prediction(
    q2_low: np.ndarray,
    q2_high: np.ndarray,
    flux_E: np.ndarray,
    flux_phi: np.ndarray,
    config: CrossSectionConfig,
    beam: str = "antineutrino",
    nE: int = 90,
    nQ2: int = 50,
    Ev_max: float = 20.0,
    apply_cuts: bool = True,
) -> np.ndarray:
    """Predicción bin a bin de <dσ/dQ²> plegada en flujo."""
    flux_E = np.asarray(flux_E, dtype=float)
    flux_phi = np.asarray(flux_phi, dtype=float)

    mask = (flux_E > 0.0) & (flux_E <= Ev_max) & np.isfinite(flux_phi) & (flux_phi > 0.0)
    E_raw = flux_E[mask]
    phi_raw = flux_phi[mask]

    if len(E_raw) < 5:
        raise ValueError("Flujo vacío o mal leído. Revisa el CSV del flujo.")

    E = np.linspace(float(np.min(E_raw)), float(np.max(E_raw)), int(nE))
    phi = np.interp(E, E_raw, phi_raw)
    phi_tot = np.trapezoid(phi, E)

    if phi_tot <= 0.0:
        raise ValueError("Normalización de flujo <= 0.")

    pred = np.zeros(len(q2_low), dtype=float)

    for i, (lo, hi) in enumerate(zip(q2_low, q2_high)):
        q2_grid = np.linspace(float(lo), float(hi), int(nQ2))
        integrand_E = np.zeros_like(E)

        for j, (Ev, w_flux) in enumerate(zip(E, phi)):
            vals_q2 = dsigma_dQ2_tensor_minerva_units(
                float(Ev),
                q2_grid,
                config=config,
                beam=beam,
                apply_cuts=apply_cuts,
            )
            int_q2 = np.trapezoid(vals_q2, q2_grid)
            integrand_E[j] = w_flux * int_q2

        pred[i] = (np.trapezoid(integrand_E, E) / phi_tot) / (float(hi) - float(lo))

    return pred


def covariance_chi2(data: np.ndarray, model: np.ndarray, V: np.ndarray) -> float:
    """χ² con matriz de covarianza total."""
    r = np.asarray(data, dtype=float) - np.asarray(model, dtype=float)
    Vinv = np.linalg.pinv(V, rcond=1.0e-12)
    return float(r @ Vinv @ r)


def build_step_xy(q2_low: np.ndarray, q2_high: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construye una curva escalonada constante por bin."""
    xs: list[float] = []
    ys: list[float] = []
    for lo, hi, val in zip(q2_low, q2_high, y):
        xs.extend([float(lo), float(hi)])
        ys.extend([float(val), float(val)])
    return np.asarray(xs), np.asarray(ys)


def make_minerva_model_config(
    base: CrossSectionConfig,
    vector_model: str,
    axial_model: str,
) -> CrossSectionConfig:
    """Configura el modelo usado en la pestaña MINERvA."""
    return CrossSectionConfig(
        vector_model=vector_model,
        axial_model=axial_model,
        m_a=base.m_a,
        m_a_tilde=base.m_a_tilde,
        use_cabibbo=True,
        include_muon_mass=True,
    )


@st.cache_data(show_spinner=True)
def compute_minerva_tensor_comparison_cached(
    vector_model: str,
    axial_model: str,
    m_a: float,
    m_a_tilde: float,
    nE: int,
    nQ2: int,
    Ev_max: float,
    apply_cuts: bool,
) -> tuple[pd.DataFrame, float, float, float, float]:
    """Calcula predicción MINERvA con el formalismo tensorial y devuelve estadísticos.

    Se devuelven:
        DataFrame bin a bin,
        chi2 con covarianza total,
        chi2_total/ndof,
        chi2 con covarianza estadística,
        chi2_stat/ndof.

    Como aquí no se ajustan parámetros a los datos, se toma ndof = N_bins.
    """
    bins_df, flux_df, Vtot, Vstat = load_minerva_hydrogen_stats_inputs()

    cfg = CrossSectionConfig(
        vector_model=vector_model,
        axial_model=axial_model,
        m_a=m_a,
        m_a_tilde=m_a_tilde,
        use_cabibbo=True,
        include_muon_mass=True,
    )

    pred = flux_folded_minerva_tensor_prediction(
        q2_low=bins_df["q2_low"].to_numpy(float),
        q2_high=bins_df["q2_high"].to_numpy(float),
        flux_E=flux_df["E"].to_numpy(float),
        flux_phi=flux_df["phi"].to_numpy(float),
        config=cfg,
        beam="antineutrino",
        nE=int(nE),
        nQ2=int(nQ2),
        Ev_max=float(Ev_max),
        apply_cuts=bool(apply_cuts),
    )

    out = bins_df.copy()
    out["model"] = pred
    out["ratio"] = out["xsec"] / np.where(np.abs(out["model"]) > 0.0, out["model"], np.nan)
    out["ratio_err_tot"] = out["err_tot_cov"] / np.where(np.abs(out["model"]) > 0.0, out["model"], np.nan)
    out["ratio_err_stat"] = out["err_stat_cov"] / np.where(np.abs(out["model"]) > 0.0, out["model"], np.nan)

    chi2_tot = covariance_chi2(out["xsec"].to_numpy(float), out["model"].to_numpy(float), Vtot)
    chi2_stat = covariance_chi2(out["xsec"].to_numpy(float), out["model"].to_numpy(float), Vstat)

    ndof = max(len(out), 1)

    return out, float(chi2_tot), float(chi2_tot / ndof), float(chi2_stat), float(chi2_stat / ndof)

def render_minerva_hydrogen_tab(config: CrossSectionConfig) -> None:
    section_header(
        "MINERvA: hidrógeno",
        "Comparación con los datos de MINERvA en hidrógeno usando el formalismo tensorial del TFG.",
    )

    st.markdown(
        "En esta pestaña se calcula el observable publicado por MINERvA, "
        r"\(\langle d\sigma/dQ^2\rangle_\Phi\), plegando la predicción teórica con el flujo "
        "de antineutrinos y aplicando los cortes cinemáticos del análisis. "
        r"**La predicción se construye desde las funciones \(W_i\) y la contracción tensorial de esta app; "
        "no se usa la fórmula compacta de Llewellyn--Smith.**"
    )

    with st.expander("Aclaración sobre el paso de dσ/dEμ a dσ/dQ²"):
        st.markdown(
            r"""
En el caso elástico libre empleado en esta app se usa la relación cinemática
\[
Q^2 = 2M(E_\nu-E_\mu),
\qquad
\frac{d\sigma}{dQ^2}
=
\frac{1}{2M}\frac{d\sigma}{dE_\mu}.
\]
Por tanto, el observable de MINERvA se obtiene a partir de la distribución
\(d\sigma/dE_\mu\) calculada con la contracción tensorial del TFG. Esta relación
debe entenderse dentro de la aproximación de nucleón libre y masas nucleónicas
iguales usada en el formalismo implementado.
"""
        )

    try:
        bins_df, flux_df, Vtot, Vstat = load_minerva_hydrogen_stats_inputs()
    except Exception as exc:
        st.error(str(exc))
        st.info("Coloca los CSV en `refs/minerva_hydrogen/` dentro del proyecto.")
        return

    c1, c2, c3, c4 = st.columns([1.0, 1.0, 1.0, 1.0])
    with c1:
        vector_model = st.selectbox("Modelo vectorial", ["Galster", "GKeX"], index=1, key="minerva_vector")
    with c2:
        axial_model = st.selectbox("Modelo axial", AXIAL_MODEL_OPTIONS, index=0, key="minerva_axial")
    with c3:
        Ev_max = st.slider(r"$E_\nu^{\max}$ para el flujo (GeV)", 5.0, 30.0, 20.0, 1.0, key="minerva_evmax")
    with c4:
        apply_cuts = st.checkbox("Aplicar cortes MINERvA/MINOS", value=True, key="minerva_cuts")

    c5, c6 = st.columns([1.0, 1.0])
    with c5:
        q2_max_display = st.slider(
            r"Máximo \(Q^2\) mostrado (GeV²)",
            0.5,
            float(max(6.0, bins_df["q2_high"].max())),
            2.0,
            0.1,
            key="minerva_q2max_display",
        )
    with c6:
        ratio_max_display = st.slider(
            "Máximo del cociente Datos/modelo",
            1.5,
            10.0,
            3.0,
            0.5,
            key="minerva_ratio_max_display",
        )

    with st.expander("Precisión numérica de la integración"):
        nE = st.slider("Puntos en energía del flujo", 40, 180, 90, 10, key="minerva_nE")
        nQ2 = st.slider("Puntos en cada bin de Q²", 20, 120, 50, 10, key="minerva_nQ2")

    m_a_eff, m_a_tilde_eff = _relevant_axial_parameters(axial_model, config.m_a, config.m_a_tilde)
    df, chi2_tot, chi2_tot_ndof, chi2_stat, chi2_stat_ndof = compute_minerva_tensor_comparison_cached(
        vector_model=vector_model,
        axial_model=axial_model,
        m_a=float(m_a_eff),
        m_a_tilde=float(m_a_tilde_eff),
        nE=int(nE),
        nQ2=int(nQ2),
        Ev_max=float(Ev_max),
        apply_cuts=bool(apply_cuts),
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("χ² total/ndof", f"{chi2_tot_ndof:.2f}", help=f"χ²_total={chi2_tot:.2f}; ndof={len(df)}")
    m2.metric("χ² stat/ndof", f"{chi2_stat_ndof:.2f}", help=f"χ²_stat={chi2_stat:.2f}; ndof={len(df)}")
    m3.metric("Nº de bins", f"{len(df)}")

    q2 = df["q2_center"].to_numpy(float)
    q2err = 0.5 * df["dq2"].to_numpy(float)
    data = df["xsec"].to_numpy(float)
    err = df["err_tot_cov"].to_numpy(float)
    model = df["model"].to_numpy(float)

    x_step, y_step = build_step_xy(
        df["q2_low"].to_numpy(float),
        df["q2_high"].to_numpy(float),
        model,
    )

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.4), sharex=True, gridspec_kw={"height_ratios": [3.2, 1.35]})
    ax, axr = axes

    ax.errorbar(
        q2,
        data,
        xerr=q2err,
        yerr=err,
        fmt="o",
        markersize=5.2,
        capsize=3.0,
        elinewidth=1.4,
        label="MINERvA H",
        zorder=3,
    )
    ax.plot(x_step, y_step, linewidth=2.6, label=f"{vector_model}, {axial_label(axial_model)}", zorder=2)

    ax.set_ylabel(r"$\langle d\sigma/dQ^2\rangle_\Phi\;(10^{-38}\,\mathrm{cm}^2/\mathrm{GeV}^2)$")
    ax.set_title("Comparación con datos de MINERvA en hidrógeno")
    ax.set_xlim(0.0, q2_max_display)
    ax.grid(True, alpha=0.25)
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=False, fontsize=9)

    ratio = df["ratio"].to_numpy(float)
    ratio_err = df["ratio_err_tot"].to_numpy(float)
    axr.axhline(1.0, linewidth=1.0, alpha=0.55)
    axr.errorbar(
        q2,
        ratio,
        xerr=q2err,
        yerr=ratio_err,
        fmt="o",
        markersize=5.2,
        capsize=3.0,
        elinewidth=1.4,
        zorder=3,
    )
    axr.set_xlabel(r"$Q^2\;(\mathrm{GeV}^2)$")
    axr.set_ylabel("Datos/modelo")
    axr.set_xlim(0.0, q2_max_display)
    axr.set_ylim(0.0, ratio_max_display)
    axr.grid(True, alpha=0.25)
    axr.tick_params(direction="in", top=True, right=True)

    finalize_figure(fig)
    st.pyplot(fig, use_container_width=True)
    render_figure_download_buttons(fig, "ccqe_minerva_hidrogeno_comparacion")

    st.caption(
        r"La visualización está ampliada por defecto a bajo \(Q^2\), donde se concentran la mayor parte "
        r"de los puntos con peso estadístico. Los bins de mayor \(Q^2\) pueden verse aumentando el rango mostrado."
    )

    # -----------------------------------------------------------------
    # Comparación de modelos axiales frente al dipolo estándar
    # -----------------------------------------------------------------
    st.markdown("### Sensibilidad a modelos axiales frente al dipolo estándar")

    axial_to_compare = st.multiselect(
        "Modelos axiales a comparar",
        AXIAL_MODEL_OPTIONS,
        default=AXIAL_COMPARISON_DEFAULTS,
        key="minerva_axial_compare_models",
    )

    chi2_rows: list[dict[str, float | str]] = []

    if axial_to_compare:
        m_a_dip_eff, m_a_tilde_dip_eff = _relevant_axial_parameters("Dipolo estándar", config.m_a, config.m_a_tilde)
        df_dip, _, _, _, _ = compute_minerva_tensor_comparison_cached(
            vector_model=vector_model,
            axial_model="Dipolo estándar",
            m_a=float(m_a_dip_eff),
            m_a_tilde=float(m_a_tilde_dip_eff),
            nE=int(nE),
            nQ2=int(nQ2),
            Ev_max=float(Ev_max),
            apply_cuts=bool(apply_cuts),
        )
        model_dip = df_dip["model"].to_numpy(float)

        fig2, axes2 = plt.subplots(1, 2, figsize=(12.4, 4.4))
        ax_abs, ax_ratio_model = axes2

        ax_abs.errorbar(
            q2,
            data,
            xerr=q2err,
            yerr=err,
            fmt="o",
            markersize=4.5,
            capsize=2.5,
            elinewidth=1.2,
            label="MINERvA H",
            zorder=3,
        )

        for axial_choice in axial_to_compare:
            m_a_ax_eff, m_a_tilde_ax_eff = _relevant_axial_parameters(axial_choice, config.m_a, config.m_a_tilde)
            df_ax, chi2_t, chi2_t_ndof, chi2_s, chi2_s_ndof = compute_minerva_tensor_comparison_cached(
                vector_model=vector_model,
                axial_model=axial_choice,
                m_a=float(m_a_ax_eff),
                m_a_tilde=float(m_a_tilde_ax_eff),
                nE=int(nE),
                nQ2=int(nQ2),
                Ev_max=float(Ev_max),
                apply_cuts=bool(apply_cuts),
            )

            chi2_rows.append(
                {
                    "modelo_axial": axial_choice,
                    "chi2_total": chi2_t,
                    "chi2_total_ndof": chi2_t_ndof,
                    "chi2_stat": chi2_s,
                    "chi2_stat_ndof": chi2_s_ndof,
                    "ndof": len(df_ax),
                }
            )

            pred_ax = df_ax["model"].to_numpy(float)
            x_s, y_s = build_step_xy(
                df_ax["q2_low"].to_numpy(float),
                df_ax["q2_high"].to_numpy(float),
                pred_ax,
            )
            ax_abs.plot(x_s, y_s, linewidth=2.1, label=axial_label(axial_choice))

            if axial_choice != "Dipolo estándar":
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio_model = pred_ax / model_dip
                ratio_model[~np.isfinite(ratio_model)] = np.nan
                x_r, y_r = build_step_xy(
                    df_ax["q2_low"].to_numpy(float),
                    df_ax["q2_high"].to_numpy(float),
                    ratio_model,
                )
                ax_ratio_model.plot(x_r, y_r, linewidth=2.1, label=axial_label(axial_choice))

        ax_abs.set_xlabel(r"$Q^2\;(\mathrm{GeV}^2)$")
        ax_ratio_model.set_xlabel(r"$Q^2\;(\mathrm{GeV}^2)$")
        ax_abs.set_ylabel(r"$\langle d\sigma/dQ^2\rangle_\Phi$")
        ax_ratio_model.set_ylabel("Modelo/dipolo")
        ax_abs.set_title("Predicciones axiales")
        ax_ratio_model.set_title("Cociente respecto al dipolo estándar")
        ax_abs.set_xlim(0.0, q2_max_display)
        ax_ratio_model.set_xlim(0.0, q2_max_display)
        ax_ratio_model.axhline(1.0, linewidth=1.0, alpha=0.55)

        for axc in axes2:
            axc.grid(True, alpha=0.25)
            axc.tick_params(direction="in", top=True, right=True)
            axc.legend(frameon=False, fontsize=8)

        finalize_figure(fig2)
        st.pyplot(fig2, use_container_width=True)
        render_figure_download_buttons(fig2, "ccqe_minerva_hidrogeno_modelos_axiales")

        st.markdown("### Tratamiento estadístico")
        chi2_table = pd.DataFrame(chi2_rows)
        st.dataframe(
            chi2_table.style.format(
                {
                    "chi2_total": "{:.3g}",
                    "chi2_total_ndof": "{:.3g}",
                    "chi2_stat": "{:.3g}",
                    "chi2_stat_ndof": "{:.3g}",
                    "ndof": "{:.0f}",
                }
            ),
            use_container_width=True,
        )

        st.download_button(
            "Descargar tabla χ² de modelos axiales",
            chi2_table.to_csv(index=False).encode("utf-8"),
            file_name=f"minerva_hydrogen_chi2_axial_models_{vector_model}.csv".replace(" ", "_"),
            mime="text/csv",
        )

    with st.expander("Matriz de correlación experimental"):
        rho_tot = correlation_from_covariance(Vtot)
        rho_stat = correlation_from_covariance(Vstat)

        fig_corr, ax_corr = plt.subplots(1, 2, figsize=(10.5, 4.2))
        im0 = ax_corr[0].imshow(rho_tot, vmin=-1.0, vmax=1.0, origin="lower")
        im1 = ax_corr[1].imshow(rho_stat, vmin=-1.0, vmax=1.0, origin="lower")

        ax_corr[0].set_title(r"$\rho_{ij}^{\mathrm{tot}}$")
        ax_corr[1].set_title(r"$\rho_{ij}^{\mathrm{stat}}$")

        for axc in ax_corr:
            axc.set_xlabel("bin j")
            axc.set_ylabel("bin i")
            axc.tick_params(direction="in", top=True, right=True)

        fig_corr.colorbar(im1, ax=ax_corr.ravel().tolist(), shrink=0.86, label=r"$\rho_{ij}$")
        finalize_figure(fig_corr)
        st.pyplot(fig_corr, use_container_width=True)
        render_figure_download_buttons(fig_corr, "ccqe_minerva_hidrogeno_matrices_correlacion")

        st.markdown(
            r"""
La matriz de covarianza total contiene las incertidumbres estadísticas y sistemáticas
publicadas por MINERvA, incluyendo correlaciones bin a bin. Por eso la comparación
principal se realiza con
\[
\chi^2=(d-t)^T V^{-1}(d-t).
\]
La matriz estadística se muestra como diagnóstico complementario, pero no sustituye
a la covarianza total para valorar el acuerdo global con la medida publicada.
"""
        )

    st.markdown("### Tabla bin a bin")
    show_cols = [
        "q2_low",
        "q2_high",
        "q2_center",
        "xsec",
        "err_tot_cov",
        "err_stat_cov",
        "model",
        "ratio",
    ]
    st.dataframe(df[show_cols].style.format("{:.4g}"), use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar comparación MINERvA en CSV",
        csv,
        file_name=f"minerva_hydrogen_tensor_{vector_model}_{axial_model}.csv".replace(" ", "_"),
        mime="text/csv",
    )

    st.caption(
        r"Unidades del eje vertical: \(10^{-38}\,\mathrm{cm}^2/\mathrm{GeV}^2\). "
        "El cálculo teórico interno se evalúa en unidades naturales y se convierte a las unidades del experimento."
    )

def render_axial_models_tab(config: CrossSectionConfig) -> None:
    section_header(
        "Comparación de modelos axiales",
        "Comparación directa de los modelos de estructura axial implementados en la sección 4 del TFG.",
    )

    c1, c2, c3 = st.columns([1.0, 1.0, 1.2])
    with c1:
        e_nu = st.slider(r"$E_\nu$ (GeV)", 0.25, 5.00, 1.00, 0.01, key="axcmp_E")
    with c2:
        beam = st.radio("Canal", ["neutrino", "antineutrino"], horizontal=True, key="axcmp_beam")
    with c3:
        selected_axial = st.multiselect(
            "Modelos axiales",
            AXIAL_MODEL_OPTIONS,
            default=AXIAL_COMPARISON_DEFAULTS,
            key="axcmp_models",
        )

    if not selected_axial:
        st.warning("Selecciona al menos un modelo axial.")
        return

    st.markdown("### Sección eficaz diferencial angular")

    theta_deg = np.linspace(0.1, 120.0, 800)
    df_ang = kinematics_from_theta_grid(e_nu, theta_deg, M_MU if config.include_muon_mass else 0.0)

    fig1, axes1 = plt.subplots(1, 2, figsize=(12.4, 4.4))
    ax_q2, ax_th = axes1

    for axial_model in selected_axial:
        cfg = CrossSectionConfig(
            vector_model=config.vector_model,
            axial_model=axial_model,
            m_a=config.m_a,
            m_a_tilde=config.m_a_tilde,
            use_cabibbo=config.use_cabibbo,
            include_muon_mass=config.include_muon_mass,
        )
        ds = dsigma_domega_from_kinematics(df_ang, e_nu, cfg, beam=beam)
        ax_q2.plot(df_ang["Q2_abs"], ds, linewidth=2.1, label=axial_label(axial_model))
        ax_th.plot(df_ang["theta_deg"], ds, linewidth=2.1, label=axial_label(axial_model))

    ax_q2.set_xlabel(r"$|Q^2|\;(\mathrm{GeV}^2)$")
    ax_th.set_xlabel(r"$\theta_\mu$ (º)")
    ax_q2.set_ylabel(r"$d\sigma/d\Omega\;(10^{-13}\,\mathrm{fm}^2/\mathrm{sr})$")
    ax_th.set_ylabel(r"$d\sigma/d\Omega\;(10^{-13}\,\mathrm{fm}^2/\mathrm{sr})$")
    ax_q2.set_title(r"Dependencia con $|Q^2|$")
    ax_th.set_title("Dependencia angular")

    for ax in axes1:
        ax.grid(True, alpha=0.25)
        ax.tick_params(direction="in", top=True, right=True)
        ax.legend(frameon=False, fontsize=8)

    finalize_figure(fig1)
    st.pyplot(fig1, use_container_width=True)
    render_figure_download_buttons(fig1, "ccqe_modelos_axiales_dsigma_domega")

    st.markdown("### Cocientes angulares respecto al dipolo estándar")

    cfg_d = CrossSectionConfig(
        vector_model=config.vector_model,
        axial_model="Dipolo estándar",
        m_a=config.m_a,
        m_a_tilde=config.m_a_tilde,
        use_cabibbo=config.use_cabibbo,
        include_muon_mass=config.include_muon_mass,
    )
    ds_d_ang = dsigma_domega_from_kinematics(df_ang, e_nu, cfg_d, beam=beam)

    fig2, axes2 = plt.subplots(1, 2, figsize=(12.4, 4.4))
    ax_rq2, ax_rth = axes2

    for axial_model in selected_axial:
        if axial_model == "Dipolo estándar":
            continue
        cfg = CrossSectionConfig(
            vector_model=config.vector_model,
            axial_model=axial_model,
            m_a=config.m_a,
            m_a_tilde=config.m_a_tilde,
            use_cabibbo=config.use_cabibbo,
            include_muon_mass=config.include_muon_mass,
        )
        ds = dsigma_domega_from_kinematics(df_ang, e_nu, cfg, beam=beam)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = ds / ds_d_ang
        ratio[~np.isfinite(ratio)] = np.nan

        ax_rq2.plot(df_ang["Q2_abs"], ratio, linewidth=2.1, label=axial_label(axial_model))
        ax_rth.plot(df_ang["theta_deg"], ratio, linewidth=2.1, label=axial_label(axial_model))

    ax_rq2.axhline(1.0, linewidth=1.0, alpha=0.45)
    ax_rth.axhline(1.0, linewidth=1.0, alpha=0.45)
    ax_rq2.set_xlabel(r"$|Q^2|\;(\mathrm{GeV}^2)$")
    ax_rth.set_xlabel(r"$\theta_\mu$ (º)")
    ax_rq2.set_ylabel(r"$\sigma_{\rm modelo}/\sigma_{\rm dipolo}$")
    ax_rth.set_ylabel(r"$\sigma_{\rm modelo}/\sigma_{\rm dipolo}$")
    ax_rq2.set_title(r"Dependencia con $|Q^2|$")
    ax_rth.set_title("Dependencia angular")

    for ax in axes2:
        ax.grid(True, alpha=0.25)
        ax.tick_params(direction="in", top=True, right=True)
        ax.legend(frameon=False, fontsize=8)

    finalize_figure(fig2)
    st.pyplot(fig2, use_container_width=True)
    render_figure_download_buttons(fig2, "ccqe_modelos_axiales_cocientes_domega")

    st.markdown("### Sección eficaz diferencial respecto a Eμ")

    e_mu_grid = np.linspace(M_MU * 1.001, e_nu * 0.999, 900)
    df_emu = kinematics_from_energy_grid(e_nu, e_mu_grid, M_MU if config.include_muon_mass else 0.0)

    fig3, axes3 = plt.subplots(1, 2, figsize=(12.4, 4.4))
    ax_emu, ax_emu_q2 = axes3

    for axial_model in selected_axial:
        cfg = CrossSectionConfig(
            vector_model=config.vector_model,
            axial_model=axial_model,
            m_a=config.m_a,
            m_a_tilde=config.m_a_tilde,
            use_cabibbo=config.use_cabibbo,
            include_muon_mass=config.include_muon_mass,
        )
        ds = dsigma_demu_from_energy_grid(df_emu, e_nu, cfg, beam=beam)

        ax_emu.plot(df_emu["E_mu"], ds, linewidth=2.1, label=axial_label(axial_model))
        ax_emu_q2.plot(df_emu["Q2_abs"], ds, linewidth=2.1, label=axial_label(axial_model))

    ax_emu.set_xlabel(r"$E_\mu\;(\mathrm{GeV})$")
    ax_emu_q2.set_xlabel(r"$|Q^2|\;(\mathrm{GeV}^2)$")
    ax_emu.set_ylabel(r"$d\sigma/dE_\mu\;(10^{-13}\,\mathrm{fm}^2/\mathrm{GeV})$")
    ax_emu_q2.set_ylabel(r"$d\sigma/dE_\mu\;(10^{-13}\,\mathrm{fm}^2/\mathrm{GeV})$")
    ax_emu.set_title(r"Dependencia con $E_\mu$")
    ax_emu_q2.set_title(r"Dependencia con $|Q^2|$")

    for ax in axes3:
        ax.grid(True, alpha=0.25)
        ax.tick_params(direction="in", top=True, right=True)
        ax.legend(frameon=False, fontsize=8)

    finalize_figure(fig3)
    st.pyplot(fig3, use_container_width=True)
    render_figure_download_buttons(fig3, "ccqe_modelos_axiales_dsigma_demu")

    st.markdown("### Cocientes en dσ/dEμ respecto al dipolo estándar")

    ds_d_emu = dsigma_demu_from_energy_grid(df_emu, e_nu, cfg_d, beam=beam)

    fig4, axes4 = plt.subplots(1, 2, figsize=(12.4, 4.4))
    ax_remu, ax_remu_q2 = axes4

    for axial_model in selected_axial:
        if axial_model == "Dipolo estándar":
            continue
        cfg = CrossSectionConfig(
            vector_model=config.vector_model,
            axial_model=axial_model,
            m_a=config.m_a,
            m_a_tilde=config.m_a_tilde,
            use_cabibbo=config.use_cabibbo,
            include_muon_mass=config.include_muon_mass,
        )
        ds = dsigma_demu_from_energy_grid(df_emu, e_nu, cfg, beam=beam)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = ds / ds_d_emu
        ratio[~np.isfinite(ratio)] = np.nan

        ax_remu.plot(df_emu["E_mu"], ratio, linewidth=2.1, label=axial_label(axial_model))
        ax_remu_q2.plot(df_emu["Q2_abs"], ratio, linewidth=2.1, label=axial_label(axial_model))

    ax_remu.axhline(1.0, linewidth=1.0, alpha=0.45)
    ax_remu_q2.axhline(1.0, linewidth=1.0, alpha=0.45)
    ax_remu.set_xlabel(r"$E_\mu\;(\mathrm{GeV})$")
    ax_remu_q2.set_xlabel(r"$|Q^2|\;(\mathrm{GeV}^2)$")
    ax_remu.set_ylabel(r"$\sigma_{\rm modelo}/\sigma_{\rm dipolo}$")
    ax_remu_q2.set_ylabel(r"$\sigma_{\rm modelo}/\sigma_{\rm dipolo}$")
    ax_remu.set_title(r"Dependencia con $E_\mu$")
    ax_remu_q2.set_title(r"Dependencia con $|Q^2|$")

    for ax in axes4:
        ax.grid(True, alpha=0.25)
        ax.tick_params(direction="in", top=True, right=True)
        ax.legend(frameon=False, fontsize=8)

    finalize_figure(fig4)
    st.pyplot(fig4, use_container_width=True)
    render_figure_download_buttons(fig4, "ccqe_modelos_axiales_cocientes_demu")

    st.info(
        "Esta pestaña mantiene fijo el modelo vectorial seleccionado en la barra lateral "
        "y cambia únicamente la estructura axial. Así se ve directamente cómo las distintas "
        "parametrizaciones de G_A(Q²) se propagan tanto a dσ/dΩ como a dσ/dEμ."
    )


def render_export_tab(config: CrossSectionConfig) -> None:
    section_header("Exportar datos numéricos", "Genera una tabla CSV para una configuración concreta.")

    c1, c2, c3 = st.columns(3)
    with c1:
        e_nu = st.slider(r"$E_\nu$ (GeV)", 0.25, 5.00, 1.00, 0.01, key="export_E")
    with c2:
        beam = st.radio("Canal", ["neutrino", "antineutrino"], horizontal=True, key="export_beam")
    with c3:
        theta_max = st.slider(r"$\theta_{\max}$", 10.0, 180.0, 120.0, 1.0, key="export_thmax")

    theta_deg = np.linspace(0.1, theta_max, 600)
    df = kinematics_from_theta_grid(e_nu, theta_deg, M_MU if config.include_muon_mass else 0.0)
    ds = dsigma_domega_from_kinematics(df, e_nu, config, beam=beam)

    export = df.copy()
    export["dsigma_dOmega_1e-13_fm2_sr"] = ds
    export["beam"] = beam
    export["vector_model"] = config.vector_model
    export["axial_model"] = config.axial_model

    st.dataframe(export.head(50), use_container_width=True)

    csv = export.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar CSV",
        csv,
        file_name=f"ccqe_{beam}_{config.vector_model}_E{e_nu:.2f}.csv",
        mime="text/csv",
    )


APP_SECTIONS = [
    "Formalismo",
    "dσ/dΩ",
    "Barrido dσ/dΩ",
    "dσ/dEμ",
    "Barrido dσ/dEμ",
    "σ total",
    "MINERvA: hidrógeno",
    "Modelos axiales",
    "ν vs antiν",
    "Exportar",
]


def render_active_section(section: str, config: CrossSectionConfig) -> None:
    """Renderiza solo la sección seleccionada.

    Streamlit calcula todas las pestañas creadas con st.tabs en cada rerun.
    En esta app eso vuelve muy lenta la interacción, porque MINERvA, las secciones
    totales y la exportación de figuras son costosas. Mantener una única sección
    activa conserva el orden y el contenido de la app, pero evita recalcular las
    demás al mover un slider.
    """
    if section == "Formalismo":
        render_formalism_tab()
    elif section == "dσ/dΩ":
        render_ccqe_angular_tab(config)
    elif section == "Barrido dσ/dΩ":
        render_ccqe_energy_scan_tab(config)
    elif section == "dσ/dEμ":
        render_energy_differential_tab(config)
    elif section == "Barrido dσ/dEμ":
        render_energy_differential_scan_tab(config)
    elif section == "σ total":
        render_total_cross_section_tab(config)
    elif section == "MINERvA: hidrógeno":
        render_minerva_hydrogen_tab(config)
    elif section == "Modelos axiales":
        render_axial_models_tab(config)
    elif section == "ν vs antiν":
        render_nu_antinu_tab(config)
    elif section == "Exportar":
        render_export_tab(config)


def section_selector() -> str:
    """Selector horizontal de secciones con fallback para versiones antiguas de Streamlit."""
    if hasattr(st, "segmented_control"):
        selected = st.segmented_control(
            "Sección",
            APP_SECTIONS,
            default=APP_SECTIONS[0],
            selection_mode="single",
            label_visibility="collapsed",
            key="active_ccqe_section",
        )
        return selected or APP_SECTIONS[0]

    return st.radio(
        "Sección",
        APP_SECTIONS,
        horizontal=True,
        label_visibility="collapsed",
        key="active_ccqe_section",
    )


def main() -> None:
    setup_page()

    st.title("Secciones eficaces CCQE desde el formalismo tensorial del TFG")
    st.markdown(
        "Aplicación para conectar la estructura hadrónica, los factores de forma y las secciones eficaces "
        "en dispersión elástica neutrino--nucleón por corrientes cargadas."
    )

    config = sidebar_config()

    selected_section = section_selector()
    render_active_section(selected_section, config)


if __name__ == "__main__":
    main()
