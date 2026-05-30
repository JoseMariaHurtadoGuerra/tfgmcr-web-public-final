
from __future__ import annotations

import math
import io
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st



# Constantes físicas (unidades naturales: hbarra = c = 1)

G_A0 = -1.267     # acoplamiento axial en el límite Q^2 -> 0
M_A_STD = 1.032    # masa axial dipolar estándar en GeV
M_A_HIGH = 1.35  # valor efectivo más alto usado en algunos ajustes nucleares
M_N = 0.939    # masa media del nucleón en GeV
M_PI = 0.13957  # masa del pión cargado en GeV
M_A1 = 1.23      # masa del mesón axial a1(1260) en GeV
ALPHA_SOFTPION = 0.93  # parámetro mesónico del modelo de dos componentes
GAMMA_SOFTPION = 0.53  # GeV^{-2}
ALPHA_PCAC_2C = 1.01   # parámetro mesónico del modelo de dos componentes
GAMMA_PCAC_2C = 0.54   # GeV^{-2}
M_A_BBBA07 = 1.014    # masa axial usada en BBBA2007 en GeV
BBBA07_XI_NODES = np.array([0.0, 1.0/6.0, 1.0/3.0, 0.5, 2.0/3.0, 5.0/6.0, 1.0])
BBBA07_AFA_COEFFS = np.array([1.0000, 0.9207, 0.9795, 1.0480, 1.0516, 1.2874, 0.7707])
T_CUT_Z_AXIAL = 9.0 * M_PI**2   # umbral axial isovectorial de tres piones, en GeV^2
T0_Z_DEFAULT = -0.28            # elección habitual para mejorar la convergencia, en GeV^2
Z_PRESET_DIPOLE_LIKE = np.array([-1.69, 0.20, 2.31, -1.15])
Z_PRESET_SLOW_FALL = np.array([-1.66, 0.098, 2.2245, -0.9325])


@dataclass(frozen=True)
class CurveSpec:
    label: str
    mass: float
    linestyle: str
    linewidth: float = 2.2
    alpha: float = 1.0



#Modelo físico

def tau(q2: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(q2) / (4.0 * M_N**2)


def ga_dipole(q2: np.ndarray | float, m_a: float, g_a0: float = G_A0) -> np.ndarray:
    q2 = np.asarray(q2, dtype=float)
    return g_a0 / (1.0 + q2 / m_a**2) ** 2


def ga_monopole(q2: np.ndarray | float, m_a: float, g_a0: float = G_A0) -> np.ndarray:
    q2 = np.asarray(q2, dtype=float)
    return g_a0 / (1.0 + q2 / m_a**2)


def ga_two_component(
    q2: np.ndarray | float,
    alpha: float,
    gamma: float,
    m_a1: float = M_A1,
    g_a0: float = G_A0,
) -> np.ndarray:
    q2 = np.asarray(q2, dtype=float)
    intrinsic = 1.0 / (1.0 + gamma * q2) ** 2
    meson_cloud = 1.0 - alpha + alpha * m_a1**2 / (m_a1**2 + q2)
    return g_a0 * intrinsic * meson_cloud


def ga_two_component_softpion(q2: np.ndarray | float, g_a0: float = G_A0) -> np.ndarray:
    return ga_two_component(q2, ALPHA_SOFTPION, GAMMA_SOFTPION, g_a0=g_a0)


def ga_two_component_pcac(q2: np.ndarray | float, g_a0: float = G_A0) -> np.ndarray:
    return ga_two_component(q2, ALPHA_PCAC_2C, GAMMA_PCAC_2C, g_a0=g_a0)


def xi_nachtmann_elastic(q2: np.ndarray | float, m_n: float = M_N) -> np.ndarray:
    """Variable de Nachtmann elástica usada en BBBA2007."""
    q2 = np.asarray(q2, dtype=float)
    tau_n = q2 / (4.0 * m_n**2)
    xi = np.zeros_like(q2, dtype=float)
    mask = tau_n > 0.0
    xi[mask] = 2.0 / (1.0 + np.sqrt(1.0 + 1.0 / tau_n[mask]))
    return xi


def lagrange_interpolator(x: np.ndarray | float, nodes: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Interpolación de Lagrange evaluada en x."""
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
    """Función correctora A_FA^{25}(xi) de BBBA2007."""
    xi = xi_nachtmann_elastic(q2)
    return lagrange_interpolator(xi, BBBA07_XI_NODES, BBBA07_AFA_COEFFS)


def ga_bbba07(q2: np.ndarray | float, g_a0: float = G_A0) -> np.ndarray:
    """Factor de forma axial BBBA2007."""
    q2 = np.asarray(q2, dtype=float)
    return ga_dipole(q2, M_A_BBBA07, g_a0=g_a0) * afa_bbba07_correction(q2)

#definimos la variable z
def z_conformal(
    q2: np.ndarray | float,
    t0: float = T0_Z_DEFAULT,
    t_cut: float = T_CUT_Z_AXIAL,
) -> np.ndarray:
    """Variable conforme z(Q^2; t0, t_cut) para el factor de forma axial."""
    q2 = np.asarray(q2, dtype=float)
    sqrt_num = np.sqrt(t_cut + q2)
    sqrt_ref = np.sqrt(t_cut - t0)
    return (sqrt_num - sqrt_ref) / (sqrt_num + sqrt_ref)


def fit_z_coefficients_to_dipole(
    kmax: int,
    t0: float = T0_Z_DEFAULT,
    t_cut: float = T_CUT_Z_AXIAL,
    m_a_ref: float = M_A_STD,
    q2_fit_max: float = 10.0,
    ridge_lambda: float = 1.0e-4,
) -> np.ndarray:
    """Obtiene coeficientes c_k que aproximan el dipolo estándar para G_A/g_A.

    Se usa la forma normalizada
        G_A^z/g_A = 1 + sum_{k=1}^{kmax} c_k [z(Q^2)^k - z(0)^k],
    que conserva automáticamente G_A(0)=g_A.

    La expansión z truncada puede extrapolar mal si se ajusta solo en una
    región limitada. Por ello, el ajuste por defecto usa todo el intervalo
    visible hasta q2_fit_max y añade una regularización pequeña para evitar
    coeficientes artificialmente grandes.
    """
    q2_fit_max = max(float(q2_fit_max), 0.5)
    q2_fit = np.geomspace(1.0e-5, q2_fit_max, 1200)
    z = z_conformal(q2_fit, t0=t0, t_cut=t_cut)
    z0 = float(z_conformal(0.0, t0=t0, t_cut=t_cut))
    design = np.column_stack([(z**k - z0**k) for k in range(1, kmax + 1)])
    target = 1.0 / (1.0 + q2_fit / m_a_ref**2) ** 2 - 1.0

    gram = design.T @ design
    rhs = design.T @ target
    coeffs = np.linalg.solve(gram + ridge_lambda * np.eye(kmax), rhs)
    return coeffs


def ga_z_expansion(
    q2: np.ndarray | float,
    coeffs: np.ndarray,
    t0: float = T0_Z_DEFAULT,
    t_cut: float = T_CUT_Z_AXIAL,
    g_a0: float = G_A0,
) -> np.ndarray:
    """Factor de forma axial mediante una expansión z normalizada en Q^2=0."""
    q2 = np.asarray(q2, dtype=float)
    coeffs = np.asarray(coeffs, dtype=float)
    z = z_conformal(q2, t0=t0, t_cut=t_cut)
    z0 = float(z_conformal(0.0, t0=t0, t_cut=t_cut))

    series = np.ones_like(q2, dtype=float)
    for idx, coeff in enumerate(coeffs, start=1):
        series += coeff * (z**idx - z0**idx)

    return g_a0 * series


def axial_radius_from_z_coeffs(
    coeffs: np.ndarray,
    t0: float = T0_Z_DEFAULT,
    t_cut: float = T_CUT_Z_AXIAL,
) -> float:
    """Radio axial cuadrático en GeV^{-2} asociado a la pendiente en el origen."""
    coeffs = np.asarray(coeffs, dtype=float)
    z0 = float(z_conformal(0.0, t0=t0, t_cut=t_cut))

    # Derivada dz/dQ^2 en Q^2=0.
    eps = 1.0e-6
    dzdq2_0 = float((z_conformal(eps, t0=t0, t_cut=t_cut) - z0) / eps)

    d_norm_dq2 = 0.0
    for idx, coeff in enumerate(coeffs, start=1):
        d_norm_dq2 += idx * coeff * z0 ** (idx - 1) * dzdq2_0

    # Como G_A = g_A * norm, <r_A^2> = -6 d(norm)/dQ^2.
    return -6.0 * d_norm_dq2


def z_ratio_diagnostics(
    q2: np.ndarray,
    coeffs: np.ndarray,
    t0: float = T0_Z_DEFAULT,
    t_cut: float = T_CUT_Z_AXIAL,
) -> dict[str, float]:
    """Diagnósticos sencillos para advertir si la serie se aleja demasiado del dipolo."""
    ga_d = ga_dipole(q2, M_A_STD)
    ga_z = ga_z_expansion(q2, coeffs, t0=t0, t_cut=t_cut)
    ratio = np.abs(ga_z) / np.maximum(np.abs(ga_d), 1.0e-15)
    return {
        "ratio_min": float(np.nanmin(ratio)),
        "ratio_max": float(np.nanmax(ratio)),
        "ratio_at_q2max": float(ratio[-1]),
        "max_abs_coeff": float(np.nanmax(np.abs(coeffs))) if len(coeffs) else 0.0,
    }


def q2_from_log_bounds(q2_log_min: float, q2_log_max: float) -> tuple[float, float]:
    """Convierte límites logarítmicos a límites en GeV^2."""
    return 10.0 ** float(q2_log_min), 10.0 ** float(q2_log_max)


def gp_pcac(q2: np.ndarray | float, ga_values: np.ndarray | float) -> np.ndarray:
    q2 = np.asarray(q2, dtype=float)
    ga_values = np.asarray(ga_values, dtype=float)
    return (4.0 * M_N**2 / (q2 + M_PI**2)) * ga_values


def normalized_ga(ga_values: np.ndarray | float) -> np.ndarray:
    return np.asarray(ga_values) / G_A0


def normalized_gp(gp_values: np.ndarray | float) -> np.ndarray:
    # Esto reproduce la normalización positiva y adimensional.
    return np.asarray(gp_values) * M_PI**2 / (4.0 * M_N**2 * G_A0)


def safe_geomspace(q2_min: float, q2_max: float, n: int = 800) -> np.ndarray:
    q2_min = max(float(q2_min), 1.0e-6)
    q2_max = max(float(q2_max), q2_min * 1.001)
    return np.geomspace(q2_min, q2_max, n)


def pct_change(new: float, old: float) -> float:
    if old == 0:
        return math.nan
    return 100.0 * (new - old) / old



def make_summary_table(q2_point: float, m_a_selected: float, m_a_reference: float, model_name: str) -> pd.DataFrame:
    if model_name == "Dipolar":
        ga_fn = ga_dipole
    else:
        ga_fn = ga_monopole

    ga_sel = float(ga_fn(q2_point, m_a_selected))
    gp_sel = float(gp_pcac(q2_point, ga_sel))

    ga_ref = float(ga_fn(q2_point, m_a_reference))
    gp_ref = float(gp_pcac(q2_point, ga_ref))

    return pd.DataFrame(
        {
            "Magnitud": [
                r"G_A / g_A",
                r"G_P m_pi^2 / [4 M_N^2 g_A]",
                r"|G_A| / |G_A|_ref",
                r"|G_P| / |G_P|_ref",
            ],
            "Seleccionado": [
                normalized_ga(ga_sel),
                normalized_gp(gp_sel),
                abs(ga_sel) / abs(ga_ref),
                abs(gp_sel) / abs(gp_ref),
            ],
            "Referencia": [
                normalized_ga(ga_ref),
                normalized_gp(gp_ref),
                1.0,
                1.0,
            ],
        }
    )

def _apply_common_axes_style(ax: plt.Axes, xscale: str = "log") -> None:
    ax.set_xscale(xscale)
    ax.grid(True, which="both", alpha=0.25)
    ax.tick_params(direction="in", top=True, right=True)


def plot_form_factors(
    q2: np.ndarray,
    curve_specs: list[CurveSpec],
    model_name: str,
    show_selected_q2: float | None = None,
) -> plt.Figure:
    ga_fn = ga_dipole if model_name == "Dipolar" else ga_monopole

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    ax1, ax2 = axes

    for spec in curve_specs:
        ga_vals = ga_fn(q2, spec.mass)
        gp_vals = gp_pcac(q2, ga_vals)

        ax1.plot(
            q2,
            normalized_ga(ga_vals),
            linestyle=spec.linestyle,
            linewidth=spec.linewidth,
            alpha=spec.alpha,
            label=spec.label,
        )
        ax2.plot(
            q2,
            normalized_gp(gp_vals),
            linestyle=spec.linestyle,
            linewidth=spec.linewidth,
            alpha=spec.alpha,
            label=spec.label,
        )

    if show_selected_q2 is not None:
        ax1.axvline(show_selected_q2, linestyle=":", linewidth=1.5)
        ax2.axvline(show_selected_q2, linestyle=":", linewidth=1.5)

    _apply_common_axes_style(ax1)
    _apply_common_axes_style(ax2)

    ax1.set_xlabel(r"$|Q^2|\; (\mathrm{GeV}^2)$")
    ax2.set_xlabel(r"$|Q^2|\; (\mathrm{GeV}^2)$")

    ax1.set_ylabel(r"$G_A/g_A$")
    ax2.set_ylabel(r"$G_P\,m_\pi^2 / \left[4M_N^2 g_A\right]$")

    ax1.set_title(r"Factor de forma axial")
    ax2.set_title(r"Factor de forma pseudoescalar")

    ax1.legend(frameon=False)
    ax2.legend(frameon=False)
    fig.tight_layout()
    return fig



def plot_sensitivity_ratios(
    q2: np.ndarray,
    m_a_selected: float,
    m_a_reference: float,
    model_name: str,
    show_selected_q2: float | None = None,
) -> plt.Figure:
    """Cocientes de G_A y G_P respecto a la masa axial de referencia.

    Se evita representar magnitudes no trabajadas en el TFG, manteniendo
    la app centrada en los factores de forma axial y pseudoescalar.
    """
    ga_fn = ga_dipole if model_name == "Dipolar" else ga_monopole

    ga_sel = ga_fn(q2, m_a_selected)
    gp_sel = gp_pcac(q2, ga_sel)

    ga_ref = ga_fn(q2, m_a_reference)
    gp_ref = gp_pcac(q2, ga_ref)

    fig, ax = plt.subplots(1, 1, figsize=(7.8, 4.4))

    ax.plot(q2, np.abs(ga_sel) / np.abs(ga_ref), linewidth=2.4, label=r"$|G_A|/|G_A|_{\rm ref}$")
    ax.plot(q2, np.abs(gp_sel) / np.abs(gp_ref), linewidth=2.2, linestyle="--", label=r"$|G_P|/|G_P|_{\rm ref}$")
    ax.axhline(1.0, linewidth=1.0, alpha=0.45)

    if show_selected_q2 is not None:
        ax.axvline(show_selected_q2, linestyle=":", linewidth=1.5)

    _apply_common_axes_style(ax)

    ax.set_xlabel(r"$|Q^2|\; (\mathrm{GeV}^2)$")
    ax.set_ylabel("cociente respecto a la referencia")
    ax.set_title("Sensibilidad de los factores de forma a la masa axial")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_dipole_monopole_comparison(
    q2: np.ndarray,
    m_a_dipole: float,
    m_a_monopole: float,
    show_selected_q2: float | None = None,
) -> plt.Figure:
    """Compara directamente las parametrizaciones dipolar y monopolar para G_A y G_P."""
    ga_d = ga_dipole(q2, m_a_dipole)
    ga_m = ga_monopole(q2, m_a_monopole)

    gp_d = gp_pcac(q2, ga_d)
    gp_m = gp_pcac(q2, ga_m)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    ax1, ax2 = axes

    ax1.plot(q2, normalized_ga(ga_d), linewidth=2.4, label="Dipolar")
    ax1.plot(q2, normalized_ga(ga_m), linewidth=2.4, label="Monopolar")

    ax2.plot(q2, normalized_gp(gp_d), linewidth=2.4, label="Dipolar")
    ax2.plot(q2, normalized_gp(gp_m), linewidth=2.4, label="Monopolar")

    if show_selected_q2 is not None:
        ax1.axvline(show_selected_q2, linestyle=":", linewidth=1.5)
        ax2.axvline(show_selected_q2, linestyle=":", linewidth=1.5)

    _apply_common_axes_style(ax1)
    _apply_common_axes_style(ax2)

    ax1.set_xlabel(r"$|Q^2|\;(\mathrm{GeV}^2)$")
    ax2.set_xlabel(r"$|Q^2|\;(\mathrm{GeV}^2)$")
    ax1.set_ylabel(r"$G_A(Q^2)/g_A$")
    ax2.set_ylabel(r"$G_P(Q^2)/G_P(0)$")
    ax1.set_title(r"Factor de forma axial")
    ax2.set_title(r"Factor de forma pseudoescalar")

    ax1.set_ylim(-0.02, 1.20)
    ax2.set_ylim(-0.02, 1.20)

    ax1.legend(frameon=False)
    ax2.legend(frameon=False)
    fig.tight_layout()
    return fig



def plot_dipole_monopole_ratios(
    q2: np.ndarray,
    m_a_dipole: float,
    m_a_monopole: float,
    show_selected_q2: float | None = None,
) -> plt.Figure:
    """Muestra cuánto se separa el monopolo del dipolo en G_A y G_P."""
    ga_d = ga_dipole(q2, m_a_dipole)
    ga_m = ga_monopole(q2, m_a_monopole)

    gp_d = gp_pcac(q2, ga_d)
    gp_m = gp_pcac(q2, ga_m)

    fig, ax = plt.subplots(1, 1, figsize=(7.8, 4.4))

    ax.plot(q2, np.abs(ga_m) / np.abs(ga_d), linewidth=2.4, label=r"$|G_A^{M}|/|G_A^{D}|$")
    ax.plot(q2, np.abs(gp_m) / np.abs(gp_d), linewidth=2.0, linestyle="--", label=r"$|G_P^{M}|/|G_P^{D}|$")
    ax.axhline(1.0, linewidth=1.0, alpha=0.45)

    if show_selected_q2 is not None:
        ax.axvline(show_selected_q2, linestyle=":", linewidth=1.5)

    _apply_common_axes_style(ax)

    ax.set_xlabel(r"$|Q^2|\;(\mathrm{GeV}^2)$")
    ax.set_ylabel("cociente monopolar/dipolar")
    ax.set_title("Comparación directa de factores de forma")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def make_dipole_monopole_table(q2_point: float, m_a_dipole: float, m_a_monopole: float) -> pd.DataFrame:
    ga_d = float(ga_dipole(q2_point, m_a_dipole))
    ga_m = float(ga_monopole(q2_point, m_a_monopole))

    gp_d = float(gp_pcac(q2_point, ga_d))
    gp_m = float(gp_pcac(q2_point, ga_m))

    return pd.DataFrame(
        {
            "Magnitud": [
                r"G_A/g_A",
                r"G_P/G_P(0)",
                r"|G_A^M|/|G_A^D|",
                r"|G_P^M|/|G_P^D|",
            ],
            "Dipolar": [
                normalized_ga(ga_d),
                normalized_gp(gp_d),
                1.0,
                1.0,
            ],
            "Monopolar": [
                normalized_ga(ga_m),
                normalized_gp(gp_m),
                abs(ga_m) / abs(ga_d),
                abs(gp_m) / abs(gp_d),
            ],
        }
    )

def plot_two_component_comparison(
    q2: np.ndarray,
    show_selected_q2: float | None = None,
) -> plt.Figure:
    """Compara el dipolo estándar con el modelo de dos componentes."""
    ga_d = ga_dipole(q2, M_A_STD)
    ga_sp = ga_two_component_softpion(q2)
    ga_pc = ga_two_component_pcac(q2)

    gp_d = gp_pcac(q2, ga_d)
    gp_sp = gp_pcac(q2, ga_sp)
    gp_pc = gp_pcac(q2, ga_pc)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    ax1, ax2 = axes

    ax1.plot(q2, normalized_ga(ga_d), linewidth=2.4, label=rf"Dipolar, $M_A={M_A_STD:.3f}$ GeV")
    ax1.plot(q2, normalized_ga(ga_sp), linewidth=2.4, label="Dos componentes, Soft-Pion")
    ax1.plot(q2, normalized_ga(ga_pc), linewidth=2.4, linestyle="--", label="Dos componentes, PCAC")

    ax2.plot(q2, normalized_gp(gp_d), linewidth=2.4, label=rf"Dipolar, $M_A={M_A_STD:.3f}$ GeV")
    ax2.plot(q2, normalized_gp(gp_sp), linewidth=2.4, label="Dos componentes, Soft-Pion")
    ax2.plot(q2, normalized_gp(gp_pc), linewidth=2.4, linestyle="--", label="Dos componentes, PCAC")

    if show_selected_q2 is not None:
        ax1.axvline(show_selected_q2, linestyle=":", linewidth=1.5)
        ax2.axvline(show_selected_q2, linestyle=":", linewidth=1.5)

    _apply_common_axes_style(ax1)
    _apply_common_axes_style(ax2)

    ax1.set_xlabel(r"$|Q^2|\;(\mathrm{GeV}^2)$")
    ax2.set_xlabel(r"$|Q^2|\;(\mathrm{GeV}^2)$")
    ax1.set_ylabel(r"$G_A(Q^2)/g_A$")
    ax2.set_ylabel(r"$G_P(Q^2)/G_P(0)$")
    ax1.set_title("Factor de forma axial")
    ax2.set_title("Factor de forma pseudoescalar")
    ax1.set_ylim(-0.02, 1.10)
    ax2.set_ylim(-0.02, 1.10)

    ax1.legend(frameon=False)
    ax2.legend(frameon=False)
    fig.tight_layout()
    return fig



def plot_two_component_ratios(
    q2: np.ndarray,
    show_selected_q2: float | None = None,
) -> plt.Figure:
    """Cocientes del modelo de dos componentes respecto al dipolo estándar."""
    ga_d = ga_dipole(q2, M_A_STD)
    ga_sp = ga_two_component_softpion(q2)
    ga_pc = ga_two_component_pcac(q2)

    gp_d = gp_pcac(q2, ga_d)
    gp_sp = gp_pcac(q2, ga_sp)
    gp_pc = gp_pcac(q2, ga_pc)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    ax1, ax2 = axes

    ax1.plot(q2, np.abs(ga_sp) / np.abs(ga_d), linewidth=2.4, label=r"$|G_A^{2c,\mathrm{SP}}|/|G_A^D|$")
    ax1.plot(q2, np.abs(ga_pc) / np.abs(ga_d), linewidth=2.4, linestyle="--", label=r"$|G_A^{2c,\mathrm{PCAC}}|/|G_A^D|$")

    ax2.plot(q2, np.abs(gp_sp) / np.abs(gp_d), linewidth=2.4, label=r"$|G_P^{2c,\mathrm{SP}}|/|G_P^D|$")
    ax2.plot(q2, np.abs(gp_pc) / np.abs(gp_d), linewidth=2.4, linestyle="--", label=r"$|G_P^{2c,\mathrm{PCAC}}|/|G_P^D|$")

    if show_selected_q2 is not None:
        ax1.axvline(show_selected_q2, linestyle=":", linewidth=1.5)
        ax2.axvline(show_selected_q2, linestyle=":", linewidth=1.5)

    _apply_common_axes_style(ax1)
    _apply_common_axes_style(ax2)
    ax1.axhline(1.0, linewidth=1.0, alpha=0.45)
    ax2.axhline(1.0, linewidth=1.0, alpha=0.45)

    ax1.set_xlabel(r"$|Q^2|\;(\mathrm{GeV}^2)$")
    ax2.set_xlabel(r"$|Q^2|\;(\mathrm{GeV}^2)$")
    ax1.set_ylabel("cociente respecto al dipolo")
    ax2.set_ylabel("cociente respecto al dipolo")
    ax1.set_title("Sensibilidad de $G_A$")
    ax2.set_title("Sensibilidad de $G_P$")
    ax1.legend(frameon=False)
    ax2.legend(frameon=False)
    fig.tight_layout()
    return fig


def make_two_component_table(q2_point: float) -> pd.DataFrame:
    ga_d = float(ga_dipole(q2_point, M_A_STD))
    ga_sp = float(ga_two_component_softpion(q2_point))
    ga_pc = float(ga_two_component_pcac(q2_point))

    gp_d = float(gp_pcac(q2_point, ga_d))
    gp_sp = float(gp_pcac(q2_point, ga_sp))
    gp_pc = float(gp_pcac(q2_point, ga_pc))

    return pd.DataFrame(
        {
            "Magnitud": [
                r"G_A/g_A",
                r"G_P/G_P(0)",
                r"|G_A|/|G_A^D|",
                r"|G_P|/|G_P^D|",
            ],
            "Dipolar": [
                normalized_ga(ga_d),
                normalized_gp(gp_d),
                1.0,
                1.0,
            ],
            "Soft-Pion": [
                normalized_ga(ga_sp),
                normalized_gp(gp_sp),
                abs(ga_sp) / abs(ga_d),
                abs(gp_sp) / abs(gp_d),
            ],
            "PCAC": [
                normalized_ga(ga_pc),
                normalized_gp(gp_pc),
                abs(ga_pc) / abs(ga_d),
                abs(gp_pc) / abs(gp_d),
            ],
        }
    )

def plot_bbba07_comparison(
    q2: np.ndarray,
    show_selected_q2: float | None = None,
) -> plt.Figure:
    """Compara BBBA2007 con el dipolo estándar y con el dipolo base de BBBA07."""
    ga_std = ga_dipole(q2, M_A_STD)
    ga_base = ga_dipole(q2, M_A_BBBA07)
    ga_b = ga_bbba07(q2)

    gp_std = gp_pcac(q2, ga_std)
    gp_base = gp_pcac(q2, ga_base)
    gp_b = gp_pcac(q2, ga_b)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    ax1, ax2 = axes

    ax1.plot(q2, normalized_ga(ga_std), linewidth=2.3, label=rf"Dipolar estándar, $M_A={M_A_STD:.3f}$ GeV")
    ax1.plot(q2, normalized_ga(ga_base), linewidth=2.0, linestyle=":", label=rf"Dipolar base BBBA07, $M_A={M_A_BBBA07:.3f}$ GeV")
    ax1.plot(q2, normalized_ga(ga_b), linewidth=2.4, linestyle="--", label="BBBA2007")

    ax2.plot(q2, normalized_gp(gp_std), linewidth=2.3, label=rf"Dipolar estándar, $M_A={M_A_STD:.3f}$ GeV")
    ax2.plot(q2, normalized_gp(gp_base), linewidth=2.0, linestyle=":", label=rf"Dipolar base BBBA07, $M_A={M_A_BBBA07:.3f}$ GeV")
    ax2.plot(q2, normalized_gp(gp_b), linewidth=2.4, linestyle="--", label="BBBA2007")

    if show_selected_q2 is not None:
        ax1.axvline(show_selected_q2, linestyle=":", linewidth=1.5)
        ax2.axvline(show_selected_q2, linestyle=":", linewidth=1.5)

    _apply_common_axes_style(ax1)
    _apply_common_axes_style(ax2)

    ax1.set_xlabel(r"$|Q^2|\;(\mathrm{GeV}^2)$")
    ax2.set_xlabel(r"$|Q^2|\;(\mathrm{GeV}^2)$")
    ax1.set_ylabel(r"$G_A(Q^2)/g_A$")
    ax2.set_ylabel(r"$G_P(Q^2)/G_P(0)$")
    ax1.set_title("Factor de forma axial")
    ax2.set_title("Factor de forma pseudoescalar")
    ax1.set_ylim(-0.02, 1.10)
    ax2.set_ylim(-0.02, 1.10)
    ax1.legend(frameon=False, fontsize=9)
    ax2.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig



def plot_bbba07_ratios(
    q2: np.ndarray,
    show_selected_q2: float | None = None,
) -> plt.Figure:
    """Cocientes de BBBA2007 respecto a dipolos de referencia, sin observables."""
    ga_std = ga_dipole(q2, M_A_STD)
    ga_base = ga_dipole(q2, M_A_BBBA07)
    ga_b = ga_bbba07(q2)

    gp_std = gp_pcac(q2, ga_std)
    gp_b = gp_pcac(q2, ga_b)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    ax1, ax2 = axes

    ax1.plot(q2, np.abs(ga_b) / np.abs(ga_std), linewidth=2.4, label=rf"$|G_A^{{\rm BBBA07}}|/|G_A^D(M_A={M_A_STD:.3f})|$")
    ax1.plot(q2, np.abs(ga_b) / np.abs(ga_base), linewidth=2.2, linestyle="--", label=rf"$|G_A^{{\rm BBBA07}}|/|G_A^D(M_A={M_A_BBBA07:.3f})|$")

    ax2.plot(q2, np.abs(gp_b) / np.abs(gp_std), linewidth=2.4, label=r"$|G_P^{\rm BBBA07}|/|G_P^D|$")

    if show_selected_q2 is not None:
        ax1.axvline(show_selected_q2, linestyle=":", linewidth=1.5)
        ax2.axvline(show_selected_q2, linestyle=":", linewidth=1.5)

    _apply_common_axes_style(ax1)
    _apply_common_axes_style(ax2)
    ax1.axhline(1.0, linewidth=1.0, alpha=0.45)
    ax2.axhline(1.0, linewidth=1.0, alpha=0.45)

    ax1.set_xlabel(r"$|Q^2|\;(\mathrm{GeV}^2)$")
    ax2.set_xlabel(r"$|Q^2|\;(\mathrm{GeV}^2)$")
    ax1.set_ylabel("cociente respecto al dipolo")
    ax2.set_ylabel("cociente respecto al dipolo")
    ax1.set_title("Sensibilidad de $G_A$")
    ax2.set_title(r"Sensibilidad de $G_P$")
    ax1.legend(frameon=False, fontsize=9)
    ax2.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def make_bbba07_table(q2_point: float) -> pd.DataFrame:
    ga_std = float(ga_dipole(q2_point, M_A_STD))
    ga_base = float(ga_dipole(q2_point, M_A_BBBA07))
    ga_b = float(ga_bbba07(q2_point))

    gp_std = float(gp_pcac(q2_point, ga_std))
    gp_b = float(gp_pcac(q2_point, ga_b))

    xi = float(xi_nachtmann_elastic(q2_point))
    corr = float(afa_bbba07_correction(q2_point))

    return pd.DataFrame(
        {
            "Magnitud": [
                r"xi",
                r"A_FA^25(xi)",
                r"G_A/g_A",
                r"G_P/G_P(0)",
                r"|G_A^BBBA07|/|G_A^D_std|",
                r"|G_A^BBBA07|/|G_A^D_base|",
                r"|G_P^BBBA07|/|G_P^D_std|",
            ],
            "Dipolar estándar": [
                np.nan,
                np.nan,
                normalized_ga(ga_std),
                normalized_gp(gp_std),
                1.0,
                np.nan,
                1.0,
            ],
            "BBBA2007": [
                xi,
                corr,
                normalized_ga(ga_b),
                normalized_gp(gp_b),
                abs(ga_b) / abs(ga_std),
                abs(ga_b) / abs(ga_base),
                abs(gp_b) / abs(gp_std),
            ],
        }
    )

def plot_z_preset_tfg_comparison(
    q2: np.ndarray,
    t0: float = T0_Z_DEFAULT,
    t_cut: float = T_CUT_Z_AXIAL,
) -> plt.Figure:
    """Figura comparativa fija para el TFG: dipolo y dos elecciones z."""
    ga_d = ga_dipole(q2, M_A_STD)
    ga_z_dipole_like = ga_z_expansion(q2, Z_PRESET_DIPOLE_LIKE, t0=t0, t_cut=t_cut)
    ga_z_slow = ga_z_expansion(q2, Z_PRESET_SLOW_FALL, t0=t0, t_cut=t_cut)

    ratio_dipole_like = np.abs(ga_z_dipole_like) / np.maximum(np.abs(ga_d), 1.0e-15)
    ratio_slow = np.abs(ga_z_slow) / np.maximum(np.abs(ga_d), 1.0e-15)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    ax1, ax2 = axes

    ax1.plot(q2, normalized_ga(ga_d), linewidth=2.4, label=rf"Dipolo, $M_A={M_A_STD:.3f}$ GeV")
    ax1.plot(q2, normalized_ga(ga_z_dipole_like), linewidth=2.4, linestyle="--", label=r"Expansión $z$ cercana al dipolo")
    ax1.plot(q2, normalized_ga(ga_z_slow), linewidth=2.4, linestyle="-.", label=r"Expansión $z$ con caída más lenta")

    ax2.plot(q2, ratio_dipole_like, linewidth=2.4, linestyle="--", label=r"Expansión $z$ cercana al dipolo")
    ax2.plot(q2, ratio_slow, linewidth=2.4, linestyle="-.", label=r"Expansión $z$ con caída más lenta")
    ax2.axhline(1.0, linewidth=1.0, alpha=0.45)

    _apply_common_axes_style(ax1)
    _apply_common_axes_style(ax2)

    ax1.set_xlabel(r"$|Q^2|\;(\mathrm{GeV}^2)$")
    ax2.set_xlabel(r"$|Q^2|\;(\mathrm{GeV}^2)$")
    ax1.set_ylabel(r"$G_A(Q^2)/g_A$")
    ax2.set_ylabel(r"$|G_A^z(Q^2)|/|G_A^D(Q^2)|$")
    ax1.set_title("Factor de forma axial")
    ax2.set_title("Cociente respecto al dipolo estándar")
    ax1.set_ylim(-0.02, 1.08)
    ax2.set_ylim(0.94, 1.36)
    ax1.legend(frameon=False, fontsize=9)
    ax2.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def make_z_preset_coeff_table() -> pd.DataFrame:
    """Tabla de coeficientes fijos usados en la figura TFG."""
    return pd.DataFrame(
        {
            "Caso": ["z cercana al dipolo", "z con caída más lenta"],
            "c1": [Z_PRESET_DIPOLE_LIKE[0], Z_PRESET_SLOW_FALL[0]],
            "c2": [Z_PRESET_DIPOLE_LIKE[1], Z_PRESET_SLOW_FALL[1]],
            "c3": [Z_PRESET_DIPOLE_LIKE[2], Z_PRESET_SLOW_FALL[2]],
            "c4": [Z_PRESET_DIPOLE_LIKE[3], Z_PRESET_SLOW_FALL[3]],
        }
    )


def plot_z_expansion_comparison(
    q2: np.ndarray,
    coeffs: np.ndarray,
    t0: float,
    t_cut: float,
    show_selected_q2: float | None = None,
) -> plt.Figure:
    """Compara la expansión z con el dipolo estándar."""
    ga_d = ga_dipole(q2, M_A_STD)
    ga_z = ga_z_expansion(q2, coeffs, t0=t0, t_cut=t_cut)

    gp_d = gp_pcac(q2, ga_d)
    gp_z = gp_pcac(q2, ga_z)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    ax1, ax2 = axes

    ax1.plot(q2, normalized_ga(ga_d), linewidth=2.4, label=rf"Dipolar, $M_A={M_A_STD:.3f}$ GeV")
    ax1.plot(q2, normalized_ga(ga_z), linewidth=2.4, linestyle="--", label=rf"Expansión $z$ ($k_{{max}}={len(coeffs)}$)")

    ax2.plot(q2, normalized_gp(gp_d), linewidth=2.4, label=rf"Dipolar, $M_A={M_A_STD:.3f}$ GeV")
    ax2.plot(q2, normalized_gp(gp_z), linewidth=2.4, linestyle="--", label=rf"Expansión $z$ ($k_{{max}}={len(coeffs)}$)")

    if show_selected_q2 is not None:
        ax1.axvline(show_selected_q2, linestyle=":", linewidth=1.5)
        ax2.axvline(show_selected_q2, linestyle=":", linewidth=1.5)

    _apply_common_axes_style(ax1)
    _apply_common_axes_style(ax2)

    ax1.set_xlabel(r"$|Q^2|\;(\mathrm{GeV}^2)$")
    ax2.set_xlabel(r"$|Q^2|\;(\mathrm{GeV}^2)$")
    ax1.set_ylabel(r"$G_A(Q^2)/g_A$")
    ax2.set_ylabel(r"$G_P(Q^2)/G_P(0)$")
    ax1.set_title("Factor de forma axial")
    ax2.set_title("Factor de forma pseudoescalar")
    ax1.set_ylim(-0.05, 1.15)
    ax2.set_ylim(-0.05, 1.15)
    ax1.legend(frameon=False)
    ax2.legend(frameon=False)
    fig.tight_layout()
    return fig



def plot_z_expansion_ratios(
    q2: np.ndarray,
    coeffs: np.ndarray,
    t0: float,
    t_cut: float,
    show_selected_q2: float | None = None,
) -> plt.Figure:
    """Cocientes de la expansión z respecto al dipolo estándar para G_A y G_P."""
    ga_d = ga_dipole(q2, M_A_STD)
    ga_z = ga_z_expansion(q2, coeffs, t0=t0, t_cut=t_cut)

    gp_d = gp_pcac(q2, ga_d)
    gp_z = gp_pcac(q2, ga_z)

    fig, ax = plt.subplots(1, 1, figsize=(7.8, 4.4))

    ax.plot(q2, np.abs(ga_z) / np.abs(ga_d), linewidth=2.4, label=r"$|G_A^z|/|G_A^D|$")
    ax.plot(q2, np.abs(gp_z) / np.abs(gp_d), linewidth=2.1, linestyle="--", label=r"$|G_P^z|/|G_P^D|$")
    ax.axhline(1.0, linewidth=1.0, alpha=0.45)

    if show_selected_q2 is not None:
        ax.axvline(show_selected_q2, linestyle=":", linewidth=1.5)

    _apply_common_axes_style(ax)

    ax.set_xlabel(r"$|Q^2|\;(\mathrm{GeV}^2)$")
    ax.set_ylabel("cociente respecto al dipolo")
    ax.set_title("Sensibilidad de $G_A$ y $G_P$")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def make_z_expansion_table(q2_point: float, coeffs: np.ndarray, t0: float, t_cut: float) -> pd.DataFrame:
    ga_d = float(ga_dipole(q2_point, M_A_STD))
    ga_z = float(ga_z_expansion(q2_point, coeffs, t0=t0, t_cut=t_cut))

    gp_d = float(gp_pcac(q2_point, ga_d))
    gp_z = float(gp_pcac(q2_point, ga_z))

    z_point = float(z_conformal(q2_point, t0=t0, t_cut=t_cut))
    r2_a = float(axial_radius_from_z_coeffs(coeffs, t0=t0, t_cut=t_cut))

    return pd.DataFrame(
        {
            "Magnitud": [
                r"z(Q^2)",
                r"<r_A^2> [GeV^{-2}]",
                r"G_A/g_A",
                r"G_P/G_P(0)",
                r"|G_A^z|/|G_A^D|",
                r"|G_P^z|/|G_P^D|",
            ],
            "Dipolar estándar": [
                np.nan,
                12.0 / M_A_STD**2,
                normalized_ga(ga_d),
                normalized_gp(gp_d),
                1.0,
                1.0,
            ],
            "Expansión z": [
                z_point,
                r2_a,
                normalized_ga(ga_z),
                normalized_gp(gp_z),
                abs(ga_z) / abs(ga_d),
                abs(gp_z) / abs(gp_d),
            ],
        }
    )

def inject_css() -> None:
    st.markdown(
        """
        <style>
        .katex-display {
            overflow-x: auto;
            overflow-y: hidden;
            padding-bottom: 0.35rem;
        }
        .small-note {
            font-size: 0.95rem;
            opacity: 0.92;
        }
        .theory-card {
            border: 1px solid rgba(128,128,128,0.25);
            border-radius: 14px;
            padding: 1rem 1rem 0.85rem 1rem;
            margin-bottom: 0.75rem;
            background: rgba(250,250,250,0.02);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def figure_to_bytes(fig: plt.Figure, file_format: str) -> bytes:
    buffer = io.BytesIO()
    save_kwargs = {"format": file_format, "bbox_inches": "tight"}
    if file_format == "png":
        save_kwargs["dpi"] = 300
    fig.savefig(buffer, **save_kwargs)
    buffer.seek(0)
    return buffer.getvalue()


def render_figure_download_buttons(fig: plt.Figure, basename: str) -> None:
    safe_name = basename.replace(" ", "_").replace("/", "_")
    with st.expander("Descargar esta gráfica", expanded=False):
        col_pdf, col_png = st.columns(2)
        with col_pdf:
            st.download_button(
                "PDF vectorial",
                data=figure_to_bytes(fig, "pdf"),
                file_name=f"{safe_name}.pdf",
                mime="application/pdf",
                key=f"download_{safe_name}_pdf",
            )
        with col_png:
            st.download_button(
                "PNG 300 dpi",
                data=figure_to_bytes(fig, "png"),
                file_name=f"{safe_name}.png",
                mime="image/png",
                key=f"download_{safe_name}_png",
            )


def section_header(title: str, body: str | None = None) -> None:
    st.markdown(f"## {title}")
    if body:
        st.markdown(body)


def render_formalism_tab() -> None:
    section_header(
        "Formalismo y notación",
        "Las fórmulas siguientes siguen la notación típica de la literatura de dispersión de neutrinos y reproducen la estructura discutida en el TFG.",
    )

    st.markdown("### 1. Factor de forma axial")
    st.latex(r"G_A(Q^2) = \frac{g_A}{\left(1 + Q^2/M_A^2\right)^2}")

    st.markdown("### 2. Factor de forma pseudoescalar a partir de PCAC")
    st.latex(r"G_P(Q^2) = \frac{4M_N^2}{Q^2 + m_\pi^2}\, G_A(Q^2)")

    st.markdown("### 3. Variable adimensional útil")
    st.latex(r"\tau = \frac{|Q^2|}{4M_N^2}")

 #   with st.expander("¿Por qué la curva pseudoescalar reacciona menos a cambios en la masa axial?"):
   #     st.markdown(
   #         "Porque su comportamiento global en $Q^2$ está dominado por el denominador del polo del pión. "
    #        "Aunque $G_P$ es proporcional a $G_A$, el factor extra de baja escala que involucra la masa del pión suprime mucho más la sensibilidad visible a $M_A$ que en el propio $G_A$."
    #    )

  #  with st.expander("¿Por qué incluir una opción monopolar si la dipolar es la elección estándar?"):
    #    st.markdown(
     #       "Porque es pedagógicamente útil comparar formas funcionales. La literatura adopta habitualmente un ansatz dipolar, pero contrastarlo con un perfil monopolar ayuda al lector a ver que la cuestión no es solo el valor de la masa axial, sino también la forma funcional asumida para la dependencia axial con el momento transferido."
       # )

def render_two_component_model_tab() -> None:
    section_header(
        "Modelo de dos componentes",
        "Esta pestaña incorpora el modelo axial de dos componentes, en el que $G_A(Q^2)$ se describe mediante una contribución intrínseca de tres quarks y una contribución mesónica.",
    )

    st.markdown('<div class="theory-card">', unsafe_allow_html=True)
    st.markdown(
        r"El modelo se escribe como "
        r"$G_A^{2c}(Q^2)=G_A(0)\,g(Q^2)\left[1-\alpha+\alpha m_{a_1}^2/(m_{a_1}^2+Q^2)\right]$, "
        r"con $g(Q^2)=(1+\gamma Q^2)^{-2}$. "
        r"La función $g(Q^2)$ representa la contribución intrínseca de los 3 quarks, mientras que el término proporcional a $\alpha$ introduce la contribución mesónica:"
    )
    st.latex(
        r"G_A^{2c}(Q^2)=G_A(0)\,\frac{1}{(1+\gamma Q^2)^2}"
        r"\left[1-\alpha+\alpha\frac{m_{a_1}^2}{m_{a_1}^2+Q^2}\right]"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.0, 1.0, 1.2])

    with c1:
        st.markdown("**Parámetros usados**")
        params_df = pd.DataFrame(
            {
                "Modelo": ["Soft-Pion", "PCAC"],
                r"$\alpha$": [ALPHA_SOFTPION, ALPHA_PCAC_2C],
                r"$\gamma$ (GeV$^{-2}$)": [GAMMA_SOFTPION, GAMMA_PCAC_2C],
            }
        )
        st.dataframe(params_df, use_container_width=True, hide_index=True)

    with c2:
        q2_log_min, q2_log_max = st.slider(
            r"Rango de representación en $\log_{10}(|Q^2|/\mathrm{GeV}^2)$",
            min_value=-4.0,
            max_value=1.0,
            value=(-3.0, 0.7),
            step=0.1,
            key="two_component_q2_range",
        )

        q2_probe_log = st.slider(
            r"Punto de inspección $\log_{10}(|Q^2|/\mathrm{GeV}^2)$",
            min_value=q2_log_min,
            max_value=q2_log_max,
            value=min(-1.0, q2_log_max),
            step=0.05,
            key="two_component_q2_probe",
        )
        q2_probe = 10.0 ** q2_probe_log

    with c3:
        st.info(
            r"Las etiquetas Soft-Pion y PCAC se refieren a dos formas de interpretar los datos de electroproducción de piones "
            r"en la extracción de $G_A(Q^2)$. No deben confundirse con la relación PCAC usada para obtener $G_P(Q^2)$ a partir de $G_A(Q^2)$."
        )

    q2 = safe_geomspace(10.0 ** q2_log_min, 10.0 ** q2_log_max)

    st.markdown("### Comparación de factores de forma")
    fig_tc = plot_two_component_comparison(q2, show_selected_q2=q2_probe)
    st.pyplot(fig_tc, use_container_width=True)
    render_figure_download_buttons(fig_tc, "axial_dos_componentes_factores_forma")

    st.markdown("### Cocientes respecto al dipolo estándar")
    fig_tc_ratio = plot_two_component_ratios(q2, show_selected_q2=q2_probe)
    st.pyplot(fig_tc_ratio, use_container_width=True)
    render_figure_download_buttons(fig_tc_ratio, "axial_dos_componentes_cocientes")

    st.markdown("### Lectura numérica en el punto seleccionado")
    df = make_two_component_table(q2_probe)
    st.dataframe(
        df.style.format({"Dipolar": "{:.6f}", "Soft-Pion": "{:.6f}", "PCAC": "{:.6f}"}),
        use_container_width=True,
    )

    ga_d = float(ga_dipole(q2_probe, M_A_STD))
    ga_sp = float(ga_two_component_softpion(q2_probe))
    ga_pc = float(ga_two_component_pcac(q2_probe))

    st.markdown(
        rf"En $|Q^2|={q2_probe:.4g}\,\mathrm{{GeV}}^2$, el modelo Soft-Pion da "
        rf"$|G_A^{{2c}}|/|G_A^D|={abs(ga_sp)/abs(ga_d):.4f}$, mientras que el modelo PCAC da "
        rf"$|G_A^{{2c}}|/|G_A^D|={abs(ga_pc)/abs(ga_d):.4f}$ respecto al dipolo estándar."
    )

    with st.expander("Lectura física de esta comparación"):
        st.markdown(
            r"El interés de esta pestaña es mostrar que el modelo de dos componentes no solo cambia una masa efectiva, "
            r"sino la forma funcional de $G_A(Q^2)$. La comparación con el dipolo permite localizar las regiones de $Q^2$ "
            r"donde la descripción intrínseca más nube mesónica puede modificar de forma apreciable el factor axial y, por tanto, "
            r"las observables y secciones eficaces que dependen de él."
        )


def render_dipole_monopole_comparison_tab() -> None:
    section_header(
        "Comparación entre parametrizaciones dipolar y monopolar",
        "Esta pestaña compara directamente dos hipótesis funcionales para la dependencia en "
        r"$Q^2$ del factor de forma axial. La masa axial dipolar $M_A$ y la masa axial "
        r"monopolar $\widetilde{M}_A$ no deben compararse como si fueran el mismo parámetro, "
        "ya que pertenecen a formas funcionales distintas.",
    )

    st.markdown('<div class="theory-card">', unsafe_allow_html=True)
    st.markdown(
        r"En la forma dipolar se toma "
        r"$G_A^D(Q^2)=g_A/(1+Q^2/M_A^2)^2$, mientras que en la forma monopolar se considera "
        r"$G_A^M(Q^2)=g_A/(1+Q^2/\widetilde{M}_A^2)$. "
        r"La forma monopolar no se introduce como una parametrización experimentalmente fijada, "
        r"sino como una alternativa fenomenológica para estudiar la sensibilidad de los resultados "
        r"a la dependencia funcional asumida para $G_A(Q^2)$."
    )
    st.markdown(
        r"Siguiendo la literatura, se explora habitualmente el intervalo "
        r"$\widetilde{M}_A=0.5$--$1.0\,\mathrm{GeV}$, tomando "
        r"$\widetilde{M}_A\simeq0.75\,\mathrm{GeV}$ como valor representativo."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.0, 1.0, 1.2])

    with c1:
        m_a_dipole = st.slider(
            r"Masa axial dipolar $M_A$ (GeV)",
            min_value=1.00,
            max_value=1.40,
            value=1.032,
            step=0.001,
            key="comparison_m_a_dipole",
        )

        m_a_monopole = st.slider(
            r"Masa axial monopolar $\widetilde{M}_A$ (GeV)",
            min_value=0.50,
            max_value=1.00,
            value=0.75,
            step=0.001,
            key="comparison_m_a_monopole",
        )

    with c2:
        q2_log_min, q2_log_max = st.slider(
            r"Rango de representación en $\log_{10}(|Q^2|/\mathrm{GeV}^2)$",
            min_value=-4.0,
            max_value=1.0,
            value=(-3.0, 0.7),
            step=0.1,
            key="comparison_q2_range",
        )

        q2_probe_log = st.slider(
            r"Punto de inspección $\log_{10}(|Q^2|/\mathrm{GeV}^2)$",
            min_value=q2_log_min,
            max_value=q2_log_max,
            value=min(-1.0, q2_log_max),
            step=0.05,
            key="comparison_q2_probe",
        )

        q2_probe = 10.0 ** q2_probe_log

    with c3:
        st.info(
            r"Si se iguala aproximadamente la pendiente inicial del monopolo y del dipolo, "
            r"$\widetilde{M}_A\simeq M_A/\sqrt{2}$. "
            r"Para $M_A\simeq1.03\,\mathrm{GeV}$ esto da "
            r"$\widetilde{M}_A\simeq0.73\,\mathrm{GeV}$, muy cercano al valor representativo "
            r"$0.75\,\mathrm{GeV}$."
        )

    q2 = safe_geomspace(10.0 ** q2_log_min, 10.0 ** q2_log_max)

    st.markdown("### Factores de forma normalizados")
    fig_comparison = plot_dipole_monopole_comparison(
        q2=q2,
        m_a_dipole=m_a_dipole,
        m_a_monopole=m_a_monopole,
        show_selected_q2=q2_probe,
    )
    st.pyplot(fig_comparison, use_container_width=True)
    render_figure_download_buttons(fig_comparison, "axial_dipolo_monopolo_factores_forma")

    st.markdown("### Cocientes monopolar/dipolar")
    fig_ratios = plot_dipole_monopole_ratios(
        q2=q2,
        m_a_dipole=m_a_dipole,
        m_a_monopole=m_a_monopole,
        show_selected_q2=q2_probe,
    )
    st.pyplot(fig_ratios, use_container_width=True)
    render_figure_download_buttons(fig_ratios, "axial_dipolo_monopolo_cocientes")

    st.markdown("### Lectura numérica en el punto seleccionado")
    comparison_df = make_dipole_monopole_table(q2_probe, m_a_dipole, m_a_monopole)
    st.dataframe(
        comparison_df.style.format({"Dipolar": "{:.6f}", "Monopolar": "{:.6f}"}),
        use_container_width=True,
    )

    ga_d = float(ga_dipole(q2_probe, m_a_dipole))
    ga_m = float(ga_monopole(q2_probe, m_a_monopole))
    gp_d = float(gp_pcac(q2_probe, ga_d))
    gp_m = float(gp_pcac(q2_probe, ga_m))

    st.markdown(
        rf"En $|Q^2|={q2_probe:.4g}\,\mathrm{{GeV}}^2$, la parametrización monopolar con "
        rf"$\widetilde{{M}}_A={m_a_monopole:.3f}\,\mathrm{{GeV}}$ da "
        rf"$|G_A^M|/|G_A^D|={abs(ga_m)/abs(ga_d):.4f}$ y "
        rf"$|G_P^M|/|G_P^D|={abs(gp_m)/abs(gp_d):.4f}$ respecto al dipolo con "
        rf"$M_A={m_a_dipole:.3f}\,\mathrm{{GeV}}$."
    )

    with st.expander("Lectura física de la comparación"):
        st.markdown(
            r"La comparación muestra que la incertidumbre no reside únicamente en el valor "
            r"numérico de la masa axial, sino también en la forma funcional elegida para "
            r"describir $G_A(Q^2)$. A bajo $Q^2$ ambas parametrizaciones pueden ser parecidas, "
            r"especialmente si se escoge $\widetilde{M}_A\simeq0.75\,\mathrm{GeV}$, pero a "
            r"transferencias de momento mayores pueden separarse de forma apreciable. "
        )



def render_simulator_tab() -> None:
    st.markdown("### Controles")
    c1, c2, c3 = st.columns([1.1, 1.1, 1.2])

    with c1:
        model_name = st.radio("Parametrización axial", ["Dipolar", "Monopolar"], horizontal=True)
        m_a_selected = st.slider(r"Masa axial seleccionada $M_A$ (GeV)", 0.75, 1.60, 1.032, 0.001)
        m_a_reference = st.slider(r"Masa axial de referencia $M_A^{\rm ref}$ (GeV)", 0.75, 1.60, 1.032, 0.001)

    with c2:
        q2_log_min, q2_log_max = st.slider(
            r"Rango de representación en $\\log_{10}(|Q^2|/\\mathrm{GeV}^2)$",
            min_value=-4.0,
            max_value=1.0,
            value=(-3.0, 0.7),
            step=0.1,
        )
        q2_probe_log = st.slider(
            r"Punto de inspección $\\log_{10}(|Q^2|/\\mathrm{GeV}^2)$",
            min_value=q2_log_min,
            max_value=q2_log_max,
            value=min(-1.0, q2_log_max),
            step=0.05,
        )
        q2_probe = 10.0 ** q2_probe_log

    with c3:
        show_standard = st.checkbox("Mostrar benchmark estándar $M_A = 1.032$ GeV", value=True)
        show_high = st.checkbox("Mostrar benchmark alto $M_A = 1.35$ GeV", value=True)
        show_reference = st.checkbox("Mostrar curva de referencia personalizada", value=True)

    q2 = safe_geomspace(10.0 ** q2_log_min, 10.0 ** q2_log_max)

    curve_specs: list[CurveSpec] = [
        CurveSpec(label=fr"Seleccionada  $M_A = {m_a_selected:.3f}$ GeV", mass=m_a_selected, linestyle="-", linewidth=2.6)
    ]
    if show_reference and abs(m_a_reference - m_a_selected) > 1e-12:
        curve_specs.append(CurveSpec(label=fr"Referencia  $M_A = {m_a_reference:.3f}$ GeV", mass=m_a_reference, linestyle="--", linewidth=2.2, alpha=0.95))
    if show_standard and all(abs(spec.mass - M_A_STD) > 1e-12 for spec in curve_specs):
        curve_specs.append(CurveSpec(label=fr"Estándar  $M_A = {M_A_STD:.3f}$ GeV", mass=M_A_STD, linestyle=":", linewidth=2.0, alpha=0.95))
    if show_high and all(abs(spec.mass - M_A_HIGH) > 1e-12 for spec in curve_specs):
        curve_specs.append(CurveSpec(label=fr"Alto  $M_A = {M_A_HIGH:.2f}$ GeV", mass=M_A_HIGH, linestyle="-.", linewidth=2.0, alpha=0.95))

    st.markdown("### Factores de forma")
    fig_ff = plot_form_factors(q2, curve_specs, model_name=model_name, show_selected_q2=q2_probe)
    st.pyplot(fig_ff, use_container_width=True)
    render_figure_download_buttons(fig_ff, "axial_simulador_factores_forma")

    st.markdown("### Sensibilidad relativa de $G_A$ y $G_P$")
    fig_ratio = plot_sensitivity_ratios(
        q2=q2,
        m_a_selected=m_a_selected,
        m_a_reference=m_a_reference,
        model_name=model_name,
        show_selected_q2=q2_probe,
    )
    st.pyplot(fig_ratio, use_container_width=True)
    render_figure_download_buttons(fig_ratio, "axial_simulador_sensibilidad_ga_gp")

    st.markdown("### Lectura numérica en el momento transferido seleccionado")
    summary_df = make_summary_table(q2_probe, m_a_selected, m_a_reference, model_name)
    st.dataframe(summary_df.style.format({"Seleccionado": "{:.6f}", "Referencia": "{:.6f}"}), use_container_width=True)

    ga_fn = ga_dipole if model_name == "Dipolar" else ga_monopole
    ga_sel = float(ga_fn(q2_probe, m_a_selected))
    gp_sel = float(gp_pcac(q2_probe, ga_sel))

    ga_ref = float(ga_fn(q2_probe, m_a_reference))
    gp_ref = float(gp_pcac(q2_probe, ga_ref))

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(
            label=r"Cambio en $|G_A|$ en el $Q^2$ seleccionado",
            value=f"{abs(ga_sel)/abs(ga_ref):.4f}",
            delta=f"{pct_change(abs(ga_sel), abs(ga_ref)):+.2f}%",
        )
    with col_b:
        st.metric(
            label=r"Cambio en $|G_P|$ en el $Q^2$ seleccionado",
            value=f"{abs(gp_sel)/abs(gp_ref):.4f}",
            delta=f"{pct_change(abs(gp_sel), abs(gp_ref)):+.2f}%",
        )

    st.markdown(
        rf"En el punto de inspección $|Q^2| = {q2_probe:.4g}\,\mathrm{{GeV}}^2$, el ansatz {model_name.lower()} seleccionado con "
        rf"$M_A = {m_a_selected:.3f}\,\mathrm{{GeV}}$ da lugar a $G_A/g_A = {normalized_ga(ga_sel):.5f}$ y "
        rf"$G_P m_\pi^2 / [4M_N^2 g_A] = {normalized_gp(gp_sel):.5f}$."
    )

    with st.expander("¿Cómo deben leerse estas gráficas?"):
        st.markdown(
            "La primera figura muestra los factores de forma normalizados con el estilo del TFG. La segunda responde a una pregunta práctica: "
            "**¿cuánto cambia la física respecto a una elección de referencia de la masa axial?**"
        )

def render_bbba07_tab() -> None:
    section_header(
        "Parametrización BBBA2007",
        "Esta pestaña incorpora la parametrización axial BBBA2007 como una corrección fenomenológica a una forma dipolar base mediante la variable de Nachtmann.",
    )

    st.markdown('<div class="theory-card">', unsafe_allow_html=True)
    st.markdown(
        r"En el sector axial, BBBA2007 puede escribirse como "
        r"$G_A^{\rm BBBA07}(Q^2)=G_A^D(Q^2;M_A^{\rm BBBA07})\,A_{F_A}^{25}(\xi)$, "
        r"donde $M_A^{\rm BBBA07}=1.014\,\mathrm{GeV}$ y $A_{F_A}^{25}(\xi)$ es una función correctora "
        r"construida por interpolación de Lagrange."
    )
    st.latex(
        r"G_A^{\rm BBBA07}(Q^2)=\frac{G_A(0)}{\left[1+Q^2/(M_A^{\rm BBBA07})^2\right]^2}\,"
        r"A_{F_A}^{25}(\xi)"
    )
    st.latex(
        r"\xi=\frac{2}{1+\sqrt{1+1/\tau_N}},\qquad \tau_N=\frac{Q^2}{4M_N^2}"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.0, 1.0, 1.2])

    with c1:
        st.markdown("**Parámetros BBBA2007 usados**")
        coeff_df = pd.DataFrame(
            {
                "j": np.arange(1, 8),
                r"$\xi_j$": BBBA07_XI_NODES,
                r"$p_j$": BBBA07_AFA_COEFFS,
            }
        )
        st.dataframe(
            coeff_df.style.format({r"$\xi_j$": "{:.4f}", r"$p_j$": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )

    with c2:
        q2_log_min, q2_log_max = st.slider(
            r"Rango de representación en $\log_{10}(|Q^2|/\mathrm{GeV}^2)$",
            min_value=-4.0,
            max_value=1.0,
            value=(-3.0, 0.7),
            step=0.1,
            key="bbba07_q2_range",
        )
        q2_probe_log = st.slider(
            r"Punto de inspección $\log_{10}(|Q^2|/\mathrm{GeV}^2)$",
            min_value=q2_log_min,
            max_value=q2_log_max,
            value=min(-1.0, q2_log_max),
            step=0.05,
            key="bbba07_q2_probe",
        )
        q2_probe = 10.0 ** q2_probe_log

    with c3:
        st.info(
            r"La curva BBBA2007 no es simplemente otro dipolo: usa un dipolo base con "
            r"$M_A=1.014\,\mathrm{GeV}$ y lo multiplica por la corrección $A_{F_A}^{25}(\xi)$. "
            r"Por eso conviene comparar tanto con el dipolo estándar como con el dipolo base de BBBA2007."
        )

    q2 = safe_geomspace(10.0 ** q2_log_min, 10.0 ** q2_log_max)

    st.markdown("### Comparación de factores de forma")
    fig_bbba = plot_bbba07_comparison(q2, show_selected_q2=q2_probe)
    st.pyplot(fig_bbba, use_container_width=True)
    render_figure_download_buttons(fig_bbba, "axial_bbba2007_factores_forma")

    st.markdown("### Cocientes respecto al dipolo")
    fig_bbba_ratio = plot_bbba07_ratios(q2, show_selected_q2=q2_probe)
    st.pyplot(fig_bbba_ratio, use_container_width=True)
    render_figure_download_buttons(fig_bbba_ratio, "axial_bbba2007_cocientes")

    st.markdown("### Lectura numérica en el punto seleccionado")
    df = make_bbba07_table(q2_probe)
    st.dataframe(
        df.style.format({"Dipolar estándar": "{:.6f}", "BBBA2007": "{:.6f}"}),
        use_container_width=True,
    )

    ga_std = float(ga_dipole(q2_probe, M_A_STD))
    ga_b = float(ga_bbba07(q2_probe))
    corr = float(afa_bbba07_correction(q2_probe))

    st.markdown(
        rf"En $|Q^2|={q2_probe:.4g}\,\mathrm{{GeV}}^2$, BBBA2007 da "
        rf"$|G_A^{{\rm BBBA07}}|/|G_A^D|={abs(ga_b)/abs(ga_std):.4f}$ "
        rf"respecto al dipolo estándar con $M_A={M_A_STD:.3f}\,\mathrm{{GeV}}$. "
        rf"La corrección interpoladora vale $A_{{F_A}}^{{25}}(\xi)={corr:.4f}$."
    )

    with st.expander("Lectura física de esta comparación"):
        st.markdown(
            r"BBBA2007 separa dos efectos: por un lado, el uso de una masa axial base ligeramente distinta "
            r"($M_A=1.014\,\mathrm{GeV}$), y por otro, una corrección dependiente de $\xi$. "
            r"La gráfica de cocientes permite ver dónde la parametrización se separa realmente del dipolo estándar. "
            r"Esta comparación es útil porque BBBA2007 no modifica la normalización $G_A(0)=g_A$, sino la forma de la curva a $Q^2$ finito."
        )


def render_z_expansion_tab() -> None:
    section_header(
        "Expansión z",
        "Esta pestaña introduce una parametrización flexible del factor de forma axial. Incluye una comparación predefinida para el TFG y una zona interactiva para explorar coeficientes.",
    )

    st.markdown('<div class="theory-card">', unsafe_allow_html=True)
    st.markdown(
        r"La expansión $z$ no define una única curva axial, sino una familia de curvas dependiente de una nueva variable z. "
        r"Se transforma $Q^2$ en dicha variable $z$ y se escribe el factor de forma como una serie truncada. "
        r"En esta simulación se usa una forma normalizada que impone automáticamente $G_A(0)=g_A$:"
    )
    st.latex(
        r"z(Q^2;t_0,t_{\rm cut})="
        r"\frac{\sqrt{t_{\rm cut}+Q^2}-\sqrt{t_{\rm cut}-t_0}}"
        r"{\sqrt{t_{\rm cut}+Q^2}+\sqrt{t_{\rm cut}-t_0}}"
    )
    st.latex(
        r"\frac{G_A^z(Q^2)}{g_A}=1+\sum_{k=1}^{k_{\max}}c_k\left[z(Q^2)^k-z(0)^k\right]"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Comparación predefinida para el TFG")
    st.markdown(
        r"Esta figura usa dos elecciones fijas de coeficientes para que la app y la figura del TFG sean coherentes: "
        r"una expansión $z$ prácticamente coincidente con el dipolo y otra con una caída ligeramente más lenta. "
        r"En ambos casos se toma $t_{\rm cut}=9m_\pi^2$, $t_0=-0.28\,\mathrm{GeV}^2$ y $k_{\max}=4$."
    )

    q2_tfg = safe_geomspace(1.0e-3, 10.0 ** 0.7)
    fig_z_tfg = plot_z_preset_tfg_comparison(q2_tfg, t0=T0_Z_DEFAULT, t_cut=T_CUT_Z_AXIAL)
    st.pyplot(fig_z_tfg, use_container_width=True)
    render_figure_download_buttons(fig_z_tfg, "axial_z_tfg_comparacion_predefinida")

    preset_coeff_df = make_z_preset_coeff_table()
    st.dataframe(
        preset_coeff_df.style.format({"c1": "{:.4f}", "c2": "{:.4f}", "c3": "{:.4f}", "c4": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Lectura física de la figura predefinida"):
        st.markdown(
            r"La curva cercana al dipolo sirve como control: muestra que la expansión $z$ puede reproducir casi exactamente "
            r"el ansatz dipolar si los coeficientes se eligen de forma adecuada. La segunda curva ilustra una deformación suave "
            r"en la que $G_A^z(Q^2)$ cae algo más lentamente, por lo que el cociente $|G_A^z|/|G_A^D|$ queda por encima de la unidad "
            r"a transferencias intermedias y altas. Esta comparación debe interpretarse como una prueba de sensibilidad funcional, "
            r"no como un ajuste experimental independiente."
        )

    st.markdown("### Exploración interactiva de coeficientes")

    c1, c2, c3 = st.columns([1.05, 1.0, 1.25])

    with c1:
        kmax = st.slider(
            r"Orden de truncamiento $k_{\max}$",
            min_value=2,
            max_value=6,
            value=4,
            step=1,
            key="z_kmax",
        )
        t0 = st.slider(
            r"$t_0$ (GeV$^2$)",
            min_value=-1.00,
            max_value=0.00,
            value=T0_Z_DEFAULT,
            step=0.01,
            key="z_t0",
        )
        use_default_fit = st.checkbox(
            "Usar coeficientes que aproximan el dipolo estándar",
            value=True,
            key="z_use_default_fit",
        )

    with c2:
        q2_log_min, q2_log_max = st.slider(
            r"Rango de representación en $\log_{10}(|Q^2|/\mathrm{GeV}^2)$",
            min_value=-4.0,
            max_value=1.0,
            value=(-3.0, 0.7),
            step=0.1,
            key="z_q2_range",
        )
        q2_probe_log = st.slider(
            r"Punto de inspección $\log_{10}(|Q^2|/\mathrm{GeV}^2)$",
            min_value=q2_log_min,
            max_value=q2_log_max,
            value=min(-1.0, q2_log_max),
            step=0.05,
            key="z_q2_probe",
        )
        q2_probe = 10.0 ** q2_probe_log

    with c3:
        t_cut = st.number_input(
            r"$t_{\rm cut}$ (GeV$^2$)",
            min_value=0.01,
            max_value=2.00,
            value=float(T_CUT_Z_AXIAL),
            step=0.001,
            format="%.6f",
            key="z_tcut",
        )
        st.info(
            r"Para el canal axial isovectorial se suele tomar $t_{\rm cut}=9m_\pi^2$. "
            r"Los coeficientes $c_k$ no son constantes universales: deben ajustarse a datos o a cálculos de lattice QCD. "
            r"Por defecto se calculan coeficientes regularizados que aproximan el dipolo en el rango visible, "
            r"pero esta pestaña debe leerse como una herramienta de sensibilidad funcional, no como un ajuste experimental."
        )

    q2_fit_min, q2_fit_max = q2_from_log_bounds(q2_log_min, q2_log_max)
    default_coeffs = fit_z_coefficients_to_dipole(
        kmax,
        t0=t0,
        t_cut=t_cut,
        m_a_ref=M_A_STD,
        q2_fit_max=q2_fit_max,
        ridge_lambda=1.0e-4,
    )

    coeffs = []
    st.markdown("### Coeficientes de la serie")
    coeff_cols = st.columns(min(kmax, 3))
    for idx in range(1, kmax + 1):
        default_value = float(default_coeffs[idx - 1]) if use_default_fit else 0.0
        col = coeff_cols[(idx - 1) % len(coeff_cols)]
        with col:
            coeff = st.slider(
                rf"$c_{idx}$",
                min_value=-30.0,
                max_value=30.0,
                value=default_value,
                step=0.01,
                key=f"z_coeff_{idx}_{kmax}_{round(t0, 3)}_{round(float(t_cut), 4)}_{use_default_fit}",
            )
        coeffs.append(coeff)
    coeffs = np.array(coeffs, dtype=float)

    q2 = safe_geomspace(10.0 ** q2_log_min, 10.0 ** q2_log_max)
    diagnostics = z_ratio_diagnostics(q2, coeffs, t0=t0, t_cut=t_cut)

    if diagnostics["ratio_max"] > 2.0 or diagnostics["ratio_min"] < 0.5:
        st.warning(
            "La serie z seleccionada se aleja mucho del dipolo en el intervalo representado. "
            "Esto no implica necesariamente un modelo físico realista: una expansión truncada sin restricciones adicionales "
            "puede extrapolar mal, sobre todo a alto Q². Reduce el rango de Q², baja el orden o ajusta los coeficientes."
        )
    elif diagnostics["ratio_max"] > 1.3 or diagnostics["ratio_min"] < 0.75:
        st.info(
            "La expansión z seleccionada muestra desviaciones moderadas respecto al dipolo. "
            "Interprétalas como una prueba de sensibilidad funcional, no como un ajuste experimental definitivo."
        )

    st.markdown("### Comparación de factores de forma")
    fig_z = plot_z_expansion_comparison(q2, coeffs, t0=t0, t_cut=t_cut, show_selected_q2=q2_probe)
    st.pyplot(fig_z, use_container_width=True)
    render_figure_download_buttons(fig_z, "axial_z_factores_forma")

    st.markdown("### Cocientes respecto al dipolo estándar")
    fig_z_ratio = plot_z_expansion_ratios(q2, coeffs, t0=t0, t_cut=t_cut, show_selected_q2=q2_probe)
    st.pyplot(fig_z_ratio, use_container_width=True)
    render_figure_download_buttons(fig_z_ratio, "axial_z_cocientes")

    st.markdown("### Lectura numérica en el punto seleccionado")
    df = make_z_expansion_table(q2_probe, coeffs, t0=t0, t_cut=t_cut)
    st.dataframe(
        df.style.format({"Dipolar estándar": "{:.6f}", "Expansión z": "{:.6f}"}),
        use_container_width=True,
    )

    coeff_df = pd.DataFrame(
        {
            "k": np.arange(1, kmax + 1),
            "c_k": coeffs,
            "c_k ajustado al dipolo": default_coeffs,
        }
    )
    st.dataframe(
        coeff_df.style.format({"c_k": "{:.6f}", "c_k ajustado al dipolo": "{:.6f}"}),
        use_container_width=True,
        hide_index=True,
    )

    diag_df = pd.DataFrame(
        {
            "Diagnóstico": [
                "mínimo de |G_A^z|/|G_A^D|",
                "máximo de |G_A^z|/|G_A^D|",
                "cociente en el Q² máximo visible",
                "máximo |c_k|",
            ],
            "Valor": [
                diagnostics["ratio_min"],
                diagnostics["ratio_max"],
                diagnostics["ratio_at_q2max"],
                diagnostics["max_abs_coeff"],
            ],
        }
    )
    st.dataframe(diag_df.style.format({"Valor": "{:.6f}"}), use_container_width=True, hide_index=True)

    ga_d = float(ga_dipole(q2_probe, M_A_STD))
    ga_z = float(ga_z_expansion(q2_probe, coeffs, t0=t0, t_cut=t_cut))
    z_probe = float(z_conformal(q2_probe, t0=t0, t_cut=t_cut))
    r2_a = float(axial_radius_from_z_coeffs(coeffs, t0=t0, t_cut=t_cut))

    st.markdown(
        rf"En $|Q^2|={q2_probe:.4g}\,\mathrm{{GeV}}^2$, se tiene "
        rf"$z={z_probe:.5f}$ y $|G_A^z|/|G_A^D|={abs(ga_z)/abs(ga_d):.4f}$. "
        rf"El radio axial cuadrático asociado a la pendiente inicial es "
        rf"$\langle r_A^2\rangle={r2_a:.4f}\,\mathrm{{GeV}}^{{-2}}$."
    )

    with st.expander("Lectura física de esta pestaña"):
        st.markdown(
            r"La expansión $z$ permite modificar la forma de $G_A(Q^2)$ sin imponer desde el principio una caída dipolar. "
            r"Esto reduce el sesgo de modelo, pero introduce más parámetros. Por ese motivo, los coeficientes deben fijarse con datos "
            r"experimentales, cálculos de lattice QCD o restricciones externas. En esta app, los coeficientes ajustados al dipolo sirven "
            r"como punto de partida estable para explorar desviaciones controladas. Si los cocientes crecen demasiado a alto $Q^2$, "
            r"la lectura correcta no es que se haya descubierto una gran corrección física, sino que la serie truncada necesita "
            r"restricciones adicionales o un rango de validez más limitado."
        )


def render_axial_form_factors_page() -> None:
    inject_css()

    st.title("Factores de forma axial y pseudoescalar del nucleón")

    tabs = st.tabs(
        [
            "Formalismo",
            "Simulador interactivo",
            "Comparación dipolo-monopolo",
            "Modelo de dos componentes",
            "BBBA2007",
            "Expansión z",
        ]
    )

    with tabs[0]:
        render_formalism_tab()
    with tabs[1]:
        render_simulator_tab()
    with tabs[2]:
        render_dipole_monopole_comparison_tab()
    with tabs[3]:
        render_two_component_model_tab()
    with tabs[4]:
        render_bbba07_tab()
    with tabs[5]:
        render_z_expansion_tab()

def main() -> None:
    st.set_page_config(page_title="Factores de forma axiales del nucleón", layout="wide")
    render_axial_form_factors_page()


if __name__ == "__main__":
    main()
