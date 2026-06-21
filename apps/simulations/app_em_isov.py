from __future__ import annotations

from io import BytesIO

import inspect
import sys
from pathlib import Path  # construimos rutas de archivos de forma robusta
from typing import Any

import matplotlib.pyplot as plt  #para hacer las gráficas
import numpy as np   #para cálculos numéricos, arrays, mallas de Q2, operaciones vectorizadas...
import pandas as pd  #lee y maneja los CSV de datos experimentales. Usado luego para leer los datos de Raúl
import streamlit as st  #con esto podemos hacer la interfaz de la web interactiva. Es importante

PROJECT_ROOT = Path(__file__).resolve().parents[2] #Localiza la raíz del proyecto. parents[0]---apps/simulations, parents [1]---apps y parents[2]----raíz del proyecto en github
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Factores de forma EM e isovectoriales", layout="wide") #Título de la página y diseño de ancho

st.markdown(
    """
    <style>
    .katex-display {
        overflow-x: auto;
        overflow-y: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)     #con esto de aquí conseguimos que las ecuaciones largas de KaTeX tengan desplazamiento horizontal y no se salgan de la pantalla. Esto lo tuve que añadir porque se me cortaban las ecuaciones
#aqui definimos algunas constantes físicas
M_N = 0.93891897 #M es la masa común del nucleón (media aritmética entre la masa del proton y neutron)
MU_P = 2.793 #momento magnetico anomalo del proton 
MU_N = -1.913# del neutron
MU_V = MU_P - MU_N #momento magnetico isovectorial 
MV_STD = 0.843 #M_V masa vectorial estandar (de la parametrizacion Galster)
LAMBDA_N_STD = 5.6  #lambda estandar. (ver mi TFG)

EXPECTED_COLUMNS = {"panel", "Q2", "y"} #Cuando la app lea un CSV de puntos experimentales va a exigir al menos estas tres columnas. Q2 indica el valor de |q2|, y indica el valor experimental normalizado
#panel indica si el punto pertenece a alguno de estos Sachs:
PANEL_ORDER = ["GEp", "GMp", "GEn", "GMn"]  #Esto permite que todos los puntos experimentales se clasifiquen automáticamente en uno de los cuatro paneles de EM
PANEL_LABELS = {
    "GEp": r"$G_E^p/G_D^{\rm ref}$",
    "GMp": r"$G_M^p/(\mu_p G_D^{\rm ref})$",
    "GEn": r"$G_E^n/G_D^{\rm ref}$",
    "GMn": r"$G_M^n/(\mu_n G_D^{\rm ref})$",
}#esto define las etiquetas de los ejes y. La app no va a representar directamente los factores de forma absolutos, sino cocientes normalizados respecto al dipolo
MODEL_COLORS = {"Galster": "tab:blue", "GKeX": "tab:orange"}#nuestros colores para la gráfica.

#Aqui definimos las funciones basicas de la cinematica/dipolo, por ejemplo tau, entre otras
def tau_qe(Q2: np.ndarray | float, M_N_val: float = M_N) -> np.ndarray:
    q2 = np.asarray(Q2, dtype=float)#esto convierte el input en array de NumPy. Con esto se consigue que la función sirva tanto para un número como para un vector de muchos puntos
    return q2 / (4.0 * M_N_val**2)

#definimos el factor de escala dipolar de Galster
def dipole_gd(Q2: np.ndarray | float, M_V: float) -> np.ndarray:
    q2 = np.asarray(Q2, dtype=float)
    return 1.0 / (1.0 + q2 / M_V**2) ** 2

#definimos el parametro adimensional labda 
def lambda_d_from_mv(M_V: float, M_N_val: float = M_N) -> float:
    return 4.0 * M_N_val**2 / M_V**2

#modelo galster para los factores de forma de sachs. Expresiones (4.15) y (4.16) de mi TFG
def galster_sachs(
    Q2: np.ndarray | float,
    M_V: float = MV_STD,
    lambda_n: float = LAMBDA_N_STD,
    M_N_val: float = M_N,
    mu_p: float = MU_P,
    mu_n: float = MU_N,
) -> dict[str, np.ndarray]:
    q2 = np.asarray(Q2, dtype=float)
    tau = tau_qe(q2, M_N_val)
    gd = dipole_gd(q2, M_V)
    xi_n = 1.0 / (1.0 + lambda_n * tau)

    gep = gd # es decir, en Galster el factor eléctrico del protón es puramente dipolar
    gen = -mu_n * tau * gd * xi_n
    gmp = mu_p * gd
    gmn = mu_n * gd

    return {"GEp": gep, "GEn": gen, "GMp": gmp, "GMn": gmn}

#con esta función conseguimos leer el script para la parametrización GKeX
def _coerce_gkex_output(res: Any) -> dict[str, np.ndarray]:
    if isinstance(res, dict):
        aliases = {
            "GEp": ["GEp", "G_Ep", "gep", "GE_p"],
            "GEn": ["GEn", "G_En", "gen", "GE_n"],
            "GMp": ["GMp", "G_Mp", "gmp", "GM_p"],
            "GMn": ["GMn", "G_Mn", "gmn", "GM_n"],
        }#con esto aceptamos varios nombres posibles para la misma cantidad. Si el script GKeX cambia ligeramente los nombres la app no se rompe
        out: dict[str, np.ndarray] = {} #crea un diccionario vacío para luego:
        for target, keys in aliases.items():
            for key in keys:
                if key in res:
                    out[target] = np.asarray(res[key], dtype=float)
                    break  #para luego buscar cada facto de forma con sus posibles nombres y guardarlo por el nombre estándar o el que queremos, que viene dado por " " arriba en verde
        if len(out) == 4:
            return out #devuelve el diccionario si ha encontrado los cuatro

    if isinstance(res, (list, tuple)) and len(res) == 4: #el código asume que está en el orden:
        return {
            "GEp": np.asarray(res[0], dtype=float),
            "GMp": np.asarray(res[1], dtype=float),
            "GEn": np.asarray(res[2], dtype=float),
            "GMn": np.asarray(res[3], dtype=float),
        }

    arr = np.asarray(res, dtype=float) #si ni era diccionario ni lista, intentamos convertirlo a array
    if arr.ndim == 2 and 4 in arr.shape:#comprueba si es una matriz 2D con una dimension de tamaño 4
        if arr.shape[0] == 4:
            return {"GEp": arr[0], "GMp": arr[1], "GEn": arr[2], "GMn": arr[3]}
        if arr.shape[1] == 4:
            return {"GEp": arr[:, 0], "GMp": arr[:, 1], "GEn": arr[:, 2], "GMn": arr[:, 3]}

    raise ValueError("No se pudo interpretar la salida de sachs_gkex.") #si no reconoce el formato lanza un error 


def gkex_sachs(Q2: np.ndarray | float) -> dict[str, np.ndarray] | None:
    try:  #intenta importar la funcion sachs_gkex. Si no existe el archivo devuelve none
        from src.form_factors_gkex import sachs_gkex  
    except Exception:
        return None

    q2 = np.asarray(Q2, dtype=float) #convierte Q2 a array
    sig = inspect.signature(sachs_gkex)
    params = sig.parameters
#con esta funcion se prepara los argumentos correctos para sachs_gkex
    def _build_kwargs(q2_value: Any) -> dict[str, Any]:
        if "Q2" in params:
            return {"Q2": q2_value}#si espera Q2 devuelve Q2
        if "q2" in params:
            return {"q2": q2_value}
        if "Q2_GeV2" in params:
            return {"Q2_GeV2": q2_value}
        return {} #si la funcion no espera nada con estos nombres devuelve el diccionario vacío

    try:
        return _coerce_gkex_output(sachs_gkex(**_build_kwargs(q2 if q2.ndim == 0 else q2)))
    except Exception:
        pass #primero intenta evaluar GKeX vectorialmente, pasando todo el array Q2 de golpe. Si funciona devuelve el resultado y si falla se pasa al método punto a punto

    q2_flat = np.atleast_1d(q2).astype(float)
    values = []
    for q2i in q2_flat:
        values.append(_coerce_gkex_output(sachs_gkex(**_build_kwargs(float(q2i))))) #Aqui se evalua la funcion GKeX punto a punto

    stacked = {
        key: np.array([item[key] for item in values], dtype=float).reshape(q2_flat.shape)
        for key in ["GEp", "GMp", "GEn", "GMn"] #con esto se reconstruye arrays para cada factor de forma
    }
    if np.asarray(Q2).ndim == 0:
        return {k: v[0] for k, v in stacked.items()}
    return stacked #lógicamente, si la entrada era un numero, devuelve un numero. Si era un array, devuelve un array

#Con esto de aquí buscamos archivos CSV candidatos para cargar los datos experimentales 
def _load_points_from_candidates(candidates: list[Path]) -> pd.DataFrame | None:
    seen: set[str] = set() #guarda rutas de archivos ya probadas para no repetir
    for candidate in candidates: #recorre posibles archivos
        if not candidate.exists():
            continue #si el archivo no existe, no pasa nada, pasamos al siguiente
        resolved = str(candidate.resolve())
        if resolved in seen:
            continue
        seen.add(resolved) #con esto evitamos leer dos veces el mismo archivo
        try:
            df = pd.read_csv(candidate)
        except Exception:
            continue #con esto se lee el CSV. Si falla, no pasa nada se sigue con otro posible candidato
        if not EXPECTED_COLUMNS.issubset(set(df.columns)):
            continue #con esto comprobamos que el archuvo contenga las tres columnas que definimos al principio: panel, Q2, y
        df = df.copy()
        if "yerr" not in df.columns:
            df["yerr"] = np.nan
        df["panel"] = df["panel"].astype(str)
        df = df[df["panel"].isin(PANEL_ORDER)].sort_values(["panel", "Q2"])
        return df #devuelve el primer CSV valido encontrado y si no lo encuentra devuelve none
    return None

#En resumen: No cargo los datos de forma ciega. La app busca varios CSV candidatos, comprueba que existen, evita duplicados, intenta leerlos con Pandas, 
#valida que tengan las columnas mínimas y solo conserva los puntos cuyo panel corresponde a uno de los cuatro factores de Sachs que quiero representar.

#Con esto de aquí cargamos los puntos experimentales, una vez ya encontrado el archivo CSV deseado
def load_points_dataframe() -> pd.DataFrame | None:
    candidates = [
        PROJECT_ROOT / "refs/em_form_factors/ff_points_green.csv",
        PROJECT_ROOT / "refs/em_form_factors/ffem_points_app_visible.csv",
        PROJECT_ROOT / "refs/em_form_factors/data_points.csv",
        PROJECT_ROOT / "data/processed/em_form_factors/ff_points_green.csv",
        PROJECT_ROOT / "data/processed/em_form_factors/ffem_points_app_visible.csv",
        Path("refs/em_form_factors/ff_points_green.csv"),
        Path("refs/em_form_factors/ffem_points_app_visible.csv"),
        Path("refs/em_form_factors/data_points.csv"),
        Path("data/processed/em_form_factors/ff_points_green.csv"),
        Path("data/processed/em_form_factors/ffem_points_app_visible.csv"),
    ] #lista de posibles rutas donde podría estar el CSV, (donde yo lo guardo siempre)
    return _load_points_from_candidates(candidates) #llama a la funcion auxiliar 

#Cargamos los puntos detalladamente
def load_points_dataframe_detailed() -> pd.DataFrame | None:
    candidates = [
        PROJECT_ROOT / "refs/em_form_factors/ff_points_green_detailed.csv",
        PROJECT_ROOT / "refs/em_form_factors/ffem_points_detailed_visible.csv",
        PROJECT_ROOT / "data/processed/em_form_factors/ff_points_green_detailed.csv",
        PROJECT_ROOT / "data/processed/em_form_factors/ffem_points_detailed_visible.csv",
        Path("refs/em_form_factors/ff_points_green_detailed.csv"),
        Path("refs/em_form_factors/ffem_points_detailed_visible.csv"),
        Path("data/processed/em_form_factors/ff_points_green_detailed.csv"),
        Path("data/processed/em_form_factors/ffem_points_detailed_visible.csv"),
    ]
    df = _load_points_from_candidates(candidates)
    if df is None:
        return None
    if "dataset_label" not in df.columns:
        df["dataset_label"] = "dataset"
    df["dataset_label"] = df["dataset_label"].astype(str) #asegura que la etiqueta sea texto y devuelve el dataframe
    return df

#filtrado de datasets de g_m^n
def filter_points_dataframe(points_df: pd.DataFrame | None, selected_gmn_datasets: list[str] | None) -> pd.DataFrame | None:
    if points_df is None:
        return None
    if selected_gmn_datasets is None or "dataset_label" not in points_df.columns:
        return points_df
    mask_gmn = points_df["panel"].eq("GMn") #identifica los puntos del panel de gmn
    keep = (~mask_gmn) | points_df["dataset_label"].isin(selected_gmn_datasets)
    return points_df.loc[keep].copy() #devuelve el dataset filtrado

#cocientes normalizados e isovectores
def safe_divide(num: np.ndarray, den: np.ndarray, eps: float = 1e-15) -> np.ndarray: #divide evitando problemas cuando el denominador es cero o casi cero
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float) #convertimos numerador y denominador en arrays
    out = np.full_like(num, np.nan, dtype=float) #crea una salida entera llena de NaN
    mask = np.abs(den) > eps #solo divide donde el denominador no sea casi 0
    out[mask] = num[mask] / den[mask] #hace entonces ahora la division segura
    return out

#construimos los cocientes que vemos en los ejes de la grafica
def ratios_from_sachs(sachs: dict[str, np.ndarray], gd_ref: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "GEp": safe_divide(sachs["GEp"], gd_ref), #dividimos gep/gdref. Que por cierto gdref es gd pero con la masa vector de referencia de 0.843 simplemente
        "GMp": safe_divide(sachs["GMp"], MU_P * gd_ref),
        "GEn": safe_divide(sachs["GEn"], gd_ref),
        "GMn": safe_divide(sachs["GMn"], MU_N * gd_ref),
    } #estos cocientes son los ejes ordenados de las cuatro graficas que queremos representar, simplemente

#aqui construimos las combinaciones isovectoriales ecuacion (4.6) de mi TFG. Importante, con el factor 1/2
def isovector_from_sachs(sachs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "GEV": np.asarray(0.5*sachs["GEp"], dtype=float) - np.asarray(0.5*sachs["GEn"], dtype=float), 
        "GMV": np.asarray(0.5*sachs["GMp"], dtype=float) - np.asarray(0.5*sachs["GMn"], dtype=float),
    }

#y ahora construimos los cocientes isovectoriales
def isovector_ratios_from_sachs(sachs: dict[str, np.ndarray], gd_ref: np.ndarray) -> dict[str, np.ndarray]:
    iso = isovector_from_sachs(sachs)
    return {
        "GEV": safe_divide(iso["GEV"], 0.5*gd_ref),
        "GMV": safe_divide(iso["GMV"], 0.5*MU_V * gd_ref),
    }

#comparamos la curva teórica con los puntos experimentales. Aquí se realiza todo el tratamiento estadístico con cálculos de errores e incertidumbres
def compare_model_to_data(
    q2_grid: np.ndarray,
    model_ratios: dict[str, np.ndarray] | None,
    points_df: pd.DataFrame | None,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, float | str | int]] = [] #guardará métricas agregadas por panel
    residual_rows: list[dict[str, float | str]] = []  #guarda punto por punto el residuo modelo-dato
    if model_ratios is None or points_df is None:
        return pd.DataFrame(metrics), pd.DataFrame(residual_rows)

    for panel in PANEL_ORDER: #recorre los cuatro paneles
        sub = points_df[points_df["panel"] == panel].copy()
        if sub.empty:
            continue #selecciona solo los datos de eese panel
        sub = sub.sort_values("Q2")
        q = sub["Q2"].to_numpy(dtype=float)
        y = sub["y"].to_numpy(dtype=float)
        yerr = sub["yerr"].to_numpy(dtype=float) #en estas lineas se extrae Q2, valores experimentales y sus incertidumbres
        pred = np.interp(q, q2_grid, model_ratios[panel]) #interpola la curva del modelo en los mismos Q2 de los datos 
        resid = pred - y #calcula el residuo como y_i del modelo - y_i de los datos
        valid_err = np.isfinite(yerr) & (yerr > 0.0)#identificamos los puntos con incertidumbres válidas
        pull = np.full_like(resid, np.nan, dtype=float) #pull= (y_imodelo-y_idato)/sigma_i
        pull[valid_err] = resid[valid_err] / yerr[valid_err]
        for qv, yv, pv, rv, ev, pl in zip(q, y, pred, resid, yerr, pull): #guardamos punto por punto el modelo, panel, Q2, dato,prediccion, residuo,error y pull
            residual_rows.append(
                {
                    "model": model_name,
                    "panel": panel,
                    "Q2": float(qv),
                    "data": float(yv),
                    "pred": float(pv),
                    "resid": float(rv),
                    "yerr": float(ev) if np.isfinite(ev) else np.nan,
                    "pull": float(pl) if np.isfinite(pl) else np.nan,
                }
            )
        n = int(len(sub))
        rmse = float(np.sqrt(np.mean(resid**2))) if n > 0 else np.nan
        mean_abs = float(np.mean(np.abs(resid))) if n > 0 else np.nan #aquí calculamos la raiz del error cuadratico medio
        n_w = int(np.count_nonzero(valid_err))
        chi2 = float(np.sum((resid[valid_err] / yerr[valid_err]) ** 2)) if n_w > 0 else np.nan
        chi2_ndf = float(chi2 / n_w) if n_w > 0 else np.nan #y aqui la chi cuadrado
        mean_pull = float(np.nanmean(pull)) if n_w > 0 else np.nan
        rms_pull = float(np.sqrt(np.nanmean(pull**2))) if n_w > 0 else np.nan # y aqui la media y el root mean square
        metrics.append(
            {
                "Modelo": model_name,
                "Panel": panel,
                "N": n,
                "N con error": n_w,
                "RMSE": rmse,
                "|resid| medio": mean_abs,
                "chi2": chi2,
                "chi2/N": chi2_ndf,
                "pull medio": mean_pull,
                "RMS pull": rms_pull,
            } #aqui guardamos la tabla resumen para el modelo y panel
        )

    return pd.DataFrame(metrics), pd.DataFrame(residual_rows) # y devuelve los dataframes: métricas y residuos

#conclusion automatica
def make_conclusion_text(metrics_df: pd.DataFrame | None) -> str: #genera un texto automatico a partir de las métricas
    if metrics_df is None or metrics_df.empty:
        return "No hay métricas suficientes para generar una conclusión automática." #si no hay metricas me avisa

    score_col = "chi2/N" if metrics_df["chi2/N"].notna().any() else "RMSE" #si existe chi2/N usa esto como criterio y si no RMSE
    summary = (                                           #aqui se agrupa por modelo y calcula puntuaciones medias
        metrics_df.groupby("Modelo", as_index=False)
        .agg(
            media_score=(score_col, "mean"),
            media_rmse=("RMSE", "mean"),
            panels=("Panel", "count"),
        )
        .sort_values("media_score")
    )
    best = summary.iloc[0]["Modelo"]
    worst = summary.iloc[-1]["Modelo"] #por consiguiente, el mejor modelo es el de menor puntuacion media

    panel_lines = []
    for panel in PANEL_ORDER: # y aqui mira panel a panel qué modelo gana
        sub = metrics_df[metrics_df["Panel"] == panel].copy()
        if len(sub) < 2:
            continue
        sub = sub.sort_values(score_col)
        panel_lines.append(f"- **{panel}**: mejor acuerdo para **{sub.iloc[0]['Modelo']}**.")

    text = [
        f"Según el criterio promedio basado en **{score_col}**, el modelo que mejor sigue estos datos es **{best}**.",
        f"El contraste frente a **{worst}** ayuda a visualizar en qué paneles la fenomenología más rica realmente aporta una mejora.",
    ]
    text.extend(panel_lines)
    return "\n".join(text) #para finalmente construir el texto final, que es lo que se verá en la aplicación

#Con esto construimos los controles laterales o desplegable lateral para mi aplicación web en Streamlit
st.sidebar.header("Controles")
model_choice = st.sidebar.radio("Curvas a mostrar", ["Galster", "GKeX", "Ambas"], index=2)
q2_max = st.sidebar.slider(r"$|Q^2|_{\max}$ (GeV$^2$)", 0.10, 10.00, 10.00, 0.10)
n_points = st.sidebar.slider("Número de puntos de muestreo", 150, 1200, 500, 50)

st.sidebar.markdown("---")
st.sidebar.subheader("Parámetros de Galster")
M_V = st.sidebar.slider(r"$M_V$ (GeV)", 0.700, 1.000, MV_STD, 0.005)
lambda_n = st.sidebar.slider(r"$\lambda_n$", 3.0, 8.0, LAMBDA_N_STD, 0.1) #con esto podemos variar lambda
lambda_d = lambda_d_from_mv(M_V)
st.sidebar.latex(rf"\lambda_D^V = \frac{{4M_N^2}}{{M_V^2}} = {lambda_d:.3f}")
st.sidebar.markdown(r"$M_V$ y $\lambda_D^V$ quedan ligados automáticamente por definición.")

st.sidebar.markdown("---")
st.sidebar.subheader("Normalización del dipolo de referencia (pestaña EM)")
reference_mode = st.sidebar.radio(
    "Denominador en los cocientes EM",
    [
        "Normalización publicada (M_V^ref = 0.843 GeV)",
        "Normalización dinámica (M_V^ref = M_V actual)",
    ],
    index=0,
)

show_data = st.sidebar.checkbox("Mostrar puntos experimentales", value=True)
show_logx = st.sidebar.checkbox("Eje x logarítmico", value=True)
show_summary_table = st.sidebar.checkbox("Mostrar tabla de valores destacados", value=True)

detailed_points_raw = load_points_dataframe_detailed() if show_data else None
gmn_dataset_options: list[str] = []
selected_gmn_datasets: list[str] | None = None
if detailed_points_raw is not None:
    gmn_dataset_options = sorted(detailed_points_raw.loc[detailed_points_raw["panel"] == "GMn", "dataset_label"].dropna().unique().tolist())
    if gmn_dataset_options:
        st.sidebar.markdown("---")
        with st.sidebar.expander("Filtrar datasets de GMn", expanded=False):
            st.caption("Sirve para identificar qué subconjuntos generan la banda superior en el panel magnético del neutrón.")
            selected_gmn_datasets = st.multiselect(
                "Datasets visibles en GMn",
                gmn_dataset_options,
                default=gmn_dataset_options,
            )
#Título de mi aplicacion
st.title("Factores de forma del nucleón: EM e isovectores")
st.markdown(
    "Presentamos la aplicación interacctiva encargada de mostrar dinámicamente el comportamiento de los factores de forma EM e isovectores del nucleón. En la primera pestaña se comparan los factores de forma electromagnéticos de "
    "Sachs del protón y del neutrón. En la segunda se construyen las combinaciones isovectoriales que entran en la corriente "
    "débil vectorial mediante CVC.Los datos experimentales pertenecen a: ver referencia [10] en mi TFG."
)
#creamos la malla Q2 y calculamos Galster
Q2 = np.geomspace(1.0e-3, q2_max, n_points) if show_logx else np.linspace(1.0e-3, q2_max, n_points)
M_V_ref = MV_STD if reference_mode.startswith("Normalización publicada") else M_V
gd_ref = dipole_gd(Q2, M_V_ref)

# En la pestaña isovectorial usamos el mismo dipolo de Galster en el denominador para
##reproducir la convención gráfica de la masterthesis: con ello, G_M^V/(mu_V G_D) queda
# exactamente plano en 1 para Galster.
gd_ref_iso = dipole_gd(Q2, M_V)

galster_raw = galster_sachs(Q2, M_V=M_V, lambda_n=lambda_n) #calcula los cuatro factores de Sachs de Galster
galster_ratio = ratios_from_sachs(galster_raw, gd_ref) #aquí los respectivos cocientes EM
galster_isovector_ratio = isovector_ratios_from_sachs(galster_raw, gd_ref_iso) #y aquí los isovectoriales
#Aquí calculamos GKeX
#Se inician las variables:
gkex_ratio = None
gkex_isovector_ratio = None
gkex_error = None
if model_choice in {"GKeX", "Ambas"}: #solo calcula GKeX si el usuario de la web quiere mostrarlo también
    try:
        gkex_raw = gkex_sachs(Q2) #aqui intenta calcular los factors de sachs gkex
        if gkex_raw is not None:
            gkex_ratio = ratios_from_sachs(gkex_raw, gd_ref)
            gkex_isovector_ratio = isovector_ratios_from_sachs(gkex_raw, gd_ref_iso) #y si están disponibles, calcula tambien sus cocientes
        else:
            gkex_error = (
                "No se encontró el módulo src/form_factors_gkex.py en el proyecto actual. "
                "La parte Galster sigue funcionando y la comparación con GKeX se activará en cuanto ese archivo esté disponible." #si no los encuentra pues salta el error
            )
    except Exception as exc:
        gkex_error = str(exc)
#cargamos los datos y calculamos metrica
points_df_simple = load_points_dataframe() if show_data else None
points_df = filter_points_dataframe(detailed_points_raw if detailed_points_raw is not None else points_df_simple, selected_gmn_datasets)
metrics_frames: list[pd.DataFrame] = []
residual_frames: list[pd.DataFrame] = []

gal_metrics, gal_resid = compare_model_to_data(Q2, galster_ratio, points_df, "Galster") #compara galster con los datos
if not gal_metrics.empty:
    metrics_frames.append(gal_metrics)
if not gal_resid.empty:
    residual_frames.append(gal_resid)

if gkex_ratio is not None:
    gk_metrics, gk_resid = compare_model_to_data(Q2, gkex_ratio, points_df, "GKeX") #hace lo mismo para GKeX solo si lo pudo calcular antes
    if not gk_metrics.empty:
        metrics_frames.append(gk_metrics)
    if not gk_resid.empty:
        residual_frames.append(gk_resid)

metrics_df = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
residuals_df = pd.concat(residual_frames, ignore_index=True) if residual_frames else pd.DataFrame()

#funcion auxiliar para dibujar modelos
def _maybe_plot_model(ax: Any, model_name: str, panel: str) -> None:
    if model_name == "Galster" and model_choice in {"Galster", "Ambas"}:
        ax.plot(Q2, galster_ratio[panel], label="Galster", linewidth=2.0, color=MODEL_COLORS["Galster"]) #dibuja galster si está seleccionado y disponible
    if model_name == "GKeX" and model_choice in {"GKeX", "Ambas"} and gkex_ratio is not None:
        ax.plot(Q2, gkex_ratio[panel], label="GKeX", linewidth=2.0, color=MODEL_COLORS["GKeX"]) #dibuja GKeX si esta selescionadao y disponible


# Renderizado de tablas sin st.dataframe para evitar dependencia de pyarrow en local.
def render_records_table(records) -> None:
    if not records:
        st.info("No hay valores para mostrar.")
        return
    columns = list(records[0].keys())

    def _fmt(value) -> str:
        if isinstance(value, float):
            return f"{value:.5g}"
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = "\n".join(
        "| " + " | ".join(_fmt(row.get(col, "")) for col in columns) + " |"
        for row in records
    )
    st.markdown(header + "\n" + separator + "\n" + body)


# Exportación de figuras para la memoria/TFG.
def figure_to_bytes(fig: Any, file_format: str) -> bytes:
    """Convierte una figura de Matplotlib a bytes para st.download_button."""
    buffer = BytesIO()
    file_format = file_format.lower().strip()
    if file_format == "pdf":
        fig.savefig(buffer, format="pdf", bbox_inches="tight")
    elif file_format == "png":
        fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    else:
        raise ValueError("Formato no soportado. Usa 'pdf' o 'png'.")
    buffer.seek(0)
    return buffer.getvalue()


def render_figure_download_buttons(fig: Any, base_filename: str, key_prefix: str) -> None:
    """Muestra botones para descargar la figura actual en PDF vectorial y PNG."""
    st.caption(
        "Los archivos se generan con la configuración actual de la app: modelo, rango de |Q²|, "
        "normalización, puntos experimentales y escala del eje x."
    )
    col_pdf, col_png = st.columns(2)
    with col_pdf:
        st.download_button(
            "Descargar PDF vectorial",
            data=figure_to_bytes(fig, "pdf"),
            file_name=f"{base_filename}.pdf",
            mime="application/pdf",
            key=f"{key_prefix}_pdf",
        )
    with col_png:
        st.download_button(
            "Descargar PNG 300 dpi",
            data=figure_to_bytes(fig, "png"),
            file_name=f"{base_filename}.png",
            mime="image/png",
            key=f"{key_prefix}_png",
        )

#en esta linea creo las pestañas
tab_em, tab_iso = st.tabs(["Factores de forma EM", "Factores de forma isovectoriales"])

with tab_em:
    with st.expander("Marco teórico y lectura física", expanded=True):
        st.markdown("**El modelo dipolar de Galster** es una parametrización sencilla y transparente para explorar sensibilidad paramétrica a bajo y moderado momento transferido:")
        st.latex(r"\tau = \frac{|Q^2|}{4M_N^2}")
        st.latex(r"G_D^V(Q^2) = \frac{1}{\left(1+\frac{|Q^2|}{M_V^2}\right)^2} = \frac{1}{(1+\lambda_D^V\tau)^2}")
        st.latex(r"G_E^p = G_D^V, \qquad G_E^n = -\mu_n\,\tau\,G_D^V\,\xi_n, \qquad \xi_n = \frac{1}{1+\lambda_n\tau}")
        st.latex(r"G_M^p = \mu_p G_D^V, \qquad G_M^n = \mu_n G_D^V")
        st.markdown("**La parametrización GKeX** se utiliza como referencia, manteniendo Galster como modelo editable.")
        st.markdown(r"""
**Qué se varía aquí y por qué:**

- **$M_V$** fija la escala del dipolo: al aumentar $M_V$, la caída con $|Q^2|$ se hace más lenta.
- **$\lambda_n$** controla sobre todo la forma de $G_E^n$.
- **$|Q^2|_{\max}$** puede llevarse hasta $10\,\mathrm{GeV}^2$ y ver cuándo GKeX empieza a separarse claramente del modelo dipolar de Galster.
""")
        st.markdown(r"""
**Advertencia de normalización:**

Los cocientes pueden mostrarse respecto al dipolo publicado con $M_V^{\rm ref}=0.843\,\mathrm{GeV}$ o respecto a un dipolo dinámico actualizado con el valor deseado y establecido en la barra lateral de la izquierda de la aplicación.


""")

    fig_em, axes_em = plt.subplots(2, 2, figsize=(11.0, 8.2), constrained_layout=True)
    axes_map_em = {"GEp": axes_em[0, 0], "GMp": axes_em[0, 1], "GEn": axes_em[1, 0], "GMn": axes_em[1, 1]}
    y_limits_em = {"GEp": (0.0, 1.35), "GMp": (0.60, 1.15), "GEn": (-0.05, 0.80), "GMn": (0.60, 1.15)}

    for panel in PANEL_ORDER:
        ax = axes_map_em[panel]
        _maybe_plot_model(ax, "Galster", panel)
        _maybe_plot_model(ax, "GKeX", panel)
        if points_df is not None:
            sub = points_df[points_df["panel"] == panel]
            if len(sub) > 0:
                if sub["yerr"].notna().any():
                    ax.errorbar(
                        sub["Q2"],
                        sub["y"],
                        yerr=sub["yerr"],
                        fmt=".",
                        markersize=5,
                        color="tab:green",
                        ecolor="tab:green",
                        elinewidth=0.8,
                        capsize=2,
                        linestyle="none",
                        alpha=0.85,
                        label="Datos exp." if panel == "GEp" else None, #si hay incertidumbre ddibuja los puntos con sus barras de error
                    )
                else:
                    ax.plot(
                        sub["Q2"],
                        sub["y"],
                        ".",
                        color="tab:green",
                        markersize=5,
                        alpha=0.85,
                        label="Datos exp." if panel == "GEp" else None, #si no hay incertidumbre dibuja los puntos sin barras
                    )
        ax.set_ylabel(PANEL_LABELS[panel])
        ax.set_xlabel(r"$|Q^2|\;({\rm GeV}^2)$")
        ax.set_ylim(*y_limits_em[panel])
        if show_logx:
            ax.set_xscale("log")
        ax.grid(alpha=0.25)

    handles, labels = axes_em[0, 0].get_legend_handles_labels()
    if handles:
        axes_em[0, 0].legend(frameon=False)
    st.pyplot(fig_em, use_container_width=True)

    with st.expander("Descargar gráfica EM", expanded=False):
        render_figure_download_buttons(
            fig_em,
            "factores_forma_EM",
            "download_em",
        )

    if points_df is not None:
        counts = points_df.groupby("panel").size().reindex(PANEL_ORDER, fill_value=0).to_dict()
        st.caption("Puntos experimentales cargados: " + ", ".join(f"{panel}={counts[panel]}" for panel in PANEL_ORDER) + ".")
    if gkex_error is not None:
        st.error(gkex_error)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Lectura física rápida")
        st.markdown(r"""
- Si aumentas **$M_V$**, el dipolo cae más lentamente y la curva de Galster se endurece.
- Si aumentas **$\lambda_n$**, el factor $\xi_n=(1+\lambda_n\tau)^{-1}$ suprime más deprisa a $G_E^n$.
- La comparación con **GKeX** muestra hasta qué punto un modelo VMD se separa del clásico modelo dipolar simple de Galster.
""")
    with col2:
        st.subheader("Propósito didáctico")
        st.markdown(r"""
Esta visualización ayuda a distinguir tres ideas:
1. qué fija la normalización en $Q^2\to 0$,
2. qué controla la forma funcional de Galster, y
3. por qué GKeX actúa como referencia fenomenológica más completa.
""")

    if show_summary_table:
        st.subheader("Valores destacados")
        q2_marks = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 2.0, 5.0, 10.0])
        q2_marks = q2_marks[q2_marks <= q2_max + 1e-12]
        gal_mark = galster_sachs(q2_marks, M_V=M_V, lambda_n=lambda_n)
        gal_ratio_mark = ratios_from_sachs(gal_mark, dipole_gd(q2_marks, M_V_ref))
        gk_ratio_mark = None
        if gkex_ratio is not None:
            gk = gkex_sachs(q2_marks)
            if gk is not None:
                gk_ratio_mark = ratios_from_sachs(gk, dipole_gd(q2_marks, M_V_ref))
        rows = []
        for i, qv in enumerate(q2_marks):
            row = {"Q2 [GeV^2]": float(qv)}
            for panel in PANEL_ORDER:
                row[f"Galster {panel}"] = float(gal_ratio_mark[panel][i])
            if gk_ratio_mark is not None:
                for panel in PANEL_ORDER:
                    row[f"GKeX {panel}"] = float(gk_ratio_mark[panel][i])
            rows.append(row)
        render_records_table(rows)
#pestaña isovectorial
with tab_iso:
    with st.expander("Marco teórico y lectura física", expanded=True):
        st.markdown(
            r"""
En esta pestaña de la aplicación web se construyen las combinaciones isovectores asociadas al
**sector vectorial de la corriente débil cargada**. Siguiendo la convención usada
en el TFG, se definen como:
"""
        )

        st.latex(
            r"G_E^V(Q^2)=\frac{1}{2}\left[G_E^p(Q^2)-G_E^n(Q^2)\right]"
        )
        st.latex(
            r"G_M^V(Q^2)=\frac{1}{2}\left[G_M^p(Q^2)-G_M^n(Q^2)\right]"
        )
        st.latex(
            r"\mu_V=\mu_p-\mu_n"
        )

        st.markdown(
            r"""
**Convención adoptada aquí**

- En el código se incluye explícitamente el factor \(1/2\) en la definición de
  \(G_E^V\) y \(G_M^V\), de acuerdo con la notación empleada en el TFG.
- Para que la comparación gráfica siga normalizada a la unidad en el límite
  dipolar, se representan los cocientes
  \(G_E^V/(G_D/2)\) y \(G_M^V/[(\mu_V/2)G_D]\), con dicho factor 1/2.
"""
        )

    fig_iso, axes_iso = plt.subplots(
        1, 2, figsize=(11.0, 4.4), constrained_layout=True
    )

    fig_iso, axes_iso = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    iso_labels = {"GEV": r"$G_E^V/(G_D/2)$", "GMV": r"$G_M^V/[(\mu_V/2)G_D]$"}
    iso_limits = {"GEV": (-0.50, 1.10), "GMV": (0.85, 1.10)}

    for panel, ax in zip(["GEV", "GMV"], axes_iso):
        if model_choice in {"Galster", "Ambas"}:
            ax.plot(Q2, galster_isovector_ratio[panel], label="Galster", linewidth=2.0, color=MODEL_COLORS["Galster"])
        if model_choice in {"GKeX", "Ambas"} and gkex_isovector_ratio is not None:
            ax.plot(Q2, gkex_isovector_ratio[panel], label="GKeX", linewidth=2.0, color=MODEL_COLORS["GKeX"])
        ax.set_ylabel(iso_labels[panel])
        ax.set_xlabel(r"$|Q^2|\;({\rm GeV}^2)$")
        ax.set_ylim(*iso_limits[panel])
        if show_logx:
            ax.set_xscale("log")
        ax.grid(alpha=0.25)

    handles, labels = axes_iso[0].get_legend_handles_labels()
    if handles:
        axes_iso[0].legend(frameon=False)
    st.pyplot(fig_iso, use_container_width=True)

    with st.expander("Descargar gráfica isovectorial", expanded=False):
        render_figure_download_buttons(
            fig_iso,
            "factores_forma_isovectoriales",
            "download_iso",
        )

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Lectura física rápida")
        st.markdown(r"""
- En **Galster**, $G_M^V/[(\mu_V/2)G_D]$ queda exactamente plano en 1 porque tanto $G_M^p$ como $G_M^n$ son dipolares puros y el mismo dipolo aparece también en el denominador.
- En **GKeX**, la desviación respecto a 1 mide hasta qué punto el sector magnético deja de comportarse como un dipolo simple.
""")
    with col4:
        st.subheader("Por qué esta pestaña merece la pena:")
        st.markdown(r"""
La corriente débil cargada no utiliza por separado los factores de forma del
protón y del neutrón, sino sus **combinaciones isovectores**. Por eso esta
pestaña actúa como puente entre los factores de forma EM, conocidos a partir de la dispersión electrón--nucleón, y la parte vector de la interacción
neutrino--nucleón.

En otras palabras, aquí se ve cómo las parametrizaciones de
\(G_E^p\), \(G_E^n\), \(G_M^p\) y \(G_M^n\) se traducen en los factores
\(G_E^V\) y \(G_M^V\), que serán los que entren después en \(F_1^V\) y
\(F_2^V\). Esto permite comprobar de forma visual cuánto cambia la estructura
vector débil al pasar de una parametrización sencilla como Galster a una más compleja como GKeX.
"""
        )

    if show_summary_table:
        st.subheader("Valores destacados")
        q2_marks = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 2.0, 5.0, 10.0])
        q2_marks = q2_marks[q2_marks <= q2_max + 1e-12]
        gal_mark = galster_sachs(q2_marks, M_V=M_V, lambda_n=lambda_n)
        gal_iso_mark = isovector_ratios_from_sachs(gal_mark, dipole_gd(q2_marks, M_V))
        gk_iso_mark = None
        if gkex_isovector_ratio is not None:
            gk = gkex_sachs(q2_marks)
            if gk is not None:
                gk_iso_mark = isovector_ratios_from_sachs(gk, dipole_gd(q2_marks, M_V))
        rows_iso = []
        for i, qv in enumerate(q2_marks):
            row = {
                "Q2 [GeV^2]": float(qv),
                "Galster GEV": float(gal_iso_mark["GEV"][i]),
                "Galster GMV": float(gal_iso_mark["GMV"][i]),
            }
            if gk_iso_mark is not None:
                row["GKeX GEV"] = float(gk_iso_mark["GEV"][i])
                row["GKeX GMV"] = float(gk_iso_mark["GMV"][i])
            rows_iso.append(row)
        render_records_table(rows_iso)
