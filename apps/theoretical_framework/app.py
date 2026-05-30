# apps/theoretical_framework_full_single.py
# -*- coding: utf-8 -*-
r"""
Formalismo teórico de la interacción débil para la dispersión elástica ν-N mediada por CC — TFG MCR
Un único archivo Streamlit con toda la teoría, ecuaciones principales y apéndices técnicos.

Uso local:
    streamlit run theoretical_framework_full_single.py

Notas de estilo:
- Todas las ecuaciones se renderizan con st.latex().
- Se usa \not{p} en lugar de \slashed{p} por compatibilidad con KaTeX/Streamlit.
- El contenido está organizado por secciones navegables desde la barra lateral.
"""
from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Formalismo teórico — CC ν-N",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(r"""
<style>
.block-container { max-width: 1220px; padding-top: 1.4rem; padding-bottom: 3rem; }
.katex-display { overflow-x: auto; overflow-y: hidden; padding-bottom: 0.25rem; }
h1 { letter-spacing: -0.02em; }
h2 { margin-top: 1.8rem; border-bottom: 1px solid rgba(49, 151, 149, .25); padding-bottom: .25rem; }
h3 { margin-top: 1.2rem; }
.eq-card { border-left: 4px solid #14b8a6; background: rgba(20,184,166,.055); padding: .55rem .75rem; border-radius: .5rem; margin: .65rem 0 .25rem 0; }
.eq-num { font-weight: 700; color: #0f766e; font-size: .92rem; }
.small-note { color: #64748b; font-size: .88rem; }
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

SECTIONS = [{'key': 'home',
  'title': 'Formalismo teórico de la interacción débil para la dispersión elástica ν-N mediada por CC',
  'subtitle': '',
  'blocks': [{'type': 'p',
              'text': 'Esta página está pensada como una versión extendida y navegable del marco teórico del TFG. La '
                      'idea no es sustituir la memoria, sino ofrecer una herramienta de defensa: si surge una duda '
                      'sobre una ecuación, una contracción, una delta de Dirac, un factor de forma o el tratamiento '
                      'estadístico, aquí aparece la cadena completa de razonamiento.'},
             {'type': 'eq',
              'num': 'Procesos CCQE',
              'latex': '\\nu_\\mu+n\\to \\mu^-+p,\\qquad \\bar\\nu_\\mu+p\\to \\mu^+ + n',
              'note': ''},
             {'type': 'eq',
              'num': 'Idea central',
              'latex': '\\text{estructura axial}\\quad \\Longrightarrow\\quad G_A(Q^2),\\,G_P(Q^2)\\quad '
                       '\\Longrightarrow\\quad \\eta_{\\mu\\nu}\\,W^{\\mu\\nu}\\quad \\Longrightarrow\\quad '
                       '\\frac{d\\sigma}{d\\Omega},\\,\\frac{d\\sigma}{dE_\\mu},\\,\\left\\langle\\frac{d\\sigma}{dQ^2}\\right\\rangle_\\Phi',
              'note': ''},
]},
 {'key': 'intro',
  'title': '1. Introducción y motivación física',
  'subtitle': 'Por qué los neutrinos permiten estudiar la estructura axial del nucleón.',
  'blocks': [{'type': 'p',
              'text': 'Los neutrinos son leptones neutros que no interaccionan fuertemente. Por ello son sondas '
                      'limpias de la estructura débil hadrónica, aunque experimentalmente difíciles de detectar. En '
                      'los experimentos reales no se mide directamente el neutrino incidente, sino los productos de su '
                      'interacción con nucleones o núcleos del detector.'},
             {'type': 'p',
              'text': 'La dispersión electromagnética electrón-nucleón sólo accede a la corriente vector del nucleón. '
                      'En cambio, las interacciones débiles de corriente cargada contienen una parte vector y una '
                      'parte axial. Por eso los procesos neutrino-nucleón son especialmente importantes para estudiar '
                      'el factor de forma axial GA(Q^2).'},
             {'type': 'h3', 'text': '1.1. Neutrinos como sondas débiles'},
             {'type': 'p',
              'text': 'En corriente cargada, el intercambio de un bosón W transforma el neutrino en su leptón cargado '
                      'asociado y produce una transición de isospín en el nucleón. Para el caso muónico libre se '
                      'consideran los canales ν_mu+n→mu^-+p y anti-ν_mu+p→mu^++n.'},
             {'type': 'eq',
              'num': 'canales CC',
              'latex': '\\nu_\\mu+n\\longrightarrow \\mu^-+p,\\qquad \\bar\\nu_\\mu+p\\longrightarrow \\mu^+ + n',
              'note': ''},
             {'type': 'h3', 'text': '1.2. Oscilaciones y reconstrucción de energía'},
             {'type': 'p',
              'text': 'La física de oscilaciones depende de la relación entre autoestados de sabor y autoestados de '
                      'masa. En el TFG se recuerda esta conexión mediante la matriz PMNS.'},
             {'type': 'eq',
              'num': '(1.1)',
              'latex': '\\begin{pmatrix}\\nu_e\\\\ \\nu_\\mu\\\\ '
                       '\\nu_\\tau\\end{pmatrix}=\\begin{pmatrix}U_{e1}&U_{e2}&U_{e3}\\\\ '
                       'U_{\\mu1}&U_{\\mu2}&U_{\\mu3}\\\\ '
                       'U_{\\tau1}&U_{\\tau2}&U_{\\tau3}\\end{pmatrix}\\begin{pmatrix}\\nu_1\\\\ \\nu_2\\\\ '
                       '\\nu_3\\end{pmatrix}',
              'note': ''},
             {'type': 'p',
              'text': 'En una aproximación efectiva de dos sabores, la probabilidad de transición se escribe como:'},
             {'type': 'eq',
              'num': '(1.2)',
              'latex': 'P_{\\nu_\\alpha\\to\\nu_\\beta}=\\sin^2(2\\theta)\\,\\sin^2\\!\\left(1.27\\,\\frac{\\Delta m^2 '
                       'L}{E_\\nu}\\right)',
              'note': ''},
             {'type': 'p',
              'text': 'Esta expresión muestra que la extracción de parámetros de oscilación depende de L/Eν. Como Eν '
                      'no se mide directamente, la reconstrucción de la energía del neutrino depende de los modelos de '
                      'interacción neutrino-nucleón o neutrino-núcleo.'},
             {'type': 'h3', 'text': '1.3. MINERvA y el hidrógeno libre'},
             {'type': 'p',
              'text': 'MINERvA estudia interacciones de neutrinos y antineutrinos en el rango del GeV. En el TFG se '
                      'utiliza especialmente la medida de antineutrinos muónicos sobre hidrógeno libre, porque reduce '
                      'las complicaciones nucleares y permite contrastar más directamente la interacción elemental.'},
             {'type': 'eq',
              'num': 'canal MINERvA H',
              'latex': '\\bar\\nu_\\mu+p\\longrightarrow \\mu^+ + n',
              'note': ''},
             {'type': 'p',
              'text': 'El haz experimental no es monocromático. Por ello la comparación con datos requiere promediar '
                      'la predicción teórica sobre el flujo de antineutrinos.'},
             {'type': 'eq', 'num': 'flujo', 'latex': '\\Phi=\\int dE_{\\bar\\nu}\\,\\phi(E_{\\bar\\nu})', 'note': ''}]},
 {'key': 'kinematics',
  'title': '2.1. Cinemática de la interacción',
  'subtitle': 'Definiciones covariantes, sistema laboratorio y condición elástica.',
  'blocks': [{'type': 'p',
              'text': 'Se estudia primero el proceso inducido por neutrinos muónicos. El canal de antineutrinos se '
                      'obtiene cambiando el canal leptónico y el signo de la contribución antisimétrica '
                      'correspondiente.'},
             {'type': 'eq', 'num': '(2.1)', 'latex': '\\nu_\\mu+n\\to\\mu^-+p', 'note': ''},
             {'type': 'p', 'text': 'Los cuadrimomentos se definen como:'},
             {'type': 'eq',
              'num': '(2.2)',
              'latex': 'k_i^\\mu=(\\epsilon_i,\\mathbf{k}_i),\\qquad p_i^\\mu=(E_i,\\mathbf{p}_i),\\qquad '
                       'k_f^\\mu=(\\epsilon_f,\\mathbf{k}_f),\\qquad p_f^\\mu=(E_f,\\mathbf{p}_f)',
              'note': ''},
             {'type': 'p', 'text': 'En el sistema laboratorio el nucleón inicial está en reposo:'},
             {'type': 'eq', 'num': '(2.3)', 'latex': 'p_i^\\mu=(M,\\mathbf{0})', 'note': ''},
             {'type': 'p',
              'text': 'El cuadrimomento transferido por el bosón W se define como diferencia leptónica o hadrónica:'},
             {'type': 'eq',
              'num': '(2.4)',
              'latex': 'Q^\\mu\\equiv k_i^\\mu-k_f^\\mu=p_f^\\mu-p_i^\\mu=(\\omega,\\mathbf{q})',
              'note': ''},
             {'type': 'eq', 'num': '(2.5)', 'latex': 'Q^2=\\omega^2-|\\mathbf{q}|^2', 'note': ''},
             {'type': 'p',
              'text': 'En dispersión débil cargada sobre nucleón libre, Q^2 es de tipo espacial. En el TFG se usa '
                      'Q^2=-|Q^2|.'},
             {'type': 'eq', 'num': '(2.6)', 'latex': 'k_i^\\mu+p_i^\\mu=k_f^\\mu+p_f^\\mu', 'note': ''},
             {'type': 'p',
              'text': 'De la conservación del momento lineal en el vértice leptónico se obtiene la relación angular:'},
             {'type': 'eq',
              'num': '(2.7)',
              'latex': '|\\mathbf{q}|^2=|\\mathbf{k}_i|^2+|\\mathbf{k}_f|^2-2|\\mathbf{k}_i|\\,|\\mathbf{k}_f|\\cos\\theta_\\mu=\\epsilon_i^2+|\\mathbf{k}_f|^2-2\\epsilon_i|\\mathbf{k}_f|\\cos\\theta_\\mu',
              'note': ''},
             {'type': 'p',
              'text': 'Del vértice hadrónico, usando p_i=(M,0), la condición elástica impone una relación directa '
                      'entre transferencia de energía y cuadrimomento transferido:'},
             {'type': 'eq',
              'num': '(2.8)',
              'latex': '|\\mathbf{q}|^2=\\omega^2+2M\\omega\\qquad\\Longrightarrow\\qquad |Q^2|=2M\\omega',
              'note': ''},
             {'type': 'info',
              'text': 'Interpretación: para un nucleón libre en reposo, fijar |Q²| fija también ω. La delta de Dirac '
                      'que aparece después en la sección eficaz impondrá exactamente esta condición elástica.'}]},
 {'key': 'currents',
  'title': '2.2-2.3. Corrientes leptónica y hadrónica',
  'subtitle': 'Estructura V-A leptónica y vértice hadrónico con factores de forma.',
  'blocks': [{'type': 'h3', 'text': '2.2. Corriente leptónica'},
             {'type': 'eq',
              'num': '(2.9)',
              'latex': 'j^\\mu=\\bar\\psi_l\\gamma^\\mu(1-\\gamma^5)\\psi_{\\nu_l}',
              'note': ''},
             {'type': 'p',
              'text': 'El operador 1-gamma5 proyecta la componente izquierda de los fermiones. En el límite mν→0, '
                      'quiralidad y helicidad coinciden prácticamente.'},
             {'type': 'eq',
              'num': '(2.10)',
              'latex': '\\gamma^\\mu(1-\\gamma^5)=\\gamma^\\mu-\\gamma^\\mu\\gamma^5',
              'note': ''},
             {'type': 'p', 'text': 'Para neutrinos y antineutrinos se escribe de forma compacta:'},
             {'type': 'eq',
              'num': '(2.11)',
              'latex': 'j^\\mu=\\bar\\psi_l\\gamma^\\mu(1\\mp\\gamma^5)\\psi_{\\nu_l}',
              'note': ''},
             {'type': 'p',
              'text': 'En la convención del TFG, el signo superior corresponde a neutrinos y el inferior a '
                      'antineutrinos.'},
             {'type': 'p', 'text': 'Las funciones de onda planas normalizadas en una caja de volumen V se toman como:'},
             {'type': 'eq',
              'num': '(2.12)',
              'latex': '\\psi=\\sqrt{\\frac{m}{EV}}\\,u(p,s)e^{-ip\\cdot X}',
              'note': ''},
             {'type': 'eq',
              'num': '(2.13)',
              'latex': 'j^\\mu=\\frac{\\sqrt{m_{\\nu_\\mu}m_\\mu}}{V\\sqrt{\\epsilon_i\\epsilon_f}}\\,\\bar '
                       'u_l(k_l,s_l)\\gamma^\\mu(1-\\gamma^5)u_{\\nu_l}(k_{\\nu_l},s_{\\nu_l})\\,e^{i(k_l-k_{\\nu_l})\\cdot '
                       'X}',
              'note': ''},
             {'type': 'h3', 'text': '2.3. Corriente hadrónica débil cargada'},
             {'type': 'eq',
              'num': '(2.14)',
              'latex': 'J^\\mu=\\bar\\psi_p(p_f,s_f)\\,\\widetilde\\Gamma^\\mu\\,\\psi_n(p_i,s_i)',
              'note': ''},
             {'type': 'eq',
              'num': '(2.15)',
              'latex': 'J^\\mu=\\frac{M}{V\\sqrt{E_iE_f}}\\,\\bar u_p(p_f,s_f)\\widetilde\\Gamma^\\mu '
                       'u_n(p_i,s_i)\\,e^{i(p_f-p_i)\\cdot X}',
              'note': ''},
             {'type': 'p',
              'text': 'El vértice hadrónico contiene la estructura interna del nucleón. En el TFG se separa en sector '
                      'vector y axial:'},
             {'type': 'eq',
              'num': '(2.16)',
              'latex': '\\widetilde\\Gamma^\\mu=\\underbrace{F_1^V(Q^2)\\gamma^\\mu+i\\frac{F_2^V(Q^2)}{2M}\\sigma^{\\mu\\nu}Q_\\nu}_{\\widetilde\\Gamma_V^\\mu}+\\underbrace{G_A(Q^2)\\gamma^\\mu\\gamma^5+\\frac{G_P(Q^2)}{2M}Q^\\mu\\gamma^5}_{\\widetilde\\Gamma_A^\\mu}',
              'note': ''},
             {'type': 'p',
              'text': 'Aplicando la identidad de Gordon, la contribución vectorial puede escribirse de forma '
                      'equivalente como:'},
             {'type': 'eq',
              'num': '(2.17)',
              'latex': '\\widetilde\\Gamma_V^\\mu=\\left(F_1^V+F_2^V\\right)\\gamma^\\mu-\\frac{F_2^V}{2M}(p_i+p_f)^\\mu',
              'note': ''},
             {'type': 'info',
              'text': 'La parte vector se relaciona con factores de forma electromagnéticos mediante CVC. La parte '
                      'axial, dominada por GA(Q²), es el núcleo físico del TFG.'}]},
 {'key': 'propagator_sfi',
  'title': '2.4-2.5. Hamiltoniano, propagador y elemento de matriz',
  'subtitle': 'Del intercambio de W al elemento de matriz y la amplitud invariante.',
  'blocks': [{'type': 'h3', 'text': '2.4. Hamiltoniano de interacción y propagador débil'},
             {'type': 'eq',
              'num': '(2.18)',
              'latex': 'H_W(X)=\\left(\\frac{g}{2\\sqrt{2}}\\right)^2 j_\\mu^\\dagger(X)A^\\mu(X)',
              'note': ''},
             {'type': 'eq',
              'num': '(2.19)',
              'latex': 'A^\\mu(X)=\\int d^4Y\\,D_W^{\\mu\\nu}(X-Y)J_\\nu(Y)',
              'note': ''},
             {'type': 'eq',
              'num': '(2.20)',
              'latex': 'H_W(X)=\\left(\\frac{g}{2\\sqrt{2}}\\right)^2j_\\mu^\\dagger(X)D_W^{\\mu\\nu}(Q)J_\\nu(X)',
              'note': ''},
             {'type': 'eq',
              'num': '(2.21)',
              'latex': 'D_W^{\\mu\\nu}(Q)=\\frac{-g^{\\mu\\nu}+Q^\\mu Q^\\nu/M_W^2}{Q^2-M_W^2+i\\epsilon}',
              'note': ''},
             {'type': 'eq',
              'num': '(2.22)',
              'latex': 'D_W^{\\mu\\nu}(X-Y)=\\int\\frac{d^4Q}{(2\\pi)^4}\\,D_W^{\\mu\\nu}(Q)e^{iQ\\cdot(X-Y)}',
              'note': ''},
             {'type': 'eq',
              'num': '(2.23)',
              'latex': 'A^\\mu(X)=\\int d^4Y\\int\\frac{d^4Q}{(2\\pi)^4}\\,D_W^{\\mu\\nu}(Q)e^{iQ\\cdot(X-Y)}J_\\nu(Y)',
              'note': ''},
             {'type': 'eq', 'num': '(2.24)', 'latex': '|Q^2|\\ll M_W^2', 'note': ''},
             {'type': 'eq',
              'num': '(2.25)',
              'latex': 'D_W^{\\mu\\nu}(Q)\\simeq \\frac{g^{\\mu\\nu}}{M_W^2}',
              'note': ''},
             {'type': 'eq',
              'num': '(2.26)',
              'latex': 'A^\\mu(X)\\simeq\\int '
                       'd^4Y\\int\\frac{d^4Q}{(2\\pi)^4}\\,e^{iQ\\cdot(X-Y)}\\frac{1}{M_W^2}J^\\mu(Y)',
              'note': ''},
             {'type': 'eq',
              'num': '(2.27)',
              'latex': 'G_F=\\frac{g^2}{8M_W^2\\sqrt{2}}\\simeq1.166\\times10^{-5}\\,\\mathrm{GeV}^{-2}',
              'note': ''},
             {'type': 'h3', 'text': '2.5. Elemento de matriz Sfi'},
             {'type': 'eq', 'num': '(2.28)', 'latex': 'S_{fi}=-i\\int d^4X\\,H_W(X)', 'note': ''},
             {'type': 'eq',
              'num': '(2.29)',
              'latex': 'S_{fi}=-i\\frac{G_F}{\\sqrt{2}}\\int d^4X\\int '
                       'd^4Y\\int\\frac{d^4Q}{(2\\pi)^4}\\,j_\\mu^\\dagger e^{iQ\\cdot(X-Y)}J^\\mu(Y)',
              'note': ''},
             {'type': 'p',
              'text': 'Sustituyendo las corrientes leptónica y hadrónica, todos los factores de normalización quedan '
                      'fuera de las integrales espacio-temporales:'},
             {'type': 'eq',
              'num': '(2.30)',
              'latex': '\\begin{aligned}S_{fi}=&-i\\frac{G_F}{\\sqrt{2}}\\frac{M\\sqrt{m_{\\nu_\\mu}m_\\mu}}{V^2\\sqrt{\\epsilon_i\\epsilon_fE_iE_f}}\\left[\\bar '
                       'u_l(k_l,s_l)\\gamma_\\mu(1-\\gamma^5)u_{\\nu_l}(k_{\\nu_l},s_{\\nu_l})\\right]^\\dagger\\\\&\\times\\left[\\bar '
                       'u_p(p_f,s_f)\\widetilde\\Gamma^\\mu u_n(p_i,s_i)\\right]\\int d^4X\\int '
                       'd^4Y\\int\\frac{d^4Q}{(2\\pi)^4}e^{iQ\\cdot(X-Y)}e^{i(k_l-k_{\\nu_l})\\cdot '
                       'X}e^{i(p_f-p_i)\\cdot Y}.\\end{aligned}',
              'note': ''},
             {'type': 'p',
              'text': 'La integral sobre X, Y y Q produce la delta de conservación del cuadrimomento. Los pasos '
                      'detallados aparecen en el Apéndice A.3.'},
             {'type': 'eq',
              'num': 'integral clave',
              'latex': '\\int d^4X\\,d^4Y\\,\\frac{d^4Q}{(2\\pi)^4}\\,e^{i(Q+k_f-k_i)\\cdot X}e^{i(p_f-p_i-Q)\\cdot '
                       'Y}=(2\\pi)^4\\delta^{(4)}(k_i+p_i-k_f-p_f)',
              'note': ''},
             {'type': 'eq',
              'num': '(2.31)',
              'latex': '\\begin{aligned}S_{fi}=&-i\\frac{G_F}{\\sqrt{2}V^2}\\frac{M\\sqrt{m_{\\nu_\\mu}m_\\mu}}{\\sqrt{\\epsilon_i\\epsilon_fE_iE_f}}(2\\pi)^4\\delta^{(4)}(k_l-k_{\\nu_l}+p_f-p_i)\\\\&\\times\\left[\\bar '
                       'u_l(k_l,s_l)\\gamma_\\mu(1-\\gamma^5)u_{\\nu_l}(k_{\\nu_l},s_{\\nu_l})\\right]^\\dagger\\left[\\bar '
                       'u_p(p_f,s_f)\\widetilde\\Gamma^\\mu u_n(p_i,s_i)\\right].\\end{aligned}',
              'note': ''},
             {'type': 'p',
              'text': 'La tasa de transición por unidad de volumen se define a partir de |Sfi|²/(VT). El cuadrado de '
                      'la delta tetradimensional se reduce usando δ⁴(0)=VT/(2π)^4.'},
             {'type': 'eq',
              'num': '(2.32)',
              'latex': '\\begin{aligned}W_{fi}=\\frac{|S_{fi}|^2}{VT}=&\\frac{G_F^2}{2}\\frac{\\left[(2\\pi)^4\\delta^{(4)}(k_l-k_{\\nu_l}+p_f-p_i)\\right]^2}{VT}\\frac{1}{V^4}\\frac{m_{\\nu_\\mu}m_\\mu '
                       'M^2}{\\epsilon_i\\epsilon_fE_iE_f}|\\mathcal{M}_{fi}|^2\\\\=&\\frac{G_F^2}{2}(2\\pi)^4\\delta^{(4)}(k_l-k_{\\nu_l}+p_f-p_i)\\frac{1}{V^4}\\frac{m_{\\nu_\\mu}m_\\mu '
                       'M^2}{\\epsilon_i\\epsilon_fE_iE_f}|\\mathcal{M}_{fi}|^2.\\end{aligned}',
              'note': ''},
             {'type': 'eq',
              'num': '(2.33)',
              'latex': '\\mathcal{M}_{fi}=\\left[\\bar '
                       'u_l(k_l,s_l)\\gamma_\\mu(1-\\gamma^5)u_{\\nu_l}(k_{\\nu_\\mu},s_{\\nu_\\mu})\\right]^\\dagger\\left[\\bar '
                       'u_p(p_f,s_f)\\widetilde\\Gamma^\\mu u_n(p_i,s_i)\\right]',
              'note': ''}]},
 {'key': 'cross_section',
  'title': '2.6. Sección eficaz',
  'subtitle': 'Espacio de fases, flujo incidente y sección eficaz diferencial doble.',
  'blocks': [{'type': 'eq',
              'num': '(2.34)',
              'latex': 'd\\sigma=\\frac{|S_{fi}|^2}{t\\,\\Phi_{\\mathrm{inc}}}\\,d\\rho',
              'note': ''},
             {'type': 'eq',
              'num': '(2.35)',
              'latex': 'd\\rho=\\prod_f^N\\frac{V}{(2\\pi)^3}\\,d^3\\mathbf{p}_f',
              'note': ''},
             {'type': 'eq', 'num': '(2.36)', 'latex': '\\Phi_{\\mathrm{inc}}=\\frac{|\\mathbf{v}_i|}{V}', 'note': ''},
             {'type': 'p',
              'text': 'Después de introducir el espacio de fases final, el flujo incidente y la tasa de transición, se '
                      'obtiene:'},
             {'type': 'eq',
              'num': '(2.37)',
              'latex': 'd\\sigma=\\frac{G_F^2}{2}\\frac{m_{\\nu_l}}{|\\mathbf{k}_i|}|\\mathcal{M}_{fi}|^2\\frac{1}{4\\pi^2}\\delta^{(4)}(k_l-k_{\\nu_l}+p_f-p_i)\\frac{m_\\mu}{\\epsilon_f}d^3\\mathbf{k}_l\\frac{M}{E_f}d^3\\mathbf{p}_f',
              'note': ''},
             {'type': 'eq',
              'num': '(2.38)',
              'latex': 'd^3\\mathbf{k}_l=|\\mathbf{k}_l|^2d|\\mathbf{k}_l|d\\Omega=|\\mathbf{k}_l|\\epsilon_fd\\epsilon_fd\\Omega',
              'note': ''},
             {'type': 'p', 'text': 'La integración sobre el momento del nucleón final fija la condición elástica:'},
             {'type': 'eq',
              'num': '(2.39)',
              'latex': '\\int\\frac{d^3\\mathbf{p}_f}{2E_f}\\delta^{(4)}(p_f-p_i-Q)=\\delta(Q^2+2M\\omega)=\\frac{1}{2M}\\delta\\!\\left(\\omega-\\frac{|Q^2|}{2M}\\right)',
              'note': ''},
             {'type': 'p',
              'text': 'La amplitud invariante al cuadrado se reescribe mediante la contracción tensorial reducida:'},
             {'type': 'eq',
              'num': '(2.40)',
              'latex': '\\frac{m_{\\nu_l}m_\\mu}{2}|\\mathcal{M}_{fi}|^2=\\widetilde\\eta_{\\mu\\nu}\\widetilde '
                       'W^{\\mu\\nu}',
              'note': ''},
             {'type': 'eq',
              'num': '(2.41)',
              'latex': '\\frac{d\\sigma}{dE_l\\,d\\Omega}=\\frac{|\\mathbf{k}_l|}{|\\mathbf{k}_{\\nu_l}|}\\frac{G_F^2}{4\\pi^2}\\delta\\!\\left(\\omega-\\frac{|Q^2|}{2M}\\right)\\widetilde\\eta_{\\mu\\nu}\\widetilde '
                       'W^{\\mu\\nu}',
              'note': ''},
             ]},
 {'key': 'tensors',
  'title': '2.7. Contracción de los tensores leptónico y hadrónico',
  'subtitle': 'Trazas, funciones de estructura Wi y origen del término W3.',
  'blocks': [{'type': 'h3', 'text': 'Tensor leptónico'},
             {'type': 'eq',
              'num': '(2.42)',
              'latex': '\\eta^{\\mu\\nu}=j^{\\mu\\dagger}j^\\nu=\\sum_{s_i,s_f}\\left[\\bar u_l\\widetilde\\gamma^\\mu '
                       'u_{\\nu_l}\\right]^\\dagger\\left[\\bar u_l\\widetilde\\gamma^\\nu u_{\\nu_l}\\right]',
              'note': ''},
             {'type': 'eq',
              'num': '(2.43)',
              'latex': '\\widetilde\\gamma^\\mu\\equiv\\gamma^\\mu-\\gamma^\\mu\\gamma^5',
              'note': ''},
             {'type': 'eq',
              'num': '(2.44)',
              'latex': '\\eta^{\\mu\\nu}=\\mathrm{Tr}\\!\\left[\\frac{\\not{k}_f+m_l}{2m_l}\\widetilde\\gamma^\\mu\\frac{\\not{k}_i+m_{\\nu_l}}{2m_{\\nu_l}}\\widetilde\\gamma^\\nu\\right]=\\frac{1}{4m_\\mu '
                       'm_\\nu}\\mathrm{Tr}\\!\\left[(\\not{k}_f+m_\\mu)\\widetilde\\gamma^\\mu(\\not{k}_i+m_\\nu)\\widetilde\\gamma^\\nu\\right]',
              'note': ''},
             {'type': 'eq',
              'num': '(2.45)',
              'latex': '\\widetilde\\eta^{\\mu\\nu}=2\\left[k_f^\\mu k_i^\\nu-(k_f\\cdot k_i)g^{\\mu\\nu}+k_f^\\nu '
                       'k_i^\\mu+i\\epsilon^{\\alpha\\mu\\beta\\nu}k_{f\\alpha}k_{i\\beta}\\right]',
              'note': ''},
             {'type': 'eq',
              'num': '(2.46)',
              'latex': '\\widetilde\\eta^{\\mu\\nu}=\\widetilde\\eta_S^{\\mu\\nu}+\\widetilde\\eta_A^{\\mu\\nu}',
              'note': ''},
             {'type': 'eq',
              'num': '(2.47)',
              'latex': '\\widetilde\\eta_S^{\\mu\\nu}=2\\left[k_f^\\mu k_i^\\nu-(k_f\\cdot k_i)g^{\\mu\\nu}+k_f^\\nu '
                       'k_i^\\mu\\right]',
              'note': ''},
             {'type': 'eq',
              'num': '(2.48)',
              'latex': '\\widetilde\\eta_A^{\\mu\\nu}=2i\\epsilon^{\\alpha\\mu\\beta\\nu}k_{f\\alpha}k_{i\\beta}',
              'note': ''},
             {'type': 'h3', 'text': 'Tensor hadrónico'},
             {'type': 'eq',
              'num': '(2.49)',
              'latex': 'W^{\\mu\\nu}=J^{\\mu\\dagger}J^\\nu=\\sum_{s_i,s_f}\\left(\\bar u_p\\widetilde\\Gamma^\\mu '
                       'u_n\\right)^\\dagger\\left(\\bar u_p\\widetilde\\Gamma^\\nu u_n\\right)',
              'note': ''},
             {'type': 'eq',
              'num': '(2.50)',
              'latex': '\\widetilde '
                       'W^{\\mu\\nu}=\\frac{1}{8M^2}\\mathrm{Tr}\\!\\left[(\\not{p}_f+M)\\widetilde\\Gamma^\\mu(\\not{p}_i+M)\\widetilde\\Gamma^\\nu\\right]',
              'note': ''},
             {'type': 'eq',
              'num': '(2.51)',
              'latex': '\\widetilde\\Gamma^\\mu=\\widetilde\\Gamma_V^\\mu+\\widetilde\\Gamma_A^\\mu',
              'note': ''},
             {'type': 'eq',
              'num': '(2.52)',
              'latex': '\\widetilde\\Gamma_V^\\mu=F_1^V(Q^2)\\gamma^\\mu+i\\frac{F_2^V(Q^2)}{2M}\\sigma^{\\mu\\nu}Q_\\nu',
              'note': ''},
             {'type': 'eq',
              'num': '(2.53)',
              'latex': '\\widetilde\\Gamma_A^\\mu=G_A(Q^2)\\gamma^\\mu\\gamma^5+\\frac{G_P(Q^2)}{2M}Q^\\mu\\gamma^5=G_A(Q^2)\\gamma^\\mu\\gamma^5+F_P(Q^2)Q^\\mu\\gamma^5',
              'note': ''},
             {'type': 'eq',
              'num': '(2.54)',
              'latex': '\\widetilde W^{\\mu\\nu}=\\widetilde W_V^{\\mu\\nu}+\\widetilde W_A^{\\mu\\nu}+\\widetilde '
                       'W_{VA}^{\\mu\\nu}',
              'note': ''},
             {'type': 'h3', 'text': 'Parte vector'},
             {'type': 'eq',
              'num': '(2.55)',
              'latex': '\\widetilde '
                       'W_V^{\\mu\\nu}=\\frac{1}{8M^2}\\left[(F_1^V)^2f_{11}^{\\mu\\nu}+\\frac{(F_2^V)^2}{4M^2}f_{22}^{\\mu\\nu}+i\\frac{F_1^VF_2^V}{2M}f_{12}^{\\mu\\nu}\\right]',
              'note': ''},
             {'type': 'eq',
              'num': '(2.56)',
              'latex': 'f_{11}^{\\mu\\nu}=\\mathrm{Tr}\\!\\left[(\\not{p}_f+M)\\gamma^\\mu(\\not{p}_i+M)\\gamma^\\nu\\right]=4\\left[p_f^\\mu '
                       'p_i^\\nu+p_f^\\nu p_i^\\mu+(M^2-p_f\\cdot p_i)g^{\\mu\\nu}\\right]',
              'note': ''},
             {'type': 'eq',
              'num': '(2.57)',
              'latex': '\\begin{aligned}f_{22}^{\\mu\\nu}=4\\Big[&-Q^2(p_f^\\mu p_i^\\nu+p_f^\\nu p_i^\\mu)+(p_f\\cdot '
                       'Q)(Q^\\mu p_i^\\nu+Q^\\nu p_i^\\mu)+(p_i\\cdot Q)(Q^\\mu p_f^\\nu+Q^\\nu '
                       'p_f^\\mu)\\\\&-(p_f\\cdot p_i)Q^\\mu Q^\\nu+g^{\\mu\\nu}\\left(Q^2(p_f\\cdot p_i)-2(p_f\\cdot '
                       'Q)(p_i\\cdot Q)\\right)+M^2(g^{\\mu\\nu}Q^2-Q^\\mu Q^\\nu)\\Big]\\end{aligned}',
              'note': ''},
             {'type': 'eq',
              'num': '(2.58)',
              'latex': 'f_{12}^{\\mu\\nu}=4iM\\left[p_f^\\mu Q^\\nu+Q^\\mu p_f^\\nu-p_i^\\mu Q^\\nu-Q^\\mu '
                       'p_i^\\nu-2g^{\\mu\\nu}(p_f\\cdot Q)+2g^{\\mu\\nu}(p_i\\cdot Q)\\right]',
              'note': ''},
             {'type': 'h3', 'text': 'Parte axial'},
             {'type': 'eq',
              'num': '(2.59)',
              'latex': '\\widetilde '
                       'W_A^{\\mu\\nu}=\\frac{1}{8M^2}\\left[G_A^2f_{33}^{\\mu\\nu}+F_P^2f_{44}^{\\mu\\nu}+G_AF_Pf_{34}^{\\mu\\nu}\\right]',
              'note': ''},
             {'type': 'eq',
              'num': '(2.60)',
              'latex': 'f_{33}^{\\mu\\nu}=\\mathrm{Tr}\\!\\left[(\\not{p}_f+M)\\gamma^\\mu\\gamma^5(\\not{p}_i+M)\\gamma^\\nu\\gamma^5\\right]=4\\left[p_f^\\mu '
                       'p_i^\\nu+p_f^\\nu p_i^\\mu-g^{\\mu\\nu}(p_f\\cdot p_i+M^2)\\right]',
              'note': ''},
             {'type': 'eq',
              'num': '(2.61)',
              'latex': 'f_{44}^{\\mu\\nu}=-\\mathrm{Tr}\\!\\left[(\\not{p}_f+M)Q^\\mu\\gamma^5(\\not{p}_i+M)Q^\\nu\\gamma^5\\right]=4(p_f\\cdot '
                       'p_i-M^2)Q^\\mu Q^\\nu',
              'note': ''},
             {'type': 'eq',
              'num': '(2.62)',
              'latex': 'f_{34}^{\\mu\\nu}=4M\\left[-Q^\\mu p_f^\\nu-Q^\\nu p_f^\\mu+p_i^\\mu Q^\\nu+p_i^\\nu '
                       'Q^\\mu\\right]',
              'note': ''},
             {'type': 'h3', 'text': 'Parte vector-axial'},
             {'type': 'eq',
              'num': '(2.63)',
              'latex': '\\widetilde '
                       'W_{VA}^{\\mu\\nu}=\\frac{1}{8M^2}\\left[F_1^VG_Af_{45}^{\\mu\\nu}+i\\frac{F_2^V}{2M}G_Af_{56}^{\\mu\\nu}+F_1^VF_Pf_{67}^{\\mu\\nu}+i\\frac{F_2^V}{2M}F_Pf_{78}^{\\mu\\nu}\\right]',
              'note': ''},
             {'type': 'eq',
              'num': '(2.64)',
              'latex': 'f_{45}^{\\mu\\nu}=\\mathrm{Tr}\\!\\left[\\not{p}_f\\gamma^\\mu\\not{p}_i\\gamma^\\nu\\gamma^5\\right]+\\mathrm{Tr}\\!\\left[\\not{p}_f\\gamma^\\mu\\gamma^5\\not{p}_i\\gamma^\\nu\\right]=8i\\,p_{f\\alpha}p_{i\\beta}\\epsilon^{\\nu\\mu\\alpha\\beta}',
              'note': ''},
             {'type': 'eq',
              'num': '(2.65)',
              'latex': 'f_{56}^{\\mu\\nu}=\\mathrm{Tr}\\!\\left[(\\not{p}_f+M)\\sigma^{\\mu\\alpha}Q_\\alpha(\\not{p}_i+M)\\gamma^\\nu\\gamma^5\\right]-\\mathrm{Tr}\\!\\left[(\\not{p}_f+M)\\gamma^\\mu\\gamma^5(\\not{p}_i+M)\\sigma^{\\nu\\beta}Q_\\beta\\right]',
              'note': ''},
             {'type': 'eq',
              'num': '(2.66)',
              'latex': 'f_{56}^{\\mu\\nu}=8M\\left(p_{f\\tau}Q_\\rho\\epsilon^{\\tau\\mu\\nu\\rho}+p_{i\\tau}Q_\\rho\\epsilon^{\\rho\\mu\\tau\\nu}\\right)',
              'note': ''},
             {'type': 'eq',
              'num': '(2.67)',
              'latex': 'f_{67}^{\\mu\\nu}=\\mathrm{Tr}\\!\\left[(\\not{p}_f+M)\\gamma^\\mu(\\not{p}_i+M)Q^\\nu\\gamma^5\\right]+\\mathrm{Tr}\\!\\left[(\\not{p}_f+M)Q^\\mu\\gamma^5(\\not{p}_i+M)\\gamma^\\nu\\right]=0',
              'note': ''},
             {'type': 'eq',
              'num': '(2.68)',
              'latex': 'f_{78}^{\\mu\\nu}=4p_{f\\rho}p_{i\\sigma}Q^\\nu '
                       'Q_\\delta\\epsilon^{\\rho\\mu\\delta\\sigma}-4p_{f\\rho}p_{i\\sigma}Q^\\mu '
                       'Q_\\delta\\epsilon^{\\rho\\sigma\\mu\\delta}',
              'note': ''},
             {'type': 'h3', 'text': 'Funciones de estructura'},
             {'type': 'eq',
              'num': '(2.69)',
              'latex': '\\widetilde W^{\\mu\\nu}=-W_1g^{\\mu\\nu}+W_2\\frac{p_i^\\mu '
                       'p_i^\\nu}{M^2}+iW_3\\frac{\\epsilon^{\\mu\\nu\\beta\\alpha}p_{i\\beta}Q_\\alpha}{2M^2}+W_4\\frac{Q^\\mu '
                       'Q^\\nu}{M^2}+W_5\\frac{p_i^\\mu Q^\\nu+Q^\\mu p_i^\\nu}{2M^2}',
              'note': ''},
             {'type': 'eq',
              'num': '(2.70)',
              'latex': 'W_1=\\tau\\left[(F_1^V+F_2^V)^2+G_A^2\\right]+G_A^2',
              'note': ''},
             {'type': 'eq', 'num': '(2.71)', 'latex': 'W_2=(F_1^V)^2+\\tau(F_2^V)^2+G_A^2', 'note': ''},
             {'type': 'eq', 'num': '(2.72)', 'latex': 'W_3=2G_A(F_1^V+F_2^V)', 'note': ''},
             {'type': 'eq',
              'num': '(2.73)',
              'latex': 'W_4=\\frac{|Q^2|-4M^2}{(4M^2)^2}(F_2^V)^2-\\frac{G_AF_P}{M}+\\tau '
                       'F_P^2-\\frac{F_1^VF_2^V}{2M^2}',
              'note': ''},
             {'type': 'eq', 'num': '(2.74)', 'latex': 'W_5=W_2', 'note': ''},
             {'type': 'eq', 'num': '(2.75)', 'latex': '\\tau\\equiv\\frac{|Q^2|}{4M^2}', 'note': ''},
             {'type': 'eq',
              'num': '(2.76)',
              'latex': '\\widetilde W^{\\mu\\nu}=\\widetilde W_S^{\\mu\\nu}+\\widetilde W_A^{\\mu\\nu}',
              'note': ''},
             {'type': 'eq',
              'num': '(2.77)',
              'latex': '\\widetilde W_S^{\\mu\\nu}=-W_1g^{\\mu\\nu}+W_2\\frac{p_i^\\mu p_i^\\nu}{M^2}+W_4\\frac{Q^\\mu '
                       'Q^\\nu}{M^2}+W_5\\frac{p_i^\\mu Q^\\nu+Q^\\mu p_i^\\nu}{2M^2}',
              'note': ''},
             {'type': 'eq',
              'num': '(2.78)',
              'latex': '\\widetilde '
                       'W_A^{\\mu\\nu}=iW_3\\frac{\\epsilon^{\\mu\\nu\\beta\\alpha}p_{i\\beta}Q_\\alpha}{2M^2}',
              'note': ''},
             {'type': 'eq',
              'num': '(2.79)',
              'latex': '\\widetilde\\eta_{\\mu\\nu}\\widetilde W^{\\mu\\nu}=\\widetilde\\eta^S_{\\mu\\nu}\\widetilde '
                       'W_S^{\\mu\\nu}+\\widetilde\\eta^A_{\\mu\\nu}\\widetilde W_A^{\\mu\\nu}',
              'note': ''},
             {'type': 'p',
              'text': 'La contracción entre una parte simétrica y una antisimétrica se anula. Por eso el término W3 '
                      'sólo se acopla a la parte antisimétrica del tensor leptónico.'},
             {'type': 'eq',
              'num': '(2.80)',
              'latex': '\\begin{aligned}\\widetilde\\eta_{\\mu\\nu}\\widetilde '
                       'W^{\\mu\\nu}=&2W_1E_{\\nu_l}(E_l-|\\mathbf{k}_l|\\cos\\theta_\\mu)+W_2E_{\\nu_l}(E_l+|\\mathbf{k}_l|\\cos\\theta_\\mu)\\\\&+\\frac{W_3}{M}E_{\\nu_l}\\left[(E_{\\nu_l}+E_l)(E_l-|\\mathbf{k}_l|\\cos\\theta_\\mu)-m_l^2\\right]\\\\&+m_l^2W_4E_{\\nu_l}(E_l-|\\mathbf{k}_l|\\cos\\theta_\\mu)-\\frac{W_5}{M}m_l^2E_{\\nu_l}.\\end{aligned}',
              'note': ''},
             {'type': 'p',
              'text': 'Para antineutrinos cambia el signo del término de interferencia vector-axial proporcional a '
                      'W3.'},
             {'type': 'eq',
              'num': '(2.81)',
              'latex': '\\begin{aligned}\\frac{d\\sigma}{dE_l '
                       'd\\Omega}=&|\\mathbf{k}_l|E_l\\frac{G_F^2}{4\\pi^2}\\delta\\!\\left(\\omega-\\frac{|Q^2|}{2M}\\right)\\Bigg\\{4W_1\\sin^2\\frac{\\theta_\\mu}{2}+2W_2\\cos^2\\frac{\\theta_\\mu}{2}\\pm2W_3\\frac{E_{\\nu_l}+E_l}{M}\\sin^2\\frac{\\theta_\\mu}{2}\\\\&+\\frac{m_l^2}{E_l(E_l+|\\mathbf{k}_l|)}\\Bigg[2W_1\\cos\\theta_\\mu-W_2\\cos\\theta_\\mu\\pm\\frac{W_3}{2}\\left(\\frac{E_{\\nu_l}+E_l}{M}\\cos\\theta_\\mu-\\frac{E_l+|\\mathbf{k}_l|}{M}\\right)\\\\&+W_4\\left(m_l^2\\cos\\theta_\\mu+2E_l(E_l+|\\mathbf{k}_l|)\\sin^2\\frac{\\theta_\\mu}{2}\\right)-\\frac{W_5}{M}(E_l+|\\mathbf{k}_l|)\\Bigg]\\Bigg\\}.\\end{aligned}',
              'note': ''},
             {'type': 'eq',
              'num': '(2.82)',
              'latex': '\\frac{d\\sigma}{dE_l '
                       'd\\Omega}=|\\mathbf{k}_l|E_l\\frac{G_F^2}{2\\pi^2}\\delta\\!\\left(\\omega-\\frac{|Q^2|}{2M}\\right)\\left\\{2W_1\\sin^2\\frac{\\theta_\\mu}{2}+W_2\\cos^2\\frac{\\theta_\\mu}{2}\\pm '
                       'W_3\\frac{E_{\\nu_l}+E_l}{M}\\sin^2\\frac{\\theta_\\mu}{2}\\right\\}',
              'note': ''},
             ]},
 {'key': 'integrated',
  'title': '2.8. Sección eficaz diferencial integrada',
  'subtitle': 'Delta elástica, cosθ0, jacobiano y factor de retroceso.',
  'blocks': [{'type': 'p',
              'text': 'La expresión diferencial doble contiene una delta de Dirac que impone la condición elástica. '
                      'Para integrar respecto al ángulo se reescribe como una delta en cosθμ.'},
             {'type': 'eq',
              'num': '(2.83)',
              'latex': '\\delta\\!\\left(\\omega-\\frac{|Q^2|}{2M}\\right)=2M\\,\\delta(2M\\omega-|Q^2|)',
              'note': ''},
             {'type': 'eq',
              'num': '(2.84)',
              'latex': '\\delta(2M\\omega-|Q^2|)=\\frac{1}{2E_{\\nu_l}|\\mathbf{k}_l|}\\delta\\!\\left(\\cos\\theta_\\mu-\\frac{2(E_{\\nu_l}E_l+ME_l-ME_{\\nu_l})-m_l^2}{2E_{\\nu_l}|\\mathbf{k}_l|}\\right)',
              'note': ''},
             {'type': 'eq',
              'num': '(2.85)',
              'latex': '\\frac{d\\sigma}{dE_l '
                       'd\\Omega}=\\frac{|\\mathbf{k}_l|}{E_{\\nu_l}}\\frac{G_F^2}{4\\pi^2}\\widetilde\\eta_{\\mu\\nu}\\widetilde '
                       'W^{\\mu\\nu}\\frac{2M}{2E_{\\nu_l}|\\mathbf{k}_l|}\\delta(\\cos\\theta_\\mu-\\cos\\theta_0)',
              'note': ''},
             {'type': 'eq',
              'num': '(2.86)',
              'latex': '\\cos\\theta_0=\\frac{2(E_{\\nu_l}E_l+ME_l-ME_{\\nu_l})-m_l^2}{2E_{\\nu_l}|\\mathbf{k}_l|}',
              'note': ''},
             {'type': 'p',
              'text': 'Integrando en ángulo sólido se obtiene la distribución en energía del leptón saliente:'},
             {'type': 'eq',
              'num': '(2.87)',
              'latex': '\\frac{d\\sigma}{dE_l}=\\frac{2MG_F^2}{4\\pi '
                       'E_{\\nu_l}^2}\\left.\\widetilde\\eta_{\\mu\\nu}\\widetilde '
                       'W^{\\mu\\nu}\\right|_{\\cos\\theta_\\mu=\\cos\\theta_0}',
              'note': ''},
             {'type': 'p', 'text': 'Para pasar a una distribución angular se usa el jacobiano dEl/dcosθμ:'},
             {'type': 'eq',
              'num': '(2.88)',
              'latex': '\\frac{d\\sigma}{d\\cos\\theta_\\mu}=\\frac{d\\sigma}{dE_l}\\left(\\frac{dE_l}{d\\cos\\theta_\\mu}\\right)=\\left.\\frac{MG_F^2}{2\\pi '
                       'E_{\\nu_l}^2}\\widetilde\\eta_{\\mu\\nu}\\widetilde '
                       'W^{\\mu\\nu}\\left(\\frac{dE_l}{d\\cos\\theta_\\mu}\\right)\\right|_{\\cos\\theta_\\mu=\\cos\\theta_0}',
              'note': ''},
             {'type': 'eq',
              'num': '(2.89)',
              'latex': '\\frac{dE_l}{d\\cos\\theta_\\mu}=\\frac{|\\mathbf{k}_l|}{1+\\frac{M}{E_{\\nu_l}}-\\frac{E_l\\cos\\theta_\\mu}{|\\mathbf{k}_l|}}',
              'note': ''},
             {'type': 'eq',
              'num': '(2.90)',
              'latex': '\\frac{d\\sigma}{d\\Omega}=\\frac{1}{2\\pi}\\frac{d\\sigma}{d\\cos\\theta_\\mu}=\\left.\\frac{G_F^2|\\mathbf{k}_l|}{4\\pi^2E_{\\nu_l}}f_{\\mathrm{rec}}^{-1}\\widetilde\\eta_{\\mu\\nu}\\widetilde '
                       'W^{\\mu\\nu}\\right|_{\\cos\\theta_\\mu=\\cos\\theta_0}',
              'note': ''},
             {'type': 'eq',
              'num': '(2.91)',
              'latex': 'f_{\\mathrm{rec}}\\equiv1+\\frac{E_{\\nu_l}(|\\mathbf{k}_l|-E_l\\cos\\theta_\\mu)}{M|\\mathbf{k}_l|}',
              'note': ''},
             {'type': 'info',
              'text': 'La condición |cosθ0|≤1 define el intervalo físico de energías Eμ accesibles para una energía '
                      'incidente dada.'}]},
 {'key': 'vector',
  'title': '3.1. Estructura vector',
  'subtitle': 'Factores de Sachs, CVC, Galster, Kelly y GKeX.',
  'blocks': [{'type': 'p',
              'text': 'El sector vector del vértice débil depende de F1^V(Q²) y F2^V(Q²). Mediante la hipótesis CVC, '
                      'estos se construyen a partir de los factores electromagnéticos del protón y del neutrón.'},
             {'type': 'h3', 'text': 'Factores de Sachs'},
             {'type': 'eq',
              'num': '(3.1)',
              'latex': 'G_E^{n,p}(Q^2)\\equiv F_1^{n,p}(Q^2)-\\frac{Q^2}{4M^2}F_2^{n,p}(Q^2),\\qquad '
                       'G_M^{n,p}(Q^2)\\equiv F_1^{n,p}(Q^2)+F_2^{n,p}(Q^2)',
              'note': ''},
             {'type': 'eq', 'num': '(3.2)', 'latex': 'G_E^n(0)=0,\\qquad G_E^p(0)=1', 'note': ''},
             {'type': 'eq', 'num': '(3.3)', 'latex': 'G_M^n(0)\\simeq -1.91,\\qquad G_M^p(0)\\simeq 2.79', 'note': ''},
             {'type': 'h3', 'text': 'Combinaciones isovectores'},
             {'type': 'eq',
              'num': '(3.4)',
              'latex': 'G_E^V=\\frac{1}{2}(G_E^p-G_E^n)\\equiv F_1^V-\\frac{Q^2}{4M^2}F_2^V,\\qquad '
                       'G_M^V=\\frac{1}{2}(G_M^p-G_M^n)\\equiv F_1^V+F_2^V',
              'note': ''},
             {'type': 'p', 'text': 'Al invertir las relaciones de Sachs se obtiene:'},
             {'type': 'eq',
              'num': '(3.5)',
              'latex': 'F_1^V(Q^2)=\\frac{1}{2(1+\\tau)}\\left[(G_E^p-G_E^n)+\\tau(G_M^p-G_M^n)\\right]',
              'note': ''},
             {'type': 'eq',
              'num': '(3.6)',
              'latex': 'F_2^V(Q^2)=\\frac{1}{2(1+\\tau)}\\left[(G_M^p-G_M^n)-(G_E^p-G_E^n)\\right]',
              'note': ''},
             {'type': 'h3', 'text': 'Parametrización dipolar de Galster'},
             {'type': 'eq', 'num': '(3.7)', 'latex': 'G_D^V(Q^2)=\\frac{1}{(1+\\lambda_D^V\\tau)^2}', 'note': ''},
             {'type': 'eq',
              'num': '(3.8)',
              'latex': 'G_E^p(Q^2)=G_D^V(Q^2),\\qquad G_M^p(Q^2)=\\mu_pG_D^V(Q^2)',
              'note': ''},
             {'type': 'eq',
              'num': '(3.9)',
              'latex': 'G_E^n(Q^2)=-\\mu_n\\tau\\xi_nG_D^V(Q^2),\\qquad G_M^n(Q^2)=\\mu_nG_D^V(Q^2)',
              'note': ''},
             {'type': 'eq',
              'num': '(3.10)',
              'latex': '\\xi_n=\\frac{1}{1+\\lambda_n\\tau},\\qquad \\lambda_n=5.6',
              'note': ''},
             {'type': 'h3', 'text': 'Parametrización de Kelly'},
             {'type': 'eq',
              'num': '(3.11)',
              'latex': 'G(Q^2)\\propto\\frac{\\sum_{k=0}^{n}a_k\\tau^k}{1+\\sum_{k=1}^{n+2}b_k\\tau^k}',
              'note': ''},
             {'type': 'h3', 'text': 'Parametrización GKeX'},
             {'type': 'p',
              'text': 'GKeX es una parametrización de tipo VMD que incorpora contribuciones de mesones vectoriales '
                      "rho, rho', omega, omega' y phi, e impone el comportamiento asintótico esperado a grandes |Q²| "
                      'de acuerdo con QCD perturbativa.'},
             {'type': 'info',
              'text': 'Conclusión del bloque vector: en las secciones eficaces del TFG, las diferencias Galster/GKeX '
                      'son mucho menores que las variaciones producidas por el sector axial en la región cinemática '
                      'dominante.'}]},
 {'key': 'axial',
  'title': '3.2. Estructura axial',
  'subtitle': 'Modelos para GA(Q²) y GP(Q²): dipolo, monopolo, dos componentes, BBBA2007 y expansión z.',
  'blocks': [{'type': 'p',
              'text': 'A diferencia del sector vector, el factor de forma axial GA(Q²) no se determina mediante '
                      'dispersión electromagnética. Por eso su forma funcional es una fuente importante de '
                      'incertidumbre fenomenológica.'},
             {'type': 'h3', 'text': '3.2.1. Parametrización dipolar'},
             {'type': 'eq',
              'num': '(3.12)',
              'latex': 'G_A^D(Q^2)=\\frac{G_A(0)}{\\left(1+\\frac{|Q^2|}{M_A^2}\\right)^2}',
              'note': ''},
             {'type': 'p',
              'text': 'Se usa GA(0)=gA=-1.267 y un valor estándar MA≈1.032 GeV. Un valor efectivo mayor, como MA≈1.35 '
                      'GeV, hace que GA decrezca más lentamente y aumenta la contribución axial a transferencias '
                      'intermedias y altas.'},
             {'type': 'h3', 'text': '3.2.2. Parametrización monopolar'},
             {'type': 'eq',
              'num': '(3.13)',
              'latex': 'G_A^M(Q^2)=\\frac{G_A(0)}{1+\\frac{|Q^2|}{\\widetilde M_A^2}}',
              'note': ''},
             {'type': 'eq', 'num': '(3.14)', 'latex': '\\widetilde M_A=0.5-1.0\\;\\mathrm{GeV}', 'note': ''},
             {'type': 'p',
              'text': 'La masa monopolar no debe interpretarse como una masa axial determinada experimentalmente de '
                      'forma análoga al caso dipolar, sino como una elección fenomenológica para estudiar sensibilidad '
                      'a la forma funcional.'},
             {'type': 'h3', 'text': 'PCAC y factor pseudoescalar'},
             {'type': 'eq', 'num': '(3.15)', 'latex': 'G_P(Q^2)=\\frac{4M_N^2}{|Q^2|+m_\\pi^2}G_A(Q^2)', 'note': ''},
             {'type': 'eq',
              'num': '(3.16)',
              'latex': '\\frac{G_P(Q^2)}{G_P(0)}=\\frac{m_\\pi^2}{|Q^2|+m_\\pi^2}\\frac{G_A(Q^2)}{G_A(0)}',
              'note': ''},
             {'type': 'p',
              'text': 'El factor pseudoescalar está dominado por el polo piónico. Por eso las diferencias entre '
                      'parametrizaciones de GA se reflejan de manera más moderada en GP.'},
             {'type': 'h3', 'text': '3.2.3. Modelo de dos componentes'},
             {'type': 'eq',
              'num': '(3.17)',
              'latex': 'G_A^{2c}(Q^2)=G_A(0)\\,g(Q^2)\\left[1-\\alpha+\\alpha\\frac{m_{a_1}^2}{m_{a_1}^2+|Q^2|}\\right]',
              'note': ''},
             {'type': 'eq', 'num': '(3.18)', 'latex': 'g(Q^2)=\\frac{1}{(1+\\gamma |Q^2|)^2}', 'note': ''},
             {'type': 'eq', 'num': '(3.19)', 'latex': '1-\\alpha+\\alpha\\frac{m_{a_1}^2}{m_{a_1}^2}=1', 'note': ''},
             {'type': 'eq', 'num': '(3.20)', 'latex': 'G_A^{2c}(0)=G_A(0)=g_A', 'note': ''},
             {'type': 'table',
              'text': '| Modelo | alpha | gamma (GeV^-2) |\n'
                      '|---|---:|---:|\n'
                      '| Soft-Pion | 0.93 | 0.53 |\n'
                      '| PCAC | 1.01 | 0.54 |'},
             {'type': 'eq',
              'num': '(3.21)',
              'latex': '\\begin{array}{c|cc}\\text{Modelo}&\\alpha&\\gamma\\;(\\mathrm{GeV}^{-2})\\\\\\hline\\text{Soft\\! '
                       '-\\! Pion}&0.93&0.53\\\\\\text{PCAC}&1.01&0.54\\end{array}',
              'note': ''},
             {'type': 'h3', 'text': '3.2.4. Parametrización BBBA2007'},
             {'type': 'eq', 'num': '(3.22)', 'latex': '\\xi=\\frac{2}{1+\\sqrt{1+\\frac{1}{\\tau}}}', 'note': ''},
             {'type': 'eq',
              'num': '(3.23)',
              'latex': 'G_A^{\\mathrm{BBBA07}}(Q^2)=G_A^D(Q^2;M_A^{\\mathrm{BBBA07}})A_{FA}^{25}(\\xi)',
              'note': ''},
             {'type': 'eq',
              'num': '(3.24)',
              'latex': 'G_A^D(Q^2;M_A^{\\mathrm{BBBA07}})=\\frac{G_A(0)}{\\left(1+\\frac{|Q^2|}{(M_A^{\\mathrm{BBBA07}})^2}\\right)^2}',
              'note': ''},
             {'type': 'eq',
              'num': '(3.25)',
              'latex': 'A_{FA}^{25}(\\xi)=\\sum_{j=1}^{7}p_j\\prod_{\\substack{k=1\\\\k\\ne '
                       'j}}^{7}\\frac{\\xi-\\xi_k}{\\xi_j-\\xi_k}',
              'note': ''},
             {'type': 'eq',
              'num': '(3.26)',
              'latex': '\\xi_j=\\left\\{0,\\frac{1}{6},\\frac{1}{3},\\frac{1}{2},\\frac{2}{3},\\frac{5}{6},1\\right\\}',
              'note': ''},
             {'type': 'eq',
              'num': '(3.27)',
              'latex': '\\begin{array}{c|ccccccc}j&1&2&3&4&5&6&7\\\\\\hline '
                       'p_j&1.0000&0.9207&0.9795&1.0480&1.0516&1.2874&0.7707\\end{array}',
              'note': ''},
             {'type': 'p',
              'text': 'BBBA2007 no es simplemente un dipolo con una masa axial distinta: introduce una corrección '
                      'dependiente de la variable de Nachtmann.'},
             {'type': 'h3', 'text': '3.2.5. Expansión z'},
             {'type': 'eq',
              'num': '(3.28)',
              'latex': 'z(Q^2;t_0,t_{\\mathrm{cut}})=\\frac{\\sqrt{t_{\\mathrm{cut}}+|Q^2|}-\\sqrt{t_{\\mathrm{cut}}-t_0}}{\\sqrt{t_{\\mathrm{cut}}+|Q^2|}+\\sqrt{t_{\\mathrm{cut}}-t_0}}',
              'note': ''},
             {'type': 'p',
              'text': 'En el canal axial isovector se toma típicamente tcut=9m_pi^2, asociado al umbral de tres '
                      'piones. La elección de t0 mejora la convergencia de la serie.'},
             {'type': 'eq', 'num': '(3.29)', 'latex': 'G_A^z(Q^2)=\\sum_{k=0}^{k_{\\max}}a_k\\,z(Q^2)^k', 'note': ''},
             {'type': 'eq', 'num': '(3.30)', 'latex': 'g_A=\\sum_{k=0}^{k_{\\max}}a_k\\,z(0)^k', 'note': ''},
             {'type': 'table',
              'text': '| Caso | a1 | a2 | a3 | a4 |\n'
                      '|---|---:|---:|---:|---:|\n'
                      '| z cercana al dipolo | -1.69 | 0.20 | 2.31 | -1.15 |\n'
                      '| z con caída más lenta | -1.66 | 0.098 | 2.2245 | -0.9325 |'},
             {'type': 'info',
              'text': 'La expansión z es una familia flexible de parametrizaciones. A diferencia del dipolo, la '
                      'dependencia en Q² no queda controlada por una única masa axial.'}]},
 {'key': 'observables',
  'title': '3.3. Observables físicos y resultados',
  'subtitle': 'Cómo las parametrizaciones se trasladan a secciones eficaces.',
  'blocks': [{'type': 'p',
              'text': 'Los factores de forma aparecen en el vértice hadrónico y, por tanto, en las funciones Wi. '
                      'Cualquier modificación de GA(Q²) o de los factores vectoriales se refleja finalmente en la '
                      'contracción tensorial y en la sección eficaz.'},
             {'type': 'h3', 'text': '3.3.1. Sección eficaz angular dσ/dΩ'},
             {'type': 'p',
              'text': 'El observable angular se calcula con la expresión (2.90). En el TFG se muestra que Galster y '
                      'GKeX producen curvas casi indistinguibles para MA estándar, mientras que aumentar MA a 1.35 GeV '
                      'incrementa claramente la sección eficaz.'},
             {'type': 'h3', 'text': '3.3.2. Sección eficaz dσ/dEμ'},
             {'type': 'p',
              'text': 'El observable energético se obtiene mediante (2.87). La condición |cosθ0|≤1 determina el '
                      'intervalo físico de energías finales del muón.'},
             {'type': 'h3', 'text': '3.3.3. Sensibilidad axial'},
             {'type': 'p',
              'text': 'GA(Q²) entra cuadráticamente en W1 y W2, y linealmente en W3. Por tanto, una parametrización '
                      'que decrece más lentamente con Q² aumenta la contribución axial y puede elevar la sección '
                      'eficaz.'},
             {'type': 'eq',
              'num': 'sensibilidad en Wi',
              'latex': 'W_1\\supset (1+\\tau)G_A^2,\\qquad W_2\\supset G_A^2,\\qquad W_3=2G_A(F_1^V+F_2^V)',
              'note': ''},
             {'type': 'h3', 'text': '3.3.4. Diferencias entre neutrinos y antineutrinos'},
             {'type': 'p',
              'text': 'La diferencia ν/anti-ν aparece de forma directa en el término de interferencia vector-axial '
                      'proporcional a W3.'},
             {'type': 'eq',
              'num': '(3.31)',
              'latex': '\\pm\\frac{W_3}{M}E_\\nu\\left[(E_\\nu+E_\\mu)(E_\\mu-|\\mathbf{k}_\\mu|\\cos\\theta_\\mu)-m_\\mu^2\\right]',
              'note': ''},
             {'type': 'p',
              'text': 'Este término es constructivo para neutrinos y destructivo para antineutrinos, de acuerdo con la '
                      'convención usada en el TFG.'},
             {'type': 'h3', 'text': '3.3.5. Sección eficaz total'},
             {'type': 'eq',
              'num': '(3.32)',
              'latex': '\\sigma_{\\mathrm{tot}}(E_\\nu)=\\int_{E_\\mu^{\\min}}^{E_\\mu^{\\max}}\\frac{d\\sigma}{dE_\\mu}\\,dE_\\mu',
              'note': ''},
             {'type': 'p',
              'text': 'Los límites de integración se fijan por la condición cinemática |cosθ0|≤1. La sección total de '
                      'neutrinos es mayor que la de antineutrinos por la interferencia VA.'},
             {'type': 'h3', 'text': '3.3.6. Comparación con MINERvA'},
             {'type': 'eq', 'num': '(3.33)', 'latex': '\\bar\\nu_\\mu+p\\longrightarrow \\mu^+ + n', 'note': ''},
             {'type': 'eq',
              'num': '(3.34)',
              'latex': '\\left\\langle\\frac{d\\sigma}{dQ^2}\\right\\rangle_{\\Phi,i}\\equiv t_i=\\frac{1}{\\Delta '
                       'Q_i^2}\\frac{1}{\\Phi}\\int '
                       'dE_{\\bar\\nu}\\,\\phi(E_{\\bar\\nu})\\int_{Q^2_{i,\\mathrm{low}}}^{Q^2_{i,\\mathrm{high}}}dQ^2\\frac{d\\sigma}{dQ^2}(E_{\\bar\\nu},Q^2)',
              'note': ''},
             {'type': 'eq', 'num': '(3.35)', 'latex': '\\Phi=\\int dE_{\\bar\\nu}\\,\\phi(E_{\\bar\\nu})', 'note': ''},
             {'type': 'eq',
              'num': '(3.36)',
              'latex': '\\Delta Q_i^2=Q^2_{i,\\mathrm{high}}-Q^2_{i,\\mathrm{low}}',
              'note': ''},
             {'type': 'eq',
              'num': '(3.37)',
              'latex': '1.5\\,\\mathrm{GeV}<E_\\mu<20\\,\\mathrm{GeV},\\qquad \\theta_\\mu<20^\\circ',
              'note': ''},
             {'type': 'eq',
              'num': '(3.38)',
              'latex': '\\chi^2=\\sum_{i,j}(\\sigma_i^{\\mathrm{exp}}-\\sigma_i^{\\mathrm{th}})(V^{-1})_{ij}(\\sigma_j^{\\mathrm{exp}}-\\sigma_j^{\\mathrm{th}})',
              'note': ''},
             {'type': 'table',
              'text': '| Modelo axial | chi2_tot | chi2_tot/ndof | chi2_stat/ndof |\n'
                      '|---|---:|---:|---:|\n'
                      '| Dipolo estándar | 14.49 | 0.97 | 4.20 |\n'
                      '| Monopolar | 13.19 | 0.88 | 1.99 |\n'
                      '| Dos componentes, Soft-Pion | 13.17 | 0.88 | 3.42 |\n'
                      '| BBBA2007 | 12.07 | 0.80 | 3.82 |\n'
                      '| Expansión z | 12.87 | 0.86 | 3.46 |'},
             {'type': 'info',
              'text': 'Lectura física: usando la covarianza total, todas las parametrizaciones consideradas son '
                      'globalmente compatibles con MINERvA. Las diferencias no permiten elegir de forma concluyente '
                      'una parametrización axial única, aunque el observable es sensible a la forma funcional de '
                      'GA.'}]},
 {'key': 'appendix_A',
  'title': 'Apéndice A. Identidades auxiliares',
  'subtitle': 'Ondas planas, Gordon, integrales espacio-temporales y flujo incidente.',
  'blocks': [{'type': 'h3', 'text': 'A.1. Ondas planas y espinores de Dirac'},
             {'type': 'eq',
              'num': '(A.1)',
              'latex': '\\psi(X)=\\sqrt{\\frac{m}{EV}}\\,u(p,s)e^{-ip\\cdot X}',
              'note': ''},
             {'type': 'eq', 'num': '(A.2)', 'latex': '(\\not{p}-m)u(p,s)=0', 'note': ''},
             {'type': 'eq', 'num': '(A.3)', 'latex': '\\bar u(p,s)\\equiv u^\\dagger(p,s)\\gamma^0', 'note': ''},
             {'type': 'eq', 'num': '(A.4)', 'latex': '\\bar u(p,s)u(p,s)=1', 'note': ''},
             {'type': 'eq', 'num': '(A.5)', 'latex': '\\sum_su(p,s)\\bar u(p,s)=\\not{p}+m', 'note': ''},
             {'type': 'h3', 'text': 'A.2. Descomposición de Gordon'},
             {'type': 'eq',
              'num': '(A.6)',
              'latex': '\\bar u(p_f,s_f)\\gamma^\\mu u(p_i,s_i)=\\frac{1}{2M}\\bar '
                       'u(p_f,s_f)\\left[(p_f+p_i)^\\mu+i\\sigma^{\\mu\\nu}(p_f-p_i)_\\nu\\right]u(p_i,s_i)',
              'note': ''},
             {'type': 'eq',
              'num': '(A.7)',
              'latex': '\\sigma^{\\mu\\nu}\\equiv\\frac{i}{2}[\\gamma^\\mu,\\gamma^\\nu]',
              'note': ''},
             {'type': 'eq',
              'num': '(A.8)',
              'latex': '\\bar u(p_f,s_f)i\\sigma^{\\mu\\nu}Q_\\nu u(p_i,s_i)=\\bar '
                       'u(p_f,s_f)\\left[2M\\gamma^\\mu-(p_f+p_i)^\\mu\\right]u(p_i,s_i)',
              'note': ''},
             {'type': 'eq',
              'num': '(A.9)',
              'latex': '\\bar u(p_f,s_f)\\widetilde\\Gamma_V^\\mu u(p_i,s_i)=\\bar '
                       'u(p_f,s_f)\\left[(F_1^V+F_2^V)\\gamma^\\mu-\\frac{F_2^V}{2M}(p_i+p_f)^\\mu\\right]u(p_i,s_i)',
              'note': ''},
             {'type': 'h3', 'text': 'A.3.1. Integración sobre X, Y y Q'},
             {'type': 'eq',
              'num': '(A.10)',
              'latex': 'I\\equiv\\int d^4X\\int d^4Y\\int\\frac{d^4Q}{(2\\pi)^4}e^{iQ\\cdot(X-Y)}e^{i(k_f-k_i)\\cdot '
                       'X}e^{i(p_f-p_i)\\cdot Y}',
              'note': ''},
             {'type': 'eq',
              'num': '(A.11)',
              'latex': 'I=\\int d^4X\\int d^4Y\\int\\frac{d^4Q}{(2\\pi)^4}e^{i(Q+k_f-k_i)\\cdot '
                       'X}e^{i(p_f-p_i-Q)\\cdot Y}',
              'note': ''},
             {'type': 'eq',
              'num': '(A.12)',
              'latex': '\\int d^4X\\,e^{iK\\cdot X}=(2\\pi)^4\\delta^{(4)}(K)',
              'note': ''},
             {'type': 'eq',
              'num': '(A.13)',
              'latex': '\\int d^4X\\,e^{i(Q+k_f-k_i)\\cdot X}=(2\\pi)^4\\delta^{(4)}(Q+k_f-k_i)',
              'note': ''},
             {'type': 'eq',
              'num': '(A.14)',
              'latex': '\\int d^4Y\\,e^{i(p_f-p_i-Q)\\cdot Y}=(2\\pi)^4\\delta^{(4)}(p_f-p_i-Q)',
              'note': ''},
             {'type': 'eq',
              'num': '(A.15)',
              'latex': 'I=\\int\\frac{d^4Q}{(2\\pi)^4}(2\\pi)^4\\delta^{(4)}(Q+k_f-k_i)(2\\pi)^4\\delta^{(4)}(p_f-p_i-Q)',
              'note': ''},
             {'type': 'eq', 'num': '(A.16)', 'latex': 'Q^\\mu=k_i^\\mu-k_f^\\mu', 'note': ''},
             {'type': 'eq',
              'num': '(A.17)',
              'latex': '\\delta^{(4)}(p_f-p_i-Q)\\to\\delta^{(4)}(p_f-p_i+k_f-k_i)',
              'note': ''},
             {'type': 'eq', 'num': '(A.18)', 'latex': 'I=(2\\pi)^4\\delta^{(4)}(k_i+p_i-k_f-p_f)', 'note': ''},
             {'type': 'eq', 'num': '(A.19)', 'latex': 'k_i^\\mu+p_i^\\mu=k_f^\\mu+p_f^\\mu', 'note': ''},
             {'type': 'h3', 'text': 'A.3.2. Reducción de la delta al cuadrado'},
             {'type': 'eq',
              'num': '(A.20)',
              'latex': '\\left[(2\\pi)^4\\delta^{(4)}(P)\\right]^2,\\qquad P^\\mu\\equiv '
                       'k_i^\\mu+p_i^\\mu-k_f^\\mu-p_f^\\mu',
              'note': ''},
             {'type': 'eq',
              'num': '(A.21)',
              'latex': '\\delta^{(4)}(P)\\delta^{(4)}(P)=\\delta^{(4)}(P)\\delta^{(4)}(0)',
              'note': ''},
             {'type': 'eq', 'num': '(A.22)', 'latex': '\\delta^{(4)}(0)=\\delta(0)\\delta^{(3)}(0)', 'note': ''},
             {'type': 'eq',
              'num': '(A.23)',
              'latex': '\\delta(0)=\\frac{T}{2\\pi},\\qquad \\delta^{(3)}(0)=\\frac{V}{(2\\pi)^3}',
              'note': ''},
             {'type': 'eq', 'num': '(A.24)', 'latex': '\\delta^{(4)}(0)=\\frac{VT}{(2\\pi)^4}', 'note': ''},
             {'type': 'eq',
              'num': '(A.25)',
              'latex': '\\left[(2\\pi)^4\\delta^{(4)}(P)\\right]^2=(2\\pi)^4VT\\delta^{(4)}(P)',
              'note': ''},
             {'type': 'eq', 'num': '(A.26)', 'latex': 'W_{fi}\\equiv\\frac{|S_{fi}|^2}{VT}', 'note': ''},
             {'type': 'h3', 'text': 'A.3.3. Integración sobre el momento del nucleón final'},
             {'type': 'eq',
              'num': '(A.27)',
              'latex': '\\int\\frac{d^3\\mathbf{p}_f}{2E_f}\\delta^{(4)}(p_f-p_i-Q)',
              'note': ''},
             {'type': 'eq',
              'num': '(A.28)',
              'latex': '\\delta^{(4)}(p_f-p_i-Q)=\\delta(E_f-E_i-\\omega)\\delta^{(3)}(\\mathbf{p}_f-\\mathbf{p}_i-\\mathbf{q})',
              'note': ''},
             {'type': 'eq',
              'num': '(A.29)',
              'latex': '\\int\\frac{d^3\\mathbf{p}_f}{2E_f}\\delta(E_f-M-\\omega)\\delta^{(3)}(\\mathbf{p}_f-\\mathbf{q})=\\frac{1}{2E_f}\\delta(E_f-M-\\omega)',
              'note': ''},
             {'type': 'eq',
              'num': '(A.30)',
              'latex': 'E_f=\\sqrt{M^2+|\\mathbf{p}_f|^2}=\\sqrt{M^2+|\\mathbf{q}|^2}',
              'note': ''},
             {'type': 'eq', 'num': '(A.31)', 'latex': '(M+\\omega)^2=E_f^2=M^2+|\\mathbf{q}|^2', 'note': ''},
             {'type': 'eq', 'num': '(A.32)', 'latex': 'M^2+2M\\omega+\\omega^2=M^2+|\\mathbf{q}|^2', 'note': ''},
             {'type': 'eq', 'num': '(A.33)', 'latex': '2M\\omega+\\omega^2-|\\mathbf{q}|^2=0', 'note': ''},
             {'type': 'eq', 'num': '(A.34)', 'latex': 'Q^2=\\omega^2-|\\mathbf{q}|^2', 'note': ''},
             {'type': 'eq', 'num': '(A.35)', 'latex': 'Q^2+2M\\omega=0', 'note': ''},
             {'type': 'eq',
              'num': '(A.36)',
              'latex': '\\delta(E_f-M-\\omega)=\\delta(f(\\omega)),\\qquad f(\\omega)=E_f-M-\\omega',
              'note': ''},
             {'type': 'eq',
              'num': '(A.37)',
              'latex': "\\delta(f(x))=\\sum_i\\frac{\\delta(x-x_i)}{|f'(x_i)|}",
              'note': ''},
             {'type': 'eq',
              'num': '(A.38)',
              'latex': '\\delta(E_f-M-\\omega)=\\delta\\!\\left(\\frac{E_f^2-(M+\\omega)^2}{E_f+M+\\omega}\\right)=(E_f+M+\\omega)\\delta(E_f^2-(M+\\omega)^2)',
              'note': ''},
             {'type': 'eq',
              'num': '(A.39)',
              'latex': '\\delta(E_f-M-\\omega)=2E_f\\delta(E_f^2-(M+\\omega)^2)',
              'note': ''},
             {'type': 'eq',
              'num': '(A.40)',
              'latex': '\\delta(E_f-M-\\omega)=2E_f\\delta(|\\mathbf{q}|^2-\\omega^2-2M\\omega)=2E_f\\delta(-Q^2-2M\\omega)',
              'note': ''},
             {'type': 'eq',
              'num': '(A.41)',
              'latex': '\\frac{1}{2E_f}\\delta(E_f-M-\\omega)=\\delta(Q^2+2M\\omega)',
              'note': ''},
             {'type': 'eq',
              'num': '(A.42)',
              'latex': '\\delta(Q^2+2M\\omega)=\\delta(2M\\omega-|Q^2|)=\\frac{1}{2M}\\delta\\!\\left(\\omega-\\frac{|Q^2|}{2M}\\right)',
              'note': ''},
             {'type': 'eq',
              'num': '(A.43)',
              'latex': '\\int\\frac{d^3\\mathbf{p}_f}{2E_f}\\delta^{(4)}(p_f-p_i-Q)=\\delta(Q^2+2M\\omega)=\\frac{1}{2M}\\delta\\!\\left(\\omega-\\frac{|Q^2|}{2M}\\right)',
              'note': ''},
             {'type': 'h3', 'text': 'A.4. Flujo incidente'},
             {'type': 'eq',
              'num': '(A.44)',
              'latex': '\\Phi_{\\mathrm{inc}}=\\frac{|\\mathbf{v}_i-\\mathbf{V}_i|}{V}',
              'note': ''}]},
 {'key': 'appendix_B',
  'title': 'Apéndice B. Cálculo de trazas y contracciones',
  'subtitle': 'Identidades de Dirac y derivación de tensores.',
  'blocks': [{'type': 'h3', 'text': 'B.1. Convenciones e identidades de trazas'},
             {'type': 'eq',
              'num': '(B.1)',
              'latex': '\\{\\gamma^\\mu,\\gamma^\\nu\\}=2g^{\\mu\\nu},\\qquad '
                       '\\gamma^5=i\\gamma^0\\gamma^1\\gamma^2\\gamma^3,\\qquad \\epsilon^{0123}=+1',
              'note': ''},
             {'type': 'eq',
              'num': '(B.2)',
              'latex': '\\not{a}\\equiv\\gamma_\\alpha a^\\alpha,\\qquad '
                       '\\sigma^{\\mu\\nu}\\equiv\\frac{i}{2}[\\gamma^\\mu,\\gamma^\\nu]',
              'note': ''},
             {'type': 'eq',
              'num': '(B.3)',
              'latex': '\\mathrm{Tr}[\\gamma^\\mu\\gamma^\\nu]=4g^{\\mu\\nu}',
              'note': ''},
             {'type': 'eq',
              'num': '(B.4)',
              'latex': '\\mathrm{Tr}[\\gamma^\\mu\\gamma^\\nu\\gamma^\\rho\\gamma^\\sigma]=4(g^{\\mu\\nu}g^{\\rho\\sigma}-g^{\\mu\\rho}g^{\\nu\\sigma}+g^{\\mu\\sigma}g^{\\nu\\rho})',
              'note': ''},
             {'type': 'eq',
              'num': '(B.5)',
              'latex': '\\mathrm{Tr}[\\gamma^\\mu\\gamma^\\nu\\gamma^\\rho\\gamma^\\sigma\\gamma^5]=-4i\\epsilon^{\\mu\\nu\\rho\\sigma}',
              'note': ''},
             {'type': 'eq',
              'num': '(B.6)',
              'latex': '\\begin{aligned}\\mathrm{Tr}[\\gamma^{\\mu_1}\\gamma^{\\mu_2}\\gamma^{\\mu_3}\\gamma^{\\mu_4}\\gamma^{\\mu_5}\\gamma^{\\mu_6}]=4(&g^{\\mu_1\\mu_2}g^{\\mu_3\\mu_4}g^{\\mu_5\\mu_6}-g^{\\mu_1\\mu_2}g^{\\mu_3\\mu_5}g^{\\mu_4\\mu_6}+g^{\\mu_1\\mu_2}g^{\\mu_3\\mu_6}g^{\\mu_4\\mu_5}\\\\&-g^{\\mu_1\\mu_3}g^{\\mu_2\\mu_4}g^{\\mu_5\\mu_6}+g^{\\mu_1\\mu_3}g^{\\mu_2\\mu_5}g^{\\mu_4\\mu_6}-g^{\\mu_1\\mu_3}g^{\\mu_2\\mu_6}g^{\\mu_4\\mu_5}+\\cdots '
                       ')\\end{aligned}',
              'note': ''},
             {'type': 'eq',
              'num': '(B.7)',
              'latex': '\\sigma^{\\mu\\alpha}Q_\\alpha=\\frac{i}{2}(\\gamma^\\mu\\not{Q}-\\not{Q}\\gamma^\\mu)=i(\\gamma^\\mu\\not{Q}-Q^\\mu)',
              'note': ''},
             {'type': 'eq',
              'num': '(B.8)',
              'latex': '\\mathrm{Tr}(\\sigma^{\\mu\\alpha}\\sigma^{\\nu\\beta})=4(g^{\\mu\\nu}g^{\\alpha\\beta}-g^{\\mu\\beta}g^{\\alpha\\nu})',
              'note': ''},
             {'type': 'eq',
              'num': '(B.9)',
              'latex': '\\mathrm{Tr}(\\sigma^{\\mu\\alpha}Q_\\alpha\\sigma^{\\nu\\beta}Q_\\beta)=4(g^{\\mu\\nu}Q^2-Q^\\mu '
                       'Q^\\nu)',
              'note': ''},
             {'type': 'eq',
              'num': '(B.10)',
              'latex': '\\epsilon_{\\alpha\\mu\\beta\\nu}\\epsilon^{\\mu\\nu\\rho\\lambda}=-2(\\delta_\\alpha^{\\ '
                       '\\rho}\\delta_\\beta^{\\ \\lambda}-\\delta_\\alpha^{\\ \\lambda}\\delta_\\beta^{\\ \\rho})',
              'note': ''},
             {'type': 'h3', 'text': 'B.2. Tensor leptónico'},
             {'type': 'eq', 'num': '(B.11)', 'latex': '\\sum_su(p,s)\\bar u(p,s)=\\frac{\\not{p}+m}{2m}', 'note': ''},
             {'type': 'eq',
              'num': '(B.12)',
              'latex': '\\eta^{\\mu\\nu}=\\mathrm{Tr}\\!\\left[\\frac{\\not{k}_f+m_l}{2m_l}\\widetilde\\gamma^\\mu\\frac{\\not{k}_i+m_\\nu}{2m_\\nu}\\widetilde\\gamma^\\nu\\right]=\\frac{1}{4m_lm_\\nu}T^{\\mu\\nu}',
              'note': ''},
             {'type': 'eq',
              'num': '(B.13)',
              'latex': '\\begin{aligned}T^{\\mu\\nu}=&\\mathrm{Tr}[\\not{k}_f\\gamma^\\mu(1-\\gamma^5)\\not{k}_i\\gamma^\\nu(1-\\gamma^5)]+m_\\nu\\mathrm{Tr}[\\not{k}_f\\gamma^\\mu(1-\\gamma^5)\\gamma^\\nu(1-\\gamma^5)]\\\\&+m_l\\mathrm{Tr}[\\gamma^\\mu(1-\\gamma^5)\\not{k}_i\\gamma^\\nu(1-\\gamma^5)]+m_lm_\\nu\\mathrm{Tr}[\\gamma^\\mu(1-\\gamma^5)\\gamma^\\nu(1-\\gamma^5)].\\end{aligned}',
              'note': ''},
             {'type': 'eq',
              'num': '(B.14)',
              'latex': '(1-\\gamma^5)\\gamma^\\nu(1-\\gamma^5)=\\gamma^\\nu(1+\\gamma^5)(1-\\gamma^5)=0',
              'note': ''},
             {'type': 'eq',
              'num': '(B.15)',
              'latex': 'T^{\\mu\\nu}=\\mathrm{Tr}[\\not{k}_f\\gamma^\\mu(1-\\gamma^5)\\not{k}_i\\gamma^\\nu(1-\\gamma^5)]',
              'note': ''},
             {'type': 'eq',
              'num': '(B.16)',
              'latex': '(1-\\gamma^5)\\not{k}_i=\\not{k}_i(1+\\gamma^5),\\qquad '
                       '(1+\\gamma^5)\\gamma^\\nu=\\gamma^\\nu(1-\\gamma^5)',
              'note': ''},
             {'type': 'eq',
              'num': '(B.17)',
              'latex': 'T^{\\mu\\nu}=2\\mathrm{Tr}[\\not{k}_f\\gamma^\\mu\\not{k}_i\\gamma^\\nu(1-\\gamma^5)]=2\\mathrm{Tr}[\\not{k}_f\\gamma^\\mu\\not{k}_i\\gamma^\\nu]-2\\mathrm{Tr}[\\not{k}_f\\gamma^\\mu\\not{k}_i\\gamma^\\nu\\gamma^5]',
              'note': ''},
             {'type': 'eq',
              'num': '(B.18)',
              'latex': '\\mathrm{Tr}[\\not{k}_f\\gamma^\\mu\\not{k}_i\\gamma^\\nu]=4(k_f^\\mu k_i^\\nu+k_f^\\nu '
                       'k_i^\\mu-(k_f\\cdot k_i)g^{\\mu\\nu})',
              'note': ''},
             {'type': 'eq',
              'num': '(B.19)',
              'latex': '\\mathrm{Tr}[\\not{k}_f\\gamma^\\mu\\not{k}_i\\gamma^\\nu\\gamma^5]=-4i\\epsilon^{\\alpha\\mu\\beta\\nu}k_{f\\alpha}k_{i\\beta}',
              'note': ''},
             {'type': 'eq',
              'num': '(B.20)',
              'latex': 'T^{\\mu\\nu}=8\\left(k_f^\\mu k_i^\\nu+k_f^\\nu k_i^\\mu-(k_f\\cdot '
                       'k_i)g^{\\mu\\nu}+i\\epsilon^{\\alpha\\mu\\beta\\nu}k_{f\\alpha}k_{i\\beta}\\right)',
              'note': ''},
             {'type': 'eq',
              'num': '(B.21)',
              'latex': '\\widetilde\\eta^{\\mu\\nu}=2\\left[k_f^\\mu k_i^\\nu+k_f^\\nu k_i^\\mu-(k_f\\cdot '
                       'k_i)g^{\\mu\\nu}+i\\epsilon^{\\alpha\\mu\\beta\\nu}k_{f\\alpha}k_{i\\beta}\\right]',
              'note': ''},
             {'type': 'h3', 'text': 'B.3-B.6. Tensor hadrónico y descomposición'},
             {'type': 'eq',
              'num': '(B.22)',
              'latex': '\\widetilde '
                       'W^{\\mu\\nu}=\\frac{1}{8M^2}\\mathrm{Tr}[(\\not{p}_f+M)\\widetilde\\Gamma^\\mu(\\not{p}_i+M)\\widetilde\\Gamma^\\nu]',
              'note': ''},
             {'type': 'eq',
              'num': '(B.23)',
              'latex': '\\widetilde\\Gamma^\\mu=F_1^V\\gamma^\\mu+i\\frac{F_2^V}{2M}\\sigma^{\\mu\\alpha}Q_\\alpha+G_A\\gamma^\\mu\\gamma^5+F_PQ^\\mu\\gamma^5',
              'note': ''},
             {'type': 'eq',
              'num': '(B.24)',
              'latex': '\\widetilde\\Gamma^\\nu=F_1^V\\gamma^\\nu-i\\frac{F_2^V}{2M}\\sigma^{\\nu\\beta}Q_\\beta+G_A\\gamma^\\nu\\gamma^5-F_PQ^\\nu\\gamma^5',
              'note': ''},
             {'type': 'eq',
              'num': '(B.25)',
              'latex': '\\widetilde\\Gamma_V^\\mu=F_1^V\\gamma^\\mu+i\\frac{F_2^V}{2M}\\sigma^{\\mu\\alpha}Q_\\alpha,\\qquad '
                       '\\widetilde\\Gamma_A^\\mu=G_A\\gamma^\\mu\\gamma^5+F_PQ^\\mu\\gamma^5',
              'note': ''},
             {'type': 'eq',
              'num': '(B.26)',
              'latex': '\\widetilde W^{\\mu\\nu}=\\widetilde W_V^{\\mu\\nu}+\\widetilde W_A^{\\mu\\nu}+\\widetilde '
                       'W_{VA}^{\\mu\\nu}',
              'note': ''},
             {'type': 'eq',
              'num': '(B.27)',
              'latex': '\\widetilde '
                       'W_V^{\\mu\\nu}=\\frac{1}{8M^2}\\left[(F_1^V)^2f_{11}^{\\mu\\nu}+\\frac{(F_2^V)^2}{4M^2}f_{22}^{\\mu\\nu}+i\\frac{F_1^VF_2^V}{2M}f_{12}^{\\mu\\nu}\\right]',
              'note': ''},
             {'type': 'eq',
              'num': '(B.28)-(B.30)',
              'latex': 'f_{11}^{\\mu\\nu}=\\mathrm{Tr}[(\\not{p}_f+M)\\gamma^\\mu(\\not{p}_i+M)\\gamma^\\nu]=4[p_f^\\mu '
                       'p_i^\\nu+p_f^\\nu p_i^\\mu+(M^2-p_f\\cdot p_i)g^{\\mu\\nu}]',
              'note': ''},
             {'type': 'eq',
              'num': '(B.31)-(B.38)',
              'latex': '\\begin{aligned}f_{22}^{\\mu\\nu}=4\\Big[&-Q^2(p_f^\\mu p_i^\\nu+p_f^\\nu p_i^\\mu)+(p_f\\cdot '
                       'Q)(Q^\\mu p_i^\\nu+Q^\\nu p_i^\\mu)+(p_i\\cdot Q)(Q^\\mu p_f^\\nu+Q^\\nu '
                       'p_f^\\mu)\\\\&-(p_f\\cdot p_i)Q^\\mu Q^\\nu+g^{\\mu\\nu}(Q^2(p_f\\cdot p_i)-2(p_f\\cdot '
                       'Q)(p_i\\cdot Q))+M^2(g^{\\mu\\nu}Q^2-Q^\\mu Q^\\nu)\\Big]\\end{aligned}',
              'note': ''},
             {'type': 'eq',
              'num': '(B.39)-(B.40)',
              'latex': 'f_{12}^{\\mu\\nu}=4iM[p_f^\\mu Q^\\nu+Q^\\mu p_f^\\nu-p_i^\\mu Q^\\nu-Q^\\mu '
                       'p_i^\\nu-2g^{\\mu\\nu}(p_f\\cdot Q)+2g^{\\mu\\nu}(p_i\\cdot Q)]',
              'note': ''},
             {'type': 'eq',
              'num': '(B.41)-(B.48)',
              'latex': '\\widetilde '
                       'W_A^{\\mu\\nu}=\\frac{1}{8M^2}(G_A^2f_{33}^{\\mu\\nu}+F_P^2f_{44}^{\\mu\\nu}+G_AF_Pf_{34}^{\\mu\\nu})',
              'note': ''},
             {'type': 'eq',
              'num': 'f33-f44-f34',
              'latex': '\\begin{gathered}f_{33}^{\\mu\\nu}=4[p_f^\\mu p_i^\\nu+p_f^\\nu '
                       'p_i^\\mu-g^{\\mu\\nu}(p_f\\cdot p_i+M^2)],\\\\ f_{44}^{\\mu\\nu}=4(p_f\\cdot p_i-M^2)Q^\\mu '
                       'Q^\\nu,\\\\ f_{34}^{\\mu\\nu}=4M[-Q^\\mu p_f^\\nu-Q^\\nu p_f^\\mu+p_i^\\mu Q^\\nu+p_i^\\nu '
                       'Q^\\mu].\\end{gathered}',
              'note': ''},
             {'type': 'eq',
              'num': '(B.49)',
              'latex': '\\widetilde '
                       'W_{VA}^{\\mu\\nu}=\\frac{1}{8M^2}\\left[F_1^VG_Af_{45}^{\\mu\\nu}+i\\frac{F_2^V}{2M}G_Af_{56}^{\\mu\\nu}+F_1^VF_Pf_{67}^{\\mu\\nu}+i\\frac{F_2^V}{2M}F_Pf_{78}^{\\mu\\nu}\\right]',
              'note': ''},
             {'type': 'eq',
              'num': '(B.50)-(B.55)',
              'latex': '\\begin{gathered}f_{45}^{\\mu\\nu}=8ip_{f\\alpha}p_{i\\beta}\\epsilon^{\\nu\\mu\\alpha\\beta},\\\\ '
                       'f_{56}^{\\mu\\nu}=8M(p_{f\\tau}Q_\\rho\\epsilon^{\\tau\\mu\\nu\\rho}+p_{i\\tau}Q_\\rho\\epsilon^{\\rho\\mu\\tau\\nu}),\\\\ '
                       'f_{67}^{\\mu\\nu}=0,\\\\ f_{78}^{\\mu\\nu}=4p_{f\\rho}p_{i\\sigma}Q^\\nu '
                       'Q_\\delta\\epsilon^{\\rho\\mu\\delta\\sigma}-4p_{f\\rho}p_{i\\sigma}Q^\\mu '
                       'Q_\\delta\\epsilon^{\\rho\\sigma\\mu\\delta}.\\end{gathered}',
              'note': ''},
             {'type': 'h3', 'text': 'B.7. Simplificación cinemática de las funciones f_i'},
             {'type': 'eq',
              'num': '(B.56)',
              'latex': 'P_f^2=P_i^2=M^2,\\qquad Q^\\mu=P_f^\\mu-P_i^\\mu,\\qquad |Q^2|=-Q^2=2P_i\\cdot P_f-2M^2',
              'note': ''},
             {'type': 'eq',
              'num': '(B.57)',
              'latex': 'P_i\\cdot Q=\\frac{|Q^2|}{2},\\qquad P_f\\cdot Q=-\\frac{|Q^2|}{2},\\qquad P_i\\cdot '
                       'P_f=M^2+\\frac{|Q^2|}{2}',
              'note': ''},
             {'type': 'eq',
              'num': '(B.58)',
              'latex': 'X^\\mu\\equiv P_i^\\mu+\\frac{P_i\\cdot Q}{|Q^2|}Q^\\mu=P_i^\\mu+\\frac{1}{2}Q^\\mu',
              'note': ''},
             {'type': 'eq',
              'num': '(B.59)',
              'latex': 'P_i^\\mu=X^\\mu-\\frac{1}{2}Q^\\mu,\\qquad P_f^\\mu=X^\\mu+\\frac{1}{2}Q^\\mu',
              'note': ''},
             {'type': 'eq',
              'num': '(B.60)',
              'latex': 'P_f^\\mu P_i^\\nu+P_f^\\nu P_i^\\mu=2X^\\mu X^\\nu-\x0c'
                       'rac{1}{2}Q^\\mu Q^\\nu,\\qquad M^2-P_f\\cdot P_i=-\\frac{|Q^2|}{2}',
              'note': ''},
             {'type': 'eq',
              'num': '(B.61)',
              'latex': 'f_{11}^{\\mu\\nu}=4\\left[2X^\\mu X^\\nu-\\frac{1}{2}Q^\\mu '
                       'Q^\\nu-\\frac{|Q^2|}{2}g^{\\mu\\nu}\\right]=-2|Q^2|\\left(g^{\\mu\\nu}+\\frac{Q^\\mu '
                       'Q^\\nu}{|Q^2|}\\right)+8X^\\mu X^\\nu',
              'note': ''},
             {'type': 'eq',
              'num': '(B.62)',
              'latex': 'f_{11}^{\\mu\\nu}=-2|Q^2|\\left(g^{\\mu\\nu}+\\frac{Q^\\mu Q^\\nu}{|Q^2|}\\right)+8X^\\mu '
                       'X^\\nu',
              'note': ''},
             {'type': 'eq',
              'num': '(B.63)',
              'latex': 'f_{22}^{\\mu\\nu}=-8M^2|Q^2|\\left(g^{\\mu\\nu}+\\frac{Q^\\mu '
                       'Q^\\nu}{|Q^2|}\\right)+8|Q^2|X^\\mu X^\\nu',
              'note': ''},
             {'type': 'eq',
              'num': '(B.64)',
              'latex': 'f_{12}^{\\mu\\nu}=8iM|Q^2|\\left(g^{\\mu\\nu}+\\frac{Q^\\mu Q^\\nu}{|Q^2|}\\right)',
              'note': ''},
             {'type': 'eq',
              'num': '(B.65)',
              'latex': 'f_{33}^{\\mu\\nu}=-2|Q^2|\\left(g^{\\mu\\nu}+\\frac{Q^\\mu Q^\\nu}{|Q^2|}\\right)+8X^\\mu '
                       'X^\\nu-8M^2g^{\\mu\\nu}',
              'note': ''},
             {'type': 'eq',
              'num': '(B.66)',
              'latex': 'f_{44}^{\\mu\\nu}=2|Q^2|Q^\\mu Q^\\nu,\\qquad f_{34}^{\\mu\\nu}=-8MQ^\\mu Q^\\nu',
              'note': ''},
             {'type': 'eq',
              'num': '(B.67)',
              'latex': 'f_{45}^{\\mu\\nu}=8i\\epsilon^{\\nu\\mu\\alpha\\beta}P_{i\\beta}Q_\\alpha,\\qquad '
                       'f_{56}^{\\mu\\nu}=16M\\epsilon^{\\rho\\mu\\tau\\nu}P_{i\\tau}Q_\\rho',
              'note': ''},
             {'type': 'eq', 'num': '(B.68)', 'latex': 'f_{67}^{\\mu\\nu}=f_{78}^{\\mu\\nu}=0', 'note': ''}]},
 {'key': 'appendix_C',
  'title': 'Apéndice C. Detalles cinemáticos de la sección eficaz integrada',
  'subtitle': 'Transformación de la delta, integración angular y jacobiano.',
  'blocks': [{'type': 'eq',
              'num': '(C.1)',
              'latex': '\\frac{d\\sigma}{dE_\\mu '
                       'd\\Omega}=\\frac{|\\mathbf{k}_\\mu|}{E_\\nu}\\frac{G_F^2}{4\\pi^2}\\delta\\!\\left(\\omega-\\frac{|Q^2|}{2M}\\right)\\widetilde\\eta_{\\mu\\nu}\\widetilde '
                       'W^{\\mu\\nu}',
              'note': ''},
             {'type': 'eq',
              'num': '(C.2)',
              'latex': '|\\mathbf{q}|^2=E_\\nu^2+|\\mathbf{k}_\\mu|^2-2E_\\nu|\\mathbf{k}_\\mu|\\cos\\theta_\\mu',
              'note': ''},
             {'type': 'eq',
              'num': '(C.3)',
              'latex': '|Q^2|=E_\\nu^2+|\\mathbf{k}_\\mu|^2-2E_\\nu|\\mathbf{k}_\\mu|\\cos\\theta_\\mu-(E_\\nu-E_\\mu)^2',
              'note': ''},
             {'type': 'eq', 'num': '(C.4)', 'latex': '(E_\\nu-E_\\mu)^2=E_\\nu^2+E_\\mu^2-2E_\\nu E_\\mu', 'note': ''},
             {'type': 'eq',
              'num': '(C.5)',
              'latex': '|Q^2|=|\\mathbf{k}_\\mu|^2-E_\\mu^2+2E_\\nu E_\\mu-2E_\\nu|\\mathbf{k}_\\mu|\\cos\\theta_\\mu',
              'note': ''},
             {'type': 'eq',
              'num': '(C.6)',
              'latex': '|Q^2|=-m_\\mu^2+2E_\\nu E_\\mu-2E_\\nu|\\mathbf{k}_\\mu|\\cos\\theta_\\mu',
              'note': ''},
             {'type': 'eq', 'num': '(C.7)', 'latex': '2M\\omega-|Q^2|=0', 'note': ''},
             {'type': 'eq',
              'num': '(C.8)',
              'latex': '2ME_\\nu-2ME_\\mu+m_\\mu^2-2E_\\nu E_\\mu+2E_\\nu|\\mathbf{k}_\\mu|\\cos\\theta_\\mu=0',
              'note': ''},
             {'type': 'eq',
              'num': '(C.9)',
              'latex': '\\cos\\theta_\\mu=\\frac{2(E_\\nu E_\\mu+ME_\\mu-ME_\\nu)-m_\\mu^2}{2E_\\nu|\\mathbf{k}_\\mu|}',
              'note': ''},
             {'type': 'eq',
              'num': '(C.10)',
              'latex': '\\cos\\theta_0\\equiv\\frac{2(E_\\nu '
                       'E_\\mu+ME_\\mu-ME_\\nu)-m_\\mu^2}{2E_\\nu|\\mathbf{k}_\\mu|}',
              'note': ''},
             {'type': 'eq', 'num': '(C.11)', 'latex': 'f(\\cos\\theta_\\mu)\\equiv2M\\omega-|Q^2|', 'note': ''},
             {'type': 'eq',
              'num': '(C.12)',
              'latex': 'f(\\cos\\theta_\\mu)=2M(E_\\nu-E_\\mu)+m_\\mu^2-2E_\\nu '
                       'E_\\mu+2E_\\nu|\\mathbf{k}_\\mu|\\cos\\theta_\\mu',
              'note': ''},
             {'type': 'eq',
              'num': '(C.13)',
              'latex': '\\frac{df}{d(\\cos\\theta_\\mu)}=2E_\\nu|\\mathbf{k}_\\mu|',
              'note': ''},
             {'type': 'eq',
              'num': '(C.14)',
              'latex': '\\delta(2M\\omega-|Q^2|)=\\frac{1}{2E_\\nu|\\mathbf{k}_\\mu|}\\delta(\\cos\\theta_\\mu-\\cos\\theta_0)',
              'note': ''},
             {'type': 'eq',
              'num': '(C.15)',
              'latex': '\\frac{d\\sigma}{dE_\\mu '
                       'd\\Omega}=\\frac{G_F^2}{8\\pi^2E_\\nu^2}\\delta(\\cos\\theta_\\mu-\\cos\\theta_0)\\widetilde\\eta_{\\mu\\nu}\\widetilde '
                       'W^{\\mu\\nu}',
              'note': ''},
             {'type': 'eq',
              'num': '(C.16)',
              'latex': '\\frac{d\\sigma}{dE_\\mu}=\\int d\\Omega\\frac{d\\sigma}{dE_\\mu d\\Omega}',
              'note': ''},
             {'type': 'eq',
              'num': '(C.17)',
              'latex': '\\frac{d\\sigma}{dE_\\mu}=\\frac{G_F^2}{8\\pi^2E_\\nu^2}\\int_0^{2\\pi}d\\phi\\int_{-1}^{1}d(\\cos\\theta_\\mu)\\delta(\\cos\\theta_\\mu-\\cos\\theta_0)\\widetilde\\eta_{\\mu\\nu}\\widetilde '
                       'W^{\\mu\\nu}',
              'note': ''},
             {'type': 'eq',
              'num': '(C.18)',
              'latex': '\\frac{d\\sigma}{dE_\\mu}=\\left.\\frac{G_F^2}{4\\pi '
                       'E_\\nu^2}\\widetilde\\eta_{\\mu\\nu}\\widetilde '
                       'W^{\\mu\\nu}\\right|_{\\cos\\theta_\\mu=\\cos\\theta_0}',
              'note': ''},
             {'type': 'eq',
              'num': '(C.19)',
              'latex': '\\frac{d\\sigma}{d(\\cos\\theta_\\mu)}=\\frac{d\\sigma}{dE_\\mu}\\left|\\frac{dE_\\mu}{d(\\cos\\theta_\\mu)}\\right|',
              'note': ''},
             {'type': 'eq',
              'num': '(C.20)',
              'latex': '2E_\\nu\\left[\\frac{d|\\mathbf{k}_\\mu|}{d(\\cos\\theta_\\mu)}\\cos\\theta_\\mu+|\\mathbf{k}_\\mu|\\right]=2(E_\\nu+M)\\frac{dE_\\mu}{d(\\cos\\theta_\\mu)}',
              'note': ''},
             {'type': 'eq',
              'num': '(C.21)',
              'latex': '\\frac{d|\\mathbf{k}_\\mu|}{d(\\cos\\theta_\\mu)}=\\frac{E_\\mu}{|\\mathbf{k}_\\mu|}\\frac{dE_\\mu}{d(\\cos\\theta_\\mu)}',
              'note': ''},
             {'type': 'eq',
              'num': '(C.22)',
              'latex': '2E_\\nu\\left[\\frac{E_\\mu}{|\\mathbf{k}_\\mu|}\\cos\\theta_\\mu\\frac{dE_\\mu}{d(\\cos\\theta_\\mu)}+|\\mathbf{k}_\\mu|\\right]=2(E_\\nu+M)\\frac{dE_\\mu}{d(\\cos\\theta_\\mu)}',
              'note': ''},
             {'type': 'eq',
              'num': '(C.23)',
              'latex': 'E_\\nu|\\mathbf{k}_\\mu|=\\left[E_\\nu+M-\\frac{E_\\nu '
                       'E_\\mu}{|\\mathbf{k}_\\mu|}\\cos\\theta_\\mu\\right]\\frac{dE_\\mu}{d(\\cos\\theta_\\mu)}',
              'note': ''},
             {'type': 'eq',
              'num': '(C.24)',
              'latex': '\\frac{dE_\\mu}{d(\\cos\\theta_\\mu)}=\\frac{|\\mathbf{k}_\\mu|}{1+\\frac{M}{E_\\nu}-\\frac{E_\\mu}{|\\mathbf{k}_\\mu|}\\cos\\theta_\\mu}',
              'note': ''},
             {'type': 'eq',
              'num': '(C.25)',
              'latex': '\\frac{d\\sigma}{d(\\cos\\theta_\\mu)}=\\frac{G_F^2}{4\\pi '
                       'E_\\nu^2}\\widetilde\\eta_{\\mu\\nu}\\widetilde '
                       'W^{\\mu\\nu}\\frac{|\\mathbf{k}_\\mu|}{1+\\frac{M}{E_\\nu}-\\frac{E_\\mu}{|\\mathbf{k}_\\mu|}\\cos\\theta_\\mu}',
              'note': ''},
             {'type': 'eq',
              'num': '(C.26)',
              'latex': '\\frac{d\\sigma}{d\\Omega}=\\frac{1}{2\\pi}\\frac{d\\sigma}{d(\\cos\\theta_\\mu)}',
              'note': ''},
             {'type': 'eq',
              'num': '(C.27)',
              'latex': '\\frac{d\\sigma}{d\\Omega}=\\left.\\frac{G_F^2}{4\\pi^2}\\frac{|\\mathbf{k}_\\mu|}{E_\\nu}f_{\\mathrm{rec}}^{-1}\\widetilde\\eta_{\\mu\\nu}\\widetilde '
                       'W^{\\mu\\nu}\\right|_{\\cos\\theta_\\mu=\\cos\\theta_0}',
              'note': ''}]},
 {'key': 'appendix_D',
  'title': 'Apéndice D. Tratamiento estadístico de MINERvA',
  'subtitle': 'Reescalado de covarianzas, correlaciones, barras de error y chi cuadrado.',
  'blocks': [{'type': 'h3', 'text': 'D.1. Reescalado de las matrices de covarianza'},
             {'type': 'p',
              'text': 'Los datos de sección eficaz de MINERvA se publican numéricamente en unidades de 10^-38 '
                      'cm^2/GeV^2. Como la covarianza tiene unidades de sección eficaz al cuadrado, debe tratarse con '
                      'una escala coherente.'},
             {'type': 'eq',
              'num': '(D.1)',
              'latex': '(10^{-38}\\,\\mathrm{cm}^2/\\mathrm{GeV}^2)^2=10^{-76}\\,\\mathrm{cm}^4/\\mathrm{GeV}^4',
              'note': ''},
             {'type': 'p',
              'text': 'Si la matriz suplementaria está publicada en unidades de 10^-80, en el TFG se reescala según la '
                      'convención numérica empleada para los datos.'},
             {'type': 'eq', 'num': 'reescalado', 'latex': 'V_{ij}=10^{-4}V_{ij}^{\\mathrm{pub}}', 'note': ''},
             {'type': 'eq', 'num': '(D.2)', 'latex': '\\chi^2=(d-t)^T V^{-1}(d-t)', 'note': ''},
             {'type': 'h3', 'text': 'D.2. Chi cuadrado interna de MINERvA frente a chi cuadrado de este trabajo'},
             {'type': 'p',
              'text': 'La chi cuadrado interna de MINERvA pertenece al procedimiento de extracción experimental. La '
                      'chi cuadrado del TFG mide la compatibilidad entre el vector de datos publicado y una predicción '
                      'teórica concreta.'},
             {'type': 'h3', 'text': 'D.3. Covarianza y correlación'},
             {'type': 'p',
              'text': 'La matriz de covarianza total contiene incertidumbres estadísticas, sistemáticas y '
                      'correlaciones entre bins. La matriz estadística se usa como diagnóstico complementario.'},
             {'type': 'eq', 'num': '(D.3)', 'latex': '\\rho_{ij}=\\frac{V_{ij}}{\\sqrt{V_{ii}V_{jj}}}', 'note': ''},
             {'type': 'h3', 'text': 'D.4. Barras de error'},
             {'type': 'eq',
              'num': '(D.4)',
              'latex': '\\delta_i^{\\mathrm{tot}}=\\sqrt{V_{ii}^{\\mathrm{tot}}}',
              'note': ''},
             {'type': 'eq',
              'num': '(D.5)',
              'latex': '\\delta_i^{\\mathrm{stat}}=\\sqrt{V_{ii}^{\\mathrm{stat}}}',
              'note': ''},
             {'type': 'eq',
              'num': '(D.6)',
              'latex': '\\frac{\\Delta Q_i^2}{2}=\\frac{Q^2_{i,\\mathrm{high}}-Q^2_{i,\\mathrm{low}}}{2}',
              'note': ''},
             {'type': 'p',
              'text': 'En el cociente datos/modelo, Ri=di/ti, si no se asigna incertidumbre teórica al modelo, la '
                      'incertidumbre se propaga como delta_Ri=delta_i/t_i.'},
             {'type': 'eq',
              'num': 'cociente',
              'latex': 'R_i=\\frac{d_i}{t_i},\\qquad \\delta '
                       'R_i^{\\mathrm{tot}}=\\frac{\\delta_i^{\\mathrm{tot}}}{t_i}',
              'note': ''},
             {'type': 'h3', 'text': 'D.5. Implementación de chi cuadrado'},
             {'type': 'eq',
              'num': '(D.7)',
              'latex': '\\chi^2_{\\mathrm{tot}}=\\sum_{i,j}r_i(V^{\\mathrm{tot}})^{-1}_{ij}r_j=r^T(V^{\\mathrm{tot}})^{-1}r,\\qquad '
                       'r_i=d_i-t_i',
              'note': ''},
             {'type': 'eq',
              'num': 'estadística',
              'latex': '\\chi^2_{\\mathrm{stat}}=r^T(V^{\\mathrm{stat}})^{-1}r',
              'note': ''},
             {'type': 'p',
              'text': 'En el TFG se toma ndof=Nbins=15 porque no se ajustan parámetros libres a los datos de MINERvA: '
                      'cada parametrización se evalúa directamente como modelo fijado.'},
             {'type': 'eq', 'num': 'ndof', 'latex': '\\mathrm{ndof}=N_{\\mathrm{bins}}=15', 'note': ''},
             {'type': 'info',
              'text': 'La comparación principal debe hacerse con la covarianza total, no sólo con barras diagonales. '
                      'Las correlaciones entre bins cambian la lectura estadística del acuerdo teoría-datos.'}]},
 {'key': 'references',
  'title': 'Referencias utilizadas en el TFG',
  'subtitle': 'Bibliografía esencial que sostiene el marco teórico y las parametrizaciones.',
  'blocks': [{'type': 'p',
              'text': 'Las referencias completas aparecen en la memoria del TFG. Para la defensa, conviene recordar '
                      'especialmente el papel de cada bloque bibliográfico.'},
             {'type': 'table',
              'text': '| Bloque | Referencias clave | Papel en el TFG |\n'
                      '|---|---|---|\n'
                      '| Formalismo CC y neutrino-nucleón | Megías TFM, Mandl-Shaw, Bjorken-Drell | Corrientes, '
                      'propagador, amplitud, trazas |\n'
                      '| Estructura EM/vector | Sachs, Walecka, González-Jiménez, Kelly, GKeX | Factores de Sachs, '
                      'CVC, parametrizaciones vectoriales |\n'
                      '| Estructura axial | Bernard et al., Ahrens et al., Megías et al. PRC 2020 | Dipolo, masa '
                      'axial, dos componentes, PCAC |\n'
                      '| BBBA2007 | Bodek, Budd, Bradford, Avvakumov | Corrección con variable de Nachtmann |\n'
                      '| Expansión z y LQCD | Petti-Hill-Tomalak, Meyer-Walker-Loud-Wilkinson | Parametrización '
                      'flexible e incertidumbres modernas |\n'
                      '| MINERvA | Nature 2023 y suplemento | Datos H, flujo, covarianzas y chi cuadrado |'},
             ]}]


KEY_TO_SECTION = {s["key"]: s for s in SECTIONS}


def eq(num: str, latex: str, note: str = "") -> None:
    st.markdown(f"<div class='eq-card'><div class='eq-num'>{num}</div></div>", unsafe_allow_html=True)
    st.latex(latex)
    if note:
        st.caption(note)


def render_block(block: dict) -> None:
    t = block["type"]
    if t == "h2":
        st.header(block["text"])
    elif t == "h3":
        st.subheader(block["text"])
    elif t == "p":
        st.markdown(block["text"])
    elif t == "info":
        st.info(block["text"])
    elif t == "warning":
        st.warning(block["text"])
    elif t == "eq":
        eq(block["num"], block["latex"], block.get("note", ""))
    elif t == "table":
        st.markdown(block["text"])
    elif t == "divider":
        st.divider()


def render_section(section: dict) -> None:
    st.title(section["title"])
    if section.get("subtitle"):
        st.caption(section["subtitle"])
    for block in section["blocks"]:
        render_block(block)


def jump_script(anchor: str) -> None:
    components.html(
        f"""
        <script>
        const doc = (window.parent && window.parent.document) ? window.parent.document : document;
        const el = doc.getElementById("{anchor}");
        if (el) {{ el.scrollIntoView({{behavior: "smooth", block: "start"}}); }}
        </script>
        """,
        height=0,
    )

st.sidebar.title("📘 Theoretical Framework")
mode = st.sidebar.radio("Modo de visualización", ["Sección individual", "Todo en una página"], index=0)
labels = [s["title"] for s in SECTIONS]
selected_title = st.sidebar.selectbox("Sección", labels, index=0)
selected = next(s for s in SECTIONS if s["title"] == selected_title)

st.sidebar.divider()
st.sidebar.markdown("**Atajos útiles**")
for label, key in [
    ("Contracción y W3", "tensors"),
    ("Estructura axial", "axial"),
    ("MINERvA y chi²", "appendix_D"),
    ("Apéndice de trazas", "appendix_B"),
]:
    if st.sidebar.button(label, use_container_width=True):
        st.session_state["forced_key"] = key
        st.rerun()

if "forced_key" in st.session_state:
    selected = KEY_TO_SECTION[st.session_state.pop("forced_key")]
    mode = "Sección individual"

if mode == "Todo en una página":
    for s in SECTIONS:
        st.markdown(f"<div id='{s['key']}'></div>", unsafe_allow_html=True)
        render_section(s)
        st.divider()
else:
    render_section(selected)

st.divider()
st.caption("TFG — Estudio de la estructura axial del nucleón en interacciones con neutrinos y antineutrinos.")
