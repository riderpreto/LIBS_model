"""
Módulo de Visualização Espectroscópica e Gráficos Quimiométricos
Projeto: Detecção de Adulteração em Fertilizantes NPK via LIBS
Embrapa Instrumentação
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plotar_comparacao_filtros_ruido(
    wl: np.ndarray,
    sinal_bruto: np.ndarray,
    sinal_wavelet: np.ndarray,
    sinal_savgol: np.ndarray,
    titulo: str = "Comparativo de Pré-Processamento de Ruído",
    xlim: Optional[Tuple[float, float]] = None
) -> plt.Figure:
    """
    Plota o sinal bruto sobreposto aos sinais filtrados por Wavelet e Savitzky-Golay.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(wl, sinal_bruto, color="gray", alpha=0.5, label="Bruto (Sem Filtro)", linewidth=1)
    ax.plot(wl, sinal_wavelet, color="tab:blue", label="Wavelet (sym8, DWT)", linewidth=1.2)
    ax.plot(wl, sinal_savgol, color="tab:orange", label="Savitzky-Golay", linewidth=1.2)

    ax.set_title(titulo, fontsize=13, fontweight="bold")
    ax.set_xlabel("Comprimento de Onda (nm)", fontsize=11)
    ax.set_ylabel("Intensidade (u.a.)", fontsize=11)
    if xlim:
        ax.set_xlim(xlim)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plotar_espectros_medios(
    df: pd.DataFrame,
    col_teor: str = "Teor_Adulteracao",
    titulo: str = "Espectros Médios por Nível de Adulteração NPK",
    cores: Optional[Dict[float, str]] = None,
    cols_metadados: Optional[List[str]] = None
) -> plt.Figure:
    """
    Plota as médias espectrais para cada teor de adulteração (0%, 10%, 25%, 50%).
    """
    if cols_metadados is None:
        cols_metadados = ["Amostra_ID", "Num_Disparo", "Nome_Arquivo", "Canal_Optico", "Teor_Adulteracao", "Adulterado"]

    cols_features = [c for c in df.columns if c not in cols_metadados]
    wls = np.array([float(c) for c in cols_features])

    teores = sorted(df[col_teor].dropna().unique())
    if cores is None:
        palette = sns.color_palette("tab10", len(teores))
        cores = {t: palette[i] for i, t in enumerate(teores)}

    fig, ax = plt.subplots(figsize=(14, 6))

    for t in teores:
        df_sub = df[df[col_teor] == t]
        media = df_sub[cols_features].mean(axis=0).values
        label = f"{t:.1f}% Adulterado" if t > 0 else "0.0% (Puro / Referência)"
        ax.plot(wls, media, label=label, color=cores.get(t, "black"), linewidth=1.3)

    ax.set_title(titulo, fontsize=14, fontweight="bold")
    ax.set_xlabel("Comprimento de Onda (nm)", fontsize=12)
    ax.set_ylabel("Intensidade Normalizada (u.a.)", fontsize=12)
    ax.legend(title="Teor de Adulteração", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plotar_pca_scores(
    df_scores: pd.DataFrame,
    pc_x: str = "PC1",
    pc_y: str = "PC2",
    col_hue: str = "Teor_Adulteracao",
    titulo: str = "Gráfico de Scores PCA"
) -> plt.Figure:
    """
    Plota projeção de scores PCA com rótulos e coloração por adulteração.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    df_plot = df_scores.copy()
    df_plot["Teor_Str"] = df_plot[col_hue].apply(lambda x: f"{float(x):.1f}%")

    sns.scatterplot(
        data=df_plot,
        x=pc_x,
        y=pc_y,
        hue="Teor_Str",
        palette="viridis",
        s=80,
        alpha=0.85,
        edgecolor="black",
        ax=ax
    )

    ax.set_title(titulo, fontsize=13, fontweight="bold")
    ax.set_xlabel(pc_x, fontsize=11)
    ax.set_ylabel(pc_y, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Adulteração", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    return fig


def plotar_loadings_com_linhas_nist(
    wls: np.ndarray,
    loadings_pc: np.ndarray,
    pc_nome: str,
    var_exp: float,
    linhas_emissao: Dict[str, List[float]],
    cores_elementos: Dict[str, str]
) -> plt.Figure:
    """
    Plota os loadings de uma componente principal com marcação vertical das linhas NIST.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(wls, loadings_pc, color="black", linewidth=1.2, label=f"Loading {pc_nome}")

    wl_min, wl_max = wls.min(), wls.max()
    elementos_plotados = set()

    for elemento, lista_wl in linhas_emissao.items():
        cor = cores_elementos.get(elemento, "blue")
        for linha in lista_wl:
            if wl_min <= linha <= wl_max:
                label = elemento if elemento not in elementos_plotados else ""
                ax.axvline(x=linha, color=cor, linestyle="--", alpha=0.6, linewidth=1, label=label)
                elementos_plotados.add(elemento)

    ax.set_title(f"Loadings {pc_nome} ({var_exp:.2f}% Variância) com Linhas NIST", fontsize=13, fontweight="bold")
    ax.set_xlabel("Comprimento de Onda (nm)", fontsize=11)
    ax.set_ylabel("Pesos do Loading", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    return fig


def plotar_curva_rmsec_rmsecv(df_lvs: pd.DataFrame) -> plt.Figure:
    """
    Plota a evolução de RMSEC e RMSECV em função do número de variáveis latentes (LVs).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_lvs["LVs"], df_lvs["RMSEC"], marker="o", label="RMSEC (Calibração)", color="tab:blue")
    ax.plot(df_lvs["LVs"], df_lvs["RMSECV"], marker="s", label="RMSECV (LOGO-CV)", color="tab:red")

    ax.set_title("Otimização de Variáveis Latentes (PLSR)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Número de Variáveis Latentes (LVs)", fontsize=11)
    ax.set_ylabel("Erro Médio Quadrático (RMSE, %)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    return fig


def plotar_calibracao_real_vs_predito(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    r2: float,
    rmse: float,
    titulo: str = "Calibração Multivariada: Real vs Previsto"
) -> plt.Figure:
    """
    Plota o gráfico de dispersão Real vs Predito com reta 1:1 e métricas estatísticas.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_real, y_pred, color="tab:blue", edgecolor="black", alpha=0.7, s=60, label="Predições LOGO-CV")

    limites = [min(np.min(y_real), np.min(y_pred)) - 2, max(np.max(y_real), np.max(y_pred)) + 2]
    ax.plot(limites, limites, "r--", linewidth=1.5, label="Linha Ideal (1:1)")

    ax.set_title(titulo, fontsize=13, fontweight="bold")
    ax.set_xlabel("Teor Real de Adulteração (%)", fontsize=11)
    ax.set_ylabel("Teor Predito (%)", fontsize=11)
    ax.set_xlim(limites)
    ax.set_ylim(limites)
    ax.grid(True, linestyle="--", alpha=0.4)

    texto_metricas = f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f}%"
    ax.text(
        0.05, 0.90, texto_metricas,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig
