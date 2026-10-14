"""
Módulo de Extração de Atributos, Espectros Representativos e Razões Elementares
Projeto: Detecção de Adulteração em Fertilizantes NPK via LIBS
Embrapa Instrumentação

Este módulo implementa:
1. Biblioteca de comprimentos de onda de referência NIST (Ca, K, P, Mg, Fe, C, NH).
2. Agregação de disparos por pastilha (espectro médio representativo).
3. Integração numérica de áreas de pico espectrais (Simpson / Trapezoidal).
4. Padronização interna e cálculo de razões elementares (Ca/K, Ca/P, Mg/P).
5. Métrica de distância interclasse de Cohen (D_C-K) para seleção de razões com maior poder discriminante.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


# ==============================================================================
# 1. BIBLIOTECA DE LINHAS DE EMISSÃO ESPECTRAL (NIST ASD)
# ==============================================================================

LINHAS_EMISSAO_VIS: Dict[str, List[float]] = {
    "Ca II": [396.87, 393.37, 317.97, 315.9, 370.60, 373.69],
    "K": [766.48, 769.92],
    "Mg": [279.58],
    "Fe I": [430.82, 432.54],
    "Ca I": [420.69, 422.67, 430.26, 431.88, 442.56, 443.51, 445.47, 527.03, 612.22, 643.91],
}

CORES_ELEMENTOS_VIS: Dict[str, str] = {
    "Ca II": "purple",
    "K": "orange",
    "Mg": "magenta",
    "Fe I": "gray",
    "Ca I": "red",
}

LINHAS_EMISSAO_UV: Dict[str, List[float]] = {
    "Ca II": [396.87, 393.37, 317.97, 315.9, 370.60, 373.69, 227.55],
    "Mg II": [279.58, 280.3],
    "NH (Molecular)": [336.10, 336.15],  # Assinatura de Nitrogênio molecular
    "Mg I": [285.3],
    "C I": [247.8],
    "Ca I": [239.86, 300.68, 299.49],
    "P I": [213.61, 214.91, 253.56, 255.33],
}

CORES_ELEMENTOS_UV: Dict[str, str] = {
    "Ca II": "purple",
    "Ca I": "red",
    "NH (Molecular)": "blue",
    "Mg II": "green",
    "Mg I": "darkgreen",
    "C I": "black",
    "P I": "gold",
}

PICOS_INTERESSE_VIS: Dict[str, float] = {
    "K_693": 693.96,
    "K_404": 404.46,
    "K_580": 580.20,
    "Ca_527": 527.08,
    "Ca_315": 315.96,
}

PICOS_INTERESSE_UV: Dict[str, float] = {
    "P_253": 253.56,
    "P_214": 214.93,
    "P_213": 213.61,
    "P_255": 255.33,
    "Ca_315": 315.87,
    "Mg_279": 279.55,
    "Mg_280": 280.26,
}


# ==============================================================================
# 2. ESPECTRO MÉDIO REPRESENTATIVO POR AMOSTRA / PASTILHA
# ==============================================================================

def gerar_espectro_representativo(
    df: pd.DataFrame,
    cols_agrupamento: Optional[List[str]] = None,
    cols_descarte: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Agrupa os múltiplos disparos de uma mesma pastilha/amostra e calcula o espectro médio.

    Essa abordagem:
    1. Aumenta a relação sinal-ruído (SNR).
    2. Evita vazamento de dados de disparos da mesma pastilha entre treino e teste.
    3. Cria uma instância física representativa de cada amostra independente.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de disparos limpos após pré-processamento.
    cols_agrupamento : list, opcional
        Colunas pelas quais agrupar (padrão: ['Teor_Adulteracao', 'Amostra_ID']).
    cols_descarte : list, opcional
        Colunas não numéricas que não devem ser calculadas na média.

    Returns
    -------
    pd.DataFrame
        DataFrame condensado com 1 linha por amostra.
    """
    if cols_agrupamento is None:
        cols_agrupamento = ["Teor_Adulteracao", "Amostra_ID"]

    if cols_descarte is None:
        cols_descarte = ["Num_Disparo", "Nome_Arquivo", "Canal_Optico"]

    df_copia = df.drop(columns=[c for c in cols_descarte if c in df.columns]).copy()

    # Separa colunas numéricas (comprimentos de onda) para calcular média
    cols_agrup = [c for c in cols_agrupamento if c in df_copia.columns]
    df_rep = df_copia.groupby(cols_agrup, as_index=False).mean(numeric_only=True)

    if "Teor_Adulteracao" in df_rep.columns:
        df_rep["Adulterado"] = df_rep["Teor_Adulteracao"].apply(lambda x: 1 if float(x) != 0.0 else 0)

    return df_rep


# ==============================================================================
# 3. INTEGRAÇÃO NUMÉRICA DE PICOS DE EMISSÃO
# ==============================================================================

def calcular_area_pico(
    df: pd.DataFrame,
    centro: float,
    janela: float = 0.3,
    metodo: str = "trapezoid"
) -> np.ndarray:
    """
    Calcula a área sob o pico espectral em torno de um comprimento de onda central.

    Filtra colunas numéricas no intervalo [centro - janela, centro + janela] e aplica
    a regra trapezoidal ou de Simpson ao longo de cada espectro.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo colunas espectrais (comprimento de onda em nm).
    centro : float
        Comprimento de onda central do pico (ex: 766.48 para K ou 393.37 para Ca II).
    janela : float, default 0.3
        Raio da janela de integração em nanômetros (+/- janela).
    metodo : str, default 'trapezoid'
        Método de integração ('trapezoid' ou 'simpson').

    Returns
    -------
    np.ndarray
        Vetor com a área integrada para cada amostra (mínimo 1e-9).
    """
    # Identifica colunas que representam comprimentos de onda
    cols_wl = []
    for c in df.columns:
        try:
            val = float(c)
            cols_wl.append((c, val))
        except ValueError:
            continue

    cols_alvo = [col_str for col_str, col_float in cols_wl if (centro - janela) <= col_float <= (centro + janela)]

    if not cols_alvo:
        print(f"Aviso: Nenhuma coluna encontrada para o pico {centro} nm (janela ±{janela} nm)!")
        return np.full(len(df), np.nan)

    dados = df[cols_alvo].astype(float).values
    wls = np.array([float(c) for c in cols_alvo])

    # Ordena os comprimentos de onda caso necessário
    idx_sort = np.argsort(wls)
    wls_sorted = wls[idx_sort]
    dados_sorted = dados[:, idx_sort]

    if metodo == "simpson" and len(wls_sorted) >= 3:
        area = simpson(dados_sorted, x=wls_sorted, axis=1)
    else:
        # Usa trapezoid do numpy (compatível com NumPy >= 2.0 e versões anteriores)
        if hasattr(np, "trapezoid"):
            area = np.trapezoid(dados_sorted, x=wls_sorted, axis=1)
        else:
            area = np.trapz(dados_sorted, x=wls_sorted, axis=1)

    return np.maximum(area, 1e-9)


def extrair_areas_picos(
    df: pd.DataFrame,
    picos_dict: Dict[str, float],
    janela_nm: float = 0.3
) -> pd.DataFrame:
    """
    Gera um DataFrame contendo as áreas integradas de todos os picos de interesse.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame espectral (idealmente representativo por pastilha).
    picos_dict : dict
        Dicionário {nome_pico: comprimento_de_onda_nm}.
    janela_nm : float, default 0.3
        Janela de integração.

    Returns
    -------
    pd.DataFrame
        DataFrame contendo metadados e uma coluna para cada pico integrado.
    """
    colunas_meta = [c for c in ["Teor_Adulteracao", "Amostra_ID", "Adulterado"] if c in df.columns]
    df_areas = df[colunas_meta].copy()

    for nome_pico, centro_wl in picos_dict.items():
        df_areas[nome_pico] = calcular_area_pico(df, centro=centro_wl, janela=janela_nm)

    return df_areas


# ==============================================================================
# 4. PADRONIZAÇÃO INTERNA E RAZÕES ELEMENTARES (Ca/K, Ca/P, Mg/P)
# ==============================================================================

def calcular_razoes_elementares(
    df_areas: pd.DataFrame,
    numeradores: List[str],
    denominadores: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Calcula razões espectrais (Padronização Interna) entre elementos adulterantes e matriz.

    Exemplo:
    - No canal VIS: Ca_527/K_693, Ca_315/K_580 (Razão Cálcio/Potássio)
    - No canal UV: Ca_315/P_253, Mg_279/P_214 (Razão Cálcio/Fósforo e Magnésio/Fósforo)

    A razão cancela efeitos de matriz e flutuações de fluência do laser.

    Parameters
    ----------
    df_areas : pd.DataFrame
        DataFrame com colunas de áreas de pico calculadas.
    numeradores : list de str
        Picos a serem utilizados no numerador (ex: picos de Ca ou Mg).
    denominadores : list de str
        Picos a serem utilizados no denominador (ex: picos de K ou P).

    Returns
    -------
    df_com_razoes : pd.DataFrame
        DataFrame original acrescido das colunas de razões.
    colunas_razao : list de str
        Lista com os nomes das colunas de razão criadas.
    """
    df_res = df_areas.copy()
    colunas_razao = []

    for num in numeradores:
        if num not in df_res.columns:
            continue
        for den in denominadores:
            if den not in df_res.columns:
                continue
            nome_razao = f"{num}/{den}"
            df_res[nome_razao] = df_res[num] / df_res[den]
            colunas_razao.append(nome_razao)

    return df_res, colunas_razao


# ==============================================================================
# 5. DISTÂNCIA INTERCLASSE DE COHEN (D_C-K)
# ==============================================================================

def calcular_distancia_interclasse_cohen(
    df_areas: pd.DataFrame,
    colunas_razao: List[str],
    teor_referencia: float = 0.0
) -> pd.DataFrame:
    """
    Calcula a métrica de distância interclasse de Cohen (D_C-K) para cada razão elementar:

        D_C-K = |mean(Puro) - mean(Adulterado)| / s_pooled
        onde s_pooled = sqrt( ((n_p - 1)*s_p^2 + (n_a - 1)*s_a^2) / (n_p + n_a - 2) )

    Razões com maior D_C-K possuem maior poder de discriminação estequiométrica
    e menor sobreposição entre classes.

    Parameters
    ----------
    df_areas : pd.DataFrame
        DataFrame com colunas de razões e Teor_Adulteracao.
    colunas_razao : list de str
        Lista de razões a testar.
    teor_referencia : float, default 0.0
        Teor de referência que representa o adubo puro.

    Returns
    -------
    pd.DataFrame
        Tabela ordenada das razões pelo maior D_C-K médio.
    """
    niveis_adulteracao = sorted([
        float(x) for x in df_areas["Teor_Adulteracao"].unique()
        if float(x) != teor_referencia and not np.isnan(float(x))
    ])

    linhas_resultado = []

    for razao in colunas_razao:
        linha = {"Razão Analisada": razao}
        grupo_puro = df_areas[df_areas["Teor_Adulteracao"] == teor_referencia][razao].dropna()
        n_p = len(grupo_puro)
        var_p = grupo_puro.var(ddof=1)

        d_ck_valores = []

        for nivel in niveis_adulteracao:
            grupo_adulterado = df_areas[df_areas["Teor_Adulteracao"] == nivel][razao].dropna()
            n_a = len(grupo_adulterado)
            var_a = grupo_adulterado.var(ddof=1)

            if n_p < 2 or n_a < 2:
                d_ck = np.nan
            else:
                s_pooled = np.sqrt(((n_p - 1) * var_p + (n_a - 1) * var_a) / (n_p + n_a - 2))
                d_ck = abs(grupo_puro.mean() - grupo_adulterado.mean()) / (s_pooled + 1e-9)

            col_label = f"0.0 vs {nivel:.1f}%"
            linha[col_label] = round(d_ck, 3)
            if not np.isnan(d_ck):
                d_ck_valores.append(d_ck)

        linha["Média_D_CK"] = round(float(np.mean(d_ck_valores)), 3) if d_ck_valores else np.nan
        linhas_resultado.append(linha)

    df_tabela = pd.DataFrame(linhas_resultado)
    if "Média_D_CK" in df_tabela.columns:
        df_tabela = df_tabela.sort_values(by="Média_D_CK", ascending=False).reset_index(drop=True)

    return df_tabela
