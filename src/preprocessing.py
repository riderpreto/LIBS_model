"""
Módulo de Pré-processamento e Tratamento de Sinal LIBS
Projeto: Detecção de Adulteração em Fertilizantes NPK via LIBS
Embrapa Instrumentação

Este módulo isola todas as funções matemáticas de leitura, correção de baseline,
remoção de ruído, normalização e filtragem de outliers para dados espectrais LIBS.
"""

import os
import re
import io
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import pandas as pd
import pywt
from scipy.signal import savgol_filter
from scipy.integrate import simpson
from scipy.ndimage import white_tophat
from tqdm import tqdm


# ==============================================================================
# 1. LEITURA E PARSING DE ARQUIVOS ESPECTRAIS (.esf e ASCII)
# ==============================================================================

def ler_espectro_libs_generico(caminho_arquivo: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lê um arquivo de espectro LIBS (formato .esf ou ASCII com delimitador por espaço/tabulação).

    Para arquivos .esf contendo a seção '[Data Section]', extrai os comprimentos de onda
    e as intensidades brutas separadas por ponto-e-vírgula (;). Caso contrário, tenta a
    leitura direta como tabela ASCII de duas colunas.

    Parameters
    ----------
    caminho_arquivo : str ou Path
        Caminho do arquivo no disco.

    Returns
    -------
    wavelengths : np.ndarray
        Vetor 1D com comprimentos de onda em nanômetros (nm).
    intensities : np.ndarray
        Vetor 1D com intensidades de emissão óptica.

    Raises
    ------
    ValueError
        Se o arquivo não puder ser lido ou o formato não for reconhecido.
    """
    caminho = Path(caminho_arquivo)
    with open(caminho, "r", encoding="utf-8", errors="ignore") as f:
        conteudo = f.read()

    # Formato .esf do software LIBS (com seção [Data Section])
    if "[data section]" in conteudo.lower():
        linhas_de_dados = []
        inicio_dos_dados = False
        for linha in conteudo.split("\n"):
            if "[data section]" in linha.lower():
                inicio_dos_dados = True
                continue
            if inicio_dos_dados and linha.strip():
                linhas_de_dados.append(linha)

        if not linhas_de_dados:
            raise ValueError(f"Seção [Data Section] vazia em {caminho}")

        dados_string = "\n".join(linhas_de_dados)
        df_bruto = pd.read_csv(io.StringIO(dados_string), sep=";", header=None, decimal=".")

        if df_bruto.shape[1] >= 2:
            # Colunas típicas: Wavelength, Pixel_Index, Intensity, Diffraction_Order
            if df_bruto.shape[1] >= 3:
                df_bruto.columns = ["Wavelength", "Pixel_Index", "Intensity", "Diffraction_Order"][:df_bruto.shape[1]]
                df_bruto = df_bruto.sort_values(by="Wavelength").reset_index(drop=True)
                return df_bruto["Wavelength"].values, df_bruto["Intensity"].values.flatten()
            else:
                df_bruto.columns = ["Wavelength", "Intensity"]
                df_bruto = df_bruto.sort_values(by="Wavelength").reset_index(drop=True)
                return df_bruto["Wavelength"].values, df_bruto["Intensity"].values.flatten()

    # Formato ASCII genérico (delimitador espaço/tab)
    try:
        df_bruto = pd.read_csv(caminho, sep=r"\s+", header=None)
        wl = df_bruto.iloc[:, 0].values
        sinal = df_bruto.iloc[:, 1].values
        return wl, sinal.flatten()
    except Exception as e:
        raise ValueError(f"Formato de arquivo não suportado para {caminho}: {e}")


def extrair_metadados_arquivo(nome_arquivo: str) -> Dict[str, Union[str, int]]:
    """
    Extrai identificadores de amostra e número de disparo a partir do nome do arquivo.

    Padrões esperados:
    - amostra: 'amostra_([a-z_]+)'
    - disparo: '__(\\d+)'
    """
    match_amostra = re.search(r"amostra_([a-zA-Z0-9_]+)", nome_arquivo)
    match_disparo = re.search(r"__(\d+)", nome_arquivo)

    amostra_id = match_amostra.group(1) if match_amostra else "desconhecida"
    num_disparo = int(match_disparo.group(1)) if match_disparo else 0

    return {
        "Amostra_ID": amostra_id,
        "Num_Disparo": num_disparo,
        "Nome_Arquivo": nome_arquivo
    }


def carregar_esf_bruto(caminho_completo: Union[str, Path], canal: Optional[str] = None) -> Dict[str, any]:
    """
    Lê o arquivo do disco e extrai metadados, retornando dicionário em memória.

    Parameters
    ----------
    caminho_completo : str ou Path
        Caminho para o arquivo .esf.
    canal : str, opcional
        Canal óptico ('UV' ou 'VIS').

    Returns
    -------
    dict com:
        - Wavelength: array 1D
        - Intensity_Raw: array 1D
        - Amostra_ID: str
        - Num_Disparo: int
        - Nome_Arquivo: str
        - Canal_Optico: str
    """
    caminho = Path(caminho_completo)
    meta = extrair_metadados_arquivo(caminho.name)
    wl, sinal_bruto = ler_espectro_libs_generico(caminho)

    dados = {
        "Wavelength": wl,
        "Intensity_Raw": sinal_bruto,
        "Amostra_ID": meta["Amostra_ID"],
        "Num_Disparo": meta["Num_Disparo"],
        "Nome_Arquivo": meta["Nome_Arquivo"],
        "Canal_Optico": canal or "UNKNOWN"
    }
    return dados


def carregar_diretorio_libs(
    diretorio_raiz: Union[str, Path],
    padrao_extensao: str = ".esf"
) -> Tuple[List[Dict], List[Dict]]:
    """
    Varre recursivamente a pasta de espectros identificando teores de adulteração
    (0%, 10%, 25%, 50%) e canais ópticos (UV e VIS).

    Parameters
    ----------
    diretorio_raiz : str ou Path
        Diretório base onde se encontram as pastas de adulteração.
    padrao_extensao : str, default '.esf'
        Extensão dos arquivos espectrais.

    Returns
    -------
    lista_uv : List[Dict]
        Lista de espectros brutos do canal UV com metadados.
    lista_vis : List[Dict]
        Lista de espectros brutos do canal VIS com metadados.
    """
    raiz = Path(diretorio_raiz)
    arquivos_disparos = [p for p in raiz.rglob(f"*{padrao_extensao}") if p.is_file()]
    arquivos_disparos.sort()

    print(f"Encontrados {len(arquivos_disparos)} arquivos para processar em {raiz}...")

    lista_uv = []
    lista_vis = []

    for arq_path in tqdm(arquivos_disparos, desc="Lendo arquivos do disco", unit="tiro"):
        try:
            pasta_do_arquivo = arq_path.parent.name.upper().strip()
            pasta_acima = arq_path.parent.parent.name.lower().strip()

            nome_canal = pasta_do_arquivo if ("UV" in pasta_do_arquivo or "VIS" in pasta_do_arquivo) else "UNKNOWN"
            dados_tiro = carregar_esf_bruto(arq_path, canal=nome_canal)

            # Mapeamento do teor de adulteração baseado na nomenclatura das pastas
            if "50porcento" in pasta_acima or "50%" in pasta_acima or "_50" in pasta_acima:
                teor = 50.0
            elif "00porcento" in pasta_acima or "0%" in pasta_acima or "_00" in pasta_acima or "_0porcento" in pasta_acima:
                teor = 0.0
            elif "10porcento" in pasta_acima or "10%" in pasta_acima or "_10" in pasta_acima:
                teor = 10.0
            elif "25porcento" in pasta_acima or "25%" in pasta_acima or "_25" in pasta_acima:
                teor = 25.0
            else:
                teor = -1.0

            dados_tiro["Teor_Adulteracao"] = teor

            if "UV" in nome_canal:
                lista_uv.append(dados_tiro)
            elif "VIS" in nome_canal:
                lista_vis.append(dados_tiro)
            else:
                # Se não puder inferir pelo nome da subpasta, tenta pelo range de comprimento de onda
                wl_med = np.mean(dados_tiro["Wavelength"])
                if wl_med < 400:
                    dados_tiro["Canal_Optico"] = "UV"
                    lista_uv.append(dados_tiro)
                else:
                    dados_tiro["Canal_Optico"] = "VIS"
                    lista_vis.append(dados_tiro)

        except Exception as e:
            print(f"Erro no ficheiro {arq_path.name}: {e}")

    print(f"Total carregado: {len(lista_uv)} disparos UV | {len(lista_vis)} disparos VIS.")
    return lista_uv, lista_vis


# ==============================================================================
# 2. FILTROS DE RUÍDO (DENOISING)
# ==============================================================================

def wavelet_denoising(
    data: np.ndarray,
    wavelet: str = "sym8",
    level: int = 2,
    mode: str = "per",
    threshold_mode: str = "hard"
) -> np.ndarray:
    """
    Redução de ruído via Transformada Wavelet Discreta (DWT) com limiarização adaptativa.

    Implementa a estimativa de desvio padrão do ruído pelo estimador MAD (Median Absolute
    Deviation) dos coeficientes de detalhe do nível mais fino (Donoho & Johnstone, 1994):
        sigma = (1 / 0.6745) * median(|cD1 - median(cD1)|)
        limiar = sigma * sqrt(2 * ln(N))

    Parameters
    ----------
    data : np.ndarray
        Vetor 1D com o sinal de intensidade espectral.
    wavelet : str, default 'sym8'
        Família de wavelet mãe (Symlet 8 recomendada para picos espectrais).
    level : int, default 2
        Nível de decomposição wavelet.
    mode : str, default 'per'
        Modo de extensão nas bordas ('per' = periódico).
    threshold_mode : str, default 'hard'
        Tipo de threshold ('hard' ou 'soft').

    Returns
    -------
    np.ndarray
        Sinal filtrado com ruído atenuado.
    """
    data = np.asarray(data, dtype=float)
    coeff = pywt.wavedec(data, wavelet, mode=mode)
    actual_level = min(level, len(coeff) - 1)

    # Estimador de ruído gaussiano baseado em MAD no nível mais fino
    detalhes_finos = coeff[-1]
    mediana_detalhes = np.median(detalhes_finos)
    mad = np.median(np.abs(detalhes_finos - mediana_detalhes))
    sigma = (1.0 / 0.6745) * mad

    # Limiar universal de Donoho-Johnstone
    uthresh = sigma * np.sqrt(2.0 * np.log(len(data)))

    novos_coeficientes = [coeff[0]]
    for i in range(1, len(coeff)):
        # Aplica threshold nos níveis de detalhe desejados
        if i >= (len(coeff) - actual_level):
            c_thresh = pywt.threshold(coeff[i], value=uthresh, mode=threshold_mode)
            novos_coeficientes.append(c_thresh)
        else:
            novos_coeficientes.append(coeff[i])

    rec = pywt.waverec(novos_coeficientes, wavelet, mode=mode)
    return rec[:len(data)]


def savgol_denoising(
    data: np.ndarray,
    window_length: int = 7,
    polyorder: int = 2
) -> np.ndarray:
    """
    Filtro de suavização polinomial Savitzky-Golay.

    Preserva melhor a altura e a forma dos picos de emissão atômica estreitos.
    """
    # Garante tamanho de janela ímpar e menor que o sinal
    if window_length % 2 == 0:
        window_length += 1
    window_length = min(window_length, len(data) - (1 if len(data) % 2 == 0 else 0))
    return savgol_filter(data, window_length=window_length, polyorder=polyorder)


# ==============================================================================
# 3. CORREÇÃO DE BASELINE (FUNDO CONTÍNUO DO PLASMA)
# ==============================================================================

def tophat_baseline_correction(data: np.ndarray, footprint_size: int = 100) -> np.ndarray:
    """
    Correção de linha de base via filtro morfológico White Top-Hat.

    O operador White Top-Hat subtrai a abertura morfológica (erosão seguida de dilatação)
    do sinal original, removendo o continuum do plasma (fundo lento) e preservando
    picos espectrais estreitos.

    Parameters
    ----------
    data : np.ndarray
        Vetor 1D do sinal espectral.
    footprint_size : int, default 100
        Tamanho do elemento estruturante (deve ser maior que a largura dos picos).

    Returns
    -------
    np.ndarray
        Sinal com baseline subtraída.
    """
    footprint = np.ones(footprint_size, dtype=int)
    return white_tophat(data, footprint=footprint)


def iterative_polynomial_baseline(
    wavelengths: np.ndarray,
    intensity: np.ndarray,
    order: int = 4,
    max_iter: int = 50,
    tol: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ajuste iterativo de baseline polinomial (ModPoly / Lieber & Mahadevan-Jansen).

    Ajusta iterativamente um polinômio de ordem `order`, substituindo os pontos
    onde a intensidade supera o polinômio pelo valor ajustado, até convergir para
    a linha de base contínua.

    Parameters
    ----------
    wavelengths : np.ndarray
        Vetor de comprimentos de onda.
    intensity : np.ndarray
        Vetor de intensidade bruta.
    order : int, default 4
        Ordem do polinômio de fundo.
    max_iter : int, default 50
        Número máximo de iterações.
    tol : float, default 1e-4
        Tolerância de convergência.

    Returns
    -------
    baseline : np.ndarray
        A linha de base estimada.
    corrected_signal : np.ndarray
        O sinal com a baseline removida (intensity - baseline).
    """
    x = (wavelengths - np.mean(wavelengths)) / np.std(wavelengths)
    y_current = intensity.copy()

    for _ in range(max_iter):
        coeffs = np.polyfit(x, y_current, deg=order)
        poly_fit = np.polyval(coeffs, x)
        y_next = np.minimum(intensity, poly_fit)

        if np.max(np.abs(y_next - y_current)) < tol:
            break
        y_current = y_next

    baseline = poly_fit
    corrected_signal = np.maximum(intensity - baseline, 0.0)
    return baseline, corrected_signal


# ==============================================================================
# 4. NORMALIZAÇÃO ESPECTRAL
# ==============================================================================

def snv_normalizacao(espectro: np.ndarray) -> np.ndarray:
    """
    Aplica a normalização Standard Normal Variate (SNV).

    Cada espectro individual tem sua média subtraída e é dividido pelo seu desvio
    padrão amostral (ddof=1), corrigindo flutuações de intensidade disparo a disparo
    devido à variação de energia do laser e acoplamento plasma-amostra.

    Formula:
        y_snv = (y - mean(y)) / std(y)
    """
    espectro = np.asarray(espectro, dtype=float)
    media = np.mean(espectro)
    desvio = np.std(espectro, ddof=1)

    if desvio == 0 or np.isnan(desvio):
        return espectro - media

    return (espectro - media) / desvio


def normalizacao_area(espectro: np.ndarray, wavelengths: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Normaliza o espectro pela sua área total sob a curva usando a regra de Simpson.

    Garante que a integral do sinal sobre todo o domínio espectral seja igual a 1.
    Se wavelengths não for fornecido, a integral é calculada assumindo espaçamento unitário.
    """
    espectro = np.asarray(espectro, dtype=float)
    if wavelengths is not None:
        area = simpson(espectro, x=wavelengths)
    else:
        area = simpson(espectro)

    if area == 0 or np.isnan(area):
        return espectro

    return espectro / area


def min_max_normalizacao(espectro: np.ndarray) -> np.ndarray:
    """
    Normalização Min-Max para escala [0, 1].
    """
    min_val = np.min(espectro)
    max_val = np.max(espectro)
    if max_val - min_val == 0:
        return np.zeros_like(espectro)
    return (espectro - min_val) / (max_val - min_val)


# ==============================================================================
# 5. FILTRAGEM DE OUTLIERS VIA SPECTRAL ANGLE MAPPER (SAM)
# ==============================================================================

def filtrar_outliers_sam(
    df_consolidado: pd.DataFrame,
    threshold_percentil: float = 99.0,
    cols_metadados: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Varre o DataFrame consolidado por pastilha/amostra (Amostra_ID) e remove
    os disparos anômalos utilizando o ângulo espectral (SAM - Spectral Angle Mapper).

    O ângulo espectral compara a direção do vetor de intensidade de cada disparo
    com a média vetorial do grupo daquela mesma pastilha:
        cos(theta) = (X_i . X_mean) / (||X_i|| * ||X_mean||)
        theta = arccos(clip(cos(theta), -1.0, 1.0))

    Disparos com theta > limite_percentil são descartados como tiros degradados
    (ex: baixa quebra do plasma, crateramento, poeira).

    Parameters
    ----------
    df_consolidado : pd.DataFrame
        DataFrame contendo metadados e colunas espectrais (comprimento de onda).
    threshold_percentil : float, default 99.0
        Percentil de corte dos ângulos (ex: 95.0, 99.0 ou 100.0 para manter todos).
    cols_metadados : list, opcional
        Nomes das colunas de metadados a ignorar no cálculo vetorial.

    Returns
    -------
    pd.DataFrame
        DataFrame limpo apenas com disparos aceitos.
    """
    if cols_metadados is None:
        cols_metadados = ["Amostra_ID", "Num_Disparo", "Nome_Arquivo", "Canal_Optico", "Teor_Adulteracao", "Adulterado"]

    cols_features = [c for c in df_consolidado.columns if c not in cols_metadados]
    lista_dfs_limpos = []

    for amostra, grupo in df_consolidado.groupby("Amostra_ID"):
        X = grupo[cols_features].values.astype(float)
        media_grupo = X.mean(axis=0).reshape(1, -1)

        normas_X = np.linalg.norm(X, axis=1)
        norma_media = np.linalg.norm(media_grupo)
        dot_product = np.dot(X, media_grupo.T).flatten()

        cos_sim = dot_product / (normas_X * norma_media + 1e-9)
        angulos = np.arccos(np.clip(cos_sim, -1.0, 1.0))

        limite_corte = np.percentile(angulos, threshold_percentil)
        mascara_validos = angulos <= limite_corte
        grupo_limpo = grupo[mascara_validos].copy()
        lista_dfs_limpos.append(grupo_limpo)

        removidos = len(grupo) - len(grupo_limpo)
        if removidos > 0:
            print(f"Amostra {amostra}: {removidos}/{len(grupo)} disparos anômalos removidos (SAM > p{threshold_percentil}).")

    df_limpo_final = pd.concat(lista_dfs_limpos, ignore_index=True)
    return df_limpo_final


# ==============================================================================
# 6. PIPELINES DE PROCESSAMENTO COMPLETO
# ==============================================================================

def processar_espectro(
    dados_brutos: Dict[str, any],
    denoise_method: Optional[str] = None,
    baseline_footprint: int = 100,
    norm_method: str = "snv",
    subtrair_minimo: bool = True,
    arredondamento_wl: int = 2
) -> pd.DataFrame:
    """
    Aplica o pipeline completo de tratamento de sinal a um disparo em memória RAM.

    Etapas configuráveis:
    1. Redução de ruído (None, 'wavelet', 'savgol' ou 'wavelet+savgol')
    2. Correção de baseline (White Top-Hat)
    3. Normalização ('snv', 'area' ou 'none')
    4. Subtração do mínimo (para evitar valores negativos se requerido)

    Parameters
    ----------
    dados_brutos : dict
        Dicionário retornado por `carregar_esf_bruto`.
    denoise_method : str, opcional
        Método de denoising ('wavelet', 'savgol', 'wavelet+savgol' ou None).
    baseline_footprint : int, default 100
        Tamanho da vizinhança do White Top-Hat.
    norm_method : str, default 'snv'
        Método de normalização ('snv', 'area', ou None).
    subtrair_minimo : bool, default True
        Se True, desloca o espectro pós-SNV para base zero.
    arredondamento_wl : int, default 2
        Casas decimais para padronizar os comprimentos de onda como colunas.

    Returns
    -------
    pd.DataFrame
        DataFrame de 1 linha com colunas espectrais e metadados.
    """
    wl = dados_brutos["Wavelength"]
    sinal = dados_brutos["Intensity_Raw"].astype(float).copy()

    # 1. Tratamento de ruído
    if denoise_method == "wavelet":
        sinal = wavelet_denoising(sinal, wavelet="sym8", level=6)
    elif denoise_method == "savgol":
        sinal = savgol_denoising(sinal, window_length=7, polyorder=2)
    elif denoise_method == "wavelet+savgol":
        sinal = wavelet_denoising(sinal, wavelet="sym8", level=6)
        sinal = savgol_denoising(sinal, window_length=7, polyorder=2)

    # 2. Correção de baseline
    if baseline_footprint and baseline_footprint > 0:
        sinal = tophat_baseline_correction(sinal, footprint_size=baseline_footprint)

    # 3. Normalização
    if norm_method == "snv":
        sinal = snv_normalizacao(sinal)
        if subtrair_minimo:
            sinal = sinal - np.min(sinal)
    elif norm_method == "area":
        sinal = normalizacao_area(sinal, wavelengths=wl)

    # Arredonda comprimento de onda para consistência de colunas
    wl_nomes = [f"{w:.{arredondamento_wl}f}" for w in wl]

    df_linha = pd.DataFrame([sinal], columns=wl_nomes)
    df_linha["Amostra_ID"] = dados_brutos["Amostra_ID"]
    df_linha["Num_Disparo"] = dados_brutos["Num_Disparo"]
    df_linha["Nome_Arquivo"] = dados_brutos["Nome_Arquivo"]
    df_linha["Canal_Optico"] = dados_brutos.get("Canal_Optico", "UNKNOWN")
    df_linha["Teor_Adulteracao"] = dados_brutos.get("Teor_Adulteracao", np.nan)

    return df_linha


def processar_lote_espectros(
    lista_dicionarios: List[Dict],
    desc: str = "Processando espectros",
    **kwargs
) -> pd.DataFrame:
    """
    Processa uma lista de dicionários brutos e retorna o DataFrame consolidado.
    """
    linhas = []
    for dados in tqdm(lista_dicionarios, desc=desc):
        df_tiro = processar_espectro(dados, **kwargs)
        linhas.append(df_tiro)

    if not linhas:
        return pd.DataFrame()

    df_consolidado = pd.concat(linhas, ignore_index=True)
    return df_consolidado
