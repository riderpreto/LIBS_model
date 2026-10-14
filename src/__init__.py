"""
Package src: Pipeline Quimiométrico para Análise LIBS de Fertilizantes NPK
Embrapa Instrumentação
"""

from .preprocessing import (
    ler_espectro_libs_generico,
    carregar_esf_bruto,
    carregar_diretorio_libs,
    wavelet_denoising,
    savgol_denoising,
    tophat_baseline_correction,
    iterative_polynomial_baseline,
    snv_normalizacao,
    normalizacao_area,
    min_max_normalizacao,
    filtrar_outliers_sam,
    processar_espectro,
    processar_lote_espectros,
)

from .features import (
    LINHAS_EMISSAO_VIS,
    CORES_ELEMENTOS_VIS,
    LINHAS_EMISSAO_UV,
    CORES_ELEMENTOS_UV,
    PICOS_INTERESSE_VIS,
    PICOS_INTERESSE_UV,
    gerar_espectro_representativo,
    calcular_area_pico,
    extrair_areas_picos,
    calcular_razoes_elementares,
    calcular_distancia_interclasse_cohen,
)

from .models import (
    executar_pca,
    identificar_top_loadings,
    avaliar_knn_leave_group_out,
    validacao_monte_carlo_knn,
    otimizar_componentes_plsr,
    avaliar_plsr_final,
    avaliar_rf_regressor,
    regressao_linear_razoes,
)

from .visualization import (
    plotar_comparacao_filtros_ruido,
    plotar_espectros_medios,
    plotar_pca_scores,
    plotar_loadings_com_linhas_nist,
    plotar_curva_rmsec_rmsecv,
    plotar_calibracao_real_vs_predito,
)

__all__ = [
    # Preprocessing
    "ler_espectro_libs_generico",
    "carregar_esf_bruto",
    "carregar_diretorio_libs",
    "wavelet_denoising",
    "savgol_denoising",
    "tophat_baseline_correction",
    "iterative_polynomial_baseline",
    "snv_normalizacao",
    "normalizacao_area",
    "min_max_normalizacao",
    "filtrar_outliers_sam",
    "processar_espectro",
    "processar_lote_espectros",
    # Features
    "LINHAS_EMISSAO_VIS",
    "CORES_ELEMENTOS_VIS",
    "LINHAS_EMISSAO_UV",
    "CORES_ELEMENTOS_UV",
    "PICOS_INTERESSE_VIS",
    "PICOS_INTERESSE_UV",
    "gerar_espectro_representativo",
    "calcular_area_pico",
    "extrair_areas_picos",
    "calcular_razoes_elementares",
    "calcular_distancia_interclasse_cohen",
    # Models
    "executar_pca",
    "identificar_top_loadings",
    "avaliar_knn_leave_group_out",
    "validacao_monte_carlo_knn",
    "otimizar_componentes_plsr",
    "avaliar_plsr_final",
    "avaliar_rf_regressor",
    "regressao_linear_razoes",
    # Visualization
    "plotar_comparacao_filtros_ruido",
    "plotar_espectros_medios",
    "plotar_pca_scores",
    "plotar_loadings_com_linhas_nist",
    "plotar_curva_rmsec_rmsecv",
    "plotar_calibracao_real_vs_predito",
]
