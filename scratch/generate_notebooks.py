"""
Script para gerar os 4 notebooks Jupyter modulares e limpos.
"""

import json
from pathlib import Path

def create_notebook(cells, filepath):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Notebook criado: {filepath}")

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip().splitlines(keepends=True)
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.strip().splitlines(keepends=True)
    }

# ==============================================================================
# NOTEBOOK 1: EDA e Pré-processamento
# ==============================================================================
nb1_cells = [
    md_cell("""# 🧪 Caderno 01: Análise Exploratória e Pipeline de Pré-processamento Espectral
**Projeto:** Detecção de Adulteração em Fertilizantes NPK via LIBS  
**Instituição:** Embrapa Instrumentação  

---

### Objetivos deste Caderno:
1. **Carregamento dos dados brutos (.esf)** para os canais ópticos **UV** e **VIS**.
2. **Avaliação comparativa de filtros de ruído:** Espectro Bruto vs Wavelet (*sym8*) vs Savitzky-Golay.
3. **Avaliação da remoção de linha de base (fundo contínuo do plasma):** Filtro morfológico *White Top-Hat* vs Ajuste Polinomial Iterativo.
4. **Normalização espectral:** *Standard Normal Variate (SNV)* e Normalização por Área (Regra de Simpson).
5. **Filtragem de disparos anômalos (outliers):** Algoritmo *Spectral Angle Mapper (SAM)*.
6. **Inspeção dos espectros médios consolidados** por teor de adulteração (0%, 10%, 25%, 50%).
"""),
    code_cell("""# Adiciona a pasta raiz ao sys.path para importar o pacote src
import sys
import os
from pathlib import Path

# Ajuste do caminho base
BASE_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importação dos módulos especializados
from src.preprocessing import (
    carregar_diretorio_libs,
    carregar_esf_bruto,
    wavelet_denoising,
    savgol_denoising,
    tophat_baseline_correction,
    iterative_polynomial_baseline,
    snv_normalizacao,
    normalizacao_area,
    filtrar_outliers_sam,
    processar_espectro,
    processar_lote_espectros,
)
from src.visualization import (
    plotar_comparacao_filtros_ruido,
    plotar_espectros_medios
)

print("Módulos carregados com sucesso!")"""),
    md_cell("""## 1. Carregamento dos Arquivos Brutos (.esf)
Os arquivos espectrais devem estar organizados em pastas contendo os teores de adulteração (`00porcento`, `10porcento`, `25porcento`, `50porcento`) e subpastas para os canais (`UV` e `VIS`).
"""),
    code_cell("""# Defina o diretório onde estão os espectros brutos
# Exemplo local: BASE_DIR / 'data' / 'raw' ou caminho personalizado
CAMINHO_DADOS = BASE_DIR / "data" / "raw"

# Caso esteja executando no Google Colab, descomente e ajuste:
# CAMINHO_DADOS = Path("/content/drive/MyDrive/LIBS/Espectros/NPK_10-10-10_Fabricado")

print(f"Diretório de dados configurado: {CAMINHO_DADOS}")

# Carregamento em lote dos metadados e matrizes brutas
if CAMINHO_DADOS.exists():
    lista_uv_bruto, lista_vis_bruto = carregar_diretorio_libs(CAMINHO_DADOS)
else:
    print(f"Aviso: Diretório {CAMINHO_DADOS} não encontrado. Crie a pasta ou configure o caminho correto.")
    lista_uv_bruto, lista_vis_bruto = [], []"""),
    md_cell("""## 2. Análise do Tratamento de Ruído (Denoising)
Comparação entre o sinal bruto e dois dos filtros mais empregados em LIBS:
- **Transformada Wavelet Discreta (DWT):** Família Symlet 8 (`sym8`) com limiarização adaptativa de Donoho-Johnstone (MAD).
- **Savitzky-Golay:** Suavização polinomial local de 2ª ordem.
"""),
    code_cell("""if len(lista_vis_bruto) > 0:
    amostra_exemplo = lista_vis_bruto[0]
    wl = amostra_exemplo['Wavelength']
    sinal_bruto = amostra_exemplo['Intensity_Raw']

    sinal_wavelet = wavelet_denoising(sinal_bruto, wavelet='sym8', level=6)
    sinal_savgol = savgol_denoising(sinal_bruto, window_length=7, polyorder=2)

    fig = plotar_comparacao_filtros_ruido(
        wl, sinal_bruto, sinal_wavelet, sinal_savgol,
        titulo=f"Comparação de Filtros de Ruído - {amostra_exemplo['Nome_Arquivo']}"
    )
    plt.show()"""),
    md_cell("""## 3. Análise da Remoção de Linha de Base (Baseline)
O plasma gerado por laser produz um fundo contínuo de radiação bremsstrahlung e recombinação.
Testamos aqui o operador morfológico **White Top-Hat** (com elemento estruturante de 100 pontos) comparado ao **Ajuste Polinomial Iterativo (ModPoly)**.
"""),
    code_cell("""if len(lista_vis_bruto) > 0:
    sinal_tophat = tophat_baseline_correction(sinal_bruto, footprint_size=100)
    baseline_poly, sinal_poly = iterative_polynomial_baseline(wl, sinal_bruto, order=4)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    
    # White Top-Hat
    axes[0].plot(wl, sinal_bruto, label="Sinal Bruto", color="gray", alpha=0.6)
    axes[0].plot(wl, sinal_tophat, label="Pós White Top-Hat (Filtro Morfológico)", color="tab:blue")
    axes[0].set_title("Correção com White Top-Hat", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Polinomial Iterativo
    axes[1].plot(wl, sinal_bruto, label="Sinal Bruto", color="gray", alpha=0.6)
    axes[1].plot(wl, baseline_poly, label="Linha de Base Estimada", color="tab:red", linestyle="--")
    axes[1].plot(wl, sinal_poly, label="Pós Subtração Polinomial", color="tab:green")
    axes[1].set_title("Correção Polinomial Iterativa", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.xlabel("Comprimento de Onda (nm)")
    plt.tight_layout()
    plt.show()"""),
    md_cell("""## 4. Normalização Espectral e Filtragem de Outliers (SAM)
Aplicamos o pipeline de processamento completo em lote:
1. **White Top-Hat** (Baseline)
2. **SNV** (Standard Normal Variate) + deslocamento ao zero
3. **Filtragem de Outliers via SAM (Spectral Angle Mapper)** com corte no percentil 99.
"""),
    code_cell("""# Processamento em lote para o Canal VIS
if len(lista_vis_bruto) > 0:
    print("Processando canal VIS...")
    df_processado_vis = processar_lote_espectros(
        lista_vis_bruto,
        baseline_footprint=100,
        norm_method='snv',
        subtrair_minimo=True
    )
    print(f"Dimensões VIS antes do SAM: {df_processado_vis.shape}")

    # Filtragem de outliers via SAM
    df_limpo_vis = filtrar_outliers_sam(df_processado_vis, threshold_percentil=99.0)
    print(f"Dimensões VIS após SAM: {df_limpo_vis.shape}")
else:
    print("Sem dados brutos carregados para processamento.")"""),
    code_cell("""# Processamento em lote para o Canal UV
if len(lista_uv_bruto) > 0:
    print("Processando canal UV...")
    df_processado_uv = processar_lote_espectros(
        lista_uv_bruto,
        baseline_footprint=100,
        norm_method='snv',
        subtrair_minimo=True
    )
    print(f"Dimensões UV antes do SAM: {df_processado_uv.shape}")

    df_limpo_uv = filtrar_outliers_sam(df_processado_uv, threshold_percentil=99.0)
    print(f"Dimensões UV após SAM: {df_limpo_uv.shape}")"""),
    md_cell("""## 5. Visualização dos Espectros Médios por Teor de Adulteração
Inspecionamos os perfis médios obtidos para cada classe (0% Puro vs 10%, 25%, 50% Adulterado).
"""),
    code_cell("""if 'df_limpo_vis' in locals() and len(df_limpo_vis) > 0:
    fig_vis = plotar_espectros_medios(
        df_limpo_vis,
        titulo="Espectros Médios Normalizados (SNV) - Canal VIS"
    )
    plt.show()

if 'df_limpo_uv' in locals() and len(df_limpo_uv) > 0:
    fig_uv = plotar_espectros_medios(
        df_limpo_uv,
        titulo="Espectros Médios Normalizados (SNV) - Canal UV"
    )
    plt.show()"""),
    md_cell("""## 6. Salvamento das Matrizes Tratadas
Para evitar ter que reprocessar arquivos `.esf` a cada análise, salvamos os dados tratados em formato Parquet/CSV na pasta `data/processed`.
"""),
    code_cell("""pasta_saida = BASE_DIR / "data" / "processed"
pasta_saida.mkdir(parents=True, exist_ok=True)

if 'df_limpo_vis' in locals() and len(df_limpo_vis) > 0:
    caminho_vis = pasta_saida / "df_limpo_vis.parquet"
    df_limpo_vis.to_parquet(caminho_vis, index=False)
    print(f"VIS salvo em: {caminho_vis}")

if 'df_limpo_uv' in locals() and len(df_limpo_uv) > 0:
    caminho_uv = pasta_saida / "df_limpo_uv.parquet"
    df_limpo_uv.to_parquet(caminho_uv, index=False)
    print(f"UV salvo em: {caminho_uv}")""")
]

# ==============================================================================
# NOTEBOOK 2: PCA e Classificação KNN
# ==============================================================================
nb2_cells = [
    md_cell("""# 🔍 Caderno 02: Redução de Dimensionalidade (PCA) e Classificação com KNN
**Projeto:** Detecção de Adulteração em Fertilizantes NPK via LIBS  
**Instituição:** Embrapa Instrumentação  

---

### Objetivos deste Caderno:
1. **Redução de Dimensionalidade via PCA (Análise de Componentes Principais)** nos canais VIS e UV.
2. **Interpretação Quimiométrica dos Loadings:** Mapeamento automático dos maiores pesos espectrais com a biblioteca atômica NIST (linhas de Ca II, Ca I, K, Mg, Fe, P).
3. **Classificação Supervisionada com KNN (K-Nearest Neighbors)** no espaço latente de scores.
4. **Validação Rigorosa contra Vazamento de Dados (Data Leakage):**
   - Comparação entre split aleatório padrão e **Leave-Group-Out CV (LOGO-CV)** agrupado por pastilha (`Amostra_ID`).
   - **Validação Cruzada de Monte Carlo** com subamostragem balanceada.
"""),
    code_cell("""import sys
from pathlib import Path

BASE_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.models import (
    executar_pca,
    identificar_top_loadings,
    avaliar_knn_leave_group_out,
    validacao_monte_carlo_knn
)
from src.features import (
    LINHAS_EMISSAO_VIS,
    CORES_ELEMENTOS_VIS,
    LINHAS_EMISSAO_UV,
    CORES_ELEMENTOS_UV
)
from src.visualization import (
    plotar_pca_scores,
    plotar_loadings_com_linhas_nist
)

print("Módulos carregados com sucesso!")"""),
    md_cell("""## 1. Carregamento dos Dados Pré-processados"""),
    code_cell("""pasta_proc = BASE_DIR / "data" / "processed"
caminho_vis = pasta_proc / "df_limpo_vis.parquet"
caminho_uv = pasta_proc / "df_limpo_uv.parquet"

df_vis = pd.read_parquet(caminho_vis) if caminho_vis.exists() else None
df_uv = pd.read_parquet(caminho_uv) if caminho_uv.exists() else None

if df_vis is not None:
    print(f"Dados VIS carregados: {df_vis.shape}")
    print(df_vis.groupby(['Teor_Adulteracao', 'Amostra_ID']).size().unstack(fill_value=0))
else:
    print("Aviso: execute primeiro o Caderno 01 para gerar df_limpo_vis.parquet.")"""),
    md_cell("""## 2. PCA no Canal VIS
Executamos a PCA sobre os comprimentos de onda na faixa de 273 nm a 780 nm.
"""),
    code_cell("""if df_vis is not None:
    # Exclusão opcional de pastilhas de validação externa ou teores específicos
    df_analise = df_vis[df_vis['Amostra_ID'] != 'i_'].copy()

    pca_vis, X_pca_vis, cols_wl_vis, df_scores_vis = executar_pca(
        df_analise,
        wl_min=273.0,
        wl_max=780.0,
        n_components=5,
        scaler=False
    )

    print("Variância Explicada por Componente:")
    for i, var in enumerate(pca_vis.explained_variance_ratio_):
        print(f"  PC{i+1}: {var*100:.2f}% (Acumulado: {np.sum(pca_vis.explained_variance_ratio_[:i+1])*100:.2f}%)")

    # Gráfico de Scores PC1 vs PC2
    fig = plotar_pca_scores(df_scores_vis, pc_x="PC1", pc_y="PC2", titulo="Scores PCA - Canal VIS")
    plt.show()"""),
    md_cell("""## 3. Interpretação dos Loadings e Identificação de Linhas NIST
Analisamos quais comprimentos de onda são responsáveis pela separação observada na PC1 e PC2.
"""),
    code_cell("""if df_vis is not None:
    wls_vis = np.array([float(c) for c in cols_wl_vis])

    # Plot dos Loadings da PC1 e PC2
    fig1 = plotar_loadings_com_linhas_nist(
        wls_vis, pca_vis.components_[0], "PC1",
        pca_vis.explained_variance_ratio_[0]*100,
        LINHAS_EMISSAO_VIS, CORES_ELEMENTOS_VIS
    )
    plt.show()

    fig2 = plotar_loadings_com_linhas_nist(
        wls_vis, pca_vis.components_[1], "PC2",
        pca_vis.explained_variance_ratio_[1]*100,
        LINHAS_EMISSAO_VIS, CORES_ELEMENTOS_VIS
    )
    plt.show()

    # Tabela com os Top-10 picos identificados
    df_top_vis = identificar_top_loadings(
        pca_vis, cols_wl_vis, n_pcs=2, top_n=10,
        linhas_referencia=LINHAS_EMISSAO_VIS
    )
    print("Top Loadings Identificados:")
    display(df_top_vis) if 'display' in dir() else print(df_top_vis.to_string())"""),
    md_cell("""## 4. Classificação com KNN: Teste contra Vazamento de Dados (Data Leakage)
Em LIBS, múltiplos disparos são realizados sobre uma mesma pastilha.  
Se disparos da mesma pastilha caírem simultaneamente no treino e no teste, a acurácia é inflada artificialmente.  
Aqui testamos a validação estrita **Leave-Group-Out**, garantindo que pastilhas inteiras fiquem restritas ao teste.
"""),
    code_cell("""if df_vis is not None:
    # Definir grupos de pastilhas para treino e teste
    grupos = df_scores_vis['Amostra_ID'].values
    y = df_scores_vis['Teor_Adulteracao'].values
    X = X_pca_vis[:, :3] # Utilizando as primeiras 3 PCs

    amostras_unicas = list(np.unique(grupos))
    print(f"Pastilhas disponíveis: {amostras_unicas}")

    # Exemplo: treino nas primeiras pastilhas, teste nas últimas
    treino_ids = amostras_unicas[:-2]
    teste_ids = amostras_unicas[-2:]

    resultado_lgo = avaliar_knn_leave_group_out(
        X, y, grupos,
        ids_treino=treino_ids,
        ids_teste=teste_ids,
        n_neighbors=5
    )

    print(f"Treino IDs: {treino_ids} ({resultado_lgo['n_treino']} disparos)")
    print(f"Teste IDs:  {teste_ids} ({resultado_lgo['n_teste']} disparos)")
    print(f"Acurácia Leave-Group-Out: {resultado_lgo['acuracia']*100:.2f}%")

    plt.figure(figsize=(6, 5))
    sns.heatmap(resultado_lgo['matriz_confusao'], annot=True, fmt='d', cmap='Blues')
    plt.title("Matriz de Confusão - Leave Group Out CV")
    plt.xlabel("Predição")
    plt.ylabel("Real")
    plt.show()"""),
    md_cell("""## 5. Validação Cruzada de Monte Carlo (100 Iterações)
Subamostragem balanceada com 100 iterações para obter o intervalo de confiança empírico de 95% da acurácia.
"""),
    code_cell("""if df_vis is not None:
    res_mc = validacao_monte_carlo_knn(
        df_scores_vis,
        cols_features=['PC1', 'PC2', 'PC3'],
        col_alvo='Teor_Adulteracao',
        n_iteracoes=100,
        n_neighbors=5
    )

    print(f"Acurácia Média Monte Carlo: {res_mc['media_acuracia']*100:.2f}% ± {res_mc['desvio_acuracia']*100:.2f}%")
    print(f"Intervalo de Confiança 95%: [{res_mc['ic_95'][0]*100:.2f}%, {res_mc['ic_95'][1]*100:.2f}%]")

    plt.figure(figsize=(8, 4))
    sns.histplot(res_mc['todas_acuracias'], bins=15, kde=True, color='teal')
    plt.title("Distribuição de Acurácias - Monte Carlo CV (100 Iterações)")
    plt.xlabel("Acurácia")
    plt.show()""")
]

# ==============================================================================
# NOTEBOOK 3: Regressão Quimiométrica PLSR e Random Forest
# ==============================================================================
nb3_cells = [
    md_cell("""# 📈 Caderno 03: Calibração Multivariada Quantitativa (PLSR e Random Forest)
**Projeto:** Detecção de Adulteração em Fertilizantes NPK via LIBS  
**Instituição:** Embrapa Instrumentação  

---

### Objetivos deste Caderno:
1. **Modelagem Quantitativa Contínua** do teor de adulteração (%) utilizando o espectro completo.
2. **Partial Least Squares Regression (PLSR):**
   - Otimização do número de Variáveis Latentes (LVs) via curvas de RMSEC e RMSECV (Leave-One-Group-Out CV).
   - Treinamento do modelo final de calibração.
3. **Quantificação da Incerteza de Medição Tipo A ($u_A = s / \sqrt{n}$)** para cada teor de adulteração.
4. **Modelagem Não Linear com Random Forest Regressor:**
   - Validação cruzada agrupada por pastilha.
   - Extração da importância dos comprimentos de onda para predição quantitativa.
"""),
    code_cell("""import sys
from pathlib import Path

BASE_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.models import (
    otimizar_componentes_plsr,
    avaliar_plsr_final,
    avaliar_rf_regressor
)
from src.visualization import (
    plotar_curva_rmsec_rmsecv,
    plotar_calibracao_real_vs_predito
)

print("Módulos carregados com sucesso!")"""),
    md_cell("""## 1. Carregamento dos Dados Tratados"""),
    code_cell("""pasta_proc = BASE_DIR / "data" / "processed"
caminho_vis = pasta_proc / "df_limpo_vis.parquet"

df_vis = pd.read_parquet(caminho_vis) if caminho_vis.exists() else None

if df_vis is not None:
    # Filtragem de amostras e seleção de features
    df_pls = df_vis[df_vis['Amostra_ID'] != 'i_'].copy()
    metadados = ['Amostra_ID', 'Num_Disparo', 'Nome_Arquivo', 'Canal_Optico', 'Teor_Adulteracao', 'Adulterado']
    
    cols_features = [c for c in df_pls.columns if c not in metadados and 273.0 <= float(c) <= 780.0]
    X = df_pls[cols_features].values.astype(float)
    y = df_pls['Teor_Adulteracao'].values.astype(float)
    grupos = df_pls['Amostra_ID'].values

    print(f"Matriz X: {X.shape} (N amostras x P comprimentos de onda)")
    print(f"Classes de teores presentes: {np.unique(y)}")
else:
    print("Aviso: execute primeiro o Caderno 01 para gerar os dados tratados.")"""),
    md_cell("""## 2. Otimização de Variáveis Latentes na PLS-Regression
Calculamos RMSEC (erro de calibração) e RMSECV (erro de validação cruzada com Leave-One-Group-Out) de 1 a 15 LVs.
"""),
    code_cell("""if df_vis is not None:
    df_lvs = otimizar_componentes_plsr(X, y, grupos, max_components=12)
    print("Tabela de Desempenho por Número de LVs:")
    display(df_lvs) if 'display' in dir() else print(df_lvs.to_string())

    # Gráfico de cotovelo RMSEC vs RMSECV
    fig_lvs = plotar_curva_rmsec_rmsecv(df_lvs)
    plt.show()

    # Seleciona número ótimo pelo mínimo RMSECV
    lv_otimo = int(df_lvs.loc[df_lvs['RMSECV'].idxmin()]['LVs'])
    print(f"Número ótimo de Variáveis Latentes sugerido: {lv_otimo}")"""),
    md_cell("""## 3. Modelo Final de PLSR e Incerteza de Medição Tipo A
Avaliamos a calibração com o número ótimo de LVs e calculamos a dispersão e incerteza $u_A = \frac{s}{\sqrt{n}}$ por nível.
"""),
    code_cell("""if df_vis is not None:
    res_pls = avaliar_plsr_final(X, y, grupos, n_components=lv_otimo)

    print(f"R² de Validação Cruzada (LOGO-CV): {res_pls['R2_CV']:.4f}")
    print(f"RMSECV: {res_pls['RMSECV']:.2f}% de teor de adulteração")

    # Gráfico Real vs Predito
    fig_cal = plotar_calibracao_real_vs_predito(
        res_pls['y_real'], res_pls['y_pred_cv'],
        r2=res_pls['R2_CV'], rmse=res_pls['RMSECV'],
        titulo="Calibração PLSR (Leave-One-Group-Out CV)"
    )
    plt.show()

    print("\nTabela de Incertezas do Tipo A:")
    display(res_pls['tabela_incertezas']) if 'display' in dir() else print(res_pls['tabela_incertezas'].to_string())"""),
    md_cell("""## 4. Modelagem Não Linear com Random Forest Regressor
Comparação de desempenho utilizando Random Forest com validação cruzada por pastilha.
"""),
    code_cell("""if df_vis is not None:
    res_rf = avaliar_rf_regressor(
        X, y, grupos=grupos,
        n_estimators=200,
        max_features=0.33,
        random_state=42
    )

    print(f"Random Forest Regressor:")
    print(f"  R² (LOGO-CV): {res_rf['R2']:.4f}")
    print(f"  RMSE (LOGO-CV): {res_rf['RMSE']:.2f}%")

    fig_rf = plotar_calibracao_real_vs_predito(
        res_rf['y_real'], res_rf['y_pred'],
        r2=res_rf['R2'], rmse=res_rf['RMSE'],
        titulo="Calibração Random Forest (LOGO-CV)"
    )
    plt.show()"""),
    md_cell("""## 5. Comparativo Final de Modelos de Regressão"""),
    code_cell("""if df_vis is not None:
    comparativo = pd.DataFrame([
        {"Modelo": f"PLS-Regression ({lv_otimo} LVs)", "R²_CV": round(res_pls['R2_CV'], 4), "RMSECV (%)": round(res_pls['RMSECV'], 2)},
        {"Modelo": "Random Forest (200 árvores)", "R²_CV": round(res_rf['R2'], 4), "RMSECV (%)": round(res_rf['RMSE'], 2)}
    ])
    display(comparativo) if 'display' in dir() else print(comparativo.to_string())""")
]

# ==============================================================================
# NOTEBOOK 4: Abordagem V2 - Razões Elementares e Interpretabilidade
# ==============================================================================
nb4_cells = [
    md_cell("""# 🔬 Caderno 04: Abordagem V2 - Razões Elementares, Padronização Interna e Interpretabilidade
**Projeto:** Detecção de Adulteração em Fertilizantes NPK via LIBS  
**Instituição:** Embrapa Instrumentação  

---

### Contexto da Abordagem V2:
Modelos do espectro completo (como PLSR e RF) são eficazes, mas funcionam como "caixas-pretas" de alta dimensionalidade.  
Na **Abordagem V2**, aplicamos **padronização interna estequiométrica**:
1. **Espectro Representativo Médio** por pastilha para eliminar disparos individuais ruidosos e aumentar a razão sinal-ruído.
2. **Integração Numérica de Áreas de Pico** para linhas de emissão específicas dos elementos adulterantes e da matriz:
   - **Canal VIS:** Razões Cálcio/Potássio (**Ca/K**), ex: `Ca_527/K_693`, `Ca_315/K_580`.
   - **Canal UV:** Razões Cálcio/Fósforo (**Ca/P**) e Magnésio/Fósforo (**Mg/P**).
3. **Métrica de Distância Interclasse de Cohen ($D_{C-K}$):** Ranqueamento automático das razões que melhor separam o fertilizante genuíno dos adulterados.
4. **Modelos Parcimoniosos Lineares (OLS)** com interpretação físico-química direta da inclinação (coeficiente angular).
"""),
    code_cell("""import sys
from pathlib import Path

BASE_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from src.features import (
    gerar_espectro_representativo,
    extrair_areas_picos,
    calcular_razoes_elementares,
    calcular_distancia_interclasse_cohen,
    PICOS_INTERESSE_VIS,
    PICOS_INTERESSE_UV
)
from src.models import (
    regressao_linear_razoes
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix

print("Módulos carregados com sucesso!")"""),
    md_cell("""## 1. Carregamento dos Dados e Geração dos Espectros Representativos
Calculamos o espectro médio de cada pastilha para eliminar o efeito de disparos individuais.
"""),
    code_cell("""pasta_proc = BASE_DIR / "data" / "processed"
caminho_vis = pasta_proc / "df_limpo_vis.parquet"

df_vis = pd.read_parquet(caminho_vis) if caminho_vis.exists() else None

if df_vis is not None:
    # Remove amostras de teste ou descartadas
    df_vis_filtrado = df_vis[df_vis['Amostra_ID'] != 'i_'].copy()

    # Gera espectro representativo por pastilha
    df_rep_vis = gerar_espectro_representativo(df_vis_filtrado)
    print(f"Disparos individuais originais: {len(df_vis_filtrado)}")
    print(f"Pastilhas representativas independentes: {len(df_rep_vis)}")
    display(df_rep_vis[['Teor_Adulteracao', 'Amostra_ID', 'Adulterado']]) if 'display' in dir() else print(df_rep_vis[['Teor_Adulteracao', 'Amostra_ID']].head())
else:
    print("Aviso: execute primeiro o Caderno 01 para gerar os dados tratados.")"""),
    md_cell("""## 2. Integração Numérica das Áreas de Pico (VIS)
Integramos as linhas de emissão do Potássio (K a 693.96, 404.46, 580.20 nm) e Cálcio (Ca a 527.08, 315.96 nm).
"""),
    code_cell("""if df_vis is not None:
    df_areas_vis = extrair_areas_picos(df_rep_vis, PICOS_INTERESSE_VIS, janela_nm=0.3)
    print("Áreas integradas para cada pico:")
    display(df_areas_vis.head()) if 'display' in dir() else print(df_areas_vis.head())"""),
    md_cell("""## 3. Cálculo das Razões Elementares e Ranqueamento via Distância de Cohen ($D_{C-K}$)
Calculamos todas as combinações Ca/K e avaliamos qual razão maximiza a distância padronizada entre a amostra pura e as adulteradas.
"""),
    code_cell("""if df_vis is not None:
    picos_ca = ['Ca_527', 'Ca_315']
    picos_k = ['K_693', 'K_404', 'K_580']

    df_razoes_vis, colunas_razao = calcular_razoes_elementares(
        df_areas_vis, numeradores=picos_ca, denominadores=picos_k
    )

    # Ranking da distância interclasse de Cohen (D_C-K)
    df_tabela_dck = calcular_distancia_interclasse_cohen(df_razoes_vis, colunas_razao, teor_referencia=0.0)
    print("=== Distância Interclasse (D_C-K) por Razão ===")
    display(df_tabela_dck) if 'display' in dir() else print(df_tabela_dck.to_string())

    melhor_razao_1 = df_tabela_dck.iloc[0]['Razão Analisada']
    melhor_razao_2 = df_tabela_dck.iloc[1]['Razão Analisada']
    print(f"\\nAs duas melhores razões selecionadas: {melhor_razao_1} e {melhor_razao_2}")"""),
    md_cell("""## 4. Visualização 2D da Separação por Razões Elementares
Dispersão das amostras no espaço bidimensional das melhores razões de Ca/K.
"""),
    code_cell("""if df_vis is not None:
    df_razoes_vis['Teor_Str'] = df_razoes_vis['Teor_Adulteracao'].apply(lambda x: f"{float(x):.1f}%")

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df_razoes_vis,
        x=melhor_razao_1,
        y=melhor_razao_2,
        hue='Teor_Str',
        palette='Set1',
        s=130,
        edgecolor='black'
    )

    for i in range(len(df_razoes_vis)):
        plt.text(
            df_razoes_vis[melhor_razao_1].iloc[i],
            df_razoes_vis[melhor_razao_2].iloc[i],
            df_razoes_vis['Amostra_ID'].iloc[i],
            fontsize=9, ha='left', va='bottom'
        )

    plt.title(f"Separação 2D de Adulteração: Razões {melhor_razao_1} vs {melhor_razao_2}", fontsize=13, fontweight='bold')
    plt.xlabel(f"Razão Ca/K ({melhor_razao_1})", fontsize=11)
    plt.ylabel(f"Razão Ca/K ({melhor_razao_2})", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Teor Real", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()"""),
    md_cell("""## 5. Modelagem de Classificação com KNN nas Razões 2D (LOO-CV)"""),
    code_cell("""if df_vis is not None:
    X_2d = df_razoes_vis[[melhor_razao_1, melhor_razao_2]].values
    y_2d = df_razoes_vis['Teor_Str'].values

    loo = LeaveOneOut()
    knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    y_pred_knn = cross_val_predict(knn, X_2d, y_2d, cv=loo)

    acuracia = accuracy_score(y_2d, y_pred_knn)
    print(f"Acurácia do KNN 2D (Leave-One-Out CV): {acuracia*100:.2f}%")

    labels_unicos = sorted(list(np.unique(y_2d)))
    cm = confusion_matrix(y_2d, y_pred_knn, labels=labels_unicos)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=labels_unicos, yticklabels=labels_unicos)
    plt.title(f"Matriz de Confusão 2D (Acurácia: {acuracia*100:.1f}%)")
    plt.xlabel("Predição do KNN")
    plt.ylabel("Real")
    plt.show()"""),
    md_cell("""## 6. Regressão Linear Parcimoniosa (OLS) e Interpretabilidade
Avaliamos a relação linear direta entre a razão Ca/K e a concentração de adulteração com Leave-One-Out CV.
"""),
    code_cell("""if df_vis is not None:
    res_linear = regressao_linear_razoes(
        df_razoes_vis,
        features=[melhor_razao_1],
        col_alvo='Teor_Adulteracao'
    )

    print("--- Regressão Linear Univariada (OLS) ---")
    print(f"Equação: Teor (%) = {res_linear['coeficientes'][0]:.4f} * ({melhor_razao_1}) + {res_linear['intercept']:.4f}")
    print(f"R² (LOO-CV): {res_linear['R2_CV']:.4f}")
    print(f"RMSE (LOO-CV): {res_linear['RMSE_CV']:.2f}% de teor")

    # Gráfico de dispersão com reta ajustada
    plt.figure(figsize=(8, 5))
    sns.regplot(
        data=df_razoes_vis,
        x=melhor_razao_1,
        y='Teor_Adulteracao',
        scatter_kws={'s': 80, 'color': 'tab:blue', 'edgecolor': 'black'},
        line_kws={'color': 'tab:red', 'label': f"Reta Ajustada (R² = {res_linear['R2_CV']:.3f})"}
    )
    plt.title(f"Curva Analítica de Calibração: {melhor_razao_1} vs Teor de Adulteração", fontsize=13, fontweight='bold')
    plt.xlabel(f"Razão Espectral ({melhor_razao_1})", fontsize=11)
    plt.ylabel("Teor Real de Adulteração (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.show()"""),
    md_cell("""## 7. Regressão Linear Multivariada (Duas Razões)"""),
    code_cell("""if df_vis is not None:
    res_multi = regressao_linear_razoes(
        df_razoes_vis,
        features=[melhor_razao_1, melhor_razao_2],
        col_alvo='Teor_Adulteracao'
    )

    print("--- Regressão Linear Bivariada ---")
    print(f"R² (LOO-CV): {res_multi['R2_CV']:.4f}")
    print(f"RMSE (LOO-CV): {res_multi['RMSE_CV']:.2f}% de teor")
    print(f"Coeficientes: {res_multi['coeficientes']}")
    print(f"Intercept: {res_multi['intercept']:.4f}")""")
]

# Gerar os 4 cadernos
create_notebook(nb1_cells, "notebooks/01_eda_e_preprocessamento.ipynb")
create_notebook(nb2_cells, "notebooks/02_pca_e_classificacao_knn.ipynb")
create_notebook(nb3_cells, "notebooks/03_regressao_quimiometrica_pls_rf.ipynb")
create_notebook(nb4_cells, "notebooks/04_razoes_elementares_v2.ipynb")
print("Todos os notebooks foram criados com sucesso!")
