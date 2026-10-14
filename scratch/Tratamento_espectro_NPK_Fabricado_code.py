# %% [markdown] Cell 1
# ## Blibiotecas


# %% [code] Cell 2
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import re
import io
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import euclidean_distances


from itertools import combinations
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GroupKFold
import glob
from scipy.optimize import curve_fit
from scipy.ndimage import white_tophat
import pywt
import plotly.graph_objects as go
import os


from google.colab import drive



# %% [markdown] Cell 3
# ## Funções


# %% [code] Cell 4

# PRÉ-PROCESSAMENTO


def wavelet_denoising(data, wavelet='sym8', level=2):
    coeff = pywt.wavedec(data, wavelet, mode="per")
    actual_level = min(level, len(coeff) - 1)
    sigma = (1/0.6745) * np.median(np.abs(coeff[-1] - np.median(coeff[-1])))  #Eq. 8 Adaptive approach for variable noise suppression on laser-induced breakdown spectroscopy responses using stationary wavelet transform

    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    new_coeff = [coeff[0]] + [pywt.threshold(c, value=uthresh, mode='hard') for c in coeff[1:]]
    rec = pywt.waverec(new_coeff, wavelet, mode='per')
    return rec[:len(data)]



def normalizacao_area(y):

    area = simpson(y)
    if area == 0:
        return y
    return y / area


def snv_normalizacao(espectro):
    """
    Aplica a normalização Standard Normal Variate (SNV).
    Subtrai a média e divide pelo desvio padrão de cada espectro individual.
    """
    media = np.mean(espectro)
    desvio_padrao = np.std(espectro, ddof=1)

    if desvio_padrao == 0:
        return espectro - media

    espectro_snv = (espectro - media) / desvio_padrao
    return espectro_snv

def filtrar_outliers(df_consolidado, threshold_percentil=100):
    """
    Varre o DataFrame consolidado amostra por amostra (Amostra_ID)
    e remove os disparos que são outliers usando a métrica SAM.
    """
    lista_dfs_limpos = []

    metadados = ['Amostra_ID', 'Num_Disparo', 'Nome_Arquivo', 'Canal_Optico', 'Teor_Adulteracao']
    cols_features = [c for c in df_consolidado.columns if c not in metadados]

    for amostra, grupo in df_consolidado.groupby('Amostra_ID'):
        X = grupo[cols_features].values

        media_grupo = X.mean(axis=0).reshape(1, -1)

        normas_X = np.linalg.norm(X, axis=1)
        norma_media = np.linalg.norm(media_grupo)
        dot_product = np.dot(X, media_grupo.T).flatten()

        cos_sim = dot_product / (normas_X * norma_media + 1e-9)
        angulos = np.arccos(np.clip(cos_sim, -1.0, 1.0))

        # Define o corte: Ex. descartar os 10% dos disparos mais distantes, percentil 90
        limite_corte = np.percentile(angulos, threshold_percentil)

        grupo_limpo = grupo[angulos <= limite_corte].copy()
        lista_dfs_limpos.append(grupo_limpo)

        print(f"Amostra {amostra}: {len(grupo) - len(grupo_limpo)} disparos anômalos removidos.")

    df_limpo_final = pd.concat(lista_dfs_limpos, ignore_index=True)
    return df_limpo_final

# LEITURA DE DADOS

def ler_espectro_libs_generico(caminho_arquivo):
    with open(caminho_arquivo, "r", encoding="utf-8", errors="ignore") as f:
        conteudo = f.read()

    if "[data section]" in conteudo.lower():
        linhas_de_dados = []
        inicio_dos_dados = False
        for linha in conteudo.split('\n'):
            if "[data section]" in linha.lower():
                inicio_dos_dados = True
                continue
            if inicio_dos_dados and linha.strip():
                linhas_de_dados.append(linha)

        dados_string = "\n".join(linhas_de_dados)
        df_bruto = pd.read_csv(io.StringIO(dados_string), sep=";", header=None, decimal=".")

        if df_bruto.shape[1] >= 3:
            df_bruto.columns = ["Wavelength", "Pixel_Index", "Intensity", "Diffraction_Order"][:df_bruto.shape[1]]
            df_bruto = df_bruto.sort_values(by="Wavelength").reset_index(drop=True)
            return df_bruto["Wavelength"].values, df_bruto["Intensity"].values.flatten()

    try:
        df_bruto = pd.read_csv(caminho_arquivo, sep=r'\s+', header=None)
        wl = df_bruto.iloc[:, 0].values
        sinal = df_bruto.iloc[:, 1].values  # Pega o primeiro vetor de sinal disponível
        return wl, sinal.flatten()
    except Exception as e:
        raise ValueError(f"Formato de arquivo não suportado para {caminho_arquivo}: {e}")


# Correção: O parâmetro 'canal' foi adicionado na assinatura da função
def carregar_esf_bruto(caminho_completo, canal=None):
    """Lê o arquivo do disco e extrai metadados, retornando um dicionário bruto."""
    nome_arquivo = os.path.basename(caminho_completo)

    match_amostra = re.search(r'amostra_([a-z_]+)', nome_arquivo)
    match_disparo = re.search(r'__(\d+)', nome_arquivo)

    amostra_id = match_amostra.group(1) if match_amostra else "desconhecida"
    num_disparo = int(match_disparo.group(1)) if match_disparo else 0

    # Usa a sua função genérica que já funciona perfeitamente
    wl, sinal_bruto = ler_espectro_libs_generico(caminho_completo)

    # Retorna os arrays numpy puros (leves para a memória RAM)
    return {
        'Wavelength': wl,
        'Intensity_Raw': sinal_bruto,
        'Amostra_ID': amostra_id,
        'Num_Disparo': num_disparo,
        'Nome_Arquivo': nome_arquivo
    }
# 3. PIPELINE

def pipeline_processamento_disparo(caminho_completo):
    nome_arquivo = os.path.basename(caminho_completo)

    match_amostra = re.search(r'amostra_([a-z_]+)', nome_arquivo)
    match_disparo = re.search(r'__(\d+)', nome_arquivo)

    amostra_id = match_amostra.group(1) if match_amostra else "desconhecida"
    num_disparo = int(match_disparo.group(1)) if match_disparo else 0

    wl, sinal_bruto = ler_espectro_libs_generico(caminho_completo)

    # Ruído
    #sinal_pos_wavelet = wavelet_denoising(sinal_bruto, wavelet='sym8', level=6)
    #sinal_s_ruido = savgol_filter(sinal_pos_wavelet, window_length=7, polyorder=2)
    # Baseline

    sinal_s_baseline = white_tophat(sinal_bruto, footprint=np.ones(100))

    # Normalização Área
    sinal_final = normalizacao_area(sinal_s_baseline)


    wl = np.round(wl, 4)   # arredondamento para tentar corrigir o UV



    df_linha = pd.DataFrame([sinal_final], columns=wl)
    df_linha['Amostra_ID'] = amostra_id
    df_linha['Num_Disparo'] = num_disparo
    df_linha['Nome_Arquivo'] = nome_arquivo

    return df_linha




def processar_espectro(dados_brutos):
    """Aplica os filtros num sinal que JÁ ESTÁ na memória RAM."""
    wl = dados_brutos['Wavelength']
    sinal_bruto = dados_brutos['Intensity_Raw']

    # --- INÍCIO DA ZONA DE TESTES MATEMÁTICOS ---
    # Ruído
    #sinal_pos_wavelet = wavelet_denoising(sinal_bruto, wavelet='sym8', level=6)
    #sinal_s_ruido = savgol_filter(sinal_pos_wavelet, window_length=7, polyorder=2)

    # Baseline
    sinal_s_baseline = white_tophat(sinal_bruto, footprint=np.ones(100))

    # Normalização
    sinal_final = snv_normalizacao(sinal_s_baseline)

    minimo_atual = np.min(sinal_final)
    sinal_final = sinal_final - minimo_atual

    # Eixo X em Texto para blindar o Pandas (nossa correção de antes)
    wl_formatado_texto = [f"{w:.2f}" for w in wl]

    # Salva no dicionário ultrarrápido
    df_linha = dict(zip(wl_formatado_texto, sinal_final))


    # Monta a linha final do DataFrame usando os metadados preservados
    df_linha = pd.DataFrame([sinal_final], columns=wl)
    df_linha['Amostra_ID'] = dados_brutos['Amostra_ID']
    df_linha['Num_Disparo'] = dados_brutos['Num_Disparo']
    df_linha['Nome_Arquivo'] = dados_brutos['Nome_Arquivo']

    return df_linha


# %% [code] Cell 5
drive.mount('/content/drive', force_remount=True)

# 2. Apaga a cópia incompleta que ficou travada na memória local
!rm -rf "/content/NPK_10-10-10_Fabricado"

# 3. AUDITORIA: Vamos perguntar ao Colab se ele está enxergando a pasta UV antes de copiar
print("O que o Colab enxerga na pasta de 50% lá no Drive?")
!ls "/content/drive/MyDrive/LIBS/Espectros/NPK_10-10-10_Fabricado/Adulterado_50porcento"

# 4. Faz a cópia pesada novamente
print("\nIniciando a cópia para o disco local...")
!cp -r "/content/drive/MyDrive/LIBS/Espectros/NPK_10-10-10_Fabricado" "/content/"
print("Cópia concluída!")

# 5. Verifica se a cópia deu certo
print("\nO que foi copiado com sucesso para a máquina local:")
!ls "/content/NPK_10-10-10_Fabricado/Adulterado_50porcento"


# %% [markdown] Cell 6
# #### Código de carregamento otimizado


# %% [code] Cell 7

caminho_trabalho = "/content/NPK_10-10-10_Fabricado"
os.chdir(caminho_trabalho)

print(f"Diretório atual: {os.getcwd()}")

arquivos_disparos = []
for root, dirs, files in os.walk('.'):
    for f in files:
        if f.lower().endswith('.esf'):
            arquivos_disparos.append(os.path.join(root, f))

arquivos_disparos.sort()
print(f"A processar {len(arquivos_disparos)} disparos...")


lista_dicionarios_uv_bruto = []
lista_dicionarios_vis_bruto = []

for arq_path in tqdm(arquivos_disparos, desc="Lendo e Extraindo do Disco", unit="tiro"):
    try:
        pasta_do_arquivo = os.path.dirname(arq_path)
        pasta_acima = os.path.dirname(pasta_do_arquivo)
        nome_pasta_canal = os.path.basename(pasta_do_arquivo).upper().strip()
        nome_pasta_adulteracao = os.path.basename(pasta_acima).lower().strip()

        dados_tiro = carregar_esf_bruto(arq_path, nome_pasta_canal)

        if '50porcento' in nome_pasta_adulteracao: teor = 50.0
        elif '00porcento' in nome_pasta_adulteracao: teor = 0.0
        elif '10porcento' in nome_pasta_adulteracao: teor = 10.0
        else: teor = 25.0

        dados_tiro['Teor_Adulteracao'] = teor
        dados_tiro['Canal_Optico'] = nome_pasta_canal

        if 'UV' in nome_pasta_canal:
            lista_dicionarios_uv_bruto.append(dados_tiro)
        elif 'VIS' in nome_pasta_canal:
            lista_dicionarios_vis_bruto.append(dados_tiro)

    except Exception as e:
        print(f"\nErro no ficheiro {arq_path}: {e}")


# %% [markdown] Cell 8
# ## Visível


# %% [code] Cell 9
lista_linhas_vis = []

# for dados_brutos in tqdm(lista_dicionarios_uv_bruto, desc="Aplicando Top-Hat/Filtros no UV"):
#     df_processado = processar_espectro(dados_brutos)
#     df_processado['Teor_Adulteracao'] = dados_brutos['Teor_Adulteracao']
#     df_processado['Canal_Optico'] = dados_brutos['Canal_Optico']
#     lista_linhas_uv.append(df_processado)

for dados_brutos in tqdm(lista_dicionarios_vis_bruto, desc="Aplicando Top-Hat/Filtros no VIS"):
    df_processado = processar_espectro(dados_brutos)
    df_processado['Teor_Adulteracao'] = dados_brutos['Teor_Adulteracao']
    df_processado['Canal_Optico'] = dados_brutos['Canal_Optico']
    lista_linhas_vis.append(df_processado)

# df_final_uv = pd.concat(lista_linhas_uv, ignore_index=True)
df_final_vis = pd.concat(lista_linhas_vis, ignore_index=True)
print("Pronto para rodar")


# %% [code] Cell 10


# ==========================================
# 5. CONSOLIDAÇÃO DO CANAL VIS
# ==========================================
print("\n" + "="*40)
print("PROCESSANDO CANAL VIS")
print("="*40)
if len(lista_linhas_vis) > 0:
    df_final_VIS = pd.concat(lista_linhas_vis, ignore_index=True)
    df_limpo_VIS = filtrar_outliers(df_final_VIS, threshold_percentil=99)

    print(f"Tamanho VIS original: {df_final_VIS.shape[0]} linhas")
    print(f"Tamanho VIS após SAM: {df_limpo_VIS.shape[0]} linhas")
    print(f"Formato Matriz VIS : {df_limpo_VIS.shape}")
    print(df_limpo_VIS.groupby('Teor_Adulteracao').size())
else:
    print("Nenhum dado de VIS encontrado.")


# %% [markdown] Cell 11
# ### Dataframes e Visualizações


# %% [code] Cell 12
df_limpo_VIS.sample(10)



# %% [code] Cell 13
df_00 = df_limpo_VIS[df_limpo_VIS['Teor_Adulteracao'] == 0.0]
df_50 = df_limpo_VIS[df_limpo_VIS['Teor_Adulteracao'] == 50.0]
df_10 = df_limpo_VIS[df_limpo_VIS['Teor_Adulteracao'] == 10.0]
df_25 = df_limpo_VIS[df_limpo_VIS['Teor_Adulteracao'] == 25.0]    # Carreguei os dados como sendo -1 mas já mudei para 25.0. Trocar na próxima

cols_features = [c for c in df_limpo_VIS.columns if c not in ['Amostra_ID', 'Num_Disparo', 'Nome_Arquivo', 'Canal_Optico', 'Teor_Adulteracao']]

espectro_medio_00 = df_00[cols_features].mean(axis=0).values
espectro_medio_50 = df_50[cols_features].mean(axis=0).values
espectro_medio_10 = df_10[cols_features].mean(axis=0).values
espectro_medio_25 = df_25[cols_features].mean(axis=0).values
comprimentos_onda = np.array(cols_features, dtype=float)

df_espectros = pd.DataFrame({
    'Wavelength': comprimentos_onda,
    'Média_Puro': espectro_medio_00,
    'Média_Adulterado_50pct': espectro_medio_50,
    'Média_Adulterado_10pct': espectro_medio_10,
    'Média_Adulterado_25pct': espectro_medio_25
})

df_espectros = df_espectros.sort_values(by='Wavelength'
)

fig = px.line(
    df_espectros,
    x='Wavelength',
    y=['Média_Puro','Média_Adulterado_50pct', 'Média_Adulterado_10pct', 'Média_Adulterado_25pct'],
    title='',
    labels={'value': 'Intensidade Normalizada', 'variable': 'Espectro', 'Wavelength': 'Comprimento de Onda (nm)'}
)

fig.data[0].line.color = 'lightgray'
fig.data[0].line.width = 3
fig.data[1].line.color = 'red'
fig.data[1].line.width = 1.5

fig.update_layout(
    template='plotly_white',
    hovermode='x unified',
    legend_title_text='Legenda'
)

fig.show()


# %% [markdown] Cell 14
# #### Análise do tratamento de ruído
# 
# Aqui ploto o espectro sem filtro algum e sobreposto a ele um espectro filtrado com wavelet e outro com savgol.


# %% [code] Cell 15


metadados = ['Amostra_ID', 'Num_Disparo', 'Nome_Arquivo', 'Canal_Optico', 'Teor_Adulteracao']
cols_wl = [c for c in df_limpo_VIS.columns if c not in metadados]
wl = np.array(cols_wl, dtype=float)


# Filtra o DF pelo Teor, pedimos 1 linha aleatória (.sample) e pegamos os dados dela (.iloc[0]), para mudar a amostra, muda-se o random_state
linha_0 = df_limpo_VIS[df_limpo_VIS['Teor_Adulteracao'] == 0.0].sample(n=1, random_state=50).iloc[0]
linha_50 = df_limpo_VIS[df_limpo_VIS['Teor_Adulteracao'] == 50.0].sample(n=1, random_state=50).iloc[0]

sinal_bruto_0 = linha_0[cols_wl].values.astype(float)
sinal_bruto_50 = linha_50[cols_wl].values.astype(float)


# Inicio deo Tratamento do ruído



sinal_wave_0 = wavelet_denoising(sinal_bruto_0, wavelet='sym8', level=6)
sinal_wave_50 = wavelet_denoising(sinal_bruto_50, wavelet='sym8', level=6)

sinal_savgol_0 = savgol_filter(sinal_bruto_0, window_length=7, polyorder=2)
sinal_savgol_50 = savgol_filter(sinal_bruto_50, window_length=7, polyorder=2)

sinal_limpo_0 = savgol_filter(sinal_wave_0, window_length=7, polyorder=2)
sinal_limpo_50 = savgol_filter(sinal_wave_50, window_length=7, polyorder=2)

fig = make_subplots(rows=2, cols=1,
                    subplot_titles=(f"Antes vs Depois - 0% NPK (Arq: {linha_0['Nome_Arquivo']})",
                                    f"Antes vs Depois - 50% Adulterado (Arq: {linha_50['Nome_Arquivo']})"),
                    vertical_spacing=0.15)

fig.add_trace(go.Scatter(x=wl, y=sinal_bruto_0, mode='lines', name='Sinal DF (0%)',
                         line=dict(color='lightgray', width=1.5), opacity=0.8), row=1, col=1)
fig.add_trace(go.Scatter(x=wl, y=sinal_wave_0, mode='lines', name='Pós-Wavelet (0%)',
                         line=dict(color='darkcyan', width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=wl, y=sinal_limpo_0, mode='lines', name='Pós-Wavelet e SavGol (0%)',
                         line=dict(color='darkcyan', width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=wl, y=sinal_savgol_0, mode='lines', name='Pós-Savgol (0%)',
                         line=dict(color='crimson', width=1.5)), row=1, col=1)



fig.add_trace(go.Scatter(x=wl, y=sinal_bruto_50, mode='lines', name='Sinal DF (50%)',
                         line=dict(color='lightgray', width=1.5), opacity=0.8), row=2, col=1)
fig.add_trace(go.Scatter(x=wl, y=sinal_wave_50, mode='lines', name='Pós-Wavelet (50%)',
                         line=dict(color='darkcyan', width=1.5)), row=2, col=1)
fig.add_trace(go.Scatter(x=wl, y=sinal_limpo_50, mode='lines', name='Pós-Wavelet e SavGol (50%)',
                         line=dict(color='darkcyan', width=1.5)), row=2, col=1)
fig.add_trace(go.Scatter(x=wl, y=sinal_savgol_50, mode='lines', name='Pós-Savgol (50%)',
                         line=dict(color='crimson', width=1.5)), row=2, col=1)


fig.update_layout(height=700, width=1300, template="plotly_white",
                  title_text="Validação da Remoção de Ruído",
                  title_x=0.5, hovermode="x unified")

fig.update_xaxes(title_text="Comprimento de Onda (nm)", row=2, col=1)
fig.show()


# %% [markdown] Cell 16
# #### Análise da remoção de baseline
# 
# 
# Aqui comparo o espectro bruto com o tratamento de baseline polinomial e um outro método que descobri discutindo com um colega, apenas com o objetivo de comparar a performance.


# %% [code] Cell 17

raio_da_bola = 100




base_0 = baseline_iterativa(wl, sinal_bruto_0, ordem = 4)
sinal_baseline_0 = sinal_bruto_0 - base_0
base_50 = baseline_iterativa(wl, sinal_bruto_50, ordem = 4)
sinal_baseline_50 = sinal_bruto_50 - base_50

sinal_tophat_0 = white_tophat(sinal_bruto_0, footprint=np.ones(raio_da_bola))
sinal_tophat_50 = white_tophat(sinal_bruto_50, footprint=np.ones(raio_da_bola))




fig = make_subplots(rows=2, cols=1,
                    subplot_titles=(f"Antes vs Depois - 0% NPK (Arq: {linha_0['Nome_Arquivo']})",
                                    f"Antes vs Depois - 50% Adulterado (Arq: {linha_50['Nome_Arquivo']})"),
                    vertical_spacing=0.15)

fig.add_trace(go.Scatter(x=wl, y=sinal_bruto_0, mode='lines', name='Sinal DF (0%)',
                         line=dict(color='lightgray', width=1.5), opacity=0.8), row=1, col=1)
fig.add_trace(go.Scatter(x=wl, y=sinal_baseline_0, mode='lines', name='Baseline (0%)',
                         line=dict(color='darkcyan', width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=wl, y=sinal_tophat_0, mode='lines', name='Tophat (0%)',
                         line=dict(color='crimson', width=1.5)), row=1, col=1)




fig.add_trace(go.Scatter(x=wl, y=sinal_bruto_50, mode='lines', name='Sinal DF (50%)',
                         line=dict(color='lightgray', width=1.5), opacity=0.8), row=2, col=1)
fig.add_trace(go.Scatter(x=wl, y=sinal_baseline_50, mode='lines', name='Baseline (50%)',
                         line=dict(color='darkcyan', width=1.5)), row=2, col=1)
fig.add_trace(go.Scatter(x=wl, y=sinal_tophat_50, mode='lines', name='Tophat (50%)',
                         line=dict(color='crimson', width=1.5)), row=2, col=1)



fig.update_layout(height=700, width=1300, template="plotly_white",
                  title_text="Validação da Remoção de Baseline",
                  title_x=0.5, hovermode="x unified")

fig.update_xaxes(title_text="Comprimento de Onda (nm)", row=2, col=1)
fig.show()


# %% [markdown] Cell 18
# ### PCA - Vis


# %% [code] Cell 19

df_limpo_VIS_PCA = df_limpo_VIS[
    (df_limpo_VIS['Amostra_ID'] != 'i_') &
    (df_limpo_VIS['Teor_Adulteracao'] != 10.0) &
    (df_limpo_VIS['Teor_Adulteracao'] != 25.0)
]

metadados = ['Amostra_ID', 'Num_Disparo', 'Nome_Arquivo', 'Canal_Optico', 'Teor_Adulteracao']
wl_min = 273
wl_max = 780

cols_features= [
    c for c in df_limpo_VIS_PCA.columns
    if c not in metadados and wl_min <= float(c) <= wl_max
]


tamanho_minimo = df_limpo_VIS_PCA['Teor_Adulteracao'].value_counts().min()
print(f"Nivelando a matriz: todas as amostras terão exatamente {tamanho_minimo} disparos.")


df_balanceado = df_limpo_VIS_PCA.groupby('Teor_Adulteracao').sample(n=tamanho_minimo, random_state=42) # Undersampling: Amostrar aleatoriamente 'tamanho_minimo' linhas de cada letra

metadados = ['Amostra_ID', 'Num_Disparo', 'Nome_Arquivo', 'Canal_Optico', 'Teor_Adulteracao']
X = df_balanceado[cols_features].values
y = df_balanceado['Teor_Adulteracao'].values

pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)



# SCREE PLOT (VARIÂNCIA EXPLICADA)
# ==============================================================================
plt.figure(figsize=(7, 4))
variancia_acumulada = np.cumsum(pca.explained_variance_ratio_) * 100
plt.plot(range(1, 11), variancia_acumulada, marker='o', linestyle='--', color='darkblue', linewidth=2)
plt.xlabel('Número de Componentes Principais (PCs)', fontsize=11)
plt.ylabel('Variância Explicada Acumulada (%)', fontsize=11)
plt.title('Scree Plot: Variância Capturada pelo PCA - Visível', fontsize=12, fontweight='bold')
plt.xticks(range(1, 11))
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

print(f"-> PC1 captura sozinho: {pca.explained_variance_ratio_[0]*100:.2f}% de toda a variância química.")
print(f"-> PC2 captura sozinho: {pca.explained_variance_ratio_[1]*100:.2f}% de toda a variância química.")
print(f"-> Total acumulado (PC1 + PC2): {variancia_acumulada[1]:.2f}%")



# SCORE PLOT (PC1 VS PC2)
# ==============================================================================
df_pca = pd.DataFrame(X_pca[:, :6], columns=['PC1', 'PC2','PC3', 'PC4', 'PC5', 'PC6'])
df_pca['Teor_Adulteracao'] = y

df_pca['Amostra_ID'] = df_balanceado['Amostra_ID'].values
df_pca['Teor_Adulteracao'] = df_balanceado['Teor_Adulteracao'].values


plt.figure(figsize=(10, 6))

x_score = 'PC1'
y_score = 'PC2'

sns.scatterplot(
    data=df_pca,
    x=x_score, y=y_score,
    hue = 'Teor_Adulteracao',
    #hue='Amostra_ID',
    #style='Teor_Adulteracao',
    # markers={0.0: '*', 50.0: 'o', 10.0:'X', -1.0:'s'},
    #markers = {25.0:'X', 10.0:"*"},
    palette='turbo',
    s=80, alpha=0.9, edgecolor='black', linewidth=0.5
)

plt.xlabel(f'{x_score} ({pca.explained_variance_ratio_[int(x_score[2]) - 1]*100:.1f}%)', fontsize=11)
plt.ylabel(f'{y_score} ({pca.explained_variance_ratio_[int(y_score[2]) - 1]*100:.1f}%)', fontsize=11)
plt.title(f'Score Plot {x_score}  e {y_score} - Visível', fontsize=12, fontweight='bold')
plt.legend(title='ID da Amostra', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()


# %% [code] Cell 20
linhas_emissao = {
    'Ca II': [396.87, 393.37, 317.97, 315.9, 370.60, 373.69],
    'K': [766.48, 769.92],
    'Mg': [279.58],
    'Fe I': [430.82, 432.54],
    'Ca I': [420.69, 422.67, 430.26, 431.88, 442.56, 443.51, 445.47, 527.03, 612.22, 643.91]}

cores_elementos = {
    'Ca II': 'purple',
    'K': 'orange',
    'Mg': 'magenta',
    'Fe I': 'gray',
    'Ca I': 'red'
}

comprimentos_onda = np.array([float(c) for c in cols_features])
wl_min, wl_max = comprimentos_onda.min(), comprimentos_onda.max()

espectros_50 = df_balanceado[df_balanceado['Teor_Adulteracao'] == 50.0][cols_features].values
espectro_medio_50 = espectros_50.mean(axis=0)

# Extrair os Loadings
loadings = pca.components_

# 4. Configurar a figura com 4 andares
fig, axes = plt.subplots(4, 1, figsize=(14, 15), sharex=True)
fig.suptitle('Análise de Loadings vs. Espectro Físico (50%) e Linhas de Emissão', fontsize=16, fontweight='bold', y=0.98)

cores_pcs = ['darkred', 'darkblue', 'darkgreen']
pcs = ['PC1', 'PC2', 'PC3']


for i in range(3):
    axes[i].plot(comprimentos_onda, loadings[i, :], color=cores_pcs[i], linewidth=1.2)
    axes[i].set_title(f'Loadings da {pcs[i]} (Variância explicada: {pca.explained_variance_ratio_[i]*100:.1f}%)', fontsize=12)
    axes[i].set_ylabel('Peso no PC', fontsize=11)
    axes[i].grid(True, linestyle=':', alpha=0.5)
    axes[i].axhline(0, color='black', linewidth=1, linestyle='--')


axes[3].plot(comprimentos_onda, espectro_medio_50, color='black', linewidth=1.2)
axes[3].set_title('Espectro Médio - Amostras com 50% de Adulteração', fontsize=12)
axes[3].set_ylabel('Intensidade', fontsize=11)
axes[3].grid(True, linestyle=':', alpha=0.5)


picos_ordenados = []
for elemento, comprimentos in linhas_emissao.items():
    cor = cores_elementos[elemento]
    for wl in comprimentos:
        if wl_min <= wl <= wl_max:
            picos_ordenados.append((wl, elemento, cor))

picos_ordenados.sort(key=lambda x: x[0])


for ax in axes:
    ymin, ymax = ax.get_ylim()

    niveis_altura = [0.95, 0.82, 0.69, 0.56]

    for i, (wl, elemento, cor) in enumerate(picos_ordenados):
        ax.axvline(x=wl, color=cor, linewidth=1.5, linestyle='-.', alpha=0.7)

        altura_atual = ymax * niveis_altura[i % len(niveis_altura)]

        ax.text(wl + 0.3, altura_atual, f'{elemento} ({wl})', color=cor,
                rotation=45, verticalalignment='top', fontsize=9, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
axes[3].set_xlim(wl_min, wl_max)
axes[3].set_xlabel('Comprimento de Onda (nm)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(top=0.94)
plt.show()


# %% [code] Cell 21

top_n = 10               # Quantos picos DISTINTOS queremos
tolerancia_id = 0.5      # Margem de erro para adivinhar o elemento
distancia_minima = 1.0

dados_tabela = []
loadings = pca.components_
variancias = pca.explained_variance_ratio_ * 100
comprimentos_onda = np.array([float(c) for c in cols_features])

# 3. Varredura com Filtro de Redundância
for i in range(3):
    pc_nome = f"PC{i+1}"
    var_explicada = f"{variancias[i]:.1f}%"

    pesos_absolutos = np.abs(loadings[i])
    indices_ordenados = np.argsort(pesos_absolutos)[::-1]

    picos_selecionados_wls = [] # Guarda os comprimentos de onda já escolhidos

    for idx in indices_ordenados:
        wl_real = comprimentos_onda[idx]
        peso_real = loadings[i][idx]

        # CHECAGEM DE REDUNDÂNCIA (A MÁGICA ACONTECE AQUI)
        redundante = False
        for wl_salvo in picos_selecionados_wls:
            if abs(wl_real - wl_salvo) < distancia_minima:
                redundante = True
                break

        # Se for do mesmo pico, ignora e vai para o próximo pixel
        if redundante:
            continue

        # Se passou no filtro, guarda o pico
        picos_selecionados_wls.append(wl_real)

        # Identificar o elemento
        elemento_identificado = "-"
        for elemento, wls in linhas_emissao.items():
            for wl_teorico in wls:
                if abs(wl_real - wl_teorico) <= tolerancia_id:
                    elemento_identificado = f"{elemento} ({wl_teorico})"
                    break

        dados_tabela.append({
            "Componente": pc_nome,
            "Variância (%)": var_explicada,
            "Comprimento de Onda (nm)": round(wl_real, 2),
            "Peso (Loading)": round(peso_real, 4),
            "Elemento Atribuído": elemento_identificado
        })

        # Para quando acharmos os 5 picos realmente distintos
        if len(picos_selecionados_wls) == top_n:
            break

# 4. Mostrar tabela
df_loadings = pd.DataFrame(dados_tabela)
df_loadings.set_index(["Componente", "Variância (%)"], inplace=True)

print("Tabela de Atribuição dos Loadings (Sem Repetição de Picos):")
display(df_loadings)
df_loadings.to_excel('tabela_loadings.xlsx')


# %% [markdown] Cell 22
# ### KNN


# %% [code] Cell 23
X_pca = X_pca[:, :]   # Depois de analisar o gráfico de scores e interpretar que a PC1 continha informações que não estavam relacionadas com a química, escolheu-se filtrar aquelas que indicam a química real.


X_treino, X_teste, y_treino, y_teste = train_test_split(
    X_pca, y, test_size=0.3, random_state=45, stratify=y
)


knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_treino, y_treino)

y_pred = knn.predict(X_teste)
acuracia = accuracy_score(y_teste, y_pred)

print(f"--- TESTE DE HOMOGENEIDADE ---")
print(f"Acurácia do kNN: {acuracia * 100:.2f}%")


cm = confusion_matrix(y_teste, y_pred, labels=np.unique(y))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=np.unique(y),
            yticklabels=np.unique(y))
plt.title(f'Matriz de Confusão Visível (Acurácia: {acuracia*100:.1f}%)')
plt.xlabel('Previsão do Modelo')
plt.ylabel('Amostra Real')
plt.show()


# %% [code] Cell 24

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

X_filtrado = X_pca[:, 1:4]

teores_adulteracao = [10.0, 25.0, 50.0]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Teste de Homogeneidade Binário (Puro vs Adulterado) - Canal VIS', fontsize=16, fontweight='bold', y=1.05)

print("="*50)
print("RELATÓRIO DE MÉTRICAS (Classe Positiva = Adulterado)")
print("="*50)

for i, teor_alvo in enumerate(teores_adulteracao):

    mascara = (y == 0.0) | (y == teor_alvo)
    X_binario = X_filtrado[mascara]
    y_binario = y[mascara]

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X_binario, y_binario, test_size=0.3, random_state=45, stratify=y_binario
    )

    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(X_treino, y_treino)

    y_pred = knn.predict(X_teste)


    acuracia = accuracy_score(y_teste, y_pred)
    # O pos_label indica quem é o "Positivo" (o adulterado)
    precisao = precision_score(y_teste, y_pred, pos_label=teor_alvo)
    recall = recall_score(y_teste, y_pred, pos_label=teor_alvo)
    f1 = f1_score(y_teste, y_pred, pos_label=teor_alvo)

    print(f"\n-> PAR: 0% vs {teor_alvo}%")
    print(f"   Acurácia : {acuracia*100:.2f}%")
    print(f"   Precisão : {precisao:.4f} (Se o modelo diz que é {teor_alvo}%, qual a chance de acertar?)")
    print(f"   Recall   : {recall:.4f} (De todos os {teor_alvo}% reais, quantos o modelo encontrou?)")
    print(f"   F1-Score : {f1:.4f} (Média harmônica entre Precisão e Recall)")

    classes_par = [0.0, teor_alvo]
    cm = confusion_matrix(y_teste, y_pred, labels=classes_par)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=[f'Prev. 0%', f'Prev. {teor_alvo}%'],
                yticklabels=[f'Real 0%', f'Real {teor_alvo}%'],
                ax=axes[i], cbar=False)

    axes[i].set_title(f'0% vs {teor_alvo}%\n Acurácia: {acuracia*100:.2f}%', fontsize=12)
    axes[i].set_xlabel('Previsão do Modelo')
    axes[i].set_ylabel('Amostra Real')

plt.tight_layout()
plt.show()


# %% [markdown] Cell 25
# #### Validar com Leave Group Out
# 
# Irei deixar os grupos g e h para teste apenas, enquanto os outros são para treino.


# %% [code] Cell 26
df_limpo_VIS.groupby('Amostra_ID').size()


# %% [code] Cell 27

grupos_ids = df_balanceado['Amostra_ID'].values

ids_para_treinar = ['a_', 'b_', 'c_', 'd_', 'f_']
ids_para_testar = ['g_', 'h_']

mask_treino = np.isin(grupos_ids, ids_para_treinar)
mask_teste = np.isin(grupos_ids, ids_para_testar)

X_treino = X_pca[mask_treino]
y_treino = y[mask_treino]

X_teste = X_pca[mask_teste]
y_teste = y[mask_teste]

print(f"Disparos no Treino: {len(X_treino)} (IDs: {np.unique(grupos_ids[mask_treino])})")
print(f"Disparos no Teste: {len(X_teste)} (IDs: {np.unique(grupos_ids[mask_teste])})")


knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_treino, y_treino)

y_pred = knn.predict(X_teste)
acuracia = accuracy_score(y_teste, y_pred)

print(f"\n--- TESTE DE INDEPENDÊNCIA DE MATRIZ ---")
print(f"Acurácia do kNN em amostras INÉDITAS: {acuracia * 100:.2f}%\n")


cm = confusion_matrix(y_teste, y_pred, labels=np.unique(y))

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=np.unique(y),
            yticklabels=np.unique(y))
plt.title(f'Matriz de Confusão: Amostras {ids_para_testar}\n(Acurácia: {acuracia*100:.1f}%)', fontweight='bold')
plt.xlabel('Previsão do Modelo (kNN)')
plt.ylabel('Teor Real')
plt.show()


# %% [markdown] Cell 28
# Testar sem o vazamento de dados
# 
# a ideia é que ao criar o espaço de componentes principais com os dados de teste, voce acaba influenciando o resultado uma vez que o eixo, com os dados de teste, não tem a mesma posição que o eixo sem.
# 
# 
# 
# 


# %% [code] Cell 29

df_balanceado = df_balanceado[df_balanceado['Amostra_ID'] != 'i_']


# Defina o que você quer analisar.
# Opções válidas:
# 'todos' -> Roda a matriz inteira (0, 10, 25, 50)
# [0.0, 10.0] -> Roda apenas Puro vs 10%
# [0.0, 50.0] -> Roda apenas Puro vs 50%

alvos_analise = [50.0, 0.0]  # <--- MUDE AQUI!

if alvos_analise == 'todos':
    df_filtrado = df_balanceado.copy()
    print("Modo de Análise: MULTICLASSE (Todas as Concentrações)")
else:
    df_filtrado = df_balanceado[df_balanceado['Teor_Adulteracao'].isin(alvos_analise)].copy()
    print(f"Modo de Análise: BINÁRIO (Concentrações Alvo: {alvos_analise})")


X = df_filtrado[cols_features].values
y = df_filtrado['Teor_Adulteracao'].values
grupos_ids = df_filtrado['Amostra_ID'].values

ids_unicos = np.unique(grupos_ids)
print(f"IDs únicos disponíveis para combinação: {ids_unicos}")

tamanho_teste = 1
todas_combinacoes_teste = list(combinations(ids_unicos, tamanho_teste))

print(f"\nIniciando {len(todas_combinacoes_teste)} baterias de testes independentes...")
print("-" * 65)

acuracias = []

for teste_ids in todas_combinacoes_teste:
    teste_ids = list(teste_ids)
    treino_ids = [id_ for id_ in ids_unicos if id_ not in teste_ids]

    mask_treino = np.isin(grupos_ids, treino_ids)
    mask_teste = np.isin(grupos_ids, teste_ids)

    # 2. Separa os dados de Treino e Teste com todas as 38.000 colunas
    X_treino_bruto, y_treino = X[mask_treino], y[mask_treino]
    X_teste_bruto, y_teste = X[mask_teste], y[mask_teste]

    # ===================================================================
    #  PCA Cego
    # ===================================================================
    pca = PCA(n_components=10)

    X_treino_pca = pca.fit_transform(X_treino_bruto)

    X_teste_pca = pca.transform(X_teste_bruto)

    # Filtro de Componentes (Rejeita a PC1 [0], mantém PC2 a PC4 [1:4])
    X_treino_pca_filtrado = X_treino_pca[:,:]
    X_teste_pca_filtrado = X_teste_pca[:,:]
    # ===================================================================

    # 3. Treina o kNN com as componentes corretas
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(X_treino_pca_filtrado, y_treino)

    # Previsão
    y_pred = knn.predict(X_teste_pca_filtrado)

    acc = accuracy_score(y_teste, y_pred)
    acuracias.append(acc)

    treino_str = ", ".join(treino_ids)
    teste_str = ", ".join(teste_ids)
    print(f"Treino: [{treino_str}] | Teste: [{teste_str}] -> Acurácia: {acc*100:.1f}%")

print("-" * 65)
print(f"ACURÁCIA MÉDIA REAL: {np.mean(acuracias)*100:.2f}% ± {np.std(acuracias)*100:.2f}%")
print("-" * 65)


# %% [markdown] Cell 30
# #### Validação Cruzada de Monte Carlo


# %% [code] Cell 31

iteracoes = 100
resultados_acuracia = []
tamanho_minimo = df_limpo_VIS['Teor_Adulteracao'].value_counts().min()
chance_aleatoria = (1.0 / len(df_limpo_VIS['Teor_Adulteracao'].unique())) * 100

print(f"Iniciando Simulação de Monte Carlo ({iteracoes} iterações)...")

# Loop de variação do Random State
for i in range(iteracoes):

    metadados = ['Amostra_ID', 'Num_Disparo', 'Nome_Arquivo', 'Canal_Optico', 'Teor_Adulteracao']
    cols_features= [
        c for c in df_limpo_VIS.columns
        if c not in metadados and wl_min <= float(c) <= wl_max
    ]


    tamanho_minimo = df_limpo_VIS['Teor_Adulteracao'].value_counts().min()
    df_balanceado = df_limpo_VIS.groupby('Teor_Adulteracao').sample(n=tamanho_minimo, random_state=42)


    X = df_balanceado[cols_features].values
    y = df_balanceado['Teor_Adulteracao'].values

    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X)

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X_pca, y, test_size=0.3, random_state=i, stratify=y
    )

    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(X_treino, y_treino)
    y_pred = knn.predict(X_teste)

    acuracia = accuracy_score(y_teste, y_pred) * 100
    resultados_acuracia.append(acuracia)





# PLOT DO HISTOGRAMA
# ==========================================
plt.figure(figsize=(9, 5))
sns.histplot(resultados_acuracia, bins=5, color='darkcyan', edgecolor='black')

plt.axvline(np.mean(resultados_acuracia), color='red', linestyle='dashed', linewidth=2,
            label=f'Média Geral: {np.mean(resultados_acuracia):.1f}%')


plt.title(f'Validação de Monte Carlo: Estabilidade do Modelo ({iteracoes} Random States)')
plt.xlabel('Acurácia do kNN (%)')
plt.ylabel('Frequência (Nº de Random States)')
plt.legend()
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.show()

print(f"Acurácia Média: {np.mean(resultados_acuracia):.2f}%")
print(f"Desvio Padrão: {np.std(resultados_acuracia):.2f}%")


# %% [code] Cell 32
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(knn, X, y, cv=kf, scoring='accuracy')

print(f"Acurácia Média (5-Fold): {np.mean(scores)*100:.2f}%")
print(f"Desvio Padrão (5-Fold): {np.std(scores)*100:.2f}%")


# %% [markdown] Cell 33
# ### PLSR


# %% [code] Cell 34


df_limpo_VIS_PLS = df_limpo_VIS[df_limpo_VIS['Amostra_ID'] != 'i_']

metadados = ['Amostra_ID', 'Num_Disparo', 'Nome_Arquivo', 'Canal_Optico', 'Teor_Adulteracao']
wl_min = 273
wl_max = 780

cols_features= [
    c for c in df_limpo_VIS_PLS.columns
    if c not in metadados and wl_min <= float(c) <= wl_max
]

tamanho_minimo_pls = df_limpo_VIS_PLS['Teor_Adulteracao'].value_counts().min()
print(f"Balanceando: Serão sorteados {tamanho_minimo_pls} disparos para cada Teor de Adulteração.")

df_balanceado = df_limpo_VIS_PLS.groupby('Teor_Adulteracao').sample(n=tamanho_minimo_pls, random_state=42)

X = df_balanceado[cols_features].values
y = df_balanceado['Teor_Adulteracao'].values

grupos_ids = df_balanceado['Amostra_ID'].values

print(f"Formato final de X (Espectros brutos): {X.shape}")
print(f"Formato final de Y (Concentrações): {y.shape}")

pls = PLSRegression(n_components=6)

# CÁLCULO DO RMSECV (Validação Cruzada)
logo = LeaveOneGroupOut()
# Gera previsões para amostras que ficaram de fora do treino
y_pred_cv = cross_val_predict(pls, X, y, groups=grupos_ids, cv=logo)

r2_cv = r2_score(y, y_pred_cv)
rmsecv = np.sqrt(mean_squared_error(y, y_pred_cv))

# CÁLCULO DO RMSEC (Calibração / Treino)
pls.fit(X, y)    # Treina o modelo com TODOS os dados simultaneamente
y_pred_cal = pls.predict(X)

r2_c = r2_score(y, y_pred_cal)
rmsec = np.sqrt(mean_squared_error(y, y_pred_cal))

print("-" * 50)
print("PERFORMANCE DE CALIBRAÇÃO (Treino - Dados vistos):")
print(f"R² de Calibração: {r2_c:.4f}")
print(f"RMSEC: {rmsec:.2f}%")
print("\nPERFORMANCE DE VALIDAÇÃO CRUZADA (Teste - LOGO):")
print(f"R² de Validação: {r2_cv:.4f}")
print(f"RMSECV: {rmsecv:.2f}%")
print("-" * 50)

plt.figure(figsize=(8, 6))
plt.plot([-5, 55], [-5, 55], color='black', linestyle='--', label='Linha Ideal (100% Acerto)')

plt.scatter(y, y_pred_cv, alpha=0.6, edgecolor='k', s=60, color='royalblue')
plt.scatter(y, y_pred_cal, alpha=0.3, edgecolor='None', s=40, color='gray', label='Calibração')

plt.title('Modelo PLS: Teor Real vs. Teor Previsto', fontsize=14, fontweight='bold')
plt.xlabel('Teor Real de Adulteração (%)', fontsize=12)
plt.ylabel('Teor Previsto pelo PLS (%)', fontsize=12)
plt.xticks([0, 10, 25, 50])
plt.yticks([0, 10, 25, 50])
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()


# %% [code] Cell 35
max_comp = 15
lvs = np.arange(1, max_comp + 1)

rmsec_list = []
rmsecv_list = []
logo = LeaveOneGroupOut()

print("Calculando erros... Isso pode demorar alguns segundos.")

for i in lvs:
    pls = PLSRegression(n_components=i)

    # RMSEC (Calibração/Treino)
    pls.fit(X, y)
    y_cal = pls.predict(X)
    rmsec = np.sqrt(mean_squared_error(y, y_cal))
    rmsec_list.append(rmsec)

    # RMSECV (Validação Cruzada LOGO)
    y_cv = cross_val_predict(pls, X, y, groups=grupos_ids, cv=logo)
    rmsecv = np.sqrt(mean_squared_error(y, y_cv))
    rmsecv_list.append(rmsecv)


plt.figure(figsize=(9, 6))

plt.plot(lvs, rmsec_list, 'o-', color='#2980b9', linewidth=2, label='RMSEC (Erro de Calibração)')
plt.plot(lvs, rmsecv_list, 's--', color='#e74c3c', linewidth=2, label='RMSECV (Erro de Validação Cruzada)')




# Ajustes dos Eixos e Título
plt.title('Otimização do PLS: Evolução do Erro por Variáveis Latentes', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Número de Variáveis Latentes (LVs)', fontsize=12)
plt.ylabel('Erro Quadrático Médio (RMSE %)', fontsize=12)

# Garantir que o eixo X mostre apenas números inteiros
plt.xticks(lvs)

plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='upper right', fontsize=11, frameon=True, shadow=True)
plt.tight_layout()

plt.show()


# %% [code] Cell 36


teores_reais = np.unique(y)

medias_pred_cv = []
incertezas_tipo_a = []

for teor in teores_reais:
    predicoes_grupo = y_pred_cv[y == teor]

    media = np.mean(predicoes_grupo)

    N = len(predicoes_grupo)
    desvio_padrao = np.std(predicoes_grupo, ddof=1)
    incerteza = desvio_padrao #/ np.sqrt(N)

    medias_pred_cv.append(media)
    incertezas_tipo_a.append(incerteza)

coefs = np.polyfit(y, y_pred_cv, 1)
reta_ajuste = np.polyval(coefs, teores_reais)
equacao_ajuste = f'Ajuste Linear (y = {coefs[0]:.3f}x + {coefs[1]:.2f})'



# ==========================================
plt.figure(figsize=(8, 6))

plt.plot([-5, 55], [-5, 55], color='black', linestyle='--', alpha=0.7, label='Linha Ideal (100% Acerto)')

plt.plot(teores_reais, reta_ajuste, color='red', linestyle='-', linewidth=2, label=equacao_ajuste)
plt.errorbar(teores_reais, medias_pred_cv, yerr=incertezas_tipo_a, fmt='o',
             color='black', ecolor='black', elinewidth=2, capsize=5, capthick=2,
             markersize=8, markeredgecolor='black', label='Média Prevista ± Desv. Padrão')

plt.title('Calibração do Modelo PLS', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Teor Real de Adulteração (%)', fontsize=12)
plt.ylabel('Teor Previsto pelo PLS (%)', fontsize=12)
plt.xlim(-2, 55)
plt.ylim(-10, 60)
plt.xticks([0, 10, 25, 50])
plt.yticks([0, 10, 25, 50])

plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)
plt.tight_layout()

plt.show()


# %% [code] Cell 37
escores = pls.x_scores_
classes = np.unique(y)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes_flat = axes.flatten()

fig.suptitle('Análise das Variáveis Latentes (LV): Identificação da Separação Química', fontsize=16, fontweight='bold')

pares_lv = [(1, 2), (1, 3), (1, 4), (1, 5)]
cores = ['#2ecc71', '#f1c40f', '#0000FF', '#e74c3c']

for i, (lv_x, lv_y) in enumerate(pares_lv):
    ax = axes_flat[i]

    for idx, classe in enumerate(classes):
        mask = (y == classe)
        ax.scatter(escores[mask, lv_x-1], escores[mask, lv_y-1],
                   label=f'{int(classe)}%', alpha=0.6, edgecolors='w', s=50, color=cores[idx])

    ax.set_xlabel(f'LV{lv_x}', fontsize=12)
    ax.set_ylabel(f'LV{lv_y}', fontsize=12)
    ax.set_title(f'Plano Latente LV{lv_x} vs LV{lv_y}', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)

axes_flat[0].legend(title='Teor (%)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# %% [markdown] Cell 38
# ### Random Forest Regressor


# %% [code] Cell 39
from sklearn.ensemble import RandomForestRegressor

# 1. INICIALIZAR O RANDOM FOREST
# n_estimators = 200 árvores
# n_jobs = -1 (usa todos os núcleos do seu processador para ser rápido)
rf_modelo = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1, max_features = 0.33)

logo = LeaveOneGroupOut()
print("A treinar o Random Forest (LOGO CV)... Isto pode demorar uns segundos.")

y_pred_cv_rf = cross_val_predict(rf_modelo, X, y, groups=grupos_ids, cv=logo, n_jobs=-1)

r2_cv_rf = r2_score(y, y_pred_cv_rf)
rmsecv_rf = np.sqrt(mean_squared_error(y, y_pred_cv_rf))

print("-" * 50)
print("PERFORMANCE DE VALIDAÇÃO CRUZADA (Random Forest - LOGO):")
print(f"R² de Validação: {r2_cv_rf:.4f}")
print(f"RMSECV: {rmsecv_rf:.2f}%")
print("-" * 50)


teores_reais = np.unique(y)
medias_pred_cv = []
incertezas_tipo_a = []

for teor in teores_reais:
    predicoes_grupo = y_pred_cv_rf[y == teor]
    medias_pred_cv.append(np.mean(predicoes_grupo))
    incertezas_tipo_a.append(np.std(predicoes_grupo, ddof=1))

# Ajuste Linear do modelo
coefs = np.polyfit(y, y_pred_cv_rf, 1) # coefs[0] é a inclinação
reta_ajuste = np.polyval(coefs, teores_reais)
equacao_ajuste = f'Ajuste Linear RF (y = {coefs[0]:.3f}x + {coefs[1]:.2f})'

# ==========================================
# 5. PLOTAGEM
# ==========================================
plt.figure(figsize=(9, 7))
plt.scatter(y, y_pred_cv_rf, alpha=0.15, color='gray', s=20, label='Disparos Individuais')
plt.plot([-5, 55], [-5, 55], color='black', linestyle='--', alpha=0.7, label='Linha Ideal (1:1)')
plt.plot(teores_reais, reta_ajuste, color='forestgreen', linestyle='-', linewidth=2, label=equacao_ajuste)
plt.errorbar(teores_reais, medias_pred_cv, yerr=incertezas_tipo_a, fmt='o',
             color='darkgreen', ecolor='black', elinewidth=1.5, capsize=5, capthick=1.5,
             markersize=8, markeredgecolor='black', label='Média Prevista ± Desvio Padrão')

plt.title('Modelo Random Forest: Teor Real vs. Teor Previsto', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Teor Real de Adulteração (%)', fontsize=12)
plt.ylabel('Teor Previsto pelo Random Forest (%)', fontsize=12)
plt.xlim(-5, 55)
plt.ylim(-5, 55)
plt.xticks([0, 10, 25, 50])
plt.yticks([0, 10, 25, 50])
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left', fontsize=10, frameon=True, shadow=True)
plt.tight_layout()
plt.show()


# %% [code] Cell 40


# Cria um DataFrame rápido com as previsões e os valores reais
df_resultados_rf = pd.DataFrame({
    'Amostra_ID': grupos_ids,
    'Teor_Real': y,
    'Teor_Previsto_RF': y_pred_cv_rf
})

# Salva num ficheiro CSV
df_resultados_rf.to_csv('resultados_random_forest_LOGO.csv', index=False)
print("ALELUIA! Resultados do Random Forest salvos com sucesso no disco!")


# %% [markdown] Cell 41
# ## UV


# %% [code] Cell 42
lista_linhas_uv = []

for dados_brutos in tqdm(lista_dicionarios_uv_bruto, desc="Aplicando Top-Hat/Filtros no UV"):
    df_processado = processar_espectro(dados_brutos)
    df_processado['Teor_Adulteracao'] = dados_brutos['Teor_Adulteracao']
    df_processado['Canal_Optico'] = dados_brutos['Canal_Optico']
    lista_linhas_uv.append(df_processado)


colunas_mestre = lista_linhas_uv[0].columns
for df in lista_linhas_uv:
    df.columns = colunas_mestre

df_final_uv = pd.concat(lista_linhas_uv, ignore_index=True)



# %% [code] Cell 43
print("\n" + "="*40)
print("PROCESSANDO CANAL UV")
print("="*40)
if len(lista_linhas_uv) > 0:
    df_final_UV = pd.concat(lista_linhas_uv, ignore_index=True)
    df_limpo_UV = filtrar_outliers(df_final_UV, threshold_percentil=100)

    print(f"Tamanho UV original: {df_final_UV.shape[0]} linhas")
    print(f"Tamanho UV após SAM: {df_limpo_UV.shape[0]} linhas")
    print(f"Formato Matriz UV: {df_limpo_UV.shape}")
    print(df_limpo_UV.groupby('Teor_Adulteracao').size())
else:
    print("Nenhum dado de UV encontrado.")


# %% [code] Cell 44
df_limpo_UV


# %% [code] Cell 45
df_00 = df_final_UV[df_final_UV['Teor_Adulteracao'] == 0.0]
df_50 = df_final_UV[df_final_UV['Teor_Adulteracao'] == 50.0]
df_10 = df_final_UV[df_final_UV['Teor_Adulteracao'] == 10.0]
df_25 = df_final_UV[df_final_UV['Teor_Adulteracao'] == 25.0]

cols_features = [c for c in df_final_UV.columns if c not in ['Amostra_ID', 'Num_Disparo', 'Nome_Arquivo', 'Canal_Optico', 'Teor_Adulteracao']]

espectro_medio_00 = df_00[cols_features].mean(axis=0).values
espectro_medio_50 = df_50[cols_features].mean(axis=0).values
espectro_medio_10 = df_10[cols_features].mean(axis=0).values
espectro_medio_25 = df_25[cols_features].mean(axis=0).values
comprimentos_onda = np.array(cols_features, dtype=float)

df_espectros = pd.DataFrame({
    'Wavelength': comprimentos_onda,
    'Média_Puro': espectro_medio_00,
    'Média_Adulterado_50pct': espectro_medio_50,
    'Média_Adulterado_10pct': espectro_medio_10,
    'Média_Adulterado_25pct': espectro_medio_25
})

df_espectros = df_espectros.sort_values(by='Wavelength'
)

fig = px.line(
    df_espectros,
    x='Wavelength',
    y=['Média_Puro','Média_Adulterado_50pct', 'Média_Adulterado_10pct', 'Média_Adulterado_25pct'],
    title='',
    labels={'value': 'Intensidade Normalizada', 'variable': 'Espectro', 'Wavelength': 'Comprimento de Onda (nm)'}
)

fig.data[0].line.color = 'lightgray'
fig.data[0].line.width = 3
fig.data[1].line.color = 'red'
fig.data[1].line.width = 1.5

fig.update_layout(
    template='plotly_white',
    hovermode='x unified',
    legend_title_text='Legenda'
)

fig.show()


# %% [code] Cell 46



# %% [code] Cell 47
 linhas_emissao = {
    'Ca II': [396.87, 393.37, 317.97, 315.9, 370.60, 373.69, 227.55],
    'Mg II': [279.58, 280.3],
    'NH (Molecular)': [336.10, 336.15], # A assinatura do Nitrogênio!
    'Mg I': [285.3],
    'C I': [247.8],
    'Ca I': [239.86, 300.68,299.49],
    'P I': [213.61, 214.91, 253.56, 255.33]
}

cores_elementos = {
    'Ca II': 'purple',
    'Ca I': 'red',
    'NH (Molecular)': 'blue',
    'Mg II': 'magenta',
    'Mg I': 'orange',
    'C I': 'red',
    'P I' : 'green'

}

comprimentos_onda = np.array([float(c) for c in cols_features])
wl_min, wl_max = comprimentos_onda.min(), comprimentos_onda.max()

espectros_50 = df_balanceado[df_balanceado['Teor_Adulteracao'] == 50.0][cols_features].values
espectro_medio_50 = espectros_50.mean(axis=0)

# Extrair os Loadings
loadings = pca.components_

# 4. Configurar a figura com 4 andares
fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
fig.suptitle('Análise de Loadings vs. Espectro Físico (50%) e Linhas de Emissão', fontsize=16, fontweight='bold', y=0.98)

cores_pcs = ['darkred', 'darkblue', 'darkgreen']
pcs = ['PC1', 'PC2', 'PC3']


for i in range(1):
    axes[i].plot(comprimentos_onda, loadings[i, :], color=cores_pcs[i], linewidth=1.2)
    axes[i].set_title(f'Loadings da {pcs[i]} (Variância explicada: {pca.explained_variance_ratio_[i]*100:.1f}%)', fontsize=12)
    axes[i].set_ylabel('Peso no PC', fontsize=11)
    axes[i].grid(True, linestyle=':', alpha=0.5)
    axes[i].axhline(0, color='black', linewidth=1, linestyle='--')


axes[1].plot(comprimentos_onda, espectro_medio_50, color='black', linewidth=1.2)
axes[1].set_title('Espectro Médio - Amostras com 50% de Adulteração', fontsize=12)
axes[1].set_ylabel('Intensidade', fontsize=11)
axes[1].grid(True, linestyle=':', alpha=0.5)


picos_ordenados = []
for elemento, comprimentos in linhas_emissao.items():
    cor = cores_elementos[elemento]
    for wl in comprimentos:
        if wl_min <= wl <= wl_max:
            picos_ordenados.append((wl, elemento, cor))

picos_ordenados.sort(key=lambda x: x[0])


for ax in axes:
    ymin, ymax = ax.get_ylim()

    niveis_altura = [0.95, 0.82, 0.69, 0.56]

    for i, (wl, elemento, cor) in enumerate(picos_ordenados):
        ax.axvline(x=wl, color=cor, linewidth=1.5, linestyle='-.', alpha=0.4)

        altura_atual = ymax * niveis_altura[i % len(niveis_altura)]

        ax.text(wl + 0.3, altura_atual, f'{elemento} ({wl})', color=cor,
                rotation=45, verticalalignment='top', fontsize=9, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
axes[3].set_xlim(wl_min, wl_max)
axes[3].set_xlabel('Comprimento de Onda (nm)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(top=0.94)
plt.show()


# %% [code] Cell 48



# %% [code] Cell 49

top_n = 10               # Quantos picos DISTINTOS queremos
tolerancia_id = 0.55      # Margem de erro para adivinhar o elemento
distancia_minima = 1.0

dados_tabela = []
loadings = pca.components_
variancias = pca.explained_variance_ratio_ * 100
comprimentos_onda = np.array([float(c) for c in cols_features])

# 3. Varredura com Filtro de Redundância
for i in range(3):
    pc_nome = f"PC{i+1}"
    var_explicada = f"{variancias[i]:.1f}%"

    pesos_absolutos = np.abs(loadings[i])
    indices_ordenados = np.argsort(pesos_absolutos)[::-1]

    picos_selecionados_wls = [] # Guarda os comprimentos de onda já escolhidos

    for idx in indices_ordenados:
        wl_real = comprimentos_onda[idx]
        peso_real = loadings[i][idx]

        # CHECAGEM DE REDUNDÂNCIA (A MÁGICA ACONTECE AQUI)
        redundante = False
        for wl_salvo in picos_selecionados_wls:
            if abs(wl_real - wl_salvo) < distancia_minima:
                redundante = True
                break

        # Se for do mesmo pico, ignora e vai para o próximo pixel
        if redundante:
            continue

        # Se passou no filtro, guarda o pico
        picos_selecionados_wls.append(wl_real)

        # Identificar o elemento
        elemento_identificado = "-"
        for elemento, wls in linhas_emissao.items():
            for wl_teorico in wls:
                if abs(wl_real - wl_teorico) <= tolerancia_id:
                    elemento_identificado = f"{elemento} ({wl_teorico})"
                    break

        dados_tabela.append({
            "Componente": pc_nome,
            "Variância (%)": var_explicada,
            "Comprimento de Onda (nm)": round(wl_real, 2),
            "Peso (Loading)": round(peso_real, 4),
            "Elemento Atribuído": elemento_identificado
        })

        # Para quando acharmos os 5 picos realmente distintos
        if len(picos_selecionados_wls) == top_n:
            break

# 4. Mostrar tabela
df_loadings = pd.DataFrame(dados_tabela)
df_loadings.set_index(["Componente", "Variância (%)"], inplace=True)

print("Tabela de Atribuição dos Loadings (Sem Repetição de Picos):")
display(df_loadings)
df_loadings.to_excel('tabela_loadings_uv.xlsx')


# %% [markdown] Cell 50
# ### PCA - UV


# %% [code] Cell 51



df_limpo_UV_PCA = df_limpo_UV[
    (df_limpo_UV['Amostra_ID'] != 'i_') &
    (df_limpo_UV['Teor_Adulteracao'] != 10.0) &
    (df_limpo_UV['Teor_Adulteracao'] != 25.0)
]
metadados = ['Amostra_ID', 'Num_Disparo', 'Nome_Arquivo', 'Canal_Optico', 'Teor_Adulteracao']
wl_min = 175
wl_max = 350

cols_features= [
    c for c in df_limpo_UV_PCA.columns
    if c not in metadados and wl_min <= float(c) <= wl_max
]

n_components=10
tamanho_minimo = df_limpo_UV_PCA['Teor_Adulteracao'].value_counts().min()
print(f"Nivelando a matriz: todas as amostras terão exatamente {tamanho_minimo} disparos.")


df_balanceado = df_limpo_UV_PCA.groupby('Teor_Adulteracao').sample(n=tamanho_minimo, random_state=42) # Undersampling: Amostrar aleatoriamente 'tamanho_minimo' linhas de cada letra

metadados = ['Amostra_ID', 'Num_Disparo', 'Nome_Arquivo', 'Canal_Optico', 'Teor_Adulteracao']
X = df_balanceado[cols_features].values
y = df_balanceado['Teor_Adulteracao'].values

pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

# SCREE PLOT (VARIÂNCIA EXPLICADA)
# ==============================================================================
plt.figure(figsize=(7, 4))
variancia_acumulada = np.cumsum(pca.explained_variance_ratio_) * 100
plt.plot(range(1, 11), variancia_acumulada, marker='o', linestyle='--', color='darkblue', linewidth=2)
plt.xlabel('Número de Componentes Principais (PCs)', fontsize=11)
plt.ylabel('Variância Explicada Acumulada (%)', fontsize=11)
plt.title('Scree Plot - UV', fontsize=12, fontweight='bold')
plt.xticks(range(1, 11))
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

print(f"-> PC1 captura sozinho: {pca.explained_variance_ratio_[0]*100:.2f}% de toda a variância química.")
print(f"-> PC2 captura sozinho: {pca.explained_variance_ratio_[1]*100:.2f}% de toda a variância química.")
print(f"-> Total acumulado (PC1 + PC2): {variancia_acumulada[1]:.2f}%")



# SCORE PLOT (PC1 VS PC2)
# ==============================================================================


columns = []
for i in range(n_components):
    columns.append(f'PC{i+1}')

df_pca = pd.DataFrame(X_pca[:, :n_components], columns=columns)
df_pca['Teor_Adulteracao'] = y

df_pca['Amostra_ID'] = df_balanceado['Amostra_ID'].values
df_pca['Teor_Adulteracao'] = df_balanceado['Teor_Adulteracao'].values

plt.figure(figsize=(9, 6))


x_score = 'PC1'
y_score = 'PC2'

sns.scatterplot(
    data=df_pca,
    x=x_score, y=y_score,
    hue = 'Teor_Adulteracao',
    #hue='Amostra_ID',
    #style='Teor_Adulteracao',
    # markers={0.0: '*', 50.0: 'o', 10.0:'X', -1.0:'s'},
    #markers = {25.0:'X', 10.0:"*"},
    palette='turbo',
    s=80, alpha=0.9, edgecolor='black', linewidth=0.5
)

plt.xlabel(f'{x_score} ({pca.explained_variance_ratio_[int(x_score[2]) - 1]*100:.1f}%)', fontsize=11)
plt.ylabel(f'{y_score} ({pca.explained_variance_ratio_[int(y_score[2]) - 1]*100:.1f}%)', fontsize=11)
plt.title(f'Score Plot {x_score}  e {y_score} - UV', fontsize=12, fontweight='bold')
plt.legend(title='ID da Amostra', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()



# %% [code] Cell 52

print("Gerando o Pairplot (Isso pode levar alguns segundos dependendo da quantidade de dados)...")
df_pca['Teor_Categoria'] = df_pca['Teor_Adulteracao'].astype(str) + '%'

# Criando o Pairplot
g = sns.pairplot(
    data=df_pca,
    vars=columns,
    hue='Teor_Categoria',                            # Colorir pela concentração
    palette='turbo',                                 # Mantendo a sua paleta
    diag_kind='kde',                                 # Curvas de densidade na diagonal principal
    plot_kws={'alpha': 0.7, 's': 50, 'edgecolor': 'black', 'linewidth': 0.5} # Estilo dos pontos
)

# Ajuste do título para não sobrepor o gráfico
g.fig.suptitle('Matriz de Dispersão (Pairplot) - PC3 a PC8 (Canal UV)', y=1.05, fontsize=16, fontweight='bold')

plt.show()


# %% [code] Cell 53
X_pca_recorte = X_pca[:,:]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X_pca_recorte, y, test_size=0.3, random_state=45, stratify=y
)


# k=5 é um bom balanço para não deixar o modelo memorizar ruído extremo
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_treino, y_treino)

# Predição
y_pred = knn.predict(X_teste)
acuracia = accuracy_score(y_teste, y_pred)

print(f"--- TESTE DE HOMOGENEIDADE ---")
print(f"Acurácia do kNN: {acuracia * 100:.2f}%")


cm = confusion_matrix(y_teste, y_pred, labels=np.unique(y))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=np.unique(y),
            yticklabels=np.unique(y))
plt.title(f'Matriz de Confusão UV (Acurácia: {acuracia*100:.1f}%)')
plt.xlabel('Previsão do Modelo')
plt.ylabel('Amostra Real')
plt.show()


# %% [markdown] Cell 54
# #### Leave Group Out


# %% [code] Cell 55



# %% [code] Cell 56

df_balanceado = df_balanceado[df_balanceado['Amostra_ID'] != 'i_']


# Defina o que você quer analisar.
# Opções válidas:
# 'todos' -> Roda a matriz inteira (0, 10, 25, 50)
# [0.0, 10.0] -> Roda apenas Puro vs 10%
# [0.0, 50.0] -> Roda apenas Puro vs 50%

alvos_analise = [50.0, 0.0]  # <--- MUDE AQUI!

if alvos_analise == 'todos':
    df_filtrado = df_balanceado.copy()
    print("Modo de Análise: MULTICLASSE (Todas as Concentrações)")
else:
    df_filtrado = df_balanceado[df_balanceado['Teor_Adulteracao'].isin(alvos_analise)].copy()
    print(f"Modo de Análise: BINÁRIO (Concentrações Alvo: {alvos_analise})")


X = df_filtrado[cols_features].values
y = df_filtrado['Teor_Adulteracao'].values
grupos_ids = df_filtrado['Amostra_ID'].values

ids_unicos = np.unique(grupos_ids)
print(f"IDs únicos disponíveis para combinação: {ids_unicos}")

tamanho_teste = 1
todas_combinacoes_teste = list(combinations(ids_unicos, tamanho_teste))

print(f"\nIniciando {len(todas_combinacoes_teste)} baterias de testes independentes...")
print("-" * 65)

acuracias = []

for teste_ids in todas_combinacoes_teste:
    teste_ids = list(teste_ids)
    treino_ids = [id_ for id_ in ids_unicos if id_ not in teste_ids]

    mask_treino = np.isin(grupos_ids, treino_ids)
    mask_teste = np.isin(grupos_ids, teste_ids)

    # 2. Separa os dados de Treino e Teste com todas as 38.000 colunas
    X_treino_bruto, y_treino = X[mask_treino], y[mask_treino]
    X_teste_bruto, y_teste = X[mask_teste], y[mask_teste]

    # ===================================================================
    #  PCA Cego
    # ===================================================================
    pca = PCA(n_components=10)

    X_treino_pca = pca.fit_transform(X_treino_bruto)

    X_teste_pca = pca.transform(X_teste_bruto)

    # Filtro de Componentes (Rejeita a PC1 [0], mantém PC2 a PC4 [1:4])
    X_treino_pca_filtrado = X_treino_pca[:,:]
    X_teste_pca_filtrado = X_teste_pca[:,:]
    # ===================================================================

    # 3. Treina o kNN com as componentes corretas
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(X_treino_pca_filtrado, y_treino)

    # Previsão
    y_pred = knn.predict(X_teste_pca_filtrado)

    acc = accuracy_score(y_teste, y_pred)
    acuracias.append(acc)

    treino_str = ", ".join(treino_ids)
    teste_str = ", ".join(teste_ids)
    print(f"Treino: [{treino_str}] | Teste: [{teste_str}] -> Acurácia: {acc*100:.1f}%")

print("-" * 65)
print(f"ACURÁCIA MÉDIA REAL: {np.mean(acuracias)*100:.2f}% ± {np.std(acuracias)*100:.2f}%")
print("-" * 65)


# %% [markdown] Cell 57
# ### Validação Cruzada de Monte Carlo


# %% [code] Cell 58

iteracoes = 100
resultados_acuracia = []
tamanho_minimo = df_final_UV['Teor_Adulteracao'].value_counts().min()
chance_aleatoria = (1.0 / len(df_final_UV['Teor_Adulteracao'].unique())) * 100

print(f"Iniciando Simulação de Monte Carlo ({iteracoes} iterações)...")

# Loop de variação do Random State
for i in range(iteracoes):

    metadados = ['Amostra_ID', 'Num_Disparo', 'Nome_Arquivo', 'Canal_Optico', 'Teor_Adulteracao']

    cols_features= [
        c for c in df_final_UV.columns
        if c not in metadados and wl_min <= float(c) <= wl_max
    ]


    tamanho_minimo = df_final_UV['Teor_Adulteracao'].value_counts().min()
    df_balanceado = df_final_UV.groupby('Teor_Adulteracao').sample(n=tamanho_minimo, random_state=42)


    X = df_balanceado[cols_features].values
    y = df_balanceado['Teor_Adulteracao'].values

    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X)

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X_pca, y, test_size=0.3, random_state=i, stratify=y
    )

    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(X_treino, y_treino)
    y_pred = knn.predict(X_teste)

    acuracia = accuracy_score(y_teste, y_pred) * 100
    resultados_acuracia.append(acuracia)





# PLOT DO HISTOGRAMA
# ==========================================
plt.figure(figsize=(9, 5))
sns.histplot(resultados_acuracia, bins=5, kde=True, color='darkcyan', edgecolor='black')

plt.axvline(np.mean(resultados_acuracia), color='red', linestyle='dashed', linewidth=2,
            label=f'Média Geral: {np.mean(resultados_acuracia):.1f}%')


plt.title(f'Validação de Monte Carlo: Estabilidade do Modelo ({iteracoes} Random States)')
plt.xlabel('Acurácia do kNN (%)')
plt.ylabel('Frequência (Nº de Random States)')
plt.legend()
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.show()

print(f"Acurácia Média: {np.mean(resultados_acuracia):.2f}%")
print(f"Desvio Padrão: {np.std(resultados_acuracia):.2f}%")


# %% [code] Cell 59



# %% [markdown] Cell 60
# #### PLSR


# %% [code] Cell 61



# %% [code] Cell 62
max_comp = 15
lvs = np.arange(1, max_comp + 1)

rmsec_list = []
rmsecv_list = []
logo = LeaveOneGroupOut()

print("Calculando erros... Isso pode demorar alguns segundos.")

for i in lvs:
    pls = PLSRegression(n_components=i)

    # RMSEC (Calibração/Treino)
    pls.fit(X, y)
    y_cal = pls.predict(X)
    rmsec = np.sqrt(mean_squared_error(y, y_cal))
    rmsec_list.append(rmsec)

    # RMSECV (Validação Cruzada LOGO)
    y_cv = cross_val_predict(pls, X, y, groups=grupos_ids, cv=logo)
    rmsecv = np.sqrt(mean_squared_error(y, y_cv))
    rmsecv_list.append(rmsecv)


plt.figure(figsize=(9, 6))

plt.plot(lvs, rmsec_list, 'o-', color='#2980b9', linewidth=2, label='RMSEC (Erro de Calibração)')
plt.plot(lvs, rmsecv_list, 's--', color='#e74c3c', linewidth=2, label='RMSECV (Erro de Validação Cruzada)')




# Ajustes dos Eixos e Título
plt.title('Otimização do PLS: Evolução do Erro por Variáveis Latentes', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Número de Variáveis Latentes (LVs)', fontsize=12)
plt.ylabel('Erro Quadrático Médio (RMSE %)', fontsize=12)

# Garantir que o eixo X mostre apenas números inteiros
plt.xticks(lvs)

plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='upper right', fontsize=11, frameon=True, shadow=True)
plt.tight_layout()

plt.show()


# %% [code] Cell 63


df_limpo_UV_PLS = df_limpo_UV[df_limpo_UV['Amostra_ID'] != 'i_']

metadados = ['Amostra_ID', 'Num_Disparo', 'Nome_Arquivo', 'Canal_Optico', 'Teor_Adulteracao']

cols_features= [
    c for c in df_limpo_UV_PLS.columns
    if c not in metadados and wl_min <= float(c) <= wl_max
]

tamanho_minimo_pls = df_limpo_UV_PLS['Teor_Adulteracao'].value_counts().min()
print(f"Balanceando: Serão sorteados {tamanho_minimo_pls} disparos para cada Teor de Adulteração.")

df_balanceado = df_limpo_UV_PLS.groupby('Teor_Adulteracao').sample(n=tamanho_minimo_pls, random_state=42)

X = df_balanceado[cols_features].values
y = df_balanceado['Teor_Adulteracao'].values

grupos_ids = df_balanceado['Amostra_ID'].values

print(f"Formato final de X (Espectros brutos): {X.shape}")
print(f"Formato final de Y (Concentrações): {y.shape}")

pls = PLSRegression(n_components=6)

# CÁLCULO DO RMSECV (Validação Cruzada)
logo = LeaveOneGroupOut()
# Gera previsões para amostras que ficaram de fora do treino
y_pred_cv = cross_val_predict(pls, X, y, groups=grupos_ids, cv=logo)

r2_cv = r2_score(y, y_pred_cv)
rmsecv = np.sqrt(mean_squared_error(y, y_pred_cv))

# CÁLCULO DO RMSEC (Calibração / Treino)
pls.fit(X, y)    # Treina o modelo com TODOS os dados simultaneamente
y_pred_cal = pls.predict(X)

r2_c = r2_score(y, y_pred_cal)
rmsec = np.sqrt(mean_squared_error(y, y_pred_cal))

print("-" * 50)
print("PERFORMANCE DE CALIBRAÇÃO (Treino - Dados vistos):")
print(f"R² de Calibração: {r2_c:.4f}")
print(f"RMSEC: {rmsec:.2f}%")
print("\nPERFORMANCE DE VALIDAÇÃO CRUZADA (Teste - LOGO):")
print(f"R² de Validação: {r2_cv:.4f}")
print(f"RMSECV: {rmsecv:.2f}%")
print("-" * 50)

plt.figure(figsize=(8, 6))
plt.plot([-5, 55], [-5, 55], color='black', linestyle='--', label='Linha Ideal (100% Acerto)')

plt.scatter(y, y_pred_cv, alpha=0.6, edgecolor='k', s=60, color='royalblue')
plt.scatter(y, y_pred_cal, alpha=0.3, edgecolor='None', s=40, color='gray', label='Calibração')

plt.title('Modelo PLS UV: Teor Real vs. Teor Previsto', fontsize=14, fontweight='bold')
plt.xlabel('Teor Real de Adulteração (%)', fontsize=12)
plt.ylabel('Teor Previsto pelo PLS (%)', fontsize=12)
plt.xticks([0, 10, 25, 50])
plt.yticks([0, 10, 25, 50])
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()


# %% [code] Cell 64


teores_reais = np.unique(y)

medias_pred_cv = []
incertezas_tipo_a = []

for teor in teores_reais:
    predicoes_grupo = y_pred_cv[y == teor]

    media = np.mean(predicoes_grupo)

    N = len(predicoes_grupo)
    desvio_padrao = np.std(predicoes_grupo, ddof=1)
    incerteza = desvio_padrao #/ np.sqrt(N)

    medias_pred_cv.append(media)
    incertezas_tipo_a.append(incerteza)

coefs = np.polyfit(y, y_pred_cv, 1)
reta_ajuste = np.polyval(coefs, teores_reais)
equacao_ajuste = f'Ajuste Linear (y = {coefs[0]:.3f}x + {coefs[1]:.2f})'



# ==========================================
plt.figure(figsize=(8, 6))

plt.plot([-5, 55], [-5, 55], color='black', linestyle='--', alpha=0.7, label='Linha Ideal (100% Acerto)')

plt.plot(teores_reais, reta_ajuste, color='red', linestyle='-', linewidth=2, label=equacao_ajuste)
plt.errorbar(teores_reais, medias_pred_cv, yerr=incertezas_tipo_a, fmt='o',
             color='black', ecolor='black', elinewidth=2, capsize=5, capthick=2,
             markersize=8, markeredgecolor='black', label='Média Prevista ± Desv. Padrão')

plt.title('Calibração do Modelo PLS', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Teor Real de Adulteração (%)', fontsize=12)
plt.ylabel('Teor Previsto pelo PLS (%)', fontsize=12)
plt.xlim(-2, 55)
plt.ylim(-10, 60)
plt.xticks([0, 10, 25, 50])
plt.yticks([0, 10, 25, 50])

plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)
plt.tight_layout()

plt.show()


# %% [code] Cell 65
from sklearn.ensemble import RandomForestRegressor


rf_modelo = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1, max_features = 0.33)

logo = LeaveOneGroupOut()
print("A treinar o Random Forest (LOGO CV)... Isto pode demorar uns segundos.")

y_pred_cv_rf = cross_val_predict(rf_modelo, X, y, groups=grupos_ids, cv=logo, n_jobs=-1)

r2_cv_rf = r2_score(y, y_pred_cv_rf)
rmsecv_rf = np.sqrt(mean_squared_error(y, y_pred_cv_rf))

print("-" * 50)
print("PERFORMANCE DE VALIDAÇÃO CRUZADA (Random Forest - LOGO):")
print(f"R² de Validação: {r2_cv_rf:.4f}")
print(f"RMSECV: {rmsecv_rf:.2f}%")
print("-" * 50)


teores_reais = np.unique(y)
medias_pred_cv = []
incertezas_tipo_a = []

for teor in teores_reais:
    predicoes_grupo = y_pred_cv_rf[y == teor]
    medias_pred_cv.append(np.mean(predicoes_grupo))
    incertezas_tipo_a.append(np.std(predicoes_grupo, ddof=1))

# Ajuste Linear do modelo
coefs = np.polyfit(y, y_pred_cv_rf, 1) # coefs[0] é a inclinação
reta_ajuste = np.polyval(coefs, teores_reais)
equacao_ajuste = f'Ajuste Linear RF (y = {coefs[0]:.3f}x + {coefs[1]:.2f})'

# ==========================================
# 5. PLOTAGEM
# ==========================================
plt.figure(figsize=(9, 7))
plt.scatter(y, y_pred_cv_rf, alpha=0.15, color='gray', s=20, label='Disparos Individuais')
plt.plot([-5, 55], [-5, 55], color='black', linestyle='--', alpha=0.7, label='Linha Ideal (1:1)')
plt.plot(teores_reais, reta_ajuste, color='forestgreen', linestyle='-', linewidth=2, label=equacao_ajuste)
plt.errorbar(teores_reais, medias_pred_cv, yerr=incertezas_tipo_a, fmt='o',
             color='darkgreen', ecolor='black', elinewidth=1.5, capsize=5, capthick=1.5,
             markersize=8, markeredgecolor='black', label='Média Prevista ± Desvio Padrão')

plt.title('Modelo Random Forest: Teor Real vs. Teor Previsto', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Teor Real de Adulteração (%)', fontsize=12)
plt.ylabel('Teor Previsto pelo Random Forest (%)', fontsize=12)
plt.xlim(-5, 55)
plt.ylim(-5, 55)
plt.xticks([0, 10, 25, 50])
plt.yticks([0, 10, 25, 50])
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left', fontsize=10, frameon=True, shadow=True)
plt.tight_layout()
plt.show()


# %% [code] Cell 66



