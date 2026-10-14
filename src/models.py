"""
Módulo de Modelagem Quimiométrica e Machine Learning
Projeto: Detecção de Adulteração em Fertilizantes NPK via LIBS
Embrapa Instrumentação

Este módulo implementa:
1. PCA: Decomposição em Componentes Principais, variância explicada e identificação de picos nos Loadings.
2. KNN: Classificação supervisionada com estratégias contra vazamento de dados (Leave-Group-Out por pastilha e Monte Carlo CV).
3. PLS-Regression: Calibração multivariada, otimização de LVs (RMSEC/RMSECV) e incerteza de medição Tipo A.
4. Random Forest: Modelos de regressão e classificação com validação cruzada agrupada.
5. Regressão Linear: Ajuste uni e multivariado sobre razões elementares com LOO-CV.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    LeaveOneGroupOut,
    LeaveOneOut,
    cross_val_predict,
    train_test_split,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    r2_score,
)


# ==============================================================================
# 1. ANÁLISE DE COMPONENTES PRINCIPAIS (PCA)
# ==============================================================================

def executar_pca(
    df: pd.DataFrame,
    wl_min: Optional[float] = None,
    wl_max: Optional[float] = None,
    n_components: int = 5,
    scaler: bool = False,
    cols_metadados: Optional[List[str]] = None
) -> Tuple[PCA, np.ndarray, List[str], pd.DataFrame]:
    """
    Executa a análise PCA na matriz espectral.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo metadados e colunas espectrais.
    wl_min, wl_max : float, opcional
        Faixa de corte espectral em nm.
    n_components : int, default 5
        Número de componentes principais a extrair.
    scaler : bool, default False
        Se True, aplica StandardScaler antes da PCA. Se False (típico em espectroscopia),
        mantém a variância natural das intensidades.
    cols_metadados : list, opcional
        Colunas não espectrais.

    Returns
    -------
    pca : PCA
        Objeto PCA treinado.
    X_pca : np.ndarray
        Matriz de scores projetados (amostras x componentes).
    cols_features : list de str
        Lista de comprimentos de onda utilizados.
    df_scores : pd.DataFrame
        DataFrame com metadados e scores das componentes (PC1, PC2, ...).
    """
    if cols_metadados is None:
        cols_metadados = ["Amostra_ID", "Num_Disparo", "Nome_Arquivo", "Canal_Optico", "Teor_Adulteracao", "Adulterado"]

    cols_validas = []
    for c in df.columns:
        if c in cols_metadados:
            continue
        try:
            wl = float(c)
            if wl_min is not None and wl < wl_min:
                continue
            if wl_max is not None and wl > wl_max:
                continue
            cols_validas.append(c)
        except ValueError:
            continue

    X = df[cols_validas].values.astype(float)

    if scaler:
        X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Constrói DataFrame de scores com metadados preservados
    colunas_meta = [c for c in cols_metadados if c in df.columns]
    df_scores = df[colunas_meta].copy().reset_index(drop=True)
    for i in range(n_components):
        df_scores[f"PC{i+1}"] = X_pca[:, i]

    return pca, X_pca, cols_validas, df_scores


def identificar_top_loadings(
    pca: PCA,
    cols_features: List[str],
    n_pcs: int = 3,
    top_n: int = 10,
    tolerancia_nm: float = 0.5,
    linhas_referencia: Optional[Dict[str, List[float]]] = None,
    distancia_minima_nm: float = 1.0
) -> pd.DataFrame:
    """
    Identifica os comprimentos de onda com maiores pesos absolutos (Loadings)
    e mapeia com a biblioteca de linhas atômicas de referência (NIST).

    Aplica filtro de redundância para evitar selecionar pixels adjacentes do mesmo pico.
    """
    wls = np.array([float(c) for c in cols_features])
    loadings = pca.components_
    variancias = pca.explained_variance_ratio_ * 100

    linhas_tabela = []

    for i in range(min(n_pcs, len(loadings))):
        pc_nome = f"PC{i+1}"
        var_exp = f"{variancias[i]:.2f}%"
        pesos_abs = np.abs(loadings[i])
        indices_ordenados = np.argsort(pesos_abs)[::-1]

        picos_selecionados = []
        for idx in indices_ordenados:
            wl_candidato = wls[idx]
            peso_real = loadings[i][idx]

            # Verifica se já selecionamos um pixel vizinho
            muito_proximo = any(abs(wl_candidato - p["wl"]) < distancia_minima_nm for p in picos_selecionados)
            if muito_proximo:
                continue

            elemento_encontrado = "Desconhecido"
            if linhas_referencia:
                for elem, lista_wl in linhas_referencia.items():
                    for wl_ref in lista_wl:
                        if abs(wl_candidato - wl_ref) <= tolerancia_nm:
                            elemento_encontrado = f"{elem} ({wl_ref:.2f} nm)"
                            break
                    if elemento_encontrado != "Desconhecido":
                        break

            picos_selecionados.append({
                "Componente": pc_nome,
                "Variância Explicada": var_exp,
                "wl": wl_candidato,
                "Wavelength (nm)": round(wl_candidato, 2),
                "Loading": round(peso_real, 4),
                "Elemento Atribuído": elemento_encontrado
            })

            if len(picos_selecionados) >= top_n:
                break

        for p in picos_selecionados:
            del p["wl"]
            linhas_tabela.append(p)

    return pd.DataFrame(linhas_tabela)


# ==============================================================================
# 2. CLASSIFICAÇÃO COM KNN E VALIDAÇÃO SEM VAZAMENTO DE DADOS
# ==============================================================================

def avaliar_knn_leave_group_out(
    X: np.ndarray,
    y: np.ndarray,
    grupos: np.ndarray,
    ids_treino: List[str],
    ids_teste: List[str],
    n_neighbors: int = 5,
    metric: str = "euclidean"
) -> Dict[str, any]:
    """
    Avalia o classificador KNN dividindo explicitamente as pastilhas (Amostra_ID)
    entre treino e teste, eliminando vazamento de dados por proximidade espacial de disparos.
    """
    mask_treino = np.isin(grupos, ids_treino)
    mask_teste = np.isin(grupos, ids_teste)

    X_train, y_train = X[mask_treino], y[mask_treino]
    X_test, y_test = X[mask_teste], y[mask_teste]

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    return {
        "modelo": knn,
        "acuracia": accuracy_score(y_test, y_pred),
        "matriz_confusao": confusion_matrix(y_test, y_pred),
        "y_teste": y_test,
        "y_pred": y_pred,
        "n_treino": len(y_train),
        "n_teste": len(y_test)
    }


def validacao_monte_carlo_knn(
    df: pd.DataFrame,
    cols_features: List[str],
    col_alvo: str = "Teor_Adulteracao",
    col_grupo: str = "Amostra_ID",
    n_iteracoes: int = 100,
    n_neighbors: int = 5,
    test_size: float = 0.3
) -> Dict[str, any]:
    """
    Executa validação cruzada Monte Carlo balanceando disparos e calculando
    distribuição empírica de acurácia.
    """
    tamanho_minimo = df[col_alvo].value_counts().min()
    acuracias = []

    for _ in range(n_iteracoes):
        df_sub = df.groupby(col_alvo, group_keys=False).apply(
            lambda x: x.sample(n=tamanho_minimo)
        ).reset_index(drop=True)

        X = df_sub[cols_features].values
        y = df_sub[col_alvo].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y
        )

        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        acuracias.append(accuracy_score(y_test, preds))

    return {
        "media_acuracia": float(np.mean(acuracias)),
        "desvio_acuracia": float(np.std(acuracias)),
        "ic_95": (float(np.percentile(acuracias, 2.5)), float(np.percentile(acuracias, 97.5))),
        "todas_acuracias": acuracias
    }


# ==============================================================================
# 3. CALIBRAÇÃO MULTIVARIADA VIA PLS-REGRESSION E INCERTEZA TIPO A
# ==============================================================================

def otimizar_componentes_plsr(
    X: np.ndarray,
    y: np.ndarray,
    grupos: np.ndarray,
    max_components: int = 15
) -> pd.DataFrame:
    """
    Avalia o desempenho da PLS-Regression de 1 a max_components usando
    Leave-One-Group-Out CV (LOGO-CV) sobre as pastilhas para calcular RMSEC e RMSECV.

    Parameters
    ----------
    X : np.ndarray
        Matriz espectral (N disparos x P comprimentos de onda).
    y : np.ndarray
        Vetor com teores de adulteração contínuos (%).
    grupos : np.ndarray
        Identificador das pastilhas (Amostra_ID) para evitar vazamento.
    max_components : int, default 15
        Número máximo de variáveis latentes (LVs).

    Returns
    -------
    pd.DataFrame
        Tabela com LVs, RMSEC, RMSECV e R2_CV.
    """
    lvs = np.arange(1, min(max_components + 1, X.shape[1], len(np.unique(grupos))))
    logo = LeaveOneGroupOut()
    resultados = []

    for n_comp in lvs:
        pls = PLSRegression(n_components=n_comp)

        # RMSEC (Calibração / Ajuste com todos os dados)
        pls.fit(X, y)
        y_cal = pls.predict(X).ravel()
        rmsec = np.sqrt(mean_squared_error(y, y_cal))

        # RMSECV via Leave-One-Group-Out
        y_cv = cross_val_predict(pls, X, y, groups=grupos, cv=logo).ravel()
        rmsecv = np.sqrt(mean_squared_error(y, y_cv))
        r2_cv = r2_score(y, y_cv)

        resultados.append({
            "LVs": n_comp,
            "RMSEC": round(rmsec, 3),
            "RMSECV": round(rmsecv, 3),
            "R2_CV": round(r2_cv, 4)
        })

    return pd.DataFrame(resultados)


def avaliar_plsr_final(
    X: np.ndarray,
    y: np.ndarray,
    grupos: np.ndarray,
    n_components: int = 5
) -> Dict[str, any]:
    """
    Treina o modelo PLSR final, calcula predições cruzadas via LOGO-CV
    e quantifica a incerteza de medição do Tipo A (u_A = s / sqrt(n)) por nível de adulteração.
    """
    pls = PLSRegression(n_components=n_components)
    logo = LeaveOneGroupOut()

    y_pred_cv = cross_val_predict(pls, X, y, groups=grupos, cv=logo).ravel()

    r2_cv = r2_score(y, y_pred_cv)
    rmsecv = np.sqrt(mean_squared_error(y, y_pred_cv))

    # Incerteza do Tipo A por nível
    teores_unicos = np.unique(y)
    tabela_incertezas = []

    for teor in teores_unicos:
        mask = (y == teor)
        preds_nivel = y_pred_cv[mask]
        n_obs = len(preds_nivel)
        s = np.std(preds_nivel, ddof=1) if n_obs > 1 else 0.0
        u_a = s / np.sqrt(n_obs) if n_obs > 0 else 0.0

        tabela_incertezas.append({
            "Teor_Real": teor,
            "Média_Predita": round(float(np.mean(preds_nivel)), 2),
            "Desvio_Padrao": round(float(s), 2),
            "N_Disparos": n_obs,
            "Incerteza_Tipo_A": round(float(u_a), 2)
        })

    return {
        "modelo": pls,
        "n_components": n_components,
        "y_real": y,
        "y_pred_cv": y_pred_cv,
        "R2_CV": r2_cv,
        "RMSECV": rmsecv,
        "tabela_incertezas": pd.DataFrame(tabela_incertezas)
    }


# ==============================================================================
# 4. RANDOM FOREST REGRESSOR & CLASSIFIER
# ==============================================================================

def avaliar_rf_regressor(
    X: np.ndarray,
    y: np.ndarray,
    grupos: Optional[np.ndarray] = None,
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    max_features: Union[str, float] = 0.33,
    random_state: int = 42
) -> Dict[str, any]:
    """
    Avalia regressão não linear com Random Forest Regressor usando Leave-One-Group-Out.
    """
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1
    )

    if grupos is not None:
        cv_strat = LeaveOneGroupOut()
        y_pred = cross_val_predict(rf, X, y, groups=grupos, cv=cv_strat, n_jobs=-1)
    else:
        cv_strat = LeaveOneOut()
        y_pred = cross_val_predict(rf, X, y, cv=cv_strat, n_jobs=-1)

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    rf.fit(X, y)

    return {
        "modelo": rf,
        "y_real": y,
        "y_pred": y_pred,
        "R2": r2,
        "RMSE": rmse,
        "importancias": rf.feature_importances_
    }


# ==============================================================================
# 5. REGRESSÃO LINEAR EM RAZÕES ELEMENTARES (ABORDAGEM V2)
# ==============================================================================

def regressao_linear_razoes(
    df_areas: pd.DataFrame,
    features: List[str],
    col_alvo: str = "Teor_Adulteracao"
) -> Dict[str, any]:
    """
    Ajusta regressão linear (OLS) uni ou multivariada sobre razões elementares
    com validação cruzada Leave-One-Out (LOO-CV).
    """
    X = df_areas[features].values
    y = df_areas[col_alvo].astype(float).values

    modelo = LinearRegression()
    loo = LeaveOneOut()

    y_pred = cross_val_predict(modelo, X, y, cv=loo)
    rmse_cv = np.sqrt(mean_squared_error(y, y_pred))
    r2_cv = r2_score(y, y_pred)

    modelo.fit(X, y)

    return {
        "modelo": modelo,
        "features": features,
        "coeficientes": modelo.coef_,
        "intercept": modelo.intercept_,
        "RMSE_CV": rmse_cv,
        "R2_CV": r2_cv,
        "y_real": y,
        "y_pred_cv": y_pred
    }
