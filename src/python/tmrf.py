import pandas as pd
import numpy as np
from typing import Sequence, Any
import numpy as np
import numpy.typing as npt
from collections import Counter

# ─────────────────────────────────────────────
# Funções de apoio
# ─────────────────────────────────────────────

def change_class_to_id(
    results: Sequence[Any],
    relevant: Any
) -> list[int]:
    """Converte lista de resultados em vetor binário de relevância.

    Gera um vetor binário onde:
        - 1 indica objeto relevante
        - 0 indica objeto irrelevante

    Args:
        results (Sequence[Any]): Lista ou sequência de identificadores retornados.
        relevant (Any): Identificador considerado relevante.

    Returns:
        List[int]: Lista binária representando relevância.

    Notes:
        - A comparação é feita por igualdade (`==`).
        - Útil para métricas como Precision@K e MAP.

    Example:
        >>> change_class_to_id([1, 2, 3], 2)
        [0, 1, 0]
    """
    new_order = []

    for value in results:
        if value == relevant:
            new_order.append(1)
        else:
            new_order.append(0)

    return new_order

# -------------------------------------------------------------------------------#

# ─────────────────────────────────────────────
# Funções de Feedback Relevance
# ─────────────────────────────────────────────
def get_feedback_by_class(
    dataset: pd.DataFrame,
    results: Sequence[int],
    category_relevance: str
) -> pd.DataFrame:
    """Simula realimentação de relevância baseada na classe da consulta.

    Filtra os resultados retornados por uma consulta, mantendo apenas os
    objetos cuja classe coincide com a classe considerada relevante
    (geralmente a classe da query).

    O resultado é um DataFrame contendo apenas os vetores relevantes,
    com uma nova coluna binária 'relevante' indicando relevância (1).

    Args:
        dataset (pd.DataFrame): DataFrame completo contendo os vetores de
            características, além de colunas de metadados como 'category'
            e 'file'.
        results (Sequence[int]): Lista ou sequência de índices retornados
            pela consulta (ranking).
        category_relevance (str): Classe considerada relevante (classe da query).

    Returns:
        pd.DataFrame: DataFrame contendo apenas os vetores relevantes,
        com a coluna adicional:
            - 'relevante' (int): valor 1 para todos os elementos retornados.

        As colunas de metadados ('category' e 'file') são removidas.

    Raises:
        KeyError: Se as colunas 'category' ou 'file' não existirem no dataset.

    Notes:
        - A função realiza uma varredura dupla (O(n * k)), onde n é o tamanho
          do dataset e k o número de resultados.
        - Apenas itens relevantes são retornados; itens não relevantes são descartados.
        - Assume que os índices em `results` correspondem diretamente aos índices
          do DataFrame `dataset`.

    Example:
        >>> df_feedback = get_feedback_by_class(
        ...     dataset=df,
        ...     results=[0, 3, 5],
        ...     category_relevance="COVID"
        ... )
        >>> df_feedback.head()
    """
    feedback = []
    relevants = []

    for index1, value1 in dataset.iterrows():
        for index2, value2 in enumerate(results):
            if (
                index1 == value2 and
                category_relevance == dataset.loc[index1]['category']
            ):
                feedback.append(value1)
                relevants.append(1)

    feedback = pd.DataFrame(feedback)

    feedback['relevante'] = relevants

    # Remove metadados
    del feedback['category']
    del feedback['file']

    return feedback

# -------------------------------------------------------------------------------#

def get_feedback_by_class(
    dataset: pd.DataFrame,
    results: Sequence[int],
    category_relevance: str
) -> pd.DataFrame:
    """Simula realimentação de relevância baseada na classe da consulta.

    Filtra os resultados retornados por uma consulta, mantendo apenas os
    objetos cuja classe coincide com a classe considerada relevante.

    O resultado é um DataFrame contendo apenas os vetores relevantes,
    com uma coluna binária adicional 'relevant'.

    Args:
        dataset (pd.DataFrame): DataFrame contendo vetores de características
            e metadados, incluindo as colunas 'category' e 'file'.
        results (Sequence[int]): Índices retornados pela consulta (ranking).
        category_relevance (str): Classe considerada relevante.

    Returns:
        pd.DataFrame: DataFrame contendo apenas os vetores relevantes, com:
            - coluna 'relevant' (int) = 1
            - sem colunas 'category' e 'file'

    Raises:
        KeyError: Se 'category' ou 'file' não existirem no dataset.

    Notes:
        - Complexidade O(n * k), onde n é o tamanho do dataset e k o número de resultados.
        - Assume que os índices em `results` correspondem aos índices do DataFrame.

    Example:
        >>> df_feedback = get_feedback_by_class(df, [0, 3, 5], "COVID")
    """
    feedback = []
    relevants = []

    for index1, value1 in dataset.iterrows():
        for value2 in results:
            if (
                index1 == value2 and
                category_relevance == dataset.loc[index1]['category']
            ):
                feedback.append(value1)
                relevants.append(1)

    feedback = pd.DataFrame(feedback)

    feedback['relevant'] = relevants

    del feedback['category']
    del feedback['file']

    return feedback

# -------------------------------------------------------------------------------#

def feedback_relevance(
    arr_relevante: Sequence[npt.ArrayLike],
    len_dimensao: int
) -> npt.NDArray[np.float64]:
    """Calcula vetor de pesos para realimentação de relevância.

    Os pesos são calculados com base no desvio padrão de cada dimensão,
    sendo inversamente proporcionais à variabilidade (menor variância →
    maior peso).

    Args:
        arr_relevante (Sequence[ArrayLike]): Coleção de vetores relevantes.
        len_dimensao (int): Número de dimensões dos vetores.

    Returns:
        np.ndarray: Vetor de pesos normalizado (norma L1 = 1).

    Notes:
        - Dimensões com desvio padrão zero recebem tratamento especial.
        - Se todos os desvios forem zero, retorna pesos uniformes.
        - Normalização final é feita com norma L1.

    Example:
        >>> feedback_relevance([[1,2], [1,3]], 2)
        array([...])
    """
    stdM = []

    arr_relevante_np = np.array(arr_relevante)

    # 1. Desvio padrão por dimensão
    for i in range(len_dimensao):
        standarddev = np.std(arr_relevante_np[:, i])
        stdM.append(standarddev)

    stdM = np.array(stdM)
    weightList = []

    # 2. Peso inversamente proporcional ao desvio padrão
    for i in range(len(stdM)):
        if stdM[i] == 0:
            avg = arr_relevante_np[:, i].mean()

            if avg != 0:
                non_zero_stds = stdM[stdM != 0]
                if len(non_zero_stds) > 0:
                    minimum = min(non_zero_stds)
                    weight = 1 / (0.5 * minimum)
                else:
                    weight = 1 / 0.5
            else:
                weight = 0
        else:
            weight = 1 / stdM[i]

        weightList.append(weight)

    # 3. Normalização L1
    weightM = np.array(weightList)
    norm_val = np.linalg.norm(weightM, ord=1)

    if norm_val != 0:
        weightM = weightM / norm_val
    else:
        weightM = np.ones(len_dimensao) / len_dimensao

    return weightM

# -------------------------------------------------------------------------------#
