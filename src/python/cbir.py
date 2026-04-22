import pandas as pd
import numpy as np
import sys
from typing import Sequence, Optional
import numpy as np
import numpy.typing as npt


# ─────────────────────────────────────────────
# Funções de apoio
# ─────────────────────────────────────────────

def load_dataset(
    path: str, 
    remove_classes=[]):
    """Carrega um dataset a partir de um arquivo CSV e organiza em estruturas separadas.

    Args:
        path (str): Caminho para o arquivo CSV do dataset.
        remove_classes (list, optional): Lista opcional de classes a serem removidas do dataset. Por default é uma lista vazia [].

    Returns:
        tuple: Uma tupla contendo (df_data_all, df_data, df_class, df_files). Onde:
            - df_data_all: dataset completo filtrado
            - df_data: apenas vetores de características (espaço métrico)
            - df_class: rótulos das classes
            - df_files: nomes dos arquivos das imagens
    """    
    df_data_all = pd.read_csv(path)

    # Remove classes especificadas
    for _, value in enumerate(remove_classes):
        df_data_all.drop(
            df_data_all[df_data_all['category'] == value].index,
            inplace=True
        )

    # Reorganiza índices após remoção
    df_data_all = df_data_all.reset_index()
    df_data_all = df_data_all.drop('index', axis=1)

    df_data = df_data_all.copy()

    # Separa classes e nomes dos arquivos
    df_class = pd.DataFrame(df_data["category"])
    df_files = pd.DataFrame(df_data['file'])

    # Remove colunas não pertencentes ao espaço métrico
    del df_data['category']
    del df_data['file']

    return (df_data_all, df_data, df_class, df_files)

# -------------------------------------------------------------------------------#

# ─────────────────────────────────────────────
# Funções de consulta sequencial
# ─────────────────────────────────────────────
def consult(
    query: npt.ArrayLike, 
    dataset: Sequence[npt.ArrayLike], 
    p: int=1, 
    w: Optional[npt.ArrayLike]=None
) -> list[float]:
    """Realiza consulta por similaridade utilizando métrica Lp ponderada.

    Calcula a distância entre um vetor de consulta e todos os vetores de um
    dataset utilizando uma métrica Lp com pesos por dimensão.

    A distância é definida como:

        d(q, x) = sum_i ( w_i * |q_i - x_i|^p )

    Para p = 1, corresponde à distância Manhattan.
    Para p = 2, corresponde à distância Euclidiana (com raiz aplicada).

    Args:
        query (array-like): Vetor de características da consulta.
        dataset (array-like): Coleção (lista ou matriz) de vetores de características.
        p (int, optional): Ordem da métrica Lp. Exemplos:
            - 1: Manhattan
            - 2: Euclidiana
            Default é 1.
        w (array-like, optional): Vetor de pesos para cada dimensão. Se None,
            pesos unitários são utilizados.

    Returns:
        list[float]: Lista contendo as distâncias entre o vetor de consulta
        e cada vetor do dataset.

    Raises:
        ValueError: Se as dimensões de `query`, `dataset` e `w` forem incompatíveis.

    Notes:
        - Se `w` não for fornecido, assume-se um vetor de pesos unitários.
        - A variável `distance` é inicializada com um valor máximo inteiro
          (`sys.maxsize`), mas é sempre sobrescrita durante o cálculo.
        - A implementação realiza busca sequencial (linear scan), com custo O(n).

    Example:
        >>> q = [1, 2, 3]
        >>> data = [[1, 2, 4], [2, 3, 4]]
        >>> consult(q, data, p=2)
        [1.0, 1.732...]
    """
    all_distances = []

    query = np.array(query)

    # Se não houver pesos, utiliza pesos unitários
    if w is None:
        w = np.ones(query.shape[0])
    else:
        w = np.array(w)

    for feature in dataset:
        distance = sys.maxsize  # valor máximo inteiro

        feature = np.array(feature)

        value = np.abs(query - feature)
        sum_val = (w * (value ** p)).sum()

        if p == 1 or p == 0:
            distance = sum_val
        else:
            distance = np.sqrt(sum_val)

        all_distances.append(distance)

    return all_distances

# -------------------------------------------------------------------------------#
