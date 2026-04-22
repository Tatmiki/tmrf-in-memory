from typing import Sequence, Any, List, Tuple, Literal
import numpy as np
import pandas as pd
from cbir import *
from tmrf import *


# ─────────────────────────────────────────────
# Funções de cálculo de métricas de desempenho
# ─────────────────────────────────────────────

def apk(
    results: Sequence[Any],
    relevant_class: Any,
    k: int = 10
) -> float:
    """Calcula a métrica Average Precision@K (AP@K) para uma consulta.

    A métrica AP@K considera tanto a presença quanto a posição dos itens
    relevantes nos top-K resultados de uma consulta. Para cada posição `i`
    onde um item relevante é encontrado, calcula-se a precisão parcial:

        Precision(i) = TP_i / (TP_i + FP_i)

    O valor final é a média das precisões nas posições relevantes:

        AP@K = (soma das precisões nas posições relevantes) / (número de relevantes encontrados)

    Se nenhum item relevante for encontrado, o valor retornado será 0.0.

    Args:
        results (Sequence[Any]): Lista ou sequência contendo as classes dos itens
            retornados, ordenados por similaridade (ranking).
        relevant_class (Any): Classe considerada relevante (classe da consulta).
        k (int, optional): Número máximo de resultados considerados. Default é 10.

    Returns:
        float: Valor de Average Precision@K arredondado para duas casas decimais.

    Notes:
        - Apenas os primeiros `k` elementos de `results` são considerados.
        - A comparação de relevância é feita por igualdade (`==`).
        - Complexidade O(k).

    Example:
        >>> apk(["A", "B", "A", "C"], relevant_class="A", k=3)
        0.83
    """
    tp = 0
    fp = 0
    mean_precision = 0.0

    for index, value in enumerate(results):
        if index < k:
            if value == relevant_class:
                tp += 1
                mean_precision += (tp / (tp + fp))
            else:
                fp += 1

    if tp != 0:
        return round(mean_precision / tp, 2)
    else:
        return 0.0
    
# -------------------------------------------------------------------------------#

def calculate_mean_precision(
    crude_dataset: pd.DataFrame,
    df_dataset: pd.DataFrame,
    df_classes: pd.DataFrame,
    type_rr: Literal['std'] = 'std',
    p: int = 1,
    qtd: int = 10,
    strategy: Literal['global', 'in_memory'] = 'global',
    memory_size: int = 50
) -> List[Tuple[str, List[float]]]:
    """Calcula a média de Average Precision@K (AP@K) com realimentação de relevância.

    Executa um processo iterativo de consulta por similaridade com realimentação
    (Relevance Feedback - RR), avaliando o desempenho ao longo de múltiplas iterações.

    A primeira iteração sempre é realizada no dataset completo. Nas iterações
    seguintes, a busca pode ocorrer globalmente ou em memória (subconjunto).

    Args:
        crude_dataset (pd.DataFrame): Dataset original contendo vetores e metadados.
        df_dataset (pd.DataFrame): Dataset contendo apenas os vetores de características.
        df_classes (pd.DataFrame): DataFrame contendo a coluna 'category'.
        type_rr (Literal['std'], optional): Tipo de realimentação de relevância.
            Atualmente suporta apenas 'std'. Default é 'std'.
        p (int, optional): Ordem da métrica Lp utilizada na consulta. Default é 1.
        qtd (int, optional): Valor de K (top-K) para cálculo de AP@K. Default é 10.
        strategy (Literal['global', 'in_memory'], optional):
            Estratégia de busca:
                - 'global': busca em todo o dataset a cada iteração.
                - 'in_memory': restringe busca ao Top-N inicial.
            Default é 'global'.
        memory_size (int, optional): Número de instâncias mantidas em memória
            na estratégia 'in_memory'. Default é 50.

    Returns:
        List[Tuple[str, List[float]]]:
            Lista onde cada elemento contém:
                - categoria (str)
                - lista de AP@K por iteração (List[float])

    Notes:
        - Executa 5 iterações (1 inicial + 4 com feedback).
        - Remove a própria query dos resultados antes do cálculo de AP@K.
        - A estratégia 'in_memory' reduz custo computacional, mas pode impactar recall.
        - Complexidade elevada devido ao uso de busca sequencial.

    Example:
        >>> results = calculate_mean_precision(df_raw, df_feat, df_class)
    """
    classes = pd.DataFrame(df_classes)
    media_apk = []

    for category in classes['category'].value_counts().index:
        ids = classes[classes['category'] == category].index

        for id_query in ids:
            precisions: List[float] = []
            weights = []
            dimensao = len(df_dataset.columns)

            # 1ª iteração (global)
            resultados_globais = consult(
                np.array(df_dataset.iloc[[id_query]]),
                np.array(df_dataset),
                p=p,
                w=weights
            )

            if strategy == 'in_memory':
                idx_in_memory = np.argsort(resultados_globais)[:(memory_size + 1)]
                resultados_iteracao = idx_in_memory[:(qtd + 1)]
                dataset_target = df_dataset.iloc[idx_in_memory].values
            else:
                resultados_iteracao = np.argsort(resultados_globais)[:(qtd + 1)]
                dataset_target = np.array(df_dataset)

            new_a = np.delete(resultados_iteracao, np.where(resultados_iteracao == id_query))
            scores = [df_classes.iloc[idx] for idx in new_a]

            avg_precision = apk(scores, category, k=qtd)
            precisions.append(avg_precision)

            if type_rr == 'std':
                feedback = get_feedback_by_class(crude_dataset, resultados_iteracao, category)
                weights = feedback_relevance(feedback, dimensao)

            # Iterações seguintes
            for _ in range(4):
                resultados_ciclo = consult(
                    np.array(df_dataset.iloc[[id_query]]),
                    dataset_target,
                    p=p,
                    w=weights
                )

                if strategy == 'in_memory':
                    idx_locais = np.argsort(resultados_ciclo)[:(qtd + 1)]
                    resultados_iteracao = np.array([idx_in_memory[loc] for loc in idx_locais])
                else:
                    resultados_iteracao = np.argsort(resultados_ciclo)[:(qtd + 1)]

                new_a = np.delete(resultados_iteracao, np.where(resultados_iteracao == id_query))
                scores = [df_classes.iloc[idx] for idx in new_a]

                avg_precision = apk(scores, category, k=qtd)
                precisions.append(avg_precision)

                if type_rr == 'std':
                    feedback = get_feedback_by_class(crude_dataset, resultados_iteracao, category)
                    weights = feedback_relevance(feedback, dimensao)

            media_apk.append((category, precisions))

    return media_apk

# -------------------------------------------------------------------------------#

def metricas_avaliacao(
    crude_dataset: pd.DataFrame,
    df_data: pd.DataFrame,
    df_class: pd.DataFrame,
    p: int = 1,
    qtd: int = 10,
    rr: str = 'std',
    strategy: str = 'global',
    memory_size: int = 50
) -> Tuple[List[float], List[Tuple[List[float], str]]]:
    """Calcula métricas de avaliação agregadas com base em AP@K.

    Executa o pipeline completo de avaliação, incluindo realimentação de relevância,
    e calcula médias de precisão globais e por classe ao longo das iterações.

    Args:
        crude_dataset (pd.DataFrame): Dataset original com metadados.
        df_data (pd.DataFrame): Dataset contendo vetores de características.
        df_class (pd.DataFrame): DataFrame contendo coluna 'category'.
        p (int, optional): Ordem da métrica Lp. Default é 1.
        qtd (int, optional): Valor de K (top-K). Default é 10.
        rr (str, optional): Tipo de realimentação ('std'). Default é 'std'.
        strategy (str, optional): Estratégia de busca ('global' ou 'in_memory').
            Default é 'global'.
        memory_size (int, optional): Tamanho da memória para estratégia 'in_memory'.
            Default é 50.

    Returns:
        Tuple:
            - List[float]: Precisões médias globais por iteração.
            - List[Tuple[List[float], str]]: Precisões por classe e iteração.

    Notes:
        - Executa 5 iterações de avaliação.
        - Exibe resultados no console.
        - Utiliza média aritmética das precisões por iteração.

    Example:
        >>> global_prec, class_prec = metricas_avaliacao(df_raw, df_feat, df_class)
    """
    results = calculate_mean_precision(
        crude_dataset,
        df_data,
        df_class,
        p=p,
        qtd=qtd,
        type_rr=rr,
        strategy=strategy,
        memory_size=memory_size
    )

    average_precision = []
    class_precision = []

    for value in results:
        class_precision.append(value[0])
        average_precision.append(value[1])

    df_results = pd.DataFrame(average_precision, columns=['1°','2°','3°','4°','5°'])
    df_results['class'] = class_precision

    print(f'TYPE_RR: {rr} | STRATEGY: {strategy}')
    print('AVERAGE PRECISION BY ALL DATASET')

    base_precisions = []
    for col in ['1°','2°','3°','4°','5°']:
        mean_val = df_results[col].mean()
        base_precisions.append(mean_val)
        print(f'Average Precision in {col} Iteration: {round(mean_val * 100, 2)}%')

    print('-----------------------------------')

    base_class_precisions = []
    categories = df_class['category'].unique()

    print('AVERAGE PRECISION BY CLASS')
    for category in categories:
        df_class_filter = df_results[df_results['class'] == category]

        means_precisions = [
            round(df_class_filter[col].mean() * 100, 2)
            for col in ['1°','2°','3°','4°','5°']
        ]

        base_class_precisions.append((means_precisions, category))
        print(f'Precisions Itr: {means_precisions}, Category: {category}')

    print('-----------------------------------\n')

    return base_precisions, base_class_precisions