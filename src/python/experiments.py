from typing import List, Tuple
import numpy as np
import pandas as pd
import time
from cbir import *
from tmrf import *
from metrics import *


def executar_experimento(crude_dataset, df_dataset, df_classes, type_rr='std', p=1, qtd=10, strategy='global', memory_size=50):
    """
    Executa o pipeline de busca e Relevance Feedback, coletando métricas de
    desempenho (MAP, Precisão, Recall, F1) e tempos de execução computacional.
    """
    classes = pd.DataFrame(df_classes)

    # Estruturas para armazenar as métricas de cada uma das 5 iterações
    metricas = {
        'map': [[], [], [], [], []],
        'precision': [[], [], [], [], []],
        'recall': [[], [], [], [], []],
        'f1': [[], [], [], [], []],
        'tempo_busca': [[], [], [], [], []],
        'tempo_rr': [[], [], [], [], []],
        'tempo_total': [[], [], [], [], []]
    }

    dimensao = len(df_dataset.columns)
    categorias = classes['category'].value_counts().index

    # Iterar sobre cada classe e cada imagem como consulta
    for category in categorias:
        ids_query = classes[classes['category'] == category].index
        total_relevantes_dataset = len(ids_query) # Necessário para o Recall absoluto

        for id_query in ids_query:
            weights = None

            # ==========================================
            # ITERAÇÃO 0 - BUSCA INICIAL
            # ==========================================
            t_busca_inicio = time.time()
            resultados_globais = consult(np.array(df_dataset.iloc[[id_query]]), np.array(df_dataset), p=p, w=weights)
            t_busca_fim = time.time()

            t_busca = t_busca_fim - t_busca_inicio
            t_rr = 0.0

            if strategy == 'in_memory':
                idx_in_memory = np.argsort(resultados_globais)[:(memory_size + 1)]
                resultados_iteracao = idx_in_memory[:(qtd + 1)]
                dataset_target = df_dataset.iloc[idx_in_memory].values
            else:
                resultados_iteracao = np.argsort(resultados_globais)[:(qtd + 1)]
                dataset_target = np.array(df_dataset)

            # Exclui a imagem de consulta do Top-K
            new_a = np.delete(resultados_iteracao, np.where(resultados_iteracao == id_query))
            scores_raw = [df_classes.iloc[idx]['category'] for idx in new_a]
            scores_array = [np.array([s]) for s in scores_raw] # Formato exigido pela função apk()

            # Cálculos de Métricas
            tp = sum(1 for s in scores_raw if s == category)
            prec = tp / qtd
            rec = tp / total_relevantes_dataset
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            ap = apk(scores_array, category, k=qtd)

            metricas['precision'][0].append(prec)
            metricas['recall'][0].append(rec)
            metricas['f1'][0].append(f1)
            metricas['map'][0].append(ap)
            metricas['tempo_busca'][0].append(t_busca)

            # Cálculo do tempo do Relevance Feedback
            t_rr_inicio = time.time()
            if type_rr == 'std':
                feedback = get_feedback_by_class(crude_dataset, resultados_iteracao, category)
                weights = feedback_relevance(feedback, dimensao)
            t_rr_fim = time.time()
            t_rr = t_rr_fim - t_rr_inicio

            metricas['tempo_rr'][0].append(t_rr)
            metricas['tempo_total'][0].append(t_busca + t_rr)

            # =============================================
            # ITERAÇÕES 1 A 4 - COM FEEDBACK DE RELEVÂNCIA
            # =============================================
            for itr in range(1, 5):
                t_busca_inicio = time.time()
                resultados_ciclo = consult(np.array(df_dataset.iloc[[id_query]]), dataset_target, p=p, w=weights)
                t_busca_fim = time.time()
                t_busca = t_busca_fim - t_busca_inicio

                if strategy == 'in_memory':
                    idx_locais = np.argsort(resultados_ciclo)[:(qtd + 1)]
                    resultados_iteracao = np.array([idx_in_memory[loc] for loc in idx_locais])
                else:
                    resultados_iteracao = np.argsort(resultados_ciclo)[:(qtd + 1)]

                new_a = np.delete(resultados_iteracao, np.where(resultados_iteracao == id_query))
                scores_raw = [df_classes.iloc[idx]['category'] for idx in new_a]
                scores_array = [np.array([s]) for s in scores_raw]

                tp = sum(1 for s in scores_raw if s == category)
                prec = tp / qtd
                rec = tp / total_relevantes_dataset
                f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
                ap = apk(scores_array, category, k=qtd)

                metricas['precision'][itr].append(prec)
                metricas['recall'][itr].append(rec)
                metricas['f1'][itr].append(f1)
                metricas['map'][itr].append(ap)
                metricas['tempo_busca'][itr].append(t_busca)

                t_rr_inicio = time.time()
                if type_rr == 'std':
                    feedback = get_feedback_by_class(crude_dataset, resultados_iteracao, category)
                    weights = feedback_relevance(feedback, dimensao)
                t_rr_fim = time.time()
                t_rr = t_rr_fim - t_rr_inicio

                metricas['tempo_rr'][itr].append(t_rr)
                metricas['tempo_total'][itr].append(t_busca + t_rr)

    # Consolida as médias gerais de todo o dataset para cada iteração
    resultados_finais = {}
    for key in metricas:
        # np.mean ignorando possíveis avisos de array vazio, só por segurança
        resultados_finais[key] = [np.mean(metricas[key][i]) for i in range(5)]

    return resultados_finais