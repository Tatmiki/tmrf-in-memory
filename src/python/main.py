import pandas as pd
from extractor import extract_feature_dataset
import os

# ─────────────────────────────────────────────
# Constantes de configuração do código
# ─────────────────────────────────────────────
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent / "data"

COREL_DATASET_PATH = BASE_DIR / "raw" / "elkamel_corel-images" / "dataset"
COVID_DATASET_PATH = BASE_DIR / "raw" / "tawsifurrahman_covid19-radiography-database" / "COVID-19_Radiography_Dataset"

CSV_OUTPUT_PATH = BASE_DIR / "features"

# ─────────────────────────────────────────────
# Funções de extração para cada dataset
# ─────────────────────────────────────────────

# DATASET: COVID-19 Radiography Database (https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
def extract_covid_dataset_to_csv(paths: list[str], qtd: int=0):
    """Extrai as características das imagens de radiografia de pulmões em diferentes situações, incluindo a de COVID-19, e salva em um arquivo CSV."""
    extractions = ['gray', 'fos','glcm', 'sfm', 'lte']
    for i, extractor in enumerate(extractions):
        dataset = []
        dataset_files = []
        
        print(f'>>> Extractor: {extractor}')
        for index, value in enumerate(paths):
            if qtd <= 0:
                qtd = len(os.listdir(value))
            (extractions, files) = extract_feature_dataset(paths[index], extractor=extractor, qtd=qtd)
            dataset = dataset + extractions
            dataset_files = dataset_files + files

        df_data = pd.DataFrame(dataset) # Criação do DataFrame
        df_data.columns = [*df_data.columns[:-1], 'category'] # Ajusta nome da última coluna como 'category'
        df_data['file'] = dataset_files # Adiciona coluna de arquivos
        
        # Caminho de saída
        output_path = f"{CSV_OUTPUT_PATH}/covid_{extractor}.csv"
        df_data.to_csv(output_path, index=False)
        
        print(f"CSV salvo em: {output_path}\n")


# DATASET: corel_images (https://www.kaggle.com/datasets/elkamel/corel-images)
def extract_corel_dataset_to_csv(paths: list[str]):
    """Extrai características de escala de cinza de múltiplas classes de imagens e salva em CSVs."""
    extractor = 'gray'
    dataset = []
    dataset_files = []

    print(f'>>> Extractor: {extractor}')

    for path in paths:
        features_list, files = extract_feature_dataset(path, extractor=extractor, qtd=len(os.listdir(path)))

        dataset.extend(features_list)
        dataset_files.extend(files)

        df_data = pd.DataFrame(dataset) # Criação do DataFrame
        df_data.columns = [*df_data.columns[:-1], 'category'] # Ajusta nome da última coluna como 'category'
        df_data['file'] = dataset_files # Adiciona coluna de arquivos

        # Caminho de saída
        output_path = os.path.join(CSV_OUTPUT_PATH, f"corel_{extractor}.csv")
        df_data.to_csv(output_path, index=False)

        print(f"CSV salvo em: {output_path}\n")

from experiments import *
from cbir import *

import sys

# Classe para sobrescrever o print dos resultados para também sair em um arquivo.
class Tee:
    def __init__(self, *outputs):
        self.outputs = outputs

    def write(self, message):
        for output in self.outputs:
            output.write(message)

    def flush(self):
        for output in self.outputs:
            output.flush()

def main():
    # paths = [ d for d in Path(COVID_DATASET_PATH).iterdir() ]
    # extract_covid_dataset_to_csv(paths, 1000)
    
    # paths = [ d for d in Path(COREL_DATASET_PATH).iterdir() ]
    # extract_corel_dataset_to_csv(paths)
    
    # 1. Carregamento dos dados
    extractors = [
        ("gray", "Gray Histogram"),
        ("glcm", "GLCM"),
        ("sfm", "SFM"),
        ("fos", "FOS"),
        ("lte", "LTE"),
    ]
    
    lista_extratores = []

    for extractor_key, extractor_name in extractors:
        path = CSV_OUTPUT_PATH / f"covid_{extractor_key}.csv"
        
        aux_data = load_dataset(path, ['lung_opacity'])
        
        covid_data = [
            aux_data[0],
            aux_data[1],
            aux_data[2],
            extractor_name,
            'covid'
        ]
        
        lista_extratores.append(covid_data)

    # Dicionário que guardará todas as variáveis para depois plotarmos os gráficos
    resultados_completos = {}
    
    # Cria arquivo de log na mesma pasta do script
    log_path = Path(__file__).resolve().parent / "resultados.txt"
    log_file = open(log_path, "w", encoding="utf-8")

    # Redireciona stdout para terminal + arquivo
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_file)

    try:
        for dados in lista_extratores:
            crude_dataset = dados[0]
            df_data = dados[1]
            df_class = dados[2]
            nome_extrator = dados[3]

            print(f"\n{'='*80}")
            print(f"INICIANDO EXPERIMENTOS: {nome_extrator}")
            print(f"{'='*80}")

            # -------------------------------------------------------------
            # Executa a estratégia Clássica (Global)
            # -------------------------------------------------------------
            print("[...] Processando Estratégia Global...")
            inicio_global = time.time()
            metricas_global = executar_experimento(crude_dataset, df_data, df_class, p=1, qtd=10, strategy='global')
            fim_global = time.time()
            tempo_total_global_absoluto = fim_global - inicio_global

            # -------------------------------------------------------------
            # Executa a estratégia Em Memória
            # -------------------------------------------------------------
            print("[...] Processando Estratégia Em Memória...")
            inicio_local = time.time()
            metricas_local = executar_experimento(crude_dataset, df_data, df_class, p=1, qtd=10, strategy='in_memory', memory_size=100)
            fim_local = time.time()
            tempo_total_local_absoluto = fim_local - inicio_local

            # Salva no dicionário global para uso posterior nos gráficos
            resultados_completos[nome_extrator] = {
                'global': metricas_global,
                'in_memory': metricas_local
            }

            # Exibir no terminal comparando as duas estratégias
            print(f"\n{'-'*80}")
            print(f"{'Métrica (Média das 5 Iterações)':<32} | {'Estratégia Global':<20} | {'Estratégia Em Memória'}")
            print(f"{'-'*80}")

            # Extraindo as médias gerais das 5 iterações para simplificar a visualização em texto
            map_glob_mean  = np.mean(metricas_global['map']) * 100
            map_loc_mean   = np.mean(metricas_local['map']) * 100

            prec_glob_mean = np.mean(metricas_global['precision']) * 100
            prec_loc_mean  = np.mean(metricas_local['precision']) * 100

            rec_glob_mean  = np.mean(metricas_global['recall']) * 100
            rec_loc_mean   = np.mean(metricas_local['recall']) * 100

            f1_glob_mean   = np.mean(metricas_global['f1']) * 100
            f1_loc_mean    = np.mean(metricas_local['f1']) * 100

            # Tempos formatados em segundos (s) (Tempo médio por consulta)
            tb_glob_mean   = np.mean(metricas_global['tempo_busca'])
            tb_loc_mean    = np.mean(metricas_local['tempo_busca'])

            tr_glob_mean   = np.mean(metricas_global['tempo_rr'])
            tr_loc_mean    = np.mean(metricas_local['tempo_rr'])

            tt_glob_mean   = np.mean(metricas_global['tempo_total'])
            tt_loc_mean    = np.mean(metricas_local['tempo_total'])

            print(f"{'MAP (Mean Average Precision)':<32} | {map_glob_mean:>18.2f}% | {map_loc_mean:>18.2f}%")
            print(f"{'Precisão (Top-K)':<32} | {prec_glob_mean:>18.2f}% | {prec_loc_mean:>18.2f}%")
            print(f"{'Recall':<32} | {rec_glob_mean:>18.2f}% | {rec_loc_mean:>18.2f}%")
            print(f"{'F1-Score':<32} | {f1_glob_mean:>18.2f}% | {f1_loc_mean:>18.2f}%")
            print(f"{'-'*80}")
            print(f"{'Tempo de Busca (seg/consulta)':<32} | {tb_glob_mean:>18.5f}s | {tb_loc_mean:>18.5f}s")
            print(f"{'Tempo do RR (seg/consulta)':<32} | {tr_glob_mean:>18.5f}s | {tr_loc_mean:>18.5f}s")
            print(f"{'Tempo TOTAL (seg/consulta)':<32} | {tt_glob_mean:>18.5f}s | {tt_loc_mean:>18.5f}s")
            print(f"{'-'*80}")

            # -------------------------------------------------------------
            # Exibição do Tempo Absoluto do Experimento Inteiro
            # -------------------------------------------------------------
            economia = 100 - ((tempo_total_local_absoluto / tempo_total_global_absoluto) * 100)

            print(f"{'TEMPO ABSOLUTO DO EXPERIMENTO':<32} | {tempo_total_global_absoluto:>18.2f}s | {tempo_total_local_absoluto:>18.2f}s")
            print(f"{'-'*80}")
            print(f"A estratégia em memória foi {economia:.2f}% mais rápida no tempo total")
            print(f"{'='*80}\n")
    
    finally:
        # Restaura stdout e fecha arquivo
        sys.stdout = original_stdout
        log_file.close()


if __name__ == "__main__":
    main()