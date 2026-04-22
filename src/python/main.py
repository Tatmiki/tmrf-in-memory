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
def extract_covid_dataset_to_csv(paths: list[str]):
    """Extrai as características das imagens de radiografia de pulmões em diferentes situações, incluindo a de COVID-19, e salva em um arquivo CSV."""
    extractions = ['gray', 'fos','glcm', 'sfm', 'lte']
    for i, extractor in enumerate(extractions):
        dataset = []
        dataset_files = []
        
        print(f'>>> Extractor: {extractor}')
        for index, value in enumerate(paths):
            (extractions, files) = extract_feature_dataset(paths[index], extractor=extractor, qtd=len(os.listdir(value)))
            dataset = dataset + extractions
            dataset_files = dataset_files + files

        df_data = pd.DataFrame(dataset)
        
        df_data.columns = [*df_data.columns[:-1], 'category']
        
        df_data['file'] = dataset_files
        
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

        # Criação do DataFrame
        df_data = pd.DataFrame(dataset)

        # Ajusta nome da última coluna como 'category'
        df_data.columns = [*df_data.columns[:-1], 'category']

        # Adiciona coluna de arquivos
        df_data['file'] = dataset_files

        # Caminho de saída
        output_path = os.path.join(CSV_OUTPUT_PATH, f"corel_{extractor}.csv")

        df_data.to_csv(output_path, index=False)

        print(f"CSV salvo em: {output_path}\n")

def main():
    paths = [ d for d in Path(COVID_DATASET_PATH).iterdir() ]
    extract_covid_dataset_to_csv(paths)
    
    paths = [ d for d in Path(COREL_DATASET_PATH).iterdir() ]
    extract_corel_dataset_to_csv(paths)

if __name__ == "__main__":
    main()