import numpy as np
import cv2 as cv
from skimage import io
import os
import pandas as pd
import re
import sys
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
print(BASE_DIR)
sys.path.append(str(BASE_DIR / "external" / "pyfeats"))

import pyfeats as ft
# ─────────────────────────────────────────────
# Funções de apoio
# ─────────────────────────────────────────────
def sorted_nicely(l: list[str]) -> list[str]:
    """Ordena strings de forma natural (considerando números corretamente).

    Diferente da ordenação padrão, esta função trata números dentro das
    strings como valores numéricos, evitando problemas como "item10"
    aparecer antes de "item2".

    Args:
        l (list[str]): Lista de strings a serem ordenadas.

    Returns:
        list[str]: Lista ordenada de forma natural.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def normalize_l2(features: np.ndarray) -> np.ndarray:
    """Normaliza um vetor pela norma L2."""
    norm = np.linalg.norm(features)
    if norm == 0:
        return features
    return features / norm

# ─────────────────────────────────────────────
# Funções de extração de características
# ─────────────────────────────────────────────
def extract_feature_dataset(path: str, extractor: str='color', qtd: int=100):
    """Extrai características de um conjunto de imagens em um diretório.

    A função percorre um diretório contendo imagens, aplica um método de extração
    de características especificado e retorna um dataset com os vetores de
    características associados a cada imagem, juntamente com seus nomes.

    Args:
        path (str): Caminho para o diretório contendo as imagens.
        extractor (str, opcional): Método de extração de características a ser utilizado.
            Valores suportados:
                - 'color': Histogramas de cores RGB normalizados.
                - 'gray': Histograma em escala de cinza.
                - 'glcm': Gray Level Co-occurrence Matrix (GLCM).
                - 'sfm': Statistical Feature Matrix (SFM).
                - 'lte': Local Texture Energy (LTE).
                - 'fos': First Order Statistics (FOS).
            Padrão é 'color'.
        qtd (int, opcional): Quantidade máxima de imagens a serem processadas.
            Padrão é 100.

    Returns:
        tuple:
            - data (list of numpy.ndarray): Lista contendo os vetores de características
              extraídos de cada imagem. Cada vetor inclui também a categoria como último elemento.
            - images_name (list of str): Lista com os nomes das imagens processadas,
              prefixados pela categoria.

    Raises:
        FileNotFoundError: Se o caminho especificado não existir.
        ValueError: Se o método de extração especificado não for suportado.
        cv2.error: Se houver erro ao carregar ou processar uma imagem.

    Notes:
        - A categoria é inferida a partir do nome do diretório (`path`).
        - As imagens são ordenadas utilizando a função `sorted_nicely` antes do processamento.
        - Para o extrator 'color', histogramas de 256 bins são calculados para cada canal RGB
          e normalizados pela norma L2.
        - Algumas técnicas (como GLCM, SFM, LTE, FOS) dependem da biblioteca `mt_feats`.
        - A máscara utilizada nos métodos baseados em textura é uma matriz de uns com o
          mesmo tamanho da imagem em escala de cinza.

    Example:
        >>> data, names = extract_feature_dataset(
        ...     path="dataset/cats",
        ...     extractor="gray",
        ...     qtd=50
        ... )
        >>> len(data)
        50
        >>> names[0]
        'catsimage1.jpg'
    """
    images_path = os.listdir(path)
    images_path = sorted_nicely(images_path)[0:qtd]
    category = os.path.basename(os.path.normpath(path))
    data = []
    images_name = []
    categories = []
    for n, image in enumerate(tqdm(images_path, desc=f"Processando {category:<30}", unit="img")):
        category = os.path.basename(os.path.normpath(path))
        img_name = image
        image = cv.imread(os.path.join(path, image))
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        mask = np.ones((image_gray.shape[0], image_gray.shape[1]))

        features = []
        if extractor == 'color':
            dataset_hist_r = []
            dataset_hist_g = []
            dataset_hist_b = []

            color = ('r', 'g', 'b')
            for i, col in enumerate(color):
                histr = cv.calcHist([image_rgb], [i], None, [256], [0,256])
                if col == 'r':
                    dataset_hist_r.append(histr)
                if col == 'g':
                    dataset_hist_g.append(histr)
                if col == 'b':
                    dataset_hist_b.append(histr)

            X_r = np.array(dataset_hist_r)
            length = np.sqrt((X_r**2).sum(axis=1))[:, None]
            X_r = X_r / length

            X_g = np.array(dataset_hist_g)
            length = np.sqrt((X_g**2).sum(axis=1))[:, None]
            X_g = X_g / length

            X_b = np.array(dataset_hist_b)
            length = np.sqrt((X_b**2).sum(axis=1))[:, None]
            X_b = X_b / length

            X = np.concatenate((X_r, X_g, X_b), axis=1)
            X.shape
            features = X.max(2)
        elif extractor == 'gray':
            histr = cv.calcHist([image_gray], [0], None, [32], [0,256])
            features = (histr).max(1)
        elif extractor == 'glcm':
            features = ft.glcm_features(image_gray, ignore_zeros=True)[0]
        elif extractor == 'sfm':
            features = ft.sfm_features(image_gray, mask, Lr=4, Lc=4)[0]
        elif extractor == 'lte':
            features = ft.lte_measures(image_gray, mask, l=7)[0]
        elif extractor == 'fos':
            features = ft.fos(image_gray, mask)[0]
            
        images_name.append(img_name)
        categories.append(category)
        
        features = np.array(features, dtype=np.float32) # Converte para float (evita problemas com dtype)
        features = normalize_l2(features) # Normalização L2 por vetor
        features = np.append(features, category)

        data.append(features)
        
    return data, images_name
