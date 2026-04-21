import os
import json
import threading
import itertools
import time
from pathlib import Path
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Configurações do Projeto
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DESTINATION_FOLDER = PROJECT_ROOT / "data" / "raw"
ENV_PATH = PROJECT_ROOT / ".env"

# ─────────────────────────────────────────────
# Lista de Datasets a Baixar
# Adicione ou remova entradas conforme necessário.
# Formato: "owner/dataset-name" (igual à URL do Kaggle)
# ─────────────────────────────────────────────
KAGGLE_DATASETS = [
    "tawsifurrahman/covid19-radiography-database",
    "elkamel/corel-images",
]


def load_animation(message: str, stop_event: threading.Event):
    """Exibe uma animação de spinner no terminal enquanto parar_evento não for acionado."""
    spinner = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
    while not stop_event.is_set():
        print(f"\r  {next(spinner)} {message}", end="", flush=True)
        time.sleep(0.1)
    print(f"\r    [SUCESSO]: Download concluído!{' ' * 100}")  # limpa a linha ao terminar

def authenticate_kaggle() -> object | None:
    """Carrega credenciais do .env e retorna uma instância autenticada da API do Kaggle."""
    if not ENV_PATH.exists():
        print(f"\n[ERRO] Arquivo .env não encontrado em: {ENV_PATH}")
        return None

    load_dotenv(dotenv_path=ENV_PATH)

    token_str = os.getenv("KAGGLE_API_TOKEN")
    if not token_str:
        print("\n[ERRO] Variável 'KAGGLE_API_TOKEN' não encontrada no arquivo .env.")
        return None

    try:
        token_str = token_str.strip("'\"")
        token_data = json.loads(token_str)
        os.environ["KAGGLE_USERNAME"] = token_data["username"]
        os.environ["KAGGLE_KEY"] = token_data["key"]
    except (json.JSONDecodeError, KeyError):
        print("\n[ERRO] Formato inválido no KAGGLE_API_TOKEN.")
        print("Exemplo: export KAGGLE_API_TOKEN='{\"username\":\"seu_user\",\"key\":\"sua_key\"}'")
        return None

    # Importa APENAS após setar as variáveis de ambiente
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as e:
        print(f"\n[ERRO] Falha na autenticação do Kaggle: {e}")
        return None


def download_dataset(api, dataset: str, destination: Path) -> bool:
    """
    Faz o download e extração de um único dataset.
    Retorna True em caso de sucesso, False em caso de falha.
    """
    dataset_folder = destination / dataset.replace("/", "_")
    dataset_folder.mkdir(parents=True, exist_ok=True)

    stop_thread = threading.Event()
    thread = threading.Thread(
        target=load_animation,
        args=(f"Baixando '{dataset}'... ", stop_thread)
    )
    thread.start()
    try:
        api.dataset_download_files(dataset, path=dataset_folder, unzip=True)
        return True
    except Exception as e:
        print(f"    [ERRO]: Falha ao baixar '{dataset}': {e}")
        return False
    finally:
        stop_thread.set() # simboliza parada pra thread
        thread.join() # espera a thread terminar antes de continuar


def main():
    print("=" * 55)
    print("   Download de Datasets do Kaggle")
    print("=" * 55)

    if not KAGGLE_DATASETS:
        print("\n[AVISO]: A lista 'KAGGLE_DATASETS' está vazia. Nada a baixar.")
        return

    # Passo 1: Preparar pasta de destino
    DESTINATION_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"\n[1/3] Pasta de destino: {DESTINATION_FOLDER}")

    # Passo 2: Autenticar
    print("[2/3] Autenticando no Kaggle...")
    api = authenticate_kaggle()
    if api is None:
        return
    print("- Autenticação realizada com sucesso.")

    # Passo 3: Iterar e baixar cada dataset
    total = len(KAGGLE_DATASETS)
    print(f"\n[3/3] Iniciando download de {total} dataset(s)...\n")

    resultados = {"sucesso": [], "falha": []}

    for i, dataset in enumerate(KAGGLE_DATASETS, start=1):
        print(f"[{i}/{total}] Dataset: {dataset}")
        ok = download_dataset(api, dataset, DESTINATION_FOLDER)
        if ok:
            resultados["sucesso"].append(dataset)
        else:
            resultados["falha"].append(dataset)
        print()

    # Resumo final
    print("=" * 55)
    print(f"  Resumo: {len(resultados['sucesso'])}/{total} baixados com sucesso.")
    if resultados["falha"]:
        print("\n  Datasets com falha:")
        for ds in resultados["falha"]:
            print(f"    • {ds}")
    print("=" * 55)


if __name__ == "__main__":
    main()