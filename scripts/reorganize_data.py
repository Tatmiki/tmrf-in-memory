import os
import shutil
from pathlib import Path


def reorganize_corel(base_path: Path):
    dataset_path = base_path / "elkamel_corel-images" / "dataset"
    test_path = dataset_path / "test_set"
    train_path = dataset_path / "training_set"

    # Coleta todas as classes existentes
    classes = set()
    for split in [test_path, train_path]:
        if split.exists():
            classes.update(p.name for p in split.iterdir() if p.is_dir())

    # Cria as pastas de classe direto em dataset e move os arquivos
    for cls in classes:
        dest = dataset_path / cls
        dest.mkdir(exist_ok=True)

        for split in [test_path, train_path]:
            src = split / cls
            if src.exists():
                for file in src.iterdir():
                    if file.is_file():
                        target = dest / file.name
                        # Evita colisão de nomes
                        if target.exists():
                            target = dest / f"{split.name}_{file.name}"
                        shutil.move(str(file), str(target))

    # Remove as pastas test_set e training_set
    for split in [test_path, train_path]:
        if split.exists():
            shutil.rmtree(split)

    print("- elkamel_corel-images reorganizado.")


def reorganize_covid(base_path: Path):
    dataset_path = base_path / "tawsifurrahman_covid19-radiography-database" / "COVID-19_Radiography_Dataset"

    for class_dir in list(dataset_path.iterdir()):
        if not class_dir.is_dir():
            # Remove arquivos soltos na raiz do dataset
            class_dir.unlink()
            continue

        # Renomeia a pasta para lowercase com underscore
        new_name = class_dir.name.lower().replace(" ", "_").replace("-", "_")
        new_path = dataset_path / new_name
        if class_dir != new_path:
            class_dir.rename(new_path)
        class_dir = new_path

        # Remove pasta masks
        masks_path = class_dir / "masks"
        if masks_path.exists():
            shutil.rmtree(masks_path)

        # Move conteúdo de images para a pasta da classe
        images_path = class_dir / "images"
        if images_path.exists():
            for file in images_path.iterdir():
                if file.is_file():
                    shutil.move(str(file), str(class_dir / file.name))
            images_path.rmdir()

        # Remove qualquer arquivo extra que não seja imagem (ex: .csv, .xlsx)
        for file in list(class_dir.iterdir()):
            if file.is_file() and file.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                file.unlink()

    print("- tawsifurrahman_covid19-radiography-database reorganizado.")


if __name__ == "__main__":
    base = Path("./data/raw")

    reorganize_corel(base)
    reorganize_covid(base)

    print("\n## Reorganização concluída! ##")