import os
import random
import shutil
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

TRAIN_RATIO = 0.8  # 80% train / 20% test ESTO se hace teniendo en cuenta la validacion cruzada


def dividir_carpetas():
    formatos = [f for f in DATA_DIR.iterdir() if f.is_dir() and f.name.startswith("formato_")]
    for formato in formatos:
        train_dir = formato / "train"
        test_dir = formato / "test"
        test_dir.mkdir(exist_ok=True)

        # Filtrar solo los archivos .pdf (los otros se derivan del mismo nombre)
        pdf_files = sorted([f for f in train_dir.glob("*.pdf")])
        random.shuffle(pdf_files)

        # Cantidad para test
        n_test = int(len(pdf_files) * (1 - TRAIN_RATIO))
        test_pdfs = pdf_files[:n_test]

        for pdf in test_pdfs:
            base_name = pdf.stem
            for ext in [".pdf", ".png", ".json"]:
                src = train_dir / f"{base_name}{ext}"
                dst = test_dir / f"{base_name}{ext}"
                if src.exists():
                    shutil.move(str(src), str(dst))

if __name__ == "__main__":
    dividir_carpetas()
