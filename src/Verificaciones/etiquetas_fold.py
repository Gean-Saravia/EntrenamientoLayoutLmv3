from pathlib import Path
from datasets import load_from_disk
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]

DATASETS_PATH = BASE_DIR / os.getenv("DATASETS_PATH", "datasets_layoutlmv3")

for fold_num in range(1, 6):
    fold_path = DATASETS_PATH / f"fold_{fold_num}"
    if fold_path.exists():
        try:
            dataset = load_from_disk(str(fold_path))
            
            # Etiquetas en train
            all_labels_train = set(sum([ex["labels"] for ex in dataset["train"]], []))
            
            # Etiquetas en validation
            all_labels_val = set(sum([ex["labels"] for ex in dataset["validation"]], []))
            
            print(f"Fold {fold_num}:")
            print(f"   Train: {len(dataset['train'])} ejemplos")
            print(f"   Val: {len(dataset['validation'])} ejemplos")
            print(f"   Etiquetas train: {sorted(all_labels_train)}")
            print(f"   Etiquetas val: {sorted(all_labels_val)}")
            print()
            
        except Exception as e:
            print(f"Error en fold {fold_num}: {e}\n")
    else:
        print(f"Fold {fold_num} no existe\n")

#Termina la verificacion, esto nos muestra las etiquetas luego de convertirlas a layoutlmv3
#y nos permite verificar que todas las etiquetas esten presentes en los folds generados.