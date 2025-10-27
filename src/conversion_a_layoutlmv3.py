import os
import json
import torch
from pathlib import Path
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv

load_dotenv()

# Convertimos el formato jsonl a un formato compatible con LayoutLMv3
BASE_DIR = Path(__file__).resolve().parent.parent
ANNOTATIONS_DIR = BASE_DIR / os.getenv("ANNOTATIONS_PATH", "annotations")
OUTPUT_DIR = BASE_DIR / os.getenv("DATASETS_PATH", "datasets_layoutlmv3")
DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalizar_bbox(bbox, width, height):
    # Normaliza coordenadas (x1, y1, x2, y2) entre 0 y 1000
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height))
    ]

# Carga un JSONL en memoria, filtrando líneas vacías o corruptas
def cargar_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # Verificacion de estructura valida para evitar errores
                if (
                    obj is None or
                    not isinstance(obj, dict) or
                    "id" not in obj or
                    obj.get("words") is None or
                    obj.get("bboxes") is None or
                    obj.get("labels") is None
                ):
                    print(f"Linea {i} no compatible en {path.name}, omitida.")
                    continue
                data.append(obj)
            except json.JSONDecodeError:
                print(f"Error JSON en linea {i} de {path.name}, omitida.")
    return data

# Convierte una lista de ejemplos a formato LayoutLMv3
def convertir_a_layoutlmv3(data, img_folder):
    registros = []
    for item in data:
        try:
            img_name = item["id"]
            words = item["words"]
            bboxes = item["bboxes"]
            labels = item["labels"]
        except Exception as e:
            print(f"Registro no valido, omitido: {e}")
            continue

        #img_path = None
        #for root, _, files in os.walk(img_folder):
            #if img_name in files:
                #img_path = os.path.join(root, img_name)
                #break

        formato = img_name.split('_')[0] + '_' + img_name.split('_')[1]  # formato_1
        img_path = DATA_DIR / formato / "train" / img_name

        if not img_path.exists():
            # Si no está en train, buscar en test
            img_path = DATA_DIR / formato / "test" / img_name
            
        if not img_path.exists():
            print(f"No se encontró la imagen para {img_name}, omitida.")
            continue

        img_path = str(img_path)


        # Simulacion de tamaño para la normalización (1000x1000 si no se conoce) (al final usamos el tamaño fijo que se genero es 1654x2339)
        width, height = 1654, 2339
        normalized_bboxes = [normalizar_bbox(b, width, height) for b in bboxes]

        registros.append({
            "id": img_name,
            "image": img_path,
            "words": words,
            "bboxes": normalized_bboxes,
            "labels": labels
        })

    return registros

# Procesa un fold específico
def procesar_fold(fold_num):
    train_path = ANNOTATIONS_DIR / f"fold_{fold_num}_train.jsonl"
    val_path = ANNOTATIONS_DIR / f"fold_{fold_num}_val.jsonl"

    train_data = cargar_jsonl(train_path)
    val_data = cargar_jsonl(val_path)

    # Convertir los ejemplos
    train_dataset = convertir_a_layoutlmv3(train_data, DATA_DIR)
    val_dataset = convertir_a_layoutlmv3(val_data, DATA_DIR)

    # Crear datasets Hugging Face
    if not train_dataset or not val_dataset:
        print(f"Fold {fold_num} vacio o con errores. Se omite.")
        return

    ds_train = Dataset.from_list(train_dataset)
    ds_val = Dataset.from_list(val_dataset)
    dataset_dict = DatasetDict({"train": ds_train, "validation": ds_val})
    #Guarda
    output_path = OUTPUT_DIR / f"fold_{fold_num}"
    dataset_dict.save_to_disk(str(output_path))

if __name__ == "__main__":
    for fold_num in range(1, 6):
        procesar_fold(fold_num)