import os
import torch
from pathlib import Path
from datasets import load_from_disk
from PIL import Image
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
from dotenv import load_dotenv
load_dotenv()

# Configuración con las rutas del env
BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_PATH = BASE_DIR / os.getenv("DATASETS_PATH", "datasets_layoutlmv3")
OUTPUT_PATH = BASE_DIR / os.getenv("OUTPUT_PATH", "resultados")
os.makedirs(OUTPUT_PATH, exist_ok=True)

NUM_FOLDS = int(os.getenv("NUM_FOLDS", "5"))
NUM_LABELS = int(os.getenv("NUM_LABELS", "9"))
EPOCHS = int(os.getenv("EPOCHS", "4"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "5e-5"))

device = "cuda" if torch.cuda.is_available() else "cpu"

# Normaliza bbox de píxeles a rango [0, 1000] para las imagenes
def normalize_bbox(bbox, width, height):
    if all(coord <= 1000 for coord in bbox):
        return [max(0, min(1000, int(coord))) for coord in bbox]
    
    return [
        max(0, min(1000, int(1000 * bbox[0] / width))),
        max(0, min(1000, int(1000 * bbox[1] / height))),
        max(0, min(1000, int(1000 * bbox[2] / width))),
        max(0, min(1000, int(1000 * bbox[3] / height))),
    ]


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=2)

    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


# Entrena un fold (recibe el número del fold a entrenar)
def entrenar_fold(fold_num):
    fold_path = DATASETS_PATH / f"fold_{fold_num}"
    dataset = load_from_disk(str(fold_path))
    ##Aca se cartga el modelo de LayoutLMv3
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base", num_labels=NUM_LABELS
    ).to(device)

    # Congelamos las primeras 10 capas para acelerar el entrenamiento (a futuro lo podemos sacar para que entrene el modelo)
    #for name, param in model.named_parameters():
        #if "encoder.layer" in name and any(f"layer.{i}." in name for i in range(10)):
            #param.requires_grad = False

    all_labels = sorted(set(sum([ex["labels"] for ex in dataset["train"]], [])))
    label_map = {label: i for i, label in enumerate(all_labels)}
    
    model.config.id2label = {i: label for i, label in enumerate(all_labels)}
    model.config.label2id = {label: i for i, label in enumerate(all_labels)}

    def preprocess_batch(example):
        try:
            image = Image.open(example["image"]).convert("RGB")
            width, height = image.size
            
            normalized_boxes = []
            for bbox in example["bboxes"]:
                try:
                    norm_bbox = normalize_bbox(bbox, width, height)
                    normalized_boxes.append(norm_bbox)
                except Exception as e:
                    normalized_boxes.append([0, 0, 1000, 1000])

            encoding = processor(
                image,
                example["words"],
                boxes=normalized_boxes,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
            )

            example_labels = [label_map.get(l, 0) for l in example["labels"]]
            labels = example_labels + [-100] * (512 - len(example_labels))

            encoding = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in encoding.items()}
            encoding["labels"] = torch.tensor(labels[:512])
            return encoding

        except Exception as e:
            print(f"Error procesando {example.get('id', 'unknown')}: {e}")
            return {
                "input_ids": torch.zeros(512, dtype=torch.long),
                "attention_mask": torch.zeros(512, dtype=torch.long),
                "bbox": torch.zeros((512, 4), dtype=torch.long),
                "pixel_values": torch.zeros((3, 224, 224)),
                "labels": torch.full((512,), -100, dtype=torch.long)
            }
    #aplica preprocess y eleimina las columnas originales
    train_dataset = dataset["train"].map(
        preprocess_batch,
        remove_columns=dataset["train"].column_names,
        desc="Preprocesando train"
    )
    
    val_dataset = dataset["validation"].map(
        preprocess_batch,
        remove_columns=dataset["validation"].column_names,
        desc="Preprocesando val"
    )
    # Configuracion  de los argumentos de entrenamiento
    args = TrainingArguments(
        output_dir=str(OUTPUT_PATH / f"fold_{fold_num}"),
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        save_steps=500,
        eval_steps=500,
        save_total_limit=1,
        load_best_model_at_end=False,
        metric_for_best_model="f1",
        logging_dir=str(OUTPUT_PATH / f"logs_fold_{fold_num}"),
        logging_steps=50,
        report_to=[],
        dataloader_pin_memory=False,
        fp16=True if device == "cuda" else False,
    )
    #crea el trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    #entrena
    trainer.train()
    
    #se guarda el modelo entrenado 
    model.save_pretrained(OUTPUT_PATH / f"modelo_fold_{fold_num}")
    processor.save_pretrained(OUTPUT_PATH / f"modelo_fold_{fold_num}")
# Evalua el modelo
    results = trainer.evaluate()
    #nos genera las metricas
    np.save(OUTPUT_PATH / f"metricas_fold_{fold_num}.npy", results)

    return results


#pipeline que se ejecuta el archivo
if __name__ == "__main__":
    metricas_folds = []
    #for i in range(1, NUM_FOLDS + 1):
    #Cambiamos solo para la carpeta 3 debido a que nos habia dado mejor metrica con 1 epoch (Se eligio la 3 debido al rendimiento que dio con 1 epoch)
    #buscamos entrenar solo 1 carpeta con mas aprendzaje por eso mismo ses le descongelo las 10 capas.
    for i in [3]:
        try:
            res = entrenar_fold(i)
            metricas_folds.append(res)
        except Exception as e:
            print(f"Error en fold {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not metricas_folds:
        print("No se completó ningún fold exitosamente.")
        exit(1)

    # Calcular promedios para ver el rendimiento general del modelo entrenado
    f1_scores = [m.get("eval_f1", 0) for m in metricas_folds]
    acc_scores = [m.get("eval_accuracy", 0) for m in metricas_folds]

    mean_f1 = np.mean(f1_scores)
    mean_acc = np.mean(acc_scores)

    # Guardamos resumen
    resumen_path = OUTPUT_PATH / "metricas_resumen.txt"
    with open(resumen_path, "w", encoding="utf-8") as f:
        f.write("Resumen entrenamiento cruzado LayoutLMv3\n")
        f.write(f"F1 promedio: {mean_f1:.4f}\n")
        f.write(f"Accuracy promedio: {mean_acc:.4f}\n")
        f.write(f"\nFolds completados: {len(metricas_folds)}\n")
        for i, metrics in enumerate(metricas_folds, 1):
            f.write(f"\nFold {i}:\n")
            f.write(f"  F1: {metrics.get('eval_f1', 0):.4f}\n")
            f.write(f"  Accuracy: {metrics.get('eval_accuracy', 0):.4f}\n")