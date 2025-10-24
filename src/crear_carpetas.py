import json
from sklearn.model_selection import KFold
from pathlib import Path

# Este archivo se encarga de dividir los archivos en folds para cross validation

BASE_DIR = Path(__file__).resolve().parent.parent
ANNOTATIONS_DIR = BASE_DIR / "annotations"
path_train = ANNOTATIONS_DIR / "train.jsonl"
num_folds = 5

data = []
with open(path_train, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Linea con error JSON (omitida): {e}")

kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

for i, (train_idx, val_idx) in enumerate(kf.split(data)):
    train_fold = [data[j] for j in train_idx]
    val_fold = [data[j] for j in val_idx]

    with open(path_train.parent / f"fold_{i+1}_train.jsonl", "w", encoding="utf-8") as f_train:
        for item in train_fold:
            f_train.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(path_train.parent / f"fold_{i+1}_val.jsonl", "w", encoding="utf-8") as f_val:
        for item in val_fold:
            f_val.write(json.dumps(item, ensure_ascii=False) + "\n")