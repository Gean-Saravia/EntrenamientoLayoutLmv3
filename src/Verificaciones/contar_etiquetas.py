import json
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
ANNOTATIONS_DIR = BASE_DIR / os.getenv("ANNOTATIONS_PATH", "annotations")
ruta = ANNOTATIONS_DIR / "train.jsonl"

contador = Counter()
lineas_invalidas = 0

with open(ruta, "r", encoding="utf-8") as f:
    for i, linea in enumerate(f, start=1):
        try:
            linea = linea.strip()
            if not linea:
                continue
            data = json.loads(linea)

            if not isinstance(data, dict):
                lineas_invalidas += 1
                continue

            etiquetas = data.get("labels", [])
            if etiquetas is None:
                lineas_invalidas += 1
                continue

            contador.update(etiquetas)

        except json.JSONDecodeError:
            lineas_invalidas += 1
            continue

print("\nDistribución de etiquetas:")
for etiqueta, cantidad in contador.items():
    print(f"  {etiqueta}: {cantidad}")

print(f"\nTotal de etiquetas únicas: {len(contador)}")
print(f"Líneas inválidas que no se contaron: {lineas_invalidas}")