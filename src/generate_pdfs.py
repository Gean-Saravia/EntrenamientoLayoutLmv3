import os
import random
from faker import Faker
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from pdf2image import convert_from_path
from pathlib import Path
import numpy as np
import cv2
import json
from PIL import Image
from dotenv import load_dotenv 

load_dotenv()

fake = Faker("es_ES")

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / os.getenv("TEMPLATES_DIR", "plantillas_facturas")
LOGOS_DIR = BASE_DIR / os.getenv("LOGOS_DIR", "logos_facturas")
DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")

FORMATOS = [f"formato_{i}" for i in range(1, 6)]
PDFS_POR_FORMATO = int(os.getenv("PDFS_POR_FORMATO", "2000"))

FUENTES = ["Arial", "Verdana", "Times New Roman", "Courier New", "Georgia"]
COLORES_TITULO = ["#0044AA", "#AA0000", "#008844", "#000000", "#6600CC"]
COLORES_BORDE = ["#222", "#444", "#666", "#999", "#555"]


def generar_items():
    items = []
    for _ in range(random.randint(2, 6)):
        precio = round(random.uniform(500, 30000), 2)
        cantidad = random.randint(1, 5)
        items.append({
            "producto": fake.word().capitalize(),
            "cantidad": cantidad,
            "precio": precio,
            "subtotal": round(precio * cantidad, 2)
        })
    return items


def agregar_ruido(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return
    ruido = np.random.normal(0, 20, img.shape).astype(np.uint8)
    img_ruido = cv2.add(img, ruido)
    angulo = random.choice([0, 0, 0, 2, -2, 1, -1])
    if angulo != 0:
        (h, w) = img_ruido.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angulo, 1.0)
        img_ruido = cv2.warpAffine(img_ruido, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite(str(img_path), img_ruido)


def guardar_json_anotaciones(data, img_path):
    json_data = {
        "file_name": os.path.basename(img_path),
        "fields": [
            {"label": "nombre", "text": data["nombre"]},
            {"label": "apellido", "text": data["apellido"]},
            {"label": "n_cliente", "text": str(data["n_cliente"])},
            {"label": "email", "text": data["email"]},
            {"label": "n_factura", "text": str(data["n_factura"])},
            {"label": "fecha", "text": str(data["fecha"])},
            {"label": "hora", "text": str(data["hora"])},
            {"label": "monto_total", "text": str(data["monto_total"])},
        ]
    }

    json_path = img_path.replace(".png", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

def generar_factura(formato, index):
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    template_path = f"factura_{formato}.html"
    if not (TEMPLATES_DIR / template_path).exists():
        print(f" No se encontró la plantilla: {template_path}")
        return

    template = env.get_template(template_path)
    items = generar_items()
    monto_total = round(sum(i["subtotal"] for i in items), 2)

    data = {
        "nombre": fake.first_name(),
        "apellido": fake.last_name(),
        "n_cliente": fake.random_int(10000, 99999),
        "n_factura": fake.random_int(1000, 9999),
        "fecha": fake.date_this_decade(),
        "hora": fake.time(),
        "email": fake.email(),
        "items": items,
        "monto_total": monto_total,
        "logo": f"../logos_facturas/logo{random.randint(1,5)}.png",
        "fuente": random.choice(FUENTES),
        "color_titulo": random.choice(COLORES_TITULO),
        "color_borde": random.choice(COLORES_BORDE)
    }

    #html a pdf, luego de pdf a imagen y al final json de anotaciones
    html_content = template.render(**data)
    output_pdf = DATA_DIR / formato / "train" / f"{formato}_{index:05d}.pdf"
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html_content, base_url=str(TEMPLATES_DIR)).write_pdf(str(output_pdf))

    try:
        images = convert_from_path(str(output_pdf))
        img_path = str(output_pdf).replace(".pdf", ".png")
        images[0].save(img_path, "PNG")
        agregar_ruido(img_path)
        guardar_json_anotaciones(data, img_path)
    except Exception as e:
        print(f"No se puede convertir el pdf a imagen {output_pdf.name}: {e}")


if __name__ == "__main__":
    for formato in FORMATOS:
        # Genera pdf+png+json por cada formato
        for i in range(1, PDFS_POR_FORMATO + 1):
            generar_factura(formato, i)