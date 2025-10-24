import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from paddleocr import PaddleOCR
from PIL import Image
import json
from collections import defaultdict
import re
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / os.getenv("MODEL_PATH", "resultados/modelo_fold_3")
IMAGES_FOLDER = BASE_DIR / os.getenv("IMAGES_FOLDER", "test")
OUTPUT_FOLDER = BASE_DIR / os.getenv("OUTPUT_FOLDER", "resultados_json")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


#Usamos CUDA si está disponible (es mejor usar GPU para acelerar el proceso, si no el cpu es suficiente)
device = "cuda" if torch.cuda.is_available() else "cpu"

def normalize_bbox(bbox, width, height):
    return [
        int(1000 * bbox[0] / width),
        int(1000 * bbox[1] / height),
        int(1000 * bbox[2] / width),
        int(1000 * bbox[3] / height),
    ]

#Intentamos extraer solo el valor útil, quitando prefijos o palabras como 'Cliente:', 'Email:', etc.
def limpiar_valor(clave, valor):
    valor = valor.strip()

    if clave == "email":
        match = re.search(r"[\w\.-]+@[\w\.-]+", valor)
        return match.group(0) if match else valor

    if clave == "fecha":
        valor_limpio = re.sub(r"(?i)(fecha|hora):\s*", "", valor).strip()
        
        fecha_match = re.search(r"(\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})", valor_limpio)
        if fecha_match:
            return fecha_match.group(0)
        
        return valor_limpio.replace("-", "").strip()
    
    if clave == "hora":
        valor_limpio = re.sub(r"(?i)(fecha|hora|email):\s*", "", valor).strip()
        hora_match = re.search(r"\d{1,2}:\d{2}(?::\d{2})?", valor_limpio)
        if hora_match:
            return hora_match.group(0)
        
        return valor_limpio

    if clave == "n_cliente":
        match = re.search(r"\d{4,}", valor)
        return match.group(0) if match else valor

    if clave == "monto_total":
        match = re.search(r"\$?\s?\d+[\.,]?\d*", valor)
        return match.group(0).replace("$", "").strip() if match else valor

    if clave in ["nombre", "apellido"]:
        valor = re.sub(r"(?i)(cliente|nombre|apellido):\s*", "", valor)
        return valor.strip()

    return valor.strip()

#Esta funcion corrige etiquetas mal asignadas basándose en patrones comunes (@ en email, hora en fecha, etc.)
def corregir_etiquetas(campos, words, labels):
    resultado_corregido = {}

    email_encontrado = None
    for w in words:
        match = re.search(r"[\w\.-]+@[\w\.-]+", w)
        if match:
            email_encontrado = match.group(0)
            break
    
    texto_completo = " ".join(words)

    fecha_encontrada = None
    for w in words:
        match = re.search(r"(\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})", w)
        if match:
            fecha_encontrada = match.group(0)
            break
    if not fecha_encontrada:
        match = re.search(r"(\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})", texto_completo)
        if match:
            fecha_encontrada = match.group(0)
    
    hora_encontrada = None
    for w in words:
        match = re.search(r"\d{1,2}:\d{2}(?::\d{2})?", w)
        if match:
            hora_encontrada = match.group(0)
            break
    if not hora_encontrada:
        match = re.search(r"\d{1,2}:\d{2}(?::\d{2})?", texto_completo)
        if match:
            hora_encontrada = match.group(0)
    
    # Reconstruir campos corrigiendo errores comunes que pueda tener nuestro modelo
    for clave, valor in campos.items():
        # Si la etiqueta "hora" contiene un email, moverlo a "email" por eso la verificacion del @
        if clave == "hora" and "@" in valor:
            resultado_corregido["email"] = limpiar_valor("email", valor)
        elif clave == "email" and "@" not in valor:
            continue
        elif clave == "fecha":
            resultado_corregido["fecha"] = limpiar_valor("fecha", valor)
            if not hora_encontrada:
                match = re.search(r"\d{1,2}:\d{2}(?::\d{2})?", valor)
                if match:
                    hora_encontrada = match.group(0)
        else:
            resultado_corregido[clave] = valor
    

    if email_encontrado and "email" not in resultado_corregido:
        resultado_corregido["email"] = email_encontrado
    

    if fecha_encontrada and "fecha" not in resultado_corregido:
        resultado_corregido["fecha"] = fecha_encontrada

    if hora_encontrada and "hora" not in resultado_corregido:
        resultado_corregido["hora"] = hora_encontrada
    
    return resultado_corregido


# Procesa una imagen y retorna el diccionario de resultados
def procesar_imagen(image_path, ocr, processor, model, device):    
    ocr_result = ocr.ocr(image_path, cls=True)
    
    words, boxes = [], []
    for line in ocr_result[0]:
        txt, bbox = line[1][0], line[0]
        words.append(txt)
        x1, y1 = bbox[0]
        x2, y2 = bbox[2]
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
    
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    
    boxes = boxes[:len(words)]
    boxes = [normalize_bbox(b, width, height) for b in boxes]
    
    encoding = processor(
        image,
        words,
        boxes=boxes,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}
    
    with torch.no_grad():
        outputs = model(**encoding)
        preds = outputs.logits.argmax(-1).squeeze().tolist()
        id2label = model.config.id2label
    
    predicted_labels = [id2label[p] for p in preds[:len(words)]]
    
    campos = defaultdict(list)
    for w, label in zip(words, predicted_labels):
        if label != "O":
            campos[label].append(w)
    
    # Aplicar limpieza
    campos_limpios = {k: limpiar_valor(k, " ".join(v)) for k, v in campos.items()}
    resultado_final = corregir_etiquetas(campos_limpios, words, predicted_labels)
    
    return resultado_final, words, predicted_labels


# Cargamos el modelo de carpeta a utilizar (la 3 en este caso por mejores metricas)
processor = LayoutLMv3Processor.from_pretrained(MODEL_PATH, apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

ocr = PaddleOCR(use_angle_cls=True, lang="es")


extensiones_validas = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

imagenes = [f for f in Path(IMAGES_FOLDER).iterdir() 
            if f.suffix.lower() in extensiones_validas]

for idx, image_path in enumerate(imagenes, 1):
    try:
        resultado, words, labels = procesar_imagen(
            str(image_path), ocr, processor, model, device
        )
        
        nombre_base = image_path.stem
        json_filename = f"analisis_{nombre_base}.json"
        json_path = os.path.join(OUTPUT_FOLDER, json_filename)
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(resultado, f, indent=2, ensure_ascii=False)
        
    except Exception as e:
        print(f"Error procesando {image_path.name}: {str(e)}\n")
        continue
