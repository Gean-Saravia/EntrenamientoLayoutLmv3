import os
import json
import jsonlines
from tqdm import tqdm
from paddleocr import PaddleOCR
from difflib import SequenceMatcher
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / os.getenv("DATA_DIR", "data")
ANNOTATIONS_PATH = BASE_DIR / os.getenv("ANNOTATIONS_PATH", "annotations")

os.makedirs(ANNOTATIONS_PATH, exist_ok=True)

# Desactivar logs de PaddleOCR
os.environ["FLAGS_logtostderr"] = "0"
os.environ["GLOG_minloglevel"] = "3"

# Inicializar OCR con nvidia si está disponible para que no tarde
try:
    ocr = PaddleOCR(
        use_angle_cls=False, 
        lang='en', 
        show_log=False,
        use_gpu=True,
        gpu_mem=2000,
        det_db_box_thresh=0.3,
        det_db_unclip_ratio=1.5
    )
except Exception as e:
    ocr = PaddleOCR(
        use_angle_cls=False, 
        lang='en', 
        show_log=False,
        use_gpu=False
    )

def similar(a, b):
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

def normalizar_texto(txt):
    txt = re.sub(r'[^\w\s]', '', txt.lower())
    return txt.strip()

def extraer_valor_campo(text):
    text_lower = text.lower()
    patterns = [
        r'(?:cliente|nombre|apellido|email|fecha|hora|factura|total|monto)[:\s]+(.+)',
        r'n[o°]?\s*(?:cliente|factura)[:\s]+(.+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1).strip()
    return text


##Para darle una etiqueta a cada palabra detectada y que tenga nocion de la misma para reconocerla, algo "guiado"
def asignar_etiqueta_palabra(word, labels_json):
    word_norm = normalizar_texto(word)
    
    mejor_etiqueta = "O"
    mejor_score = 0.0
    
    for label, ref_text in labels_json.items():
        ref_norm = normalizar_texto(ref_text)
        
        score1 = similar(word_norm, ref_norm)
        valor_extraido = normalizar_texto(extraer_valor_campo(word))
        score2 = similar(valor_extraido, ref_norm)
        score3 = 0.6 if (word_norm in ref_norm or ref_norm in word_norm) else 0
        
        keywords = {
            'n_cliente': ['cliente', 'ncliente', 'cllente'],
            'n_factura': ['factura', 'nfactura', 'invoice'],
            'fecha': ['fecha', 'date'],
            'hora': ['hora', 'time'],
            'email': ['email', 'mail', '@'],
            'nombre': ['nombre', 'name'],
            'apellido': ['apellido', 'surname'],
            'monto_total': ['total', 'monto', 'amount'],
        }
        
        score4 = 0
        if label in keywords:
            for keyword in keywords[label]:
                if keyword in word_norm:
                    score4 = 0.5
                    break
        
        max_score = max(score1, score2, score3, score4)
        
        if max_score > mejor_score:
            mejor_score = max_score
            mejor_etiqueta = label
    
    return mejor_etiqueta if mejor_score >= 0.4 else "O"

def procesar_imagen(img_path, json_path):
    try:
        ocr_result = ocr.ocr(img_path, cls=False)
        
        if not ocr_result or not isinstance(ocr_result[0], list):
            return None

        words, bboxes = [], []
        for block in ocr_result:
            for line in block:
                if not line or len(line) < 2:
                    continue
                box = line[0]
                text, _ = line[1]
                if not text.strip():
                    continue

                x1 = min(p[0] for p in box)
                y1 = min(p[1] for p in box)
                x2 = max(p[0] for p in box)
                y2 = max(p[1] for p in box)
                words.append(text)
                bboxes.append([x1, y1, x2, y2])

        if not words:
            return None

        labels_json = {}
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for field in data.get("fields", []):
                labels_json[field["label"]] = normalizar_texto(field["text"])

        labels = [asignar_etiqueta_palabra(w, labels_json) for w in words]
        #Al final nos  retorna en formato estructurado
        return {
            "id": os.path.basename(img_path),
            "words": words,
            "bboxes": bboxes,
            "labels": labels
        }
    except Exception as e:
        print(f"Error procesando {os.path.basename(img_path)}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

def procesar_directorio(split):
    # Procesa imágenes y genera archivo JSONL
    salida_path = ANNOTATIONS_PATH / f"{split}.jsonl"

    if salida_path.exists():
        os.remove(salida_path)

    formatos = [f for f in DATA_PATH.iterdir() 
                if f.is_dir() and f.name.startswith("formato_") and f.name != "formato_5"]

    # Recolectar todas las tareas
    tareas = []
    for formato in formatos:
        formato_path = formato / split
        if not formato_path.exists():
            continue

        archivos = [f for f in formato_path.iterdir() if f.suffix == ".png"]
        for archivo in archivos:
            img_path = archivo
            json_path = img_path.with_suffix(".json")
            if json_path.exists():
                tareas.append((str(img_path), str(json_path)))
    
    if not tareas:
        return

    # Procesar y acumular en memoria
    resultados = []
    for img_path, json_path in tqdm(tareas, desc=f"OCR {split}"):
        resultado = procesar_imagen(img_path, json_path)
        if resultado:
            resultados.append(resultado)
    
    # Escribir todos al final para evitar escribir linea por linea
    with jsonlines.open(str(salida_path), mode="w") as writer:
        for res in resultados:
            writer.write(res)

if __name__ == "__main__":
    procesar_directorio("train")
    procesar_directorio("test") 