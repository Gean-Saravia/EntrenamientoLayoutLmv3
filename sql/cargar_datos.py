import json
import os
from pathlib import Path
from datetime import datetime
import pyodbc
from conexion import get_connection

#Combinafecha y hora debido a que la columna es datetime
def combinar_fecha_hora(fecha_str, hora_str):
    if not fecha_str:
        return None
    
    formatos_fecha = ['%Y-%m-%d', '%d/%m/%Y', '%Y/%m/%d']
    
    fecha_obj = None
    for formato in formatos_fecha:
        try:
            fecha_obj = datetime.strptime(fecha_str, formato)
            break
        except ValueError:
            continue
    
    if not fecha_obj:
        return None
    
    if hora_str:
        try:
            if len(hora_str.split(':')) == 2:
                hora_obj = datetime.strptime(hora_str, '%H:%M')
            else:
                hora_obj = datetime.strptime(hora_str, '%H:%M:%S')
            
            fecha_obj = fecha_obj.replace(
                hour=hora_obj.hour,
                minute=hora_obj.minute,
                second=hora_obj.second
            )
        except ValueError:
            pass
    
    return fecha_obj


#Lee un archivo JSON y lo inserta en la base de datos
def insertar_json_a_db(json_path, conn):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            datos = json.load(f)
        
        nombre_imagen = Path(json_path).stem.replace('analisis_', '')
        
        fecha_hora = combinar_fecha_hora(
            datos.get('fecha'),
            datos.get('hora')
        )
        
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO datos_imagenes 
        (nombre_imagen, n_cliente, nombre, apellido, email, fecha_factura, monto_total)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        valores = (
            nombre_imagen,
            datos.get('n_cliente'),
            datos.get('nombre'),
            datos.get('apellido'),
            datos.get('email'),
            fecha_hora,
            datos.get('monto_total')
        )
        
        cursor.execute(insert_query, valores)
        conn.commit()
        
        return True, nombre_imagen
        
    except pyodbc.IntegrityError as e:
        if "UNIQUE" in str(e):
            return False, "Duplicado"
        return False, str(e)
    except Exception as e:
        return False, str(e)

def cargar_todos_los_json(json_folder):

    conn = get_connection()
    if not conn:
        print("No se pudo conectar a la base de datos")
        return
    
    json_files = list(Path(json_folder).glob('*.json'))
    
    if not json_files:
        print("No se encontraron archivos JSON")
        conn.close()
        return
    
    exitosos = 0
    fallidos = 0
    
    for json_file in json_files:
        success, msg = insertar_json_a_db(str(json_file), conn)
        
        if success:
            exitosos += 1
        else:
            fallidos += 1
    
    conn.close()
    
    print(f"OK: Insertados: {exitosos} | Fallidos: {fallidos}")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    JSON_FOLDER = BASE_DIR / "resultados_json"
    
    cargar_todos_los_json(JSON_FOLDER)