from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "sql"))

from conexion import get_connection

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#Interpreta el mensaje del usuario y decide si necesita consultar la BD o solo conversar.
#Instruccion central del sistema
def interpretar_y_ejecutar(prompt_usuario: str):

    try:
        # Paso 1: Decidir si necesita consultar la base de datos
        response_decision = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """
                            Eres un asistente que ayuda con consultas sobre una base de datos de facturas.

                            Decide si el mensaje del usuario requiere consultar la base de datos o si es una conversacion casual.

                            La base de datos contiene:
                            - Tabla: datos_imagenes
                            - Columnas: id, nombre_imagen, n_cliente, nombre, apellido, email, fecha_factura, monto_total, fecha_subida

                            Responde SOLO con "SQL" si necesitas consultar la base de datos.
                            Responde SOLO con "CHAT" si es conversacion casual (saludos, agradecimientos, preguntas generales).

                            Ejemplos:
                            - "hola" -> CHAT
                            - "gracias" -> CHAT
                            - "como estas?" -> CHAT
                            - "cuantos clientes hay?" -> SQL
                            - "dame el nombre del cliente" -> SQL
                            - "muestrame todas las facturas" -> SQL
                            """
                },
                {"role": "user", "content": prompt_usuario}
            ]
        )
        
        decision = response_decision.choices[0].message.content.strip().upper()
        print(f"[DECISION] {decision}")
        
        # Paso 2: Actuar según la decisión
        if decision == "SQL":
            return ejecutar_consulta_sql(prompt_usuario)
        else:
            return responder_chat(prompt_usuario)
            
    except Exception as e:
        return f"ERROR: {e}"

#Instruccion interactiva para responder sin SQL
def responder_chat(prompt_usuario: str):
    """
    Responde de forma conversacional sin consultar la base de datos.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """
                        Eres un asistente amigable de un sistema de facturas. 

                        Responde de forma natural, breve y amigable.
                        Si te preguntan que puedes hacer, menciona que puedes ayudar a consultar datos de facturas.

                        Ejemplos:
                        - Si dicen "hola" -> "Hola! Como puedo ayudarte?"
                        - Si dicen "gracias" -> "De nada! Si necesitas algo mas, aqui estoy"
                        """
                },
                {"role": "user", "content": prompt_usuario}
            ]
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"ERROR: {e}"

#Instruccion para generar y ejecutar SQL y devolver una respuesta acorde a lo que pregunte el usuario(sobre la tabla de facturas)
def ejecutar_consulta_sql(prompt_usuario: str):
    """
    Genera y ejecuta una consulta SQL basada en el mensaje del usuario.
    """
    
    conn = get_connection()
    if not conn:
        return "ERROR: No se pudo conectar a la base de datos"
    
    cursor = conn.cursor()

    try:
        # Generar consulta SQL
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """
                            Eres un generador de consultas SQL para Microsoft SQL Server.

                            Usa SOLO la tabla 'datos_imagenes' en 'dbo.datos_imagenes'.
                            Columnas: id, nombre_imagen, n_cliente, nombre, apellido, email, fecha_factura, monto_total, fecha_subida

                            Reglas:
                            - Usa TOP en lugar de LIMIT
                            - NO uses LIMIT
                            - Selecciona solo columnas necesarias
                            - Responde SOLO con SQL valido, sin markdown ni comentarios
                            """
                },
                {"role": "user", "content": prompt_usuario}
            ]
        )

        sql_query = response.choices[0].message.content.strip()
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        
        # Convertir LIMIT a TOP para que sea compatible con SQL Server
        if "LIMIT" in sql_query.upper():
            import re
            match = re.search(r'LIMIT\s+(\d+)', sql_query, re.IGNORECASE)
            if match:
                limit_num = match.group(1)
                sql_query = re.sub(r'SELECT\s+', f'SELECT TOP {limit_num} ', sql_query, count=1, flags=re.IGNORECASE)
                sql_query = re.sub(r'\s*LIMIT\s+\d+', '', sql_query, flags=re.IGNORECASE)
        
        print(f"\n[SQL] {sql_query}\n")

        # Parametros de seguridad: prevenir consultas peligrosas
        palabras_prohibidas = ["delete", "drop", "update", "insert", "alter"]
        if any(x in sql_query.lower() for x in palabras_prohibidas):
            return "AVISO: Consulta peligrosa detectada"

        cursor.execute(sql_query)
        rows = cursor.fetchall()

        if not rows:
            return "No se encontraron resultados."

        columnas = [desc[0] for desc in cursor.description]
        resultados = [dict(zip(columnas, row)) for row in rows]

        response2 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "Responde de forma clara, concisa y amigable en español. Resume los datos de forma legible."
                },
                {
                    "role": "user", 
                    "content": f"El usuario pregunto: '{prompt_usuario}'\n\nResultado de la consulta: {resultados}\n\nResponde de forma natural y directa."
                }
            ]
        )

        return response2.choices[0].message.content

    except Exception as e:
        return f"ERROR al ejecutar SQL: {e}"
    
    finally:
        conn.close()