import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
EVALUAR_SCRIPT = BASE_DIR / "src" / "evaluar_modelo.py"
CARGAR_SCRIPT = BASE_DIR / "sql" / "cargar_datos.py"

def ejecutar_script(ruta_script, mostrar_salida=False):
    try:
        result = subprocess.run(
            [sys.executable, str(ruta_script)], 
            check=True,
            capture_output=True,
            text=True
        )
        if mostrar_salida and result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR en {ruta_script.name}:")
        if e.stderr:
            print(e.stderr)
        return False

if __name__ == "__main__":

    # Paso 1
    print("[1/3] Procesando facturas con LayoutLMv3...")
    if not ejecutar_script(EVALUAR_SCRIPT):
        sys.exit(1)

    # Paso 2
    print("[2/3] Cargando datos a SQL Server...")
    if not ejecutar_script(CARGAR_SCRIPT, mostrar_salida=True):
        sys.exit(1)

    #Inicio del html
    print("[3/3] Iniciando interfaz web...")
    subprocess.run([sys.executable, str(BASE_DIR / "src" / "app_web.py")])