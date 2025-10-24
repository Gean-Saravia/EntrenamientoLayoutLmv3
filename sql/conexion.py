import pyodbc
from dotenv import load_dotenv
import os

load_dotenv()

def get_connection():
    server = os.getenv("DB_SERVER", "(localdb)\\MSSQLLocalDB")
    database = os.getenv("DB_NAME", "facturas_db")
    trusted = os.getenv("DB_TRUSTED_CONNECTION", "yes")
    driver = os.getenv("DRIVER", "{ODBC Driver 17 for SQL Server}")
    
    try:
        conn = pyodbc.connect(
            f"DRIVER={driver};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"Trusted_Connection={trusted};"
        )
        return conn
    except Exception as e:
        print(f"Error de conexion: {e}")
        return None