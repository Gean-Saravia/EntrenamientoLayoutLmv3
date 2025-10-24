from flask import Flask, render_template, request, jsonify
from chat_sql import interpretar_y_ejecutar
import webbrowser
import threading

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/consulta', methods=['POST'])
def consulta():
    try:
        data = request.json
        pregunta = data.get('pregunta', '')
        
        if not pregunta:
            return jsonify({'error': 'Pregunta vacía'}), 400
        
        respuesta = interpretar_y_ejecutar(pregunta)
        return jsonify({'respuesta': respuesta})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def abrir_navegador():
    """Abre el navegador después de 1 segundo"""
    import time
    time.sleep(1)
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    # Abrir navegador automáticamente
    threading.Thread(target=abrir_navegador, daemon=True).start()
    
    print("\n" + "="*60)
    print("SERVIDOR WEB INICIADO")
    print("Abriendo navegador en http://127.0.0.1:5000")
    print("Presiona Ctrl+C para detener")
    print("="*60 + "\n")
    
    app.run(debug=False, port=5000)