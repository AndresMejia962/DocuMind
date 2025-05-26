from flask import Flask, render_template, request, jsonify
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import logging
from typing import Tuple, Dict, Any

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constantes
ARCHIVO_DATOS = 'documentos_balanceado_v2.csv'
CATEGORIAS = ['Legal', 'Educativo']

app = Flask(__name__)

class ClasificadorTexto:
    def __init__(self):
        self.modelo = None
        self.vectorizer = None
        self.stop_words = None
        self.lemmatizer = None
        self._inicializar_recursos()
        self._entrenar_modelo()

    def _inicializar_recursos(self) -> None:
        """Inicializa los recursos necesarios para el procesamiento de texto."""
        try:
            # Descargar recursos de NLTK si no existen
            for recurso in ['punkt', 'stopwords', 'wordnet']:
                nltk.download(recurso, quiet=True)
            
            self.stop_words = set(stopwords.words('spanish'))
            self.lemmatizer = WordNetLemmatizer()
            logger.info("Recursos NLTK inicializados correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar recursos NLTK: {str(e)}")
            raise

    def _preprocesar_texto(self, texto: str) -> str:
        """Preprocesa el texto para su clasificación."""
        if not isinstance(texto, str):
            return ""
        
        try:
            # Convertir a minúsculas y eliminar caracteres especiales
            texto = texto.lower()
            texto = re.sub(r'[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ\s]', '', texto)
            
            # Tokenización y lematización
            palabras = texto.split()
            palabras_procesadas = [
                self.lemmatizer.lemmatize(palabra)
                for palabra in palabras
                if palabra not in self.stop_words
            ]
            
            return ' '.join(palabras_procesadas)
        except Exception as e:
            logger.error(f"Error en preprocesamiento de texto: {str(e)}")
            return ""

    def _entrenar_modelo(self) -> None:
        """Entrena el modelo de clasificación."""
        try:
            # Cargar y preparar datos
            data = pd.read_csv(ARCHIVO_DATOS)
            data['texto_procesado'] = data['texto'].apply(self._preprocesar_texto)
            
            # Vectorización
            self.vectorizer = TfidfVectorizer()
            X = self.vectorizer.fit_transform(data['texto_procesado'])
            y = data['categoria']
            
            # Entrenamiento
            self.modelo = MultinomialNB()
            self.modelo.fit(X, y)
            logger.info("Modelo entrenado correctamente")
        except Exception as e:
            logger.error(f"Error en entrenamiento del modelo: {str(e)}")
            raise

    def clasificar(self, texto: str) -> Dict[str, Any]:
        """Clasifica un texto y retorna el resultado."""
        try:
            texto_procesado = self._preprocesar_texto(texto)
            if not texto_procesado:
                return {'error': 'El texto no pudo ser procesado'}

            X = self.vectorizer.transform([texto_procesado])
            prediccion = self.modelo.predict(X)[0]
            probabilidades = self.modelo.predict_proba(X)[0]
            
            return {
                'categoria': prediccion,
                'confianza': float(max(probabilidades)),
                'error': None
            }
        except Exception as e:
            logger.error(f"Error en clasificación: {str(e)}")
            return {'error': 'Error al clasificar el texto'}

# Inicializar el clasificador
clasificador = ClasificadorTexto()

@app.route('/')
def home():
    """Renderiza la página principal."""
    return render_template('index.html')

@app.route('/clasificar', methods=['POST'])
def clasificar():
    """Endpoint para clasificar texto."""
    try:
        if not request.is_json:
            return jsonify({'error': 'Se requiere formato JSON'}), 400

        texto = request.json.get('texto')
        if not texto:
            return jsonify({'error': 'No se proporcionó texto'}), 400

        resultado = clasificador.clasificar(texto)
        if resultado.get('error'):
            return jsonify(resultado), 400

        return jsonify(resultado)
    except Exception as e:
        logger.error(f"Error en endpoint /clasificar: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

if __name__ == '__main__':
    app.run(debug=True) 