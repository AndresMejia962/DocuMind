from flask import Flask, render_template, request, jsonify
import pandas as pd
import nltk
from nltk.corpus import stopwords
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import logging
from typing import Tuple, Dict, Any
import joblib
import os
import numpy as np

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constantes
ARCHIVO_DATOS = 'documentos_balanceado_v2.csv'
ARCHIVO_MODELO = 'modelo_nb.joblib'
ARCHIVO_VECTORIZER = 'vectorizador_tfidf.joblib'
CATEGORIAS = ['Legal', 'Educativo']

# Límites de texto para prevenir DoS
MIN_LONGITUD_TEXTO = 10  # Caracteres mínimos
MAX_LONGITUD_TEXTO = 10000  # Caracteres máximos
MAX_PALABRAS = 2000  # Palabras máximas
MAX_TOKENS = 5000  # Tokens máximos después del preprocesamiento

# Parámetros del vectorizador
VECTORIZER_PARAMS = {
    'ngram_range': (1, 2),  # Unigramas y bigramas
    'max_df': 0.85,         # Ignorar términos que aparecen en más del 85% de los documentos
    'min_df': 2,            # Ignorar términos que aparecen en menos de 2 documentos
    'max_features': 10000,  # Máximo número de características
    'sublinear_tf': True,   # Aplicar escala sublineal a la frecuencia de términos
    'use_idf': True,        # Usar IDF (Inverse Document Frequency)
    'smooth_idf': True,     # Suavizar IDF
    'norm': 'l2'            # Normalización L2
}

app = Flask(__name__)

class ClasificadorTexto:
    def __init__(self):
        self.modelo = None
        self.vectorizer = None
        self.nlp = None
        self._inicializar_recursos()
        self._cargar_o_entrenar_modelo()

    def _inicializar_recursos(self) -> None:
        """Inicializa los recursos necesarios para el procesamiento de texto."""
        try:
            # Cargar modelo de spaCy
            logger.info("Cargando modelo de spaCy...")
            try:
                self.nlp = spacy.load("es_core_news_sm")
            except OSError:
                logger.info("Modelo no encontrado, descargando...")
                spacy.cli.download("es_core_news_sm")
                self.nlp = spacy.load("es_core_news_sm")
            
            logger.info("Recursos inicializados correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar recursos: {str(e)}")
            raise

    def _validar_longitud_texto(self, texto: str) -> Tuple[bool, str]:
        """Valida la longitud del texto para prevenir DoS."""
        if len(texto) < MIN_LONGITUD_TEXTO:
            return False, f'El texto es demasiado corto (mínimo {MIN_LONGITUD_TEXTO} caracteres)'
        
        if len(texto) > MAX_LONGITUD_TEXTO:
            return False, f'El texto es demasiado largo (máximo {MAX_LONGITUD_TEXTO} caracteres)'
        
        palabras = texto.split()
        if len(palabras) > MAX_PALABRAS:
            return False, f'El texto contiene demasiadas palabras (máximo {MAX_PALABRAS} palabras)'
        
        return True, ''

    def _preprocesar_texto(self, texto: str) -> str:
        """Preprocesa el texto para su clasificación con técnicas avanzadas de NLP."""
        if not isinstance(texto, str):
            return ""
        
        try:
            # Validar longitud del texto
            es_valido, mensaje = self._validar_longitud_texto(texto)
            if not es_valido:
                logger.warning(f"Texto rechazado por longitud: {mensaje}")
                return ""
            
            # Procesar el texto con spaCy
            doc = self.nlp(texto.lower())
            
            # Filtrar y lematizar tokens
            tokens = [
                token.lemma_ for token in doc 
                if not token.is_stop  # Eliminar stopwords
                and token.is_alpha    # Solo palabras
                and len(token) >= 3   # Longitud mínima
                and not token.is_punct # No puntuación
            ]
            
            # Verificar límite de tokens
            if len(tokens) > MAX_TOKENS:
                logger.warning(f"Texto truncado: excede el límite de {MAX_TOKENS} tokens")
                tokens = tokens[:MAX_TOKENS]
            
            return ' '.join(tokens)
        except Exception as e:
            logger.error(f"Error en preprocesamiento de texto: {str(e)}")
            return ""

    def _cargar_o_entrenar_modelo(self) -> None:
        """Carga el modelo guardado o entrena uno nuevo si no existe."""
        try:
            if os.path.exists(ARCHIVO_MODELO) and os.path.exists(ARCHIVO_VECTORIZER):
                logger.info("Cargando modelo existente...")
                self.modelo = joblib.load(ARCHIVO_MODELO)
                self.vectorizer = joblib.load(ARCHIVO_VECTORIZER)
                logger.info("Modelo cargado correctamente")
            else:
                logger.info("No se encontró modelo guardado, entrenando nuevo modelo...")
                self._entrenar_modelo()
        except Exception as e:
            logger.error(f"Error al cargar modelo: {str(e)}")
            logger.info("Entrenando nuevo modelo...")
            self._entrenar_modelo()

    def _guardar_modelo(self) -> None:
        """Guarda el modelo y el vectorizer en archivos."""
        try:
            # Guardar modelo y vectorizador con compresión
            joblib.dump(self.modelo, ARCHIVO_MODELO, compress=3)
            joblib.dump(self.vectorizer, ARCHIVO_VECTORIZER, compress=3)
            
            # Verificar que los archivos se guardaron correctamente
            if os.path.exists(ARCHIVO_MODELO) and os.path.exists(ARCHIVO_VECTORIZER):
                logger.info(f"Modelo guardado en {ARCHIVO_MODELO}")
                logger.info(f"Vectorizador guardado en {ARCHIVO_VECTORIZER}")
            else:
                raise Exception("Error al verificar archivos guardados")
                
        except Exception as e:
            logger.error(f"Error al guardar modelo: {str(e)}")
            raise

    def _entrenar_modelo(self) -> None:
        """Entrena el modelo de clasificación inicial."""
        try:
            # Cargar y preparar datos
            data = pd.read_csv(ARCHIVO_DATOS)
            data['texto_procesado'] = data['texto'].apply(self._preprocesar_texto)
            
            # Vectorización inicial con parámetros optimizados
            self.vectorizer = TfidfVectorizer(**VECTORIZER_PARAMS)
            X = self.vectorizer.fit_transform(data['texto_procesado'])
            y = data['categoria']
            
            # Entrenamiento inicial
            self.modelo = MultinomialNB()
            self.modelo.fit(X, y)
            
            # Guardar el modelo entrenado
            self._guardar_modelo()
            
            # Logging de información del vectorizador
            logger.info(f"Vectorizador configurado con {len(self.vectorizer.get_feature_names_out())} características")
            logger.info(f"Rango de n-gramas: {VECTORIZER_PARAMS['ngram_range']}")
            logger.info(f"Max DF: {VECTORIZER_PARAMS['max_df']}, Min DF: {VECTORIZER_PARAMS['min_df']}")
            
            logger.info("Modelo inicial entrenado y guardado correctamente")
        except Exception as e:
            logger.error(f"Error en entrenamiento del modelo: {str(e)}")
            raise

    def entrenar_con_nuevo_texto(self, texto: str, categoria: str, forzar: bool = False) -> Dict[str, Any]:
        """Entrena el modelo con un nuevo texto usando partial_fit.
        
        Args:
            texto: El texto a entrenar
            categoria: La categoría del texto
            forzar: Si es True, fuerza el entrenamiento incluso con baja confianza
        """
        try:
            # Validar longitud del texto
            es_valido, mensaje = self._validar_longitud_texto(texto)
            if not es_valido:
                return {'error': mensaje}

            # Preprocesar el nuevo texto
            texto_procesado = self._preprocesar_texto(texto)
            if not texto_procesado:
                return {'error': 'El texto no pudo ser procesado'}

            # Validar el texto antes de entrenar
            clasificacion = self.clasificar(texto)
            if clasificacion.get('error'):
                return {
                    'error': f'El texto no cumple con los requisitos mínimos: {clasificacion["error"]}',
                    'advertencia': 'Se recomienda proporcionar un texto más descriptivo'
                }

            # Verificar la confianza de la clasificación actual
            confianza_actual = clasificacion.get('confianza', 0)
            categoria_actual = clasificacion.get('categoria')
            
            if confianza_actual < 0.5:
                if not forzar:
                    return {
                        'error': 'Confianza muy baja. No se entrenará el modelo con este texto.',
                        'confianza': confianza_actual,
                        'categoria_actual': categoria_actual,
                        'mensaje': 'Si estás seguro de la categoría, puedes forzar el entrenamiento con forzar=True'
                    }
                else:
                    logger.warning(f"Entrenamiento forzado con confianza baja ({confianza_actual:.2f})")

            # Transformar el nuevo texto usando el vectorizer existente
            X_nuevo = self.vectorizer.transform([texto_procesado])
            y_nuevo = np.array([categoria])

            # Si el modelo no tiene clases, inicializarlo
            if not hasattr(self.modelo, 'classes_'):
                self.modelo.partial_fit(X_nuevo, y_nuevo, classes=np.array(CATEGORIAS))
            else:
                # Actualizar el modelo con el nuevo texto
                self.modelo.partial_fit(X_nuevo, y_nuevo)

            # Guardar el modelo actualizado
            self._guardar_modelo()

            # Actualizar el archivo CSV
            data = pd.read_csv(ARCHIVO_DATOS)
            nuevo_dato = pd.DataFrame({
                'texto': [texto],
                'categoria': [categoria]
            })
            data = pd.concat([data, nuevo_dato], ignore_index=True)
            data.to_csv(ARCHIVO_DATOS, index=False)
            
            # Verificar la confianza después del entrenamiento
            clasificacion_final = self.clasificar(texto)
            confianza_final = clasificacion_final.get('confianza', 0)
            
            return {
                'mensaje': 'Modelo actualizado correctamente con partial_fit',
                'error': None,
                'confianza_inicial': confianza_actual,
                'confianza_final': confianza_final,
                'mejora_confianza': confianza_final - confianza_actual,
                'entrenamiento_forzado': forzar and confianza_actual < 0.5
            }
        except Exception as e:
            logger.error(f"Error en entrenamiento con nuevo texto: {str(e)}")
            return {'error': 'Error al actualizar el modelo'}

    def clasificar(self, texto: str) -> Dict[str, Any]:
        """Clasifica un texto y retorna el resultado."""
        try:
            # Validar longitud del texto
            es_valido, mensaje = self._validar_longitud_texto(texto)
            if not es_valido:
                return {'error': mensaje}

            # Preprocesar el texto
            texto_procesado = self._preprocesar_texto(texto)
            if not texto_procesado:
                return {'error': 'El texto no pudo ser procesado'}

            # Validar texto después del preprocesamiento
            palabras_procesadas = texto_procesado.split()
            if len(palabras_procesadas) < 2:
                return {'error': 'El texto es demasiado corto después del preprocesamiento'}

            X = self.vectorizer.transform([texto_procesado])
            prediccion = self.modelo.predict(X)[0]
            probabilidades = self.modelo.predict_proba(X)[0]
            
            # Validar nivel de confianza
            confianza = float(max(probabilidades))
            if confianza < 0.6:
                return {
                    'error': 'El nivel de confianza es muy bajo para hacer una clasificación precisa',
                    'categoria': prediccion,
                    'confianza': confianza
                }
            
            return {
                'categoria': prediccion,
                'confianza': confianza,
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

@app.route('/entrenar', methods=['POST'])
def entrenar():
    """Endpoint para entrenar el modelo con nuevo texto."""
    try:
        if not request.is_json:
            return jsonify({'error': 'Se requiere formato JSON'}), 400

        texto = request.json.get('texto')
        categoria = request.json.get('categoria')
        forzar = request.json.get('forzar', False)

        if not texto or not categoria:
            return jsonify({'error': 'Se requiere texto y categoría'}), 400

        if categoria not in CATEGORIAS:
            return jsonify({'error': f'Categoría debe ser una de: {", ".join(CATEGORIAS)}'}), 400

        resultado = clasificador.entrenar_con_nuevo_texto(texto, categoria, forzar)
        if resultado.get('error'):
            return jsonify(resultado), 400

        return jsonify(resultado)
    except Exception as e:
        logger.error(f"Error en endpoint /entrenar: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

if __name__ == '__main__':
    app.run(debug=True) 