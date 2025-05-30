import pandas as pd
import nltk
from nltk.corpus import stopwords
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import re
import numpy as np
import logging
import joblib
import matplotlib.pyplot as plt

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Parámetros para GridSearchCV
PARAM_GRID = {
    'alpha': [0.1, 0.5, 1.0, 2.0],  # Parámetro de suavizado para Naive Bayes
    'fit_prior': [True, False]      # Si se debe aprender las probabilidades a priori
}

try:
    # Cargar modelo de spaCy
    logger.info("Cargando modelo de spaCy...")
    try:
        nlp = spacy.load("es_core_news_sm")
    except OSError:
        logger.info("Modelo no encontrado, descargando...")
        spacy.cli.download("es_core_news_sm")
        nlp = spacy.load("es_core_news_sm")

    # Función de preprocesamiento de texto usando spaCy
    def preprocess(text):
        if not isinstance(text, str):
            return ""
        
        # Procesar el texto con spaCy
        doc = nlp(text.lower())
        
        # Filtrar y lematizar tokens
        tokens = [
            token.lemma_ for token in doc 
            if not token.is_stop  # Eliminar stopwords
            and token.is_alpha    # Solo palabras
            and len(token) >= 3   # Longitud mínima
            and not token.is_punct # No puntuación
        ]
        
        return ' '.join(tokens)

    # Cargar el archivo CSV
    logger.info("Cargando archivo CSV...")
    data = pd.read_csv('documentos_balanceado_v2.csv')
    logger.info(f"Archivo cargado. Número de filas: {len(data)}")

    # Aplicar preprocesamiento a la columna "texto"
    logger.info("Preprocesando textos...")
    data['texto_procesado'] = data['texto'].apply(preprocess)

    # Convertir textos a vectores TF-IDF
    logger.info("Vectorizando textos...")
    vectorizer = TfidfVectorizer(**VECTORIZER_PARAMS)
    X = vectorizer.fit_transform(data['texto_procesado'])
    y = data['categoria']

    # Dividir los datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
    logger.info("Dividiendo datos en conjuntos de entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Validación cruzada
    logger.info("\nRealizando validación cruzada...")
    cv_scores = cross_val_score(MultinomialNB(), X, y, cv=5)
    logger.info(f"Puntuaciones de validación cruzada: {cv_scores}")
    logger.info(f"Precisión media: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Búsqueda de hiperparámetros óptimos
    logger.info("\nBuscando hiperparámetros óptimos...")
    grid_search = GridSearchCV(MultinomialNB(), PARAM_GRID, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Mejores parámetros: {grid_search.best_params_}")
    logger.info(f"Mejor puntuación: {grid_search.best_score_:.4f}")

    # Entrenar el modelo con los mejores parámetros
    logger.info("\nEntrenando modelo con parámetros óptimos...")
    model = MultinomialNB(**grid_search.best_params_)
    model.fit(X_train, y_train)

    # Curva de aprendizaje
    logger.info("\nGenerando curva de aprendizaje...")
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Guardar modelo y vectorizador
    logger.info("Guardando modelo y vectorizador...")
    joblib.dump(model, 'modelo_nb.joblib', compress=3)
    joblib.dump(vectorizer, 'vectorizador_tfidf.joblib', compress=3)
    logger.info("Modelo y vectorizador guardados correctamente")

    # Hacer predicciones en el conjunto de prueba
    logger.info("Realizando predicciones...")
    y_pred = model.predict(X_test)
    probabilidades = model.predict_proba(X_test)
    confianza = np.max(probabilidades, axis=1)

    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    # Calcular estadísticas de confianza
    confianza_promedio = np.mean(confianza)
    confianza_min = np.min(confianza)
    confianza_max = np.max(confianza)
    textos_baja_confianza = np.sum(confianza < 0.6)

    # Imprimir métricas de evaluación
    logger.info("\nResultados del modelo:")
    logger.info(f'Precisión: {accuracy:.4f}')
    logger.info(f'\nConfianza promedio: {confianza_promedio:.4f}')
    logger.info(f'Confianza mínima: {confianza_min:.4f}')
    logger.info(f'Confianza máxima: {confianza_max:.4f}')
    logger.info(f'Textos con baja confianza (< 0.6): {textos_baja_confianza}')
    
    logger.info('\nMatriz de confusión:')
    logger.info(conf_matrix)
    
    logger.info('\nReporte de clasificación:')
    logger.info(class_report)

    # Graficar curva de aprendizaje
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Precisión de entrenamiento')
    plt.plot(train_sizes, test_mean, label='Precisión de validación')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
    plt.xlabel('Tamaño del conjunto de entrenamiento')
    plt.ylabel('Precisión')
    plt.title('Curva de Aprendizaje')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('curva_aprendizaje.png')
    logger.info("\nCurva de aprendizaje guardada en 'curva_aprendizaje.png'")

except Exception as e:
    logger.error(f"Error durante la ejecución: {str(e)}")
    import traceback
    logger.error(traceback.format_exc())