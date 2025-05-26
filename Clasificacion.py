import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import re

try:
    # Descargar recursos de nltk (solo necesario la primera vez)
    print("Descargando recursos de NLTK...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    # Cargar el archivo CSV
    print("Cargando archivo CSV...")
    data = pd.read_csv('documentos_balanceado_v2.csv')
    print(f"Archivo cargado. Número de filas: {len(data)}")

    # Definir stopwords en español y el lematizador
    stop_words = set(stopwords.words('spanish'))
    lemmatizer = WordNetLemmatizer()

    # Función de preprocesamiento de texto
    def preprocess(text):
        if not isinstance(text, str):
            return ""
        # Convertir a minúsculas
        text = text.lower()
        # Eliminar caracteres especiales y números
        text = re.sub(r'[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ\s]', '', text)
        # Dividir en palabras
        words = text.split()
        # Filtrar stopwords y lematizar
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return ' '.join(words)

    # Aplicar preprocesamiento a la columna "texto"
    print("Preprocesando textos...")
    data['texto_procesado'] = data['texto'].apply(preprocess)

    # Convertir textos a vectores TF-IDF
    print("Vectorizando textos...")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['texto_procesado'])
    y = data['categoria']

    # Dividir los datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
    print("Dividiendo datos en conjuntos de entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el modelo Naive Bayes
    print("Entrenando modelo...")
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Hacer predicciones en el conjunto de prueba
    print("Realizando predicciones...")
    y_pred = model.predict(X_test)

    # Imprimir métricas de evaluación
    print("\nResultados del modelo:")
    print('Precisión:', accuracy_score(y_test, y_pred))
    print('\nMatriz de confusión:')
    print(confusion_matrix(y_test, y_pred))
    print('\nReporte de clasificación:')
    print(classification_report(y_test, y_pred))

except Exception as e:
    print(f"Error durante la ejecución: {str(e)}")
    import traceback
    print(traceback.format_exc())