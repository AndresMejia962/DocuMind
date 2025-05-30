# DocuMind

## Descripción
DocuMind es un proyecto que implementa un clasificador de textos utilizando técnicas de Procesamiento de Lenguaje Natural (NLP) y Machine Learning. El objetivo es determinar si un documento pertenece a la categoría Legal o Educativa.

## Características
- Clasificación de textos en dos categorías: Legal y Educativo.
- Interfaz web amigable.
- Preprocesamiento de texto en español.
- Visualización del nivel de confianza de la clasificación.
- Manejo de errores robusto.

## Requisitos
- Python 3.8 o superior.
- Flask.
- scikit-learn.
- pandas.
- nltk.
- spacy.
- joblib.
- matplotlib.

## Instalación
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/AndresMejia962/DocuMind.git
   cd DocuMind
   ```

2. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Descargar los recursos de NLTK:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## Uso
1. Ejecutar la aplicación:
   ```bash
   python app.py
   ```

2. Abrir el navegador en `http://localhost:5000`.
3. Ingresar el texto a clasificar y hacer clic en "Clasificar".

## Estructura del Proyecto
```
DocuMind/
├── app.py              # Aplicación Flask y lógica del clasificador
├── templates/          # Plantillas HTML
│   └── index.html     # Interfaz de usuario
├── documentos_balanceado_v2.csv  # Datos de entrenamiento
├── requirements.txt    # Dependencias del proyecto
└── README.md          # Este archivo
```

## Tecnologías Utilizadas
- Flask: Framework web.
- scikit-learn: Machine Learning.
- NLTK: Procesamiento de Lenguaje Natural.
- spaCy: Procesamiento de Lenguaje Natural.
- Bootstrap: Interfaz de usuario.
- JavaScript: Interactividad del frontend.

## Contribuir
1. Fork el proyecto.
2. Crear una rama para tu feature (`git checkout -b feature/AmazingFeature`).
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`).
4. Push a la rama (`git push origin feature/AmazingFeature`).
5. Abrir un Pull Request.

## Licencia
Este proyecto está bajo la Licencia MIT - ver el archivo LICENSE para más detalles. 