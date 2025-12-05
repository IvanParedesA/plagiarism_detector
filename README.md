# Sistema de Detección de Similitud en Código Fuente

Este proyecto implementa un sistema completo para analizar similitud en programas escritos en Java. Combina técnicas de procesamiento de código, métricas cuantitativas y un sistema de votación basado en múltiples modelos. El objetivo es apoyar la detección de posibles casos de plagio, reutilización no autorizada o similitud anómala entre archivos fuente.

---

## Modelos incluidos en el sistema

El sistema utiliza cuatro enfoques complementarios para medir similitud:

### 1. TF‑IDF + Similitud del Coseno
Convierte cada archivo en un vector TF‑IDF usando `TfidfVectorizer` y compara pares mediante similitud del coseno. Permite capturar similitud textual general después de limpiar y normalizar el código.

### 2. Jaccard con k‑shingles
Fragmenta cada archivo en shingles (subcadenas de longitud fija). Compara conjuntos de shingles para obtener una medida de similitud mediante el índice de Jaccard. Es útil para detectar reordenamientos o modificaciones menores.

### 3. Similitud estructural mediante AST
Genera el Abstract Syntax Tree (AST) de cada archivo y extrae una representación estructural. La similitud se calcula comparando las estructuras internas de los programas.

### 4. Similitud de Secuencia (Tokens)
Convierte cada archivo en una secuencia de tokens y calcula la similitud utilizando medidas basadas en coincidencia parcial y alineamiento de secuencias. Permite detectar cambios leves conservando patrones lógicos.

### Ensemble de modelos
Combina los cuatro métodos usando un sistema de votación configurable. Un par de archivos se considera sospechoso si supera el número mínimo de votos definidos por el usuario.

---

## Estructura del proyecto

- **src/**
  - `features_tfidf.py` — Modelo TF‑IDF.
  - `shingles_jaccard.py` — Modelo Jaccard con k‑shingles.
  - `ast_structural.py` — Modelo estructural basado en AST.
  - `sequence_similarity.py` — Modelo de similitud de secuencia.
  - `ensemble_models.py` — Combinación de modelos y votación.
  - `io_loader.py`, `preproc.py` — Utilidades de lectura y normalización.
- **run_detector.py** — Ejecuta el pipeline completo.
- **datasets/** — Conjunto de datos de ejemplo (incluye IR‑Plag y un dataset pequeño).
- **outputs/** — Carpetas generadas por los modelos.
- **app.py** — Interfaz en Streamlit.

---

## Requisitos

Instalar dependencias:

```
pip install -r requirements.txt
```

---

## Ejecución del sistema

### Ejecutar todo el pipeline con un solo comando

```
python run_detector.py --input datasets/IR-Plag-Dataset/case-01 --out outputs_case01
```

Si no se especifican parámetros, `run_detector.py` utilizará rutas por defecto. También puede ejecutarse directamente desde el botón *Run* en VSCode.

---

## Ejecución de cada modelo por separado

### 1. TF‑IDF + Coseno
```
python -m src.features_tfidf --input_dir datasets/IR-Plag-Dataset/case-01 --output_dir outputs_tfidf
```

### 2. Jaccard (k‑shingles)
```
python -m src.shingles_jaccard --input_dir datasets/IR-Plag-Dataset/case-01 --output_dir outputs_jaccard
```

### 3. AST estructural
```
python -m src.ast_structural --input_dir datasets/IR-Plag-Dataset/case-01 --output_dir outputs_ast --threshold 0.7
```

### 4. Similitud de Secuencia (Tokens)
```
python -m src.sequence_similarity --input_dir datasets/IR-Plag-Dataset/case-01 --output_dir outputs_seq --threshold 0.8
```

### 5. Ensemble (votación)
```
python -m src.ensemble_models \
    --cosine_dir outputs_tfidf \
    --jaccard_dir outputs_jaccard \
    --ast_dir outputs_ast \
    --seq_dir outputs_seq \
    --out_dir outputs_ensemble \
    --cosine_thr 0.75 \
    --jaccard_thr 0.5 \
    --ast_thr 0.7 \
    --seq_thr 0.8
```

---

## Uso en Streamlit

El proyecto incluye una interfaz en `app.py`.

```
streamlit run app.py
```

La aplicación permite:

- Ejecutar el pipeline completo.
- Procesar datasets locales o cargar archivos ZIP.
- Visualizar resultados por modelo.
- Descargar los reportes generados.

---

## Datasets incluidos

### IR‑Plag Dataset
Fuente: https://github.com/oscarkarnalim/sourcecodeplagiarismdataset  
Incluye múltiples casos organizados por niveles de transformación.

### small_example
Dataset pequeño para pruebas rápidas.

---

## Notas finales

- El sistema está diseñado para ser extensible con modelos supervisados.
- El ensemble permite ajustar umbrales y estrategias de decisión.
- Todos los reportes se guardan en carpetas generadas automáticamente.

