# Sistema de Detección de Similitud en Código Fuente

Este proyecto implementa un sistema para apoyar la detección de posibles infracciones de derechos de autor en programas de computadora, combinando:

- *Compiladores* (análisis léxico/sintáctico, normalización y estructura del código),
- *Inteligencia Artificial* (representaciones vectoriales y modelos supervisados),
- *Métodos Cuantitativos* (métricas de similitud, umbrales y sistemas de votación).

En su primera fase, el sistema genera una representación *TF-IDF* del código (usando `TfidfVectorizer` de Scikit-Learn) y calcula la similitud mediante *cosine similarity*, produciendo un puntaje objetivo entre cada par de archivos.

---

## Características principales

- Limpieza y normalización del código (comentarios, strings, identificadores).
- Extracción de características mediante TF-IDF.
- Cálculo de similitud numérica entre archivos.
- Generación de reportes:
  - Matriz de similitud
  - CSV de pares sospechosos
- Arquitectura preparada para integrar más modelos:
  - Jaccard sobre k-shingles  
  - Similitud estructural mediante AST  
  - Clasificadores supervisados (LogReg / MLP)  
  - Sistemas de votación (majority / weighted)

---

## Estructura del proyecto

- **src/**
  - `io_loader.py`: lectura de archivos desde carpetas.
  - `preproc.py`: limpieza y normalización del código.
  - `features_tfidf.py`: construcción de la matriz TF-IDF.
  - `ensemble.py`: cálculo de similitud y futura votación multimodelo.
  - `cli.py`: punto de entrada por línea de comandos.

- **data/**: ejemplos de código a analizar.  
- **outputs/**: reportes generados (matrices y CSV).  
- **docs/**: documentación del diseño y notas del reto.

---

## Requisitos

```bash
pip install -r requirements.txt

---

## Ejecución del proyecto

### 1. Ejecutar modelo TF-IDF + Coseno
```bash
python -m src.features_tfidf --input_dir data --output_dir outputs
```

### 2. Ejecutar modelo Jaccard (k-shingles)
```bash
python -m src.shingles_jaccard --input_dir data --output_dir outputs_jaccard
```

### 3. Ejecutar ensemble (combinación de modelos)
```bash
python -m src.ensemble_models \
    --cosine_dir outputs \
    --jaccard_dir outputs_jaccard \
    --out_dir outputs_ensemble \
    --cosine_thr 0.75 \
    --jaccard_thr 0.5
```

Esto generará los reportes en las carpetas `outputs/`, `outputs_jaccard/` y `outputs_ensemble/`.
