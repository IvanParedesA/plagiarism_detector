import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def simple_cosine_report(files, X, out_dir):
    """
    Genera reportes de similitud usando únicamente TF-IDF + coseno.

    - pairs_cosine.csv: todos los pares (i, j) con su similitud coseno.
    - sospechosos_cosine.csv: pares por encima de un umbral práctico.
    - matriz_cosine.csv: matriz completa de similitud (N x N).

    Por ahora usamos:
      - Umbral estadístico: mu + 2*sigma (solo informativo)
      - Umbral práctico fijo: 0.75
    """

    # Matriz de similitud coseno
    S = cosine_similarity(X)
    n = len(files)

    # Construimos la tabla de pares (i < j)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            rows.append(
                {
                    "file_i": files[i],
                    "file_j": files[j],
                    "cosine": float(S[i, j]),
                }
            )

    df = pd.DataFrame(rows).sort_values("cosine", ascending=False)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "pairs_cosine.csv"), index=False)

    # Estadísticos básicos sobre las similitudes
    mu = df["cosine"].mean()
    sigma = df["cosine"].std(ddof=0)  # var poblacional; con pocos datos da más estable
    thr_stats = mu + 2 * sigma        # umbral estadístico de referencia

    # Umbral práctico (por ahora fijo). Se puede ajustar más adelante
    thr = 0.75

    sospechosos = df[df["cosine"] >= thr]
    sospechosos.to_csv(
        os.path.join(out_dir, "sospechosos_cosine.csv"), index=False
    )

    # Guardamos también la matriz completa
    mat = pd.DataFrame(S, index=files, columns=files)
    mat.to_csv(os.path.join(out_dir, "matriz_cosine.csv"))

    print(f"Umbral estadístico mu+2σ = {thr_stats:.3f}")
    print(f"Umbral práctico usado   = {thr:.3f}")
    print(f"Pares sospechosos encontrados: {len(sospechosos)}")