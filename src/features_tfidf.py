# src/features_tfidf.py

import os
from itertools import combinations
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preproc import normalize_java


def load_java_files(input_dir: str) -> Tuple[List[str], List[str]]:
    """
    Recorre recursivamente `input_dir` y carga todos los archivos .java.

    Devuelve:
        paths: lista de rutas completas
        texts: lista de códigos normalizados (strings tokenizados)
    """
    paths: List[str] = []
    texts: List[str] = []

    for root, _, files in os.walk(input_dir):
        for fname in files:
            if fname.endswith(".java"):
                full_path = os.path.join(root, fname)
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    raw_code = f.read()
                norm_code = normalize_java(raw_code)
                paths.append(full_path)
                texts.append(norm_code)

    return paths, texts


def build_tfidf_matrix(texts: List[str]):
    """
    Construye la matriz TF-IDF a partir de texto YA normalizado/tokenizado.

    Importante:
    - tokenizer=str.split  → usamos los tokens que ya vienen separados.
    - lowercase=False      → ya tratamos los tokens tal cual.
    - ngram_range=(1, 3)   → unigrams, bigrams y trigrams de tokens.
    """
    vectorizer = TfidfVectorizer(
        analyzer="word",
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 3),
        min_df=1,
    )
    X = vectorizer.fit_transform(texts)
    return X, vectorizer


def compute_cosine_matrix(X) -> np.ndarray:
    """
    Calcula la matriz de similitud Cosine NxN.
    """
    return cosine_similarity(X)


def export_reports(
    paths: List[str],
    sim_matrix: np.ndarray,
    output_dir: str,
    threshold: float = 0.75,
    metric_name: str = "cosine",
) -> None:
    """
    Genera los 3 CSV:

    - matriz_{metric_name}.csv       (NxN)
    - pairs_{metric_name}.csv        (lista de pares únicos)
    - sospechosos_{metric_name}.csv  (pares con similitud >= threshold)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Normalizamos los paths a tipo Unix para que se vean parecidos a tu ejemplo.
    file_ids = [p.replace("\\", "/") for p in paths]

    # 1) Matriz completa
    df_mat = pd.DataFrame(sim_matrix, index=file_ids, columns=file_ids)
    matriz_path = os.path.join(output_dir, f"matriz_{metric_name}.csv")
    df_mat.to_csv(matriz_path)

    # 2) Lista de pares únicos (i < j)
    rows = []
    n = len(file_ids)
    for i in range(n):
        for j in range(i + 1, n):
            rows.append(
                (file_ids[i], file_ids[j], float(sim_matrix[i, j]))
            )

    df_pairs = pd.DataFrame(rows, columns=["file_i", "file_j", metric_name])
    df_pairs.sort_values(by=metric_name, ascending=False, inplace=True)
    pairs_path = os.path.join(output_dir, f"pairs_{metric_name}.csv")
    df_pairs.to_csv(pairs_path, index=False)

    # 3) Pairs sospechosos (>= threshold)
    df_susp = df_pairs[df_pairs[metric_name] >= threshold].copy()
    sospechosos_path = os.path.join(output_dir, f"sospechosos_{metric_name}.csv")
    df_susp.to_csv(sospechosos_path, index=False)

    # Log número de pares sospechosos (para Streamlit y consola)
    print(f"[TFIDF] Pares sospechosos encontrados: {len(df_susp)}")


def run_tfidf_pipeline(
    input_dir: str = "data",
    output_dir: str = "outputs",
    threshold: float = 0.75,
) -> None:
    """
    Pipeline completo:

    1. Lee archivos .java de `input_dir`.
    2. Normaliza con normalize_java.
    3. Construye TF-IDF con n-gramas 1-3.
    4. Calcula matriz Cosine.
    5. Exporta CSVs a `output_dir`.
    """
    paths, texts = load_java_files(input_dir)

    if not paths:
        raise SystemExit(f"No se encontraron archivos .java en el directorio: {input_dir}")

    X, _ = build_tfidf_matrix(texts)
    sim_matrix = compute_cosine_matrix(X)
    export_reports(paths, sim_matrix, output_dir, threshold=threshold)


if __name__ == "__main__":
    # Punto de entrada directo por CLI, opcional
    import argparse

    parser = argparse.ArgumentParser(
        description="Genera matriz y reportes de similitud TF-IDF (cosine) para archivos Java."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data",
        help="Directorio raíz donde se buscan archivos .java (por defecto: data)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directorio donde se guardan los CSV (por defecto: outputs)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Umbral de similitud para considerar pares sospechosos (por defecto: 0.75)",
    )
    args = parser.parse_args()

    run_tfidf_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
    )
