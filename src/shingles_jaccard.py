"""
Correr con:
python -m src.shingles_jaccard --input_dir data --output_dir outputs_jaccard
"""

import os
from typing import List, Set, Tuple

import numpy as np
import pandas as pd

from src.preproc import normalize_java
from pathlib import Path


def load_java_files(input_dir: str) -> Tuple[List[str], List[str]]:
    """
    Carga archivos .java desde un directorio y devuelve:
    - files: lista de rutas
    - texts_norm: lista de textos normalizados (tokens separados por espacios)
    """
    files = []
    texts_norm = []

    input_path = Path(input_dir)
    for path in input_path.rglob("*.java"):
        try:
            code = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            # Intento con latin-1 si hay problemas raros de encoding
            code = path.read_text(encoding="latin-1", errors="ignore")

        norm = normalize_java(code)
        if norm:  # evitar vacíos
            files.append(str(path))
            texts_norm.append(norm)

    return files, texts_norm


def make_shingles(tokens: List[str], k: int = 5) -> Set[str]:
    """
    Genera un conjunto de k-shingles (ventanas de k tokens consecutivos)
    a partir de una lista de tokens.
    """
    if len(tokens) < k:
        # Si hay muy pocos tokens, usamos todo como un único shingle
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def compute_jaccard_matrix(texts_norm: List[str], k: int = 5) -> np.ndarray:
    """
    A partir de textos normalizados (tokens separados por espacios),
    construye la matriz NxN de similitud de Jaccard sobre k-shingles.
    """
    # Convertimos cada texto a lista de tokens y luego a conjunto de shingles
    shingles_list: List[Set[str]] = []
    for t in texts_norm:
        tokens = t.split()
        shingles = make_shingles(tokens, k=k)
        shingles_list.append(shingles)

    n = len(shingles_list)
    J = np.zeros((n, n), dtype=float)

    for i in range(n):
        J[i, i] = 1.0
        for j in range(i + 1, n):
            a = shingles_list[i]
            b = shingles_list[j]
            if not a and not b:
                sim = 0.0
            else:
                inter = len(a & b)
                union = len(a | b) or 1
                sim = inter / union
            J[i, j] = sim
            J[j, i] = sim

    return J


def export_jaccard_reports(
    files: List[str],
    J: np.ndarray,
    output_dir: str,
    threshold: float = 0.5,
) -> None:
    """
    Genera tres reportes basados en la matriz de Jaccard:

    - matriz_jaccard.csv: matriz completa NxN
    - pairs_jaccard.csv: lista de pares (i < j) y su similitud
    - sospechosos_jaccard.csv: pares con similitud >= threshold
    """
    os.makedirs(output_dir, exist_ok=True)

    # Matriz completa
    mat_df = pd.DataFrame(J, index=files, columns=files)
    mat_df.to_csv(os.path.join(output_dir, "matriz_jaccard.csv"))

    # Pares
    n = len(files)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            rows.append(
                {
                    "file_i": files[i],
                    "file_j": files[j],
                    "jaccard": float(J[i, j]),
                }
            )

    df_pairs = pd.DataFrame(rows).sort_values("jaccard", ascending=False)
    df_pairs.to_csv(os.path.join(output_dir, "pairs_jaccard.csv"), index=False)

    # Sospechosos según threshold
    sospechosos = df_pairs[df_pairs["jaccard"] >= threshold]
    sospechosos.to_csv(
        os.path.join(output_dir, "sospechosos_jaccard.csv"), index=False
    )

    print(f"[Jaccard] Umbral usado: {threshold:.3f}")
    print(f"[Jaccard] Pares sospechosos encontrados: {len(sospechosos)}")


def run_jaccard_pipeline(input_dir: str, output_dir: str, k: int = 5, threshold: float = 0.5) -> None:
    """
    Pipeline completo para:
      - cargar .java,
      - normalizarlos,
      - calcular matriz Jaccard con k-shingles,
      - exportar reportes.
    """
    print("[Jaccard] Cargando archivos .java...")
    files, texts_norm = load_java_files(input_dir)
    if not files:
        print("[Jaccard] No se encontraron archivos .java en el directorio indicado.")
        return

    print(f"[Jaccard] Archivos cargados: {len(files)}")
    print("[Jaccard] Calculando matriz de similitud Jaccard...")
    J = compute_jaccard_matrix(texts_norm, k=k)

    print("[Jaccard] Exportando reportes...")
    export_jaccard_reports(files, J, output_dir, threshold=threshold)
    print("[Jaccard] Listo. Revisa la carpeta de salida.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline de similitud Jaccard sobre k-shingles para código Java."
    )
    parser.add_argument("--input_dir", required=True, help="Carpeta raíz con archivos .java")
    parser.add_argument("--output_dir", default="outputs_jaccard", help="Carpeta de salida")
    parser.add_argument("--k", type=int, default=5, help="Tamaño de los k-shingles (default=5)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Umbral para marcar sospechosos")
    args = parser.parse_args()

    run_jaccard_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        k=args.k,
        threshold=args.threshold,
    )