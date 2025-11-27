"""
Modelo 4: Similitud de secuencia de tokens para código Java.

- Reutiliza la normalización de `preproc.normalize_java`.
- Convierte cada archivo en una secuencia de tokens.
- Usa difflib.SequenceMatcher para medir similitud de secuencia.
- Exporta:
  - matriz_seq.csv
  - pairs_seq.csv
  - sospechosos_seq.csv

Ejemplo de uso:

python -m src.sequence_similarity --input_dir data --output_dir outputs_seq --threshold 0.8
"""

import os
from pathlib import Path
from typing import List, Tuple
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from src.preproc import normalize_java  # importante: usar el import con src.


# -----------------------------
# Carga de archivos .java
# -----------------------------

def load_java_files(input_dir: str) -> Tuple[List[str], List[str]]:
    """
    Carga archivos .java desde un directorio (recursivo) y regresa:

    - files: lista de rutas (str)
    - texts_norm: lista de textos normalizados (tokens separados por espacios)
    """
    files: List[str] = []
    texts_norm: List[str] = []

    root = Path(input_dir)
    for path in root.rglob("*.java"):
        try:
            code = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            code = path.read_text(encoding="latin-1", errors="ignore")

        norm = normalize_java(code)
        if norm:
            files.append(str(path))
            texts_norm.append(norm)

    return files, texts_norm


# -----------------------------
# Similitud de secuencia
# -----------------------------

def compute_sequence_matrix(texts_norm: List[str]) -> np.ndarray:
    """
    A partir de textos normalizados (tokens separados por espacios),
    construye la matriz NxN de similitud de secuencia usando
    difflib.SequenceMatcher sobre listas de tokens.
    """
    tokens_list: List[List[str]] = [t.split() for t in texts_norm]

    n = len(tokens_list)
    S = np.zeros((n, n), dtype=float)

    for i in range(n):
        S[i, i] = 1.0
        for j in range(i + 1, n):
            a = tokens_list[i]
            b = tokens_list[j]
            if not a and not b:
                sim = 0.0
            else:
                sim = SequenceMatcher(None, a, b).ratio()
            S[i, j] = sim
            S[j, i] = sim

    return S


# -----------------------------
# Reportes
# -----------------------------

def export_sequence_reports(
    files: List[str],
    S: np.ndarray,
    output_dir: str,
    threshold: float = 0.8,
) -> None:
    """
    Genera reportes basados en la matriz de similitud de secuencia:

    - matriz_seq.csv         (NxN completa)
    - pairs_seq.csv          (pares i < j con su similitud)
    - sospechosos_seq.csv    (pares con similitud >= threshold)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Matriz completa
    mat_df = pd.DataFrame(S, index=files, columns=files)
    mat_df.to_csv(os.path.join(output_dir, "matriz_seq.csv"))

    # Pares (i < j)
    n = len(files)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            rows.append(
                {
                    "file_i": files[i],
                    "file_j": files[j],
                    "seq_sim": float(S[i, j]),
                }
            )

    df_pairs = pd.DataFrame(rows).sort_values("seq_sim", ascending=False)
    df_pairs.to_csv(os.path.join(output_dir, "pairs_seq.csv"), index=False)

    # Sospechosos
    sospechosos = df_pairs[df_pairs["seq_sim"] >= threshold]
    sospechosos.to_csv(
        os.path.join(output_dir, "sospechosos_seq.csv"), index=False
    )

    print(f"[SEQ] Umbral usado: {threshold:.3f}")
    print(f"[SEQ] Pares sospechosos encontrados: {len(sospechosos)}")


# -----------------------------
# Pipeline completo
# -----------------------------

def run_sequence_pipeline(input_dir: str, output_dir: str, threshold: float = 0.8) -> None:
    """
    Pipeline completo del modelo de secuencia:

    1. Cargar archivos .java.
    2. Normalizarlos (normalize_java).
    3. Calcular matriz de similitud de secuencia.
    4. Exportar reportes.
    """
    print("[SEQ] Cargando archivos .java...")
    files, texts_norm = load_java_files(input_dir)

    if not files:
        print("[SEQ] No se encontraron archivos .java en el directorio indicado.")
        return

    print(f"[SEQ] Archivos cargados: {len(files)}")

    print("[SEQ] Calculando matriz de similitud de secuencia...")
    S = compute_sequence_matrix(texts_norm)

    print("[SEQ] Exportando reportes...")
    export_sequence_reports(files, S, output_dir, threshold=threshold)
    print("[SEQ] Listo. Revisa la carpeta de salida.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Modelo de similitud de secuencia de tokens para código Java."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Carpeta raíz con archivos .java",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs_seq",
        help="Carpeta de salida para los reportes de secuencia",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Umbral para marcar pares sospechosos según similitud de secuencia",
    )

    args = parser.parse_args()

    run_sequence_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
    )