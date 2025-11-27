"""
Modelo 3: Similitud estructural basada en AST para código Java.

- Usa javalang para parsear el código fuente.
- Extrae vectores de conteo de tipos de nodos del AST.
- Calcula similitud coseno entre perfiles estructurales.
- Genera CSVs: matriz_ast.csv, pairs_ast.csv, sospechosos_ast.csv.

Ejemplo de uso:

python -m src.ast_structural --input_dir data --output_dir outputs_ast --threshold 0.7
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import javalang
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Carga de archivos .java
# -----------------------------

def load_java_files(input_dir: str) -> Tuple[List[str], List[str]]:
    """
    Carga archivos .java desde un directorio (recursivo) y regresa:

    - files: lista de rutas (str)
    - codes: lista de contenidos de archivo (str)
    """
    files: List[str] = []
    codes: List[str] = []

    root = Path(input_dir)
    for path in root.rglob("*.java"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = path.read_text(encoding="latin-1", errors="ignore")
        files.append(str(path))
        codes.append(text)

    return files, codes


# -----------------------------
# Extracción de features AST
# -----------------------------

def extract_ast_features(code: str) -> Dict[str, int]:
    """
    Parsea el código Java con javalang y construye un perfil
    estructural simple: conteo de tipos de nodos en el AST.

    Si el parseo falla, regresa un dict vacío.
    """
    features: Dict[str, int] = {}

    try:
        tree = javalang.parse.parse(code)
    except Exception:
        # Código mal formado o errores de parseo
        return features

    # Recorremos el árbol: tree es un CompilationUnit
    for path, node in tree:
        node_type = type(node).__name__

        # Contar todos los tipos de nodo puede ser útil,
        # pero si quieres puedes filtrar algunos específicos:
        # if node_type in {"IfStatement", "ForStatement", ...}
        features[node_type] = features.get(node_type, 0) + 1

    return features


def build_ast_matrix(codes: List[str]) -> Tuple[np.ndarray, DictVectorizer, List[Dict[str, int]]]:
    """
    A partir de una lista de códigos fuente, genera:

    - X: matriz de features (N x D)
    - vec: DictVectorizer usado para mapear features
    - feats: lista de dicts de features por archivo
    """
    feats: List[Dict[str, int]] = []
    for code in codes:
        f = extract_ast_features(code)
        feats.append(f)

    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(feats)

    return X, vec, feats


# -----------------------------
# Reportes de similitud AST
# -----------------------------

def export_ast_reports(
    files: List[str],
    S: np.ndarray,
    output_dir: str,
    threshold: float = 0.7,
) -> None:
    """
    Genera reportes basados en la matriz de similitud estructural (coseno sobre AST):

    - matriz_ast.csv         (NxN completa)
    - pairs_ast.csv          (pares i < j con su similitud)
    - sospechosos_ast.csv    (pares con similitud >= threshold)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Matriz completa
    mat_df = pd.DataFrame(S, index=files, columns=files)
    mat_df.to_csv(os.path.join(output_dir, "matriz_ast.csv"))

    # Pares (i < j)
    n = len(files)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            rows.append(
                {
                    "file_i": files[i],
                    "file_j": files[j],
                    "ast_cosine": float(S[i, j]),
                }
            )

    df_pairs = pd.DataFrame(rows).sort_values("ast_cosine", ascending=False)
    df_pairs.to_csv(os.path.join(output_dir, "pairs_ast.csv"), index=False)

    # Sospechosos
    sospechosos = df_pairs[df_pairs["ast_cosine"] >= threshold]
    sospechosos.to_csv(
        os.path.join(output_dir, "sospechosos_ast.csv"), index=False
    )

    print(f"[AST] Umbral usado: {threshold:.3f}")
    print(f"[AST] Pares sospechosos encontrados: {len(sospechosos)}")


# -----------------------------
# Pipeline completo
# -----------------------------

def run_ast_pipeline(input_dir: str, output_dir: str, threshold: float = 0.7) -> None:
    """
    Pipeline completo del modelo estructural:

    1. Cargar archivos .java.
    2. Extraer vectores AST (perfil estructural).
    3. Calcular matriz de similitud coseno.
    4. Exportar reportes.
    """
    print("[AST] Cargando archivos .java...")
    files, codes = load_java_files(input_dir)

    if not files:
        print("[AST] No se encontraron archivos .java en el directorio indicado.")
        return

    print(f"[AST] Archivos cargados: {len(files)}")

    print("[AST] Extrayendo características estructurales (AST)...")
    X, vec, feats = build_ast_matrix(codes)

    print("[AST] Calculando matriz de similitud coseno...")
    S = cosine_similarity(X)

    print("[AST] Exportando reportes...")
    export_ast_reports(files, S, output_dir, threshold=threshold)
    print("[AST] Listo. Revisa la carpeta de salida.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Modelo de similitud estructural basado en AST para código Java."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Carpeta raíz con archivos .java",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs_ast",
        help="Carpeta de salida para los reportes AST",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Umbral para marcar pares sospechosos según similitud estructural",
    )

    args = parser.parse_args()

    run_ast_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
    )