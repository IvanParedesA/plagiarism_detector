"""
Ensamble de modelos de similitud (TF-IDF + Coseno, Jaccard, etc.)

Correr con, por ejemplo:

python -m src.ensemble_models \
    --cosine_dir outputs \
    --jaccard_dir outputs_jaccard \
    --out_dir outputs_ensemble \
    --cosine_thr 0.75 \
    --jaccard_thr 0.5
"""

import os
from typing import Optional

import pandas as pd


def _load_pairs(
    csv_path: str,
    score_col: str,
    source_name: str,
) -> pd.DataFrame:
    """
    Carga un archivo de pares (file_i, file_j, score) y normaliza
    una llave de par para poder hacer merges consistentes.

    Agrega columnas:
      - pair_key: tupla ordenada (min(file_i, file_j), max(file_i, file_j)) como string
      - source: nombre del modelo (para debug)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el archivo: {csv_path}")

    df = pd.read_csv(csv_path)

    if not {"file_i", "file_j", score_col}.issubset(df.columns):
        raise ValueError(
            f"El archivo {csv_path} debe contener las columnas "
            f"'file_i', 'file_j' y '{score_col}'."
        )

    # Normalizar llave de par (independiente del orden i/j)
    def make_key(row):
        a, b = row["file_i"], row["file_j"]
        return f"{min(a, b)}||{max(a, b)}"

    df["pair_key"] = df.apply(make_key, axis=1)
    df["source"] = source_name
    return df


def build_ensemble(
    cosine_dir: str,
    jaccard_dir: str,
    out_dir: str,
    cosine_thr: float = 0.75,
    jaccard_thr: float = 0.5,
) -> None:
    """
    Construye un ensemble sencillo entre:

      - Modelo 1: TF-IDF + Coseno
      - Modelo 2: Jaccard (k-shingles)

    Lee los CSV de pares de cada modelo, los fusiona por (file_i, file_j)
    y aplica un esquema de votación basado en umbrales.

    Salidas:
      - ensemble_pairs.csv: todos los pares con scores y votos
      - ensemble_sospechosos.csv: solo pares marcados como sospechosos
    """

    os.makedirs(out_dir, exist_ok=True)

    # 1) Cargar pares de coseno
    cosine_csv = os.path.join(cosine_dir, "pairs_cosine.csv")
    df_cos = _load_pairs(cosine_csv, "cosine", "cosine")

    # 2) Cargar pares de jaccard
    jaccard_csv = os.path.join(jaccard_dir, "pairs_jaccard.csv")
    df_jac = _load_pairs(jaccard_csv, "jaccard", "jaccard")

    # 3) Merge por llave común (inner join: pares presentes en ambos modelos)
    df = pd.merge(
        df_cos[["pair_key", "file_i", "file_j", "cosine"]],
        df_jac[["pair_key", "jaccard"]],
        on="pair_key",
        how="inner",
    )

    # 4) Aplicar votos por umbral
    df["vote_cosine"] = (df["cosine"] >= cosine_thr).astype(int)
    df["vote_jaccard"] = (df["jaccard"] >= jaccard_thr).astype(int)

    # Número total de votos por par
    df["votes"] = df["vote_cosine"] + df["vote_jaccard"]

    # Regla de decisión (por ahora: al menos 1 modelo lo ve como sospechoso)
    df["decision"] = (df["votes"] >= 2).astype(int)

    # Ordenamos por un score combinado simple (promedio de ambos)
    df["score_mean"] = (df["cosine"] + df["jaccard"]) / 2.0
    df = df.sort_values("score_mean", ascending=False)

    # 5) Guardar todos los pares
    out_pairs = os.path.join(out_dir, "ensemble_pairs.csv")
    df.to_csv(out_pairs, index=False)

    # 6) Guardar solo sospechosos
    df_sos = df[df["decision"] == 1].copy()
    out_sos = os.path.join(out_dir, "ensemble_sospechosos.csv")
    df_sos.to_csv(out_sos, index=False)

    print(f"[Ensemble] Pares totales: {len(df)}")
    print(f"[Ensemble] Pares sospechosos: {len(df_sos)}")
    print(f"[Ensemble] Archivo completo: {out_pairs}")
    print(f"[Ensemble] Archivo sospechosos: {out_sos}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ensamble de modelos de similitud (coseno + Jaccard)."
    )
    parser.add_argument(
        "--cosine_dir",
        default="outputs",
        help="Carpeta donde se encuentra pairs_cosine.csv (TF-IDF + coseno).",
    )
    parser.add_argument(
        "--jaccard_dir",
        default="outputs_jaccard",
        help="Carpeta donde se encuentra pairs_jaccard.csv (Jaccard).",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs_ensemble",
        help="Carpeta donde se guardarán los CSV del ensemble.",
    )
    parser.add_argument(
        "--cosine_thr",
        type=float,
        default=0.75,
        help="Umbral de similitud coseno para voto del modelo de TF-IDF.",
    )
    parser.add_argument(
        "--jaccard_thr",
        type=float,
        default=0.5,
        help="Umbral de similitud Jaccard para voto del modelo de shingles.",
    )

    args = parser.parse_args()

    build_ensemble(
        cosine_dir=args.cosine_dir,
        jaccard_dir=args.jaccard_dir,
        out_dir=args.out_dir,
        cosine_thr=args.cosine_thr,
        jaccard_thr=args.jaccard_thr,
    )