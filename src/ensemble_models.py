"""
Ensamble de modelos de similitud (TF-IDF + Coseno, Jaccard, AST y secuencia de tokens).

Correr con, por ejemplo:

python -m src.ensemble_models \
    --cosine_dir outputs \
    --jaccard_dir outputs_jaccard \
    --ast_dir outputs_ast \
    --seq_dir outputs_seq \
    --out_dir outputs_ensemble \
    --cosine_thr 0.75 \
    --jaccard_thr 0.5 \
    --ast_thr 0.7 \
    --seq_thr 0.8
"""

import os
import pandas as pd


def _load_pairs(csv_path: str, score_col: str, source_name: str) -> pd.DataFrame:
    """
    Carga un archivo de pares (file_i, file_j, score) y normaliza
    una llave de par para poder hacer merges consistentes.

    Agrega columnas:
      - pair_key: (min(file_i, file_j), max(file_i, file_j)) como string
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

    def make_key(row):
        a, b = row["file_i"], row["file_j"]
        return f"{min(a, b)}||{max(a, b)}"

    df["pair_key"] = df.apply(make_key, axis=1)
    df["source"] = source_name
    return df


def build_ensemble(
    cosine_dir: str,
    jaccard_dir: str,
    ast_dir: str,
    seq_dir: str,
    out_dir: str,
    cosine_thr: float = 0.75,
    jaccard_thr: float = 0.5,
    ast_thr: float = 0.7,
    seq_thr: float = 0.8,
) -> None:
    """
    Construye un ensemble entre:

      - Modelo 1: TF-IDF + Coseno (pairs_cosine.csv)
      - Modelo 2: Jaccard (pairs_jaccard.csv)
      - Modelo 3: AST estructural (pairs_ast.csv)
      - Modelo 4: Secuencia de tokens (pairs_seq.csv)

    Lee los CSV de pares de cada modelo, los fusiona por (file_i, file_j)
    y aplica un esquema de votación basado en umbrales.

    Regla de decisión por defecto:
      - Un par es sospechoso si al menos 3 de los 4 modelos lo consideran sospechoso
        según sus umbrales individuales (mayoría calificada).

    Salidas:
      - ensemble_pairs.csv: todos los pares con scores y votos
      - ensemble_sospechosos.csv: solo pares marcados como sospechosos
    """

    os.makedirs(out_dir, exist_ok=True)

    # 1) Cargar pares de cada modelo
    cosine_csv = os.path.join(cosine_dir, "pairs_cosine.csv")
    jaccard_csv = os.path.join(jaccard_dir, "pairs_jaccard.csv")
    ast_csv = os.path.join(ast_dir, "pairs_ast.csv")
    seq_csv = os.path.join(seq_dir, "pairs_seq.csv")

    df_cos = _load_pairs(cosine_csv, "cosine", "cosine")
    df_jac = _load_pairs(jaccard_csv, "jaccard", "jaccard")
    df_ast = _load_pairs(ast_csv, "ast_cosine", "ast")
    df_seq = _load_pairs(seq_csv, "seq_sim", "seq")

    # 2) Merge por llave común (inner join: pares presentes en todos los modelos)
    df = (
        df_cos[["pair_key", "file_i", "file_j", "cosine"]]
        .merge(df_jac[["pair_key", "jaccard"]], on="pair_key", how="inner")
        .merge(df_ast[["pair_key", "ast_cosine"]], on="pair_key", how="inner")
        .merge(df_seq[["pair_key", "seq_sim"]], on="pair_key", how="inner")
    )

    # 3) Aplicar votos por umbral
    df["vote_cosine"] = (df["cosine"] >= cosine_thr).astype(int)
    df["vote_jaccard"] = (df["jaccard"] >= jaccard_thr).astype(int)
    df["vote_ast"] = (df["ast_cosine"] >= ast_thr).astype(int)
    df["vote_seq"] = (df["seq_sim"] >= seq_thr).astype(int)

    # Número total de votos por par
    df["votes"] = (
        df["vote_cosine"]
        + df["vote_jaccard"]
        + df["vote_ast"]
        + df["vote_seq"]
    )

    # Regla de decisión: mayoría calificada (al menos 3 de 4 modelos)
    df["decision"] = (df["votes"] >= 3).astype(int)

    # Score combinado simple (promedio de los 4 modelos)
    df["score_mean"] = (
        df["cosine"] + df["jaccard"] + df["ast_cosine"] + df["seq_sim"]
    ) / 4.0
    df = df.sort_values("score_mean", ascending=False)

    # 4) Guardar todos los pares
    out_pairs = os.path.join(out_dir, "ensemble_pairs.csv")
    df.to_csv(out_pairs, index=False)

    # 5) Guardar solo sospechosos
    df_sos = df[df["decision"] == 1].copy()
    out_sos = os.path.join(out_dir, "ensemble_sospechosos.csv")
    df_sos.to_csv(out_sos, index=False)

    print(f"[Ensemble] Pares totales: {len(df)}")
    print(f"[Ensemble] Pares sospechosos (votes >= 3): {len(df_sos)}")
    print(f"[Ensemble] Archivo completo: {out_pairs}")
    print(f"[Ensemble] Archivo sospechosos: {out_sos}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Ensamble de modelos de similitud "
            "(coseno TF-IDF, Jaccard, AST, secuencia de tokens)."
        )
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
        "--ast_dir",
        default="outputs_ast",
        help="Carpeta donde se encuentra pairs_ast.csv (AST estructural).",
    )
    parser.add_argument(
        "--seq_dir",
        default="outputs_seq",
        help="Carpeta donde se encuentra pairs_seq.csv (secuencia de tokens).",
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
    parser.add_argument(
        "--ast_thr",
        type=float,
        default=0.7,
        help="Umbral de similitud coseno para voto del modelo estructural AST.",
    )
    parser.add_argument(
        "--seq_thr",
        type=float,
        default=0.8,
        help="Umbral de similitud de secuencia para voto del modelo de tokens.",
    )

    args = parser.parse_args()

    build_ensemble(
        cosine_dir=args.cosine_dir,
        jaccard_dir=args.jaccard_dir,
        ast_dir=args.ast_dir,
        seq_dir=args.seq_dir,
        out_dir=args.out_dir,
        cosine_thr=args.cosine_thr,
        jaccard_thr=args.jaccard_thr,
        ast_thr=args.ast_thr,
        seq_thr=args.seq_thr,
    )