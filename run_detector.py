# run_detector.py
import subprocess

def run(cmd: str):
    print(f"\n>>> Ejecutando:\n{cmd}\n")
    subprocess.run(cmd, shell=True, check=True)

def run_pipeline(input_dir: str, out_prefix: str):
    base_in = input_dir
    base_out = out_prefix

    tfidf_out   = f"{base_out}_tfidf"
    jaccard_out = f"{base_out}_jaccard"
    ast_out     = f"{base_out}_ast"
    seq_out     = f"{base_out}_seq"
    ens_out     = f"{base_out}_ensemble"

    # 1. TF-IDF + Coseno
    run(f"python -m src.features_tfidf --input_dir {base_in} --output_dir {tfidf_out}")

    # 2. Jaccard
    run(f"python -m src.shingles_jaccard --input_dir {base_in} --output_dir {jaccard_out}")

    # 3. AST
    run(f"python -m src.ast_structural --input_dir {base_in} --output_dir {ast_out}")

    # 4. Secuencia de tokens
    run(f"python -m src.sequence_similarity --input_dir {base_in} --output_dir {seq_out} --threshold 0.8")

    # 5. Ensemble final
    run(
        f"python -m src.ensemble_models "
        f"--cosine_dir {tfidf_out} "
        f"--jaccard_dir {jaccard_out} "
        f"--ast_dir {ast_out} "
        f"--seq_dir {seq_out} "
        f"--out_dir {ens_out} "
        f"--cosine_thr 0.75 --jaccard_thr 0.5 --ast_thr 0.7 --seq_thr 0.8"
    )

    print("\n Pipeline completado exitosamente.")
    print(f" Resultados finales en: {ens_out}")
    return ens_out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="datasets/IR-Plag-Dataset/case-01")
    parser.add_argument("--out",   default="outputs_autorun")
    args = parser.parse_args()

    run_pipeline(args.input, args.out)