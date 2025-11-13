import argparse
import os
from io_loader import load_folder
from preproc import normalize
from features_tfidf import build_tfidf_matrix
from ensemble import simple_cosine_report

def main():
    parser = argparse.ArgumentParser(
        description="MVP: detección de similitud en código fuente con TF-IDF + coseno."
    )
    parser.add_argument("--input", required=True, help="Carpeta con archivos de código")
    parser.add_argument("--out", default="outputs", help="Carpeta de salida para reportes")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    files, texts = load_folder(args.input)
    if not files:
        print(" No se encontraron archivos en la carpeta de entrada.")
        return

    texts_norm = [normalize(t) for t in texts]
    X, _ = build_tfidf_matrix(texts_norm)

    simple_cosine_report(files, X, args.out)
    print(f" Análisis completado. Revisa la carpeta: {args.out}")

if __name__ == "__main__":
    main()