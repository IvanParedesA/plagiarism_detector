import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def simple_cosine_report(files, X, out_dir):
    S = cosine_similarity(X)
    n = len(files)
    rows = []
    for i in range(n):
        for j in range(i+1, n):
            rows.append({
                "file_i": files[i],
                "file_j": files[j],
                "cosine": float(S[i, j])
            })

    df = pd.DataFrame(rows).sort_values("cosine", ascending=False)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "pairs_cosine.csv"), index=False)

    mu, sigma = df["cosine"].mean(), df["cosine"].std()
    thr = mu + 2 * sigma
    df[df["cosine"] >= thr].to_csv(os.path.join(out_dir, "sospechosos_cosine.csv"), index=False)

    mat = pd.DataFrame(S, index=files, columns=files)
    mat.to_csv(os.path.join(out_dir, "matriz_cosine.csv"))

    print(f"Umbral sugerido: {thr:.3f}")