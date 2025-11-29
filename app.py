# app.py
import streamlit as st
import subprocess
from pathlib import Path
from datetime import datetime
import re
import pandas as pd

def parse_summary(stdout: str) -> dict:
    summary = {
        "tfidf_suspicious": None,
        "jaccard_suspicious": None,
        "ast_suspicious": None,
        "seq_suspicious": None,
        "ensemble_total_pairs": None,
        "ensemble_suspicious": None,
    }
    for line in stdout.splitlines():
        line = line.strip()
        # Jaccard suspicious pairs
        if "[Jaccard]" in line and "Pares sospechosos encontrados" in line:
            try:
                summary["jaccard_suspicious"] = int(line.split(":")[-1].strip())
            except ValueError:
                pass
        # AST suspicious pairs
        if "[AST]" in line and "Pares sospechosos encontrados" in line:
            try:
                summary["ast_suspicious"] = int(line.split(":")[-1].strip())
            except ValueError:
                pass
        # Sequence suspicious pairs
        if "[SEQ]" in line and "Pares sospechosos encontrados" in line:
            try:
                summary["seq_suspicious"] = int(line.split(":")[-1].strip())
            except ValueError:
                pass
        # TF-IDF suspicious pairs
        if "[TF-IDF]" in line and "Pares sospechosos encontrados" in line:
            try:
                summary["tfidf_suspicious"] = int(line.split(":")[-1].strip())
            except ValueError:
                pass
        # Ensemble total pairs
        if line.startswith("[Ensemble] Pares totales"):
            try:
                summary["ensemble_total_pairs"] = int(line.split(":")[-1].strip())
            except ValueError:
                pass
        # Ensemble suspicious pairs
        if line.startswith("[Ensemble] Pares sospechosos"):
            # The line looks like: [Ensemble] Pares sospechosos (votes >= 3): 493
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    summary["ensemble_suspicious"] = int(parts[-1].strip())
                except ValueError:
                    pass
    return summary

def run_pipeline(input_dir: str, out_prefix: str):
    cmd = ["python", "run_detector.py", "--input", input_dir, "--out", out_prefix]
    result = subprocess.run(cmd, capture_output=True, text=True)
    summary = parse_summary(result.stdout or "")
    return result, summary

st.title("Plagiarism Detector 💻📚")

st.markdown("""
Sube una carpeta con archivos `.java` (o usa uno de los datasets de ejemplo) y 
ejecuta todos los modelos de similitud para ver posibles casos sospechosos.
""")

# --- Inicializar estado ---
if "run_info" not in st.session_state:
    st.session_state.run_info = None  # aquí guardaremos logs y out_prefix

# Inputs básicos
input_dir = st.text_input(
    "Carpeta de entrada",
    value="datasets/IR-Plag-Dataset/case-01"
)

out_prefix = st.text_input(
    "Prefijo de salida",
    value="outputs_case01"
)

# --- Botón para lanzar el análisis ---
if st.button("Ejecutar análisis"):
    with st.spinner("Corriendo análisis... esto puede tardar un poquito ⏳"):
        result, summary = run_pipeline(input_dir, out_prefix)

    # Guardamos en session_state para que no se pierda al hacer clic en otros botones
    st.session_state.run_info = {
        "input_dir": input_dir,
        "out_prefix": out_prefix,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "summary": summary,
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

# --- Mostrar resultados de la última corrida (si existe) ---
run_info = st.session_state.run_info

if run_info is not None:
    summary = run_info.get("summary", {}) or {}
    st.subheader("Resumen del análisis")

    # Mostrar el nombre completo del dataset arriba
    dataset_name = Path(run_info["input_dir"]).name
    st.markdown(f"<h3>Dataset analizado: <span style='color:#4CAF50;'>{dataset_name}</span></h3>", unsafe_allow_html=True)

    st.markdown("")  # pequeño espacio

    # Métricas rápidas (sin el dataset aquí para que tengan más espacio)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("TF-IDF sospechosos", summary.get("tfidf_suspicious", "–"))
    col2.metric("Jaccard sospechosos", summary.get("jaccard_suspicious", "–"))
    col3.metric("AST sospechosos", summary.get("ast_suspicious", "–"))
    col4.metric("Secuencia sospechosos", summary.get("seq_suspicious", "–"))
    col5.metric("Pares sospechosos (ensemble)", summary.get("ensemble_suspicious", "–"))

    # Reporte en texto
    ensemble_susp = summary.get("ensemble_suspicious", "N/D")
    jaccard_susp = summary.get("jaccard_suspicious", "N/D")
    ast_susp = summary.get("ast_suspicious", "N/D")
    seq_susp = summary.get("seq_suspicious", "N/D")
    tfidf_susp = summary.get("tfidf_suspicious", "N/D")
    total_pairs = summary.get("ensemble_total_pairs", "N/D")

    report_text = f"""Plagiarism Detector - Resumen de análisis
Fecha y hora: {run_info.get("run_time", "N/D")}
Carpeta de entrada: {run_info["input_dir"]}
Prefijo de salida: {run_info["out_prefix"]}

Resultados (ensemble):
- Pares totales analizados: {total_pairs}
- Pares sospechosos (ensemble): {ensemble_susp}

Resultados por modelo:
- TF-IDF: {tfidf_susp} pares sospechosos
- Jaccard: {jaccard_susp} pares sospechosos
- AST estructural: {ast_susp} pares sospechosos
- Secuencia: {seq_susp} pares sospechosos
"""

    st.download_button(
        "Descargar reporte (TXT)",
        report_text.encode("utf-8"),
        file_name=f"{run_info['out_prefix']}_reporte.txt",
        mime="text/plain",
    )

    with st.expander("Ver log completo de ejecución"):
        st.code(run_info["stdout"] or "(sin salida stdout)")

    if run_info["stderr"]:
        with st.expander("Ver errores / advertencias"):
            st.code(run_info["stderr"])

    # Usamos el prefijo de ESA corrida, no lo que esté ahora en el input
    csv_path = Path(f"{run_info['out_prefix']}_ensemble/ensemble_sospechosos.csv")

    if csv_path.exists():
        st.success(f"Pipeline completado. Leyendo: {csv_path}")
        df = pd.read_csv(csv_path)

        # Hacer el CSV más legible: separar el par de archivos
        if "pair_key" in df.columns:
            files = df["pair_key"].str.split("|", n=1, expand=True)
            df["file_a"] = files[0].apply(lambda p: Path(p).name if isinstance(p, str) else p)
            df["file_b"] = files[1].apply(lambda p: Path(p).name if isinstance(p, str) else p)

            cols = ["file_a", "file_b"] + [c for c in df.columns if c not in ["pair_key", "file_a", "file_b"]]
            df = df[cols]

        st.subheader("Pares sospechosos (ensemble)")

        # Renombrar columnas para que sean más legibles en la tabla y el CSV
        rename_map = {
            "file_a": "Archivo A",
            "file_b": "Archivo B",
            "file_i": "ID archivo A",
            "file_j": "ID archivo B",
            "cosine": "Similitud TF-IDF (coseno)",
            "jaccard": "Similitud Jaccard (k-shingles)",
            "ast_cosine": "Similitud estructural (AST)",
            "seq_sim": "Similitud de secuencia",
            "vote_cosine": "Voto TF-IDF",
            "vote_jaccard": "Voto Jaccard",
            "vote_ast": "Voto AST",
            "vote_seq": "Voto Secuencia",
            "votes": "Total de votos",
            "decision": "Decisión final (1 = sospechoso)",
            "score_mean": "Promedio de similitud",
        }

        df_display = df.rename(columns=rename_map)

        st.dataframe(df_display)

        st.download_button(
            "Descargar CSV",
            df_display.to_csv(index=False).encode("utf-8"),
            file_name=f"{run_info['out_prefix']}_ensemble_sospechosos.csv",
            mime="text/csv",
        )
    else:
        st.error(f"No encontré {csv_path}. Revisa las rutas de salida.")
else:
    st.info("Ejecuta el análisis para ver resultados.")