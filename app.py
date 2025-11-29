# app.py
import streamlit as st
import subprocess
from pathlib import Path
from datetime import datetime
import re
import pandas as pd
import zipfile
import time

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

# --- Selección de origen del dataset ---
mode = st.radio(
    "Origen del dataset",
    ["Dataset de ejemplo", "Subir ZIP con archivos .java"],
    index=0,
)

uploaded_zip = None

if mode == "Dataset de ejemplo":
    # Mapear etiquetas amigables a rutas reales
    example_options = {
        "Small example (demo rápido)": "datasets/data",
        "IR-Plag - case-01": "datasets/IR-Plag-Dataset/case-01",
        "IR-Plag - case-02": "datasets/IR-Plag-Dataset/case-02",
        "IR-Plag - case-03": "datasets/IR-Plag-Dataset/case-03",
        "IR-Plag - case-04": "datasets/IR-Plag-Dataset/case-04",
        "IR-Plag - case-05": "datasets/IR-Plag-Dataset/case-05",
        "IR-Plag - case-06": "datasets/IR-Plag-Dataset/case-06",
        "IR-Plag - case-07": "datasets/IR-Plag-Dataset/case-07",
        "IR-Plag - case-08": "datasets/IR-Plag-Dataset/case-08",
        "IR-Plag - case-09": "datasets/IR-Plag-Dataset/case-09",
        "IR-Plag - case-10": "datasets/IR-Plag-Dataset/case-10",
        "IR-Plag completo (todos los cases)": "datasets/IR-Plag-Dataset",
    }

    selected_label = st.selectbox(
        "Selecciona un dataset de ejemplo",
        options=list(example_options.keys()),
        index=0,
    )

    # Ruta por defecto según la opción elegida
    default_input_dir = example_options[selected_label]

    # Permitimos editar la ruta en caso de que el usuario quiera ajustar algo
    input_dir = st.text_input(
        "Carpeta de entrada",
        value=default_input_dir,
        help="Ruta en el servidor donde están los archivos .java del dataset.",
    )
else:
    st.info(
        "Sube un archivo .zip que contenga tus archivos .java. "
        "El contenido se descomprimirá en el servidor y se analizará automáticamente."
    )
    uploaded_zip = st.file_uploader("Archivo ZIP", type=["zip"])
    # En modo ZIP, la ruta real de entrada se calculará al descomprimir
    input_dir = ""

out_prefix = st.text_input(
    "Prefijo de salida",
    value="outputs_case01"
)

# --- Botón para lanzar el análisis ---
if st.button("Ejecutar análisis"):
    effective_input_dir = None

    # Determinar la carpeta de entrada según el modo
    if mode == "Dataset de ejemplo":
        effective_input_dir = input_dir
    else:
        # Modo ZIP: validar y descomprimir
        if uploaded_zip is None:
            st.error("Por favor sube un archivo ZIP con tus archivos .java antes de ejecutar el análisis.")
        else:
            uploads_root = Path("uploads")
            uploads_root.mkdir(exist_ok=True)

            raw_base_name = Path(uploaded_zip.name).stem
            # Reemplazar espacios y caracteres raros por guiones bajos
            base_name = re.sub(r'[^a-zA-Z0-9_-]+', '_', raw_base_name)

            ts = int(time.time())
            temp_zip_path = uploads_root / f"{base_name}_{ts}.zip"
            extract_dir = uploads_root / f"{base_name}_{ts}"

            # Guardar el ZIP en disco
            with open(temp_zip_path, "wb") as f:
                f.write(uploaded_zip.getbuffer())

            # Descomprimir el ZIP
            with zipfile.ZipFile(temp_zip_path, "r") as zf:
                zf.extractall(extract_dir)

            # Buscar archivos .java dentro de la carpeta extraída
            java_files = list(Path(extract_dir).rglob("*.java"))

            # Ignorar basura de macOS: carpeta __MACOSX y archivos que empiezan con "._"
            java_files = [
                p for p in java_files
                if "__MACOSX" not in p.parts and not p.name.startswith("._")
            ]

            if not java_files:
                st.error("No se encontraron archivos .java válidos dentro del ZIP. "
                        "Verifica que el ZIP no esté vacío y que contenga archivos .java.")
                effective_input_dir = None
            else:
                # Si todos los .java están en la misma carpeta, usamos esa como carpeta de entrada
                parents = {p.parent for p in java_files}
                if len(parents) == 1:
                    effective_input_dir = str(next(iter(parents)))
                else:
                    # Si hay varias carpetas, usamos la raíz extraída
                    effective_input_dir = str(extract_dir)

    if effective_input_dir:
        with st.spinner("Corriendo análisis... esto puede tardar un poquito ⏳"):
            result, summary = run_pipeline(effective_input_dir, out_prefix)

        # Guardamos en session_state para que no se pierda al hacer clic en otros botones
        st.session_state.run_info = {
            "input_dir": effective_input_dir,
            "out_prefix": out_prefix,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "summary": summary,
            "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

# --- Mostrar resultados de la última corrida (si existe) ---
run_info = st.session_state.get("run_info", None)

if run_info is not None:
    summary = run_info.get("summary", {}) or {}
    st.subheader("Resumen del análisis")

    # Mostrar el nombre completo del dataset arriba
    dataset_name = Path(run_info["input_dir"]).name
    st.markdown(f"<h3>Dataset analizado: <span style='color:#4CAF50;'>{dataset_name}</span></h3>", unsafe_allow_html=True)

    st.markdown("")  # pequeño espacio

    # Métricas rápidas (sin el dataset aquí para que tengan más espacio)
    # Primera fila de métricas
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TF‑IDF – pares sospechosos", summary.get("tfidf_suspicious", "–"))
    c2.metric("Jaccard – pares sospechosos", summary.get("jaccard_suspicious", "–"))
    c3.metric("AST estructural – pares sospechosos", summary.get("ast_suspicious", "–"))
    c4.metric("Secuencia – pares sospechosos", summary.get("seq_suspicious", "–"))

    # Segunda fila para ensemble
    c5 = st.columns(1)[0]
    c5.metric("Pares sospechosos (ensemble)", summary.get("ensemble_suspicious", "–"))

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

    with st.expander("Ver contenido del reporte (vista previa)"):
        st.code(report_text)

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

        # Si el CSV está vacío, avisamos y evitamos hacer split de pair_key
        if df.empty:
            st.info("No se encontraron pares sospechosos en el ensemble (votes >= 3).")
        else:
            # Separar el par de archivos si existe la columna pair_key
            if "pair_key" in df.columns:
                files = df["pair_key"].astype(str).str.split("|", n=1, expand=True)

                # Asegurarnos de que se generaron al menos 2 columnas
                if files.shape[1] >= 2:
                    df["file_a"] = files[0].apply(
                        lambda p: Path(p).name if isinstance(p, str) else p
                    )
                    df["file_b"] = files[1].apply(
                        lambda p: Path(p).name if isinstance(p, str) else p
                    )

                    # Reordenar columnas para mostrar file_a y file_b al inicio
                    cols = ["file_a", "file_b"] + [
                        c for c in df.columns
                        if c not in ["pair_key", "file_a", "file_b"]
                    ]
                    df = df[cols]

            st.subheader("Pares sospechosos (ensemble)")

            # Renombrado de columnas para mayor claridad
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