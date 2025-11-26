# src/preproc.py

import re
from typing import List

# Palabras reservadas de Java (y algunas clases/miembros frecuentes)
KW = set("""
abstract assert boolean break byte case catch char class const continue default do
double else enum extends final finally float for goto if implements import instanceof
int interface long native new package private protected public return short static
strictfp super switch synchronized this throw throws transient try void volatile while
true false null
String System out println
""".split())

# Comentarios en Java: // ... y /* ... */
RE_COM = re.compile(r"//.*?$|/\*[\s\S]*?\*/", re.MULTILINE)

# Cadenas "..." o '...'
RE_STR = re.compile(r'(".*?"|\'.*?\')', re.DOTALL)

# Números (enteros o decimales)
RE_NUM = re.compile(r"\b\d+(\.\d+)?\b")

# Identificadores (variables, clases, métodos)
RE_ID = re.compile(r"\b([A-Za-z_$][A-Za-z_0-9$]*)\b")


def _tokenize(code: str) -> List[str]:
    r"""
    Tokenización sencilla:
    - Palabras (\w+)
    - Símbolos relevantes de Java: llaves, paréntesis, operadores sencillos, etc.
    """
    return re.findall(r"\b\w+\b|[{}();=+*\-/<>&|!%.,]", code)


def normalize_java(code: str) -> str:
    """
    Normaliza código Java para análisis de similitud / plagio:

    - Remueve comentarios.
    - Remueve/reemplaza strings por el token STR.
    - Reemplaza números por el token NUM.
    - Reemplaza identificadores no-keyword por el token ID (α-renaming ligero).
    - Devuelve una secuencia de tokens lista para alimentar a TF-IDF
      (por ejemplo, con tokenizer=str.split).
    """

    # 1) Quitar comentarios
    code = RE_COM.sub(" ", code)

    # 2) Reemplazar strings por STR
    code = RE_STR.sub(" STR ", code)

    # 3) Reemplazar números por NUM
    code = RE_NUM.sub(" NUM ", code)

    # 4) Reemplazar identificadores no-keyword por ID
    def repl_id(match: re.Match) -> str:
        word = match.group(1)
        if word in KW:
            return word
        return "ID"

    code = RE_ID.sub(repl_id, code)

    # 5) Tokenizar
    tokens = _tokenize(code)

    # 6) Unir tokens con espacios (para que el TF-IDF trabaje sobre "palabras")
    return " ".join(tokens)


def normalize_code(code: str, language: str = "java") -> str:
    """
    Punto de entrada genérico por si en el futuro quieres agregar otros lenguajes.
    De momento solo soporta Java.
    """
    lang = language.lower()
    if lang == "java":
        return normalize_java(code)
    # En el futuro podrías agregar: if lang == "python": ...
    raise ValueError(f"Lenguaje no soportado para normalización: {language}")
