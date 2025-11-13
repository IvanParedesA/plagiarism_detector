import re

# Palabras reservadas de Java
KW = set("""
abstract assert boolean break byte case catch char class const continue default do
double else enum extends final finally float for goto if implements import instanceof
int interface long native new package private protected public return short static
strictfp super switch synchronized this throw throws transient try void volatile while
true false null
""".split())

# Comentarios en Java (// y /* ... */)
RE_COM = re.compile(r"//.?$|/\[\s\S]?\/", re.MULTILINE)

# Cadenas en Java ("" o '')
RE_STR = re.compile(r"(['\"]).*?\1", re.DOTALL)

# Números (enteros o decimales simples)
RE_NUM = re.compile(r"\b\d+(\.\d+)?\b")

# Identificadores de Java
RE_ID = re.compile(r"\b([A-Za-z_$][A-Za-z_0-9$]*)\b")


def normalize(code: str) -> str:
    """
    Normaliza código Java para análisis de similitud / plagio:

    - Reemplaza cadenas por el token 'STR'
    - Elimina comentarios (// y /* ... */)
    - Reemplaza números por el token 'NUM'
    - Reemplaza nombres de variables/métodos/clases por ID_1, ID_2, ...
      (pero conserva palabras reservadas de Java y nombres TODO MAYÚSCULAS
       que normalmente representan constantes)
    - Compacta espacios en blanco en una sola línea limpia
    """

    # 1) Reemplazar todas las cadenas por 'STR'
    code = RE_STR.sub("STR", code)

    # 2) Eliminar comentarios
    code = RE_COM.sub(" ", code)

    # 3) Reemplazar números por 'NUM'
    code = RE_NUM.sub("NUM", code)

    # Diccionario para mapear identificadores originales -> ID_n
    ids = {}

    # Función para reemplazar cada identificador encontrado
    def repl(m):
        w = m.group(1)

        # Si es palabra reservada de Java o está en mayúsculas, no lo cambiamos
        if w in KW or w.isupper():
            return w

        # Si es la primera vez que vemos este identificador, le asignamos un ID nuevo
        if w not in ids:
            ids[w] = f"ID_{len(ids) + 1}"

        # Devolvemos el identificador normalizado
        return ids[w]

    # 4) Aplicar el reemplazo de identificadores
    code = RE_ID.sub(repl, code)

    # 5) Normalizar espacios en blanco y recortar extremos
    code = re.sub(r"\s+", " ", code).strip()

    return code