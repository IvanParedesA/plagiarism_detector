import os
import chardet

DEFAULT_EXTS = {".java"}

def read_text(path, max_mb=2):
    if os.path.getsize(path) > max_mb * 1024 * 1024:
        return None
    with open(path, "rb") as f:
        raw = f.read()
    enc = chardet.detect(raw)["encoding"] or "utf-8"
    try:
        return raw.decode(enc, errors="ignore")
    except Exception:
        return raw.decode("utf-8", errors="ignore")

def load_folder(root, exts=None):
    exts = set(exts or DEFAULT_EXTS)
    files, texts = [], []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in exts:
                path = os.path.join(dirpath, name)
                txt = read_text(path)
                if txt:
                    files.append(path)
                    texts.append(txt)
    return files, texts