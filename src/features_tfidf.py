from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1)
tok_vec  = TfidfVectorizer(analyzer=str.split, ngram_range=(1,2), min_df=1)

def build_tfidf_matrix(texts_norm):
    Xc = char_vec.fit_transform(texts_norm)
    Xt = tok_vec.fit_transform(texts_norm)
    X = hstack([Xc, Xt])
    return X, (char_vec, tok_vec)