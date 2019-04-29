QUERY_LANG = "cs"
DOCS_LANG = "hu"

QUERY_LANG_EMBS = "data/embeddings/wiki.cs.mapped.vectors"
DOC_LANG_EMBS = "data/embeddings/wiki.hu.mapped.vectors"
QUERY_LANG_VOCAB = "data/embeddings/wiki.cs.mapped.vocab"
DOC_LANG_VOCAB = "data/embeddings/wiki.hu.mapped.vocab"


MODEL = "models/pretrained/de-en.model"
PREDS_PATH = "output/preds.txt"

PATH_QUERIES = "data/europarl/cs-hu/test/europarl.cs-hu.clean.sents.test.cs"
PATH_DOCS = "data/europarl/cs-hu/test/europarl.cs-hu.clean.sents.test.hu"

STOPWORDS_QL = "data/stopwords/stopwords-" + QUERY_LANG + ".txt" 
STOPWORDS_DL = "data/stopwords/stopwords-" + DOCS_LANG + ".txt"

