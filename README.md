# bicla-clir
Bidirectional Attention Model for Zero-Shot Language Transfer for Cross-Lingual Sentence Retrieval

## Description

BiCLA is a tool for cross-lingual transfer for sentence-level cross-lingual information retrieval (CLIR). It allows you to use a pre-trained neural model for cross-lingual sentence retrieval/matching for one language pair (e.g, EN-DE) and apply it for CLIR over another language pair (e.g., Hungarian queries and Czech sentence collection). 

The repository contains 6 pre-trained sentence-level CLIR models and a shared multilingual embedding space for four languages (EN, DE, HU, CS) -- the model can be applied for sentence CLIR for other languages as well, given that pre-trained word embeddings of that other language are first mapped to the same multilingual embedding space. 

The code and data in this repository accompany the following publication: 

```
@InProceedings{glavavs:2019:ECIR,
  author    = {Glava\v{s}, Goran  and  Vulic\'{c}tajner, Ivan},
  title     = {Zero-Shot Language Transfer for Cross-Lingual Sentence Retrieval with the Bidirectional Attention Model},
  booktitle = {Proceedings of the 41st European Conference on Information Retrieval (ECIR '19)},
  month     = {April},
  year      = {2019},
  address   = {Cologne, Germany},
  publisher = {Springer},
  pages     = {523--538}
}

```

If you're using BiCLA code or data in your work, please cite the above publication. 

## Usage 

The tool allows for using any of the pre-trained CLIR for a new set of "queries" and "documents" (both sentences). The script for predicting the scores is simply run with: 

*python predictor.py *

### Configuration

The following parameters used by the script *predictor.py* need to be defined in the file *config.py*:

-QUERY_LANG: Language of the query sentences
-DOCS_LANG: Language of the collection sentences
-QUERY_LANG_EMBS: Path to the file containing serialized pretrained word embedding vectors of the query language (mapped in the shared multilingual embedding space)
-QUERY_LANG_VOCAB: Path to the file containing pickled vocabulary (for the pretrained word embeddings) of the query language
-DOC_LANG_EMBS: Path to the file containing serialized pretrained word embedding vectors of the collection language (mapped in the shared multilingual embedding space)
-DOC_LANG_VOCAB: Path to the file containing pickled vocabulary (for the pretrained word embeddings) of the collection language
-MODEL: The file containing the pre-trained neural CLIR model

-PATH_QUERIES: Path to the file containing query sentences for prediction (one sentence per line)
-PATH_DOCS: Path to the files containing collection sentences for prediction (one sentence per line)
-PREDS_PATH: Path to which the store the prediction (one score produced for each sentence from PATH_QUERIES paired with each sentence from PATH_DOCS)

-STOPWORDS_QL = Path to the file containing stopwords (one per line) of the query language 
-STOPWORDS_DL = Path to the file containing stopwords (one per line) of the collection language 

### Pretrained models

There are 6 available pre-trained CLIR models (all pairwise combinations of EN, DE, HU, CS), stored in the subdirectory *models/pretrained*. Path to the desired model needs to be set in the variable MODEL in *config.py*

### Multilingual Embedding Space

A shared 4-lingual embedding space has been induced by projecting monolingual DE, HU, and CS embeddings to the EN embedding space. For each language, after the projection to the shared embedding space, the space has been serialized in two files, which can be found in the subdirectory *data/embeddings*

- *wiki.lang.mapped.vectors* is an embedding matrix (serialized numpy array) containing the vectors of words
- *wiki.lang.mapped.vocab* is a vocabulary (pickled python dictionary) mapping the words of a language to indices of the corresponding embedding matrix 

For each language, the embedding matrix and vocabulary dictionary are loded using following commands: 

```
vocab = pickle.load(open(os.path.join(dirname,config.QUERY_LANG_VOCAB),"rb"))
numpy.load(os.path.join(dirname,config.DOC_LANG_EMBS))

```

If you want to make CLIR predictions involving some other language LANG you need to: 

1. Project the monolingual embeddings of LANG to the given multilingual embedding space (i.e., to the EN space)
2. Pickle the vocabulary (python dictionary) and serialize the embedding matrix (2D numpy array)
3. Set the paths in *config.py* to the serialized vocabulary and embedding matrix of LANG 

### Inputs and Predictions

Input sentences ("queries" and "documents") are specified with the parameters PATH_QUERIES and PATH_DOCS in config.py. Both files should contain one sentence per line. The model will predict a score for each pair of sentences (i.e., if PATH_QUERIES has M lines and PATH_DOCS has N lines, the output file, specified with PREDS_PATH, will have M*N scores). The output format is given as follows: 

```
0	0	-0.795404143820669
0	1	-0.9999662864925332
0	2	-0.9994514501139407 (meaning: the score for the sentence in line with index 0 in PATH_QUERIES and sentence with index 2 in PATH_DOCS is -0.9994514501139407)
...

```

Example of the predictions file (generated from 100 "queries" and 100 "documents") is given in *output/preds.txt*. 

### Prerequisites

The BiCLA tool has been (successfully) tested with the following libraries: 

- Tensorflow 1.12.0
- Numpy 1.15.4

## Data

The EuroParl datasets used for training and testing in the above publication are given in *data/europarl*





