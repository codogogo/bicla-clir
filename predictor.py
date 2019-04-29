import numpy as np
from recurrent import bidirectional
from recurrent import attention
from recurrent import rnn_cells
from embeddings import text_embeddings
from helpers import io_helper
from helpers import data_shaper
from models import sentpair_model
import tensorflow as tf
from ml import loss_functions
from ml import trainer
from recurrent import dynrnn
from sys import stdin
from helpers import data_helper
import itertools
import random
from layers import embeddings_layer
#from evaluation import appleveleval
import sys
import math
import config
import os
import pickle
import numpy as np

dirname = os.path.dirname(__file__)

########################################################################################
# loading the pre-trained model
########################################################################################

print("Forwarded arguments: ")
model_name = os.path.join(dirname, config.MODEL)
print("Model name: " + str(model_name))

lang_query = config.QUERY_LANG
lang_doc = config.DOCS_LANG
preds_path = config.PREDS_PATH

print("Prediction language pair: " + lang_query + " " + lang_doc)

print("Deserializing the model...")
model_serialization_path = model_name
hyperparams, variables = io_helper.deserialize(model_serialization_path)
print("Hyperparameters: ")
print(hyperparams)

hyp_first_enc, hyp_second_encoder, batch_size, same_encoder, cross_attention, self_attention, share_cross_attention, share_intra_attention, bilinear_product_score = hyperparams
state_size, max_len, forward_cell_type, backward_cell_type = hyp_first_enc

########################################################################################
# loading/merging word embeddings
########################################################################################

vocab_q = pickle.load(open(os.path.join(dirname,config.QUERY_LANG_VOCAB),"rb"))
vectors_q = np.load(os.path.join(dirname,config.QUERY_LANG_EMBS))
norms_q = vectors_q / np.transpose([np.linalg.norm(vectors_q, 2, 1)])

vocab_d = pickle.load(open(os.path.join(dirname,config.DOC_LANG_VOCAB),"rb"))
vectors_d = np.load(os.path.join(dirname,config.DOC_LANG_EMBS))
norms_d = vectors_d / np.transpose([np.linalg.norm(vectors_d, 2, 1)])

print("Loading embeddings of both languages...")
t_embeddings = text_embeddings.Embeddings()
t_embeddings.lang_embeddings['en'] = vectors_d
t_embeddings.lang_embeddings['de'] = vectors_q
t_embeddings.lang_vocabularies['en'] = vocab_d
t_embeddings.lang_vocabularies['de'] = vocab_q
t_embeddings.lang_emb_norms['en'] = norms_d
t_embeddings.lang_emb_norms['de'] = norms_q

#t_embeddings.load_embeddings(, 200000, language = 'en', print_loading = True, skip_first_line = True)
#t_embeddings.load_embeddings(os.path.join(dirname,config.QUERY_LANG_EMBS), 200000, language = 'de', print_loading = True, skip_first_line = True, special_tokens = ["<PAD/>", "<NUM/>"])
t_embeddings.merge_embedding_spaces(["en", "de"], 300, merge_name = "default", lang_prefix_delimiter = "__", special_tokens = ["<PAD/>", "<NUM/>"])

vocabulary_size = len(t_embeddings.lang_vocabularies["default"])
embeddings = t_embeddings.lang_embeddings["default"].astype(np.float64)
embedding_size = 300
t_embeddings.inverse_vocabularies()

########################################################################################
# loading/preparing data
########################################################################################

stopwords_de = io_helper.load_lines(os.path.join(dirname,config.STOPWORDS_QL))
stopwords_en = io_helper.load_lines(os.path.join(dirname,config.STOPWORDS_DL))

punctuation = [".", ",", "!", ":", "?", ";", "-", ")", "(", "[", "]", "{", "}", "...", "/", "\\", "''", "\"", "'"]

path_language_query_test = os.path.join(dirname,config.PATH_QUERIES)
path_language_doc_test = os.path.join(dirname,config.PATH_DOCS)

sent_lines_de_test = io_helper.load_lines(path_language_query_test)
sent_lines_en_test = io_helper.load_lines(path_language_doc_test)

sent_index_de_test = {i : sent_lines_de_test[i] for i in range(len(sent_lines_de_test))}
sent_index_en_test = {i : sent_lines_en_test[i] for i in range(len(sent_lines_en_test))}

print("Length sent_index_de_test: " + str(len(sent_index_de_test)))
print("Length sent_index_en_test: " + str(len(sent_index_en_test)))

print("Cleaning dictionaries DE test...")
for k in sent_index_de_test:
	sent_index_de_test[k] = data_helper.clean_str(sent_index_de_test[k]).split()
print("Cleaning dictionaries EN test...")
for k in sent_index_en_test:
	sent_index_en_test[k] = data_helper.clean_str(sent_index_en_test[k]).split()

print("Embedding indices lookup DE test...")
sent_index_embedded_de_test = data_shaper.prep_embeddings_lookup(sent_index_de_test, t_embeddings, stopwords = stopwords_de, punctuation = punctuation, lang = 'default', text_lang_prefix = 'de__', min_tokens = 2, num_token = "<NUM/>") 
print(len(sent_index_embedded_de_test))
print("Embedding indices lookup EN test...")
sent_index_embedded_en_test = data_shaper.prep_embeddings_lookup(sent_index_en_test, t_embeddings, stopwords = stopwords_en, punctuation = punctuation, lang = 'default', text_lang_prefix = 'en__', min_tokens = 2, num_token = "<NUM/>") 
print(len(sent_index_embedded_en_test))

print("Padding sentences...")
lengths_de_test, ml1 = data_shaper.pad_sequences(sent_index_embedded_de_test, t_embeddings.lang_vocabularies["default"]["<PAD/>"], max_len = max_len)
lengths_en_test, ml2 = data_shaper.pad_sequences(sent_index_embedded_en_test, t_embeddings.lang_vocabularies["default"]["<PAD/>"], max_len = max_len)

print("Preparing test examples...")
data_test = []
for k1 in sent_index_embedded_de_test:
	for k2 in sent_index_embedded_en_test:
		data_test.append((k1, k2, sent_index_embedded_de_test[k1], lengths_de_test[k1], sent_index_embedded_en_test[k2], lengths_en_test[k2], 1.0 if k1 == k2 else -1.0))

print("Size of the test set: " + str(len(data_test)))
test_keys = list(sent_index_embedded_en_test.keys())

########################################################################################
# model definition
########################################################################################

print("Defining embeddings layer...")
emb_layer = embeddings_layer.EmbeddingLayer(None, embeddings, embedding_size, update_embeddings = False)

print("Defining first encoder...")
first_enc_net = bidirectional.BidirectionalRNN(state_size, state_size, vocabulary_size, max_len, scope = "bidirrnn_encoder" + ("" if same_encoder else "_first"), unique_scope_addition = "_1")
first_enc_net.define_model(rnn_cells.CellType.LSTM, rnn_cells.CellType.LSTM, emb_layer, batch_size = batch_size, share_params = None)

print("Defining second encoder...")
second_enc_net = bidirectional.BidirectionalRNN(state_size, state_size, vocabulary_size, max_len, scope = "bidirrnn_encoder" + ("" if same_encoder else "_second"), unique_scope_addition = "_2")
second_enc_net.define_model(rnn_cells.CellType.LSTM, rnn_cells.CellType.LSTM, emb_layer, batch_size = batch_size, share_params = same_encoder)

print("Defining sentence pair model...")
sp_model = sentpair_model.SentencePairClassifier(first_enc_net, second_enc_net, same_encoder = same_encoder)
sp_model.define_model(cross_attention = cross_attention, self_attention = self_attention, cross_attention_bilinear = True, activation = tf.nn.tanh, batch_size = batch_size, bilinear_product_score = True, share_cross_attention = share_cross_attention, share_intra_attention = share_intra_attention)

print("Initializing session...")
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

print("Setting model variables to trained values...")
sp_model.set_variable_values(session, variables)

def build_feed_dict_func(model, data, config = None, predict = False):
	inds_de, inds_en, sents_first, lengths_first, sents_second, lengths_second, y = zip(*data)
	zeroout_mask_first = dynrnn.zeroout_mask_seqlen(lengths_first, max_len)
	zeroout_mask_second = dynrnn.zeroout_mask_seqlen(lengths_second, max_len)

	fd_first_enc = model.first_encoder.get_feed_dict(sents_first, lengths_first)
	fd_second_enc = model.second_encoder.get_feed_dict(sents_second, lengths_second)
	fd_sp_model = model.get_feed_dict(None, zeroout_mask_first, zeroout_mask_second)

	fd = {}
	fd.update(fd_first_enc)
	fd.update(fd_second_enc)
	fd.update(fd_sp_model)

	#print(y)
	#stdin.readline()
	return fd, y

def print_attention_matrix(matrix):
	string = "["	
	print(matrix.shape)
	for i in range(len(matrix)):
		string += "["
		if matrix[i][0] == matrix[i][1] and matrix[i][0] == matrix[i][2]:
			break
		for j in range(len(matrix[i])):
			if matrix[i][j] == 0:
				break
			else:
				if j > 0:
					string += ", "
				string += '{:.4f}'.format(matrix[i][j])
		string += ("]]" if i == (len(matrix) - 1) else "],\n")
	
	print(string)

path_to_write_test_predictions = preds_path
def retrieval_eval(model, session):
	num_batches = int(len(data_test) / batch_size)
	print(num_batches)
	preds = []
	for i in range(num_batches):
		print("Evaluation TEST, batch " + str(i+1) + " of " + str(num_batches))
		batch_data = data_test[i * batch_size : (i+1) * batch_size]
		fd, ys = build_feed_dict_func(model, batch_data, None, predict = True)
		preds_batch = model.preds.eval(session = session, feed_dict = fd)

		if len(preds_batch) != len(batch_data):
			raise ValueError("GG: Unexpected number of predictions returned!")
		preds.extend(preds_batch)
	
	inds_de_test, inds_en_test, sents_first_test, lengths_first_test, sents_second_test, lengths_second_test, y_test = zip(*data_test)
	predictions_with_indices = list(zip(inds_de_test, inds_en_test, preds))
	
	io_helper.write_list_tuples_separated(os.path.join(dirname,path_to_write_test_predictions), predictions_with_indices, delimiter = '\t')
	print("Predictions written to the output file: " + config.PREDS_PATH)

	ranks = []
	print("Computing the MAP / MRR (only meaningful for mate retrieval where sentences at same lines are relevant for each other)")
	for i in test_keys:
		subset_scores = [x for x in predictions_with_indices if x[0] == i]
					
		subset_scores_sorted = sorted(subset_scores, key=lambda x: x[2])
		subset_scores_sorted.reverse()
		match = [x for x in subset_scores_sorted if x[0] == i and x[1] == i]
			
		if len(match) == 1:
			rank = subset_scores_sorted.index(match[0]) + 1
		elif len(match) == 0:
			raise ValueError("Match for a positive pair not found!")
		else:
			raise ValueError("More than one match obtained for a positive pair!")
		ranks.append(rank)
	sum_prec = sum([(1.0 / x if x >= 1 else 0) for x in ranks])
	map = sum_prec / len(ranks)
	print("MAP: " + str(map))

retrieval_eval(sp_model, session)

