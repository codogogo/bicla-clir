import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops
import numpy as np
from tensorflow.python.util import nest
from recurrent import rnn_cells

def zeroout_mask_seqlen(seq_lens, max_seq_len):
	mask = []
	for i in range(len(seq_lens)):
		vec = []
		vec.extend([1.0]*seq_lens[i])
		vec.extend([0.0]*(max_seq_len - seq_lens[i]))
		mask.append(vec)
	return np.array(mask, dtype = np.float64)

def zeroout_mask_seqlen_states(seq_lens, max_seq_len, output_size):
	seq_len_tensor = []
	for i in range(len(seq_lens)):
		mat = []
		mat.extend([[1.0]*output_size]*seq_lens[i])
		mat.extend([[0.0]*output_size]*(max_seq_len - seq_lens[i]))
		seq_len_tensor.append(mat)
	return np.array(seq_len_tensor, dtype = np.float64)

class DynamicRNN(object):
	"""
	A general danymic recurrent neural network. Arbitrary cell type can be plugged in. 
	"""

	def __init__(self, state_size, output_size, vocab_size, max_seq_len, scope = "dynrnn", unique_scope_addition = "_1", parent = None):
		self.vocab_size = vocab_size
		self.max_seq_len = max_seq_len
		self.state_size = state_size
		self.output_size = output_size
		self.scope = scope
		self.unique_scope_addition = unique_scope_addition
		self.parent = parent

	def define_model(self, rnn_cell_type, embeddings_layer, batch_size = 50, activation = None, previous_layer = None, backward = False, share_params = None):
		self.batch_size = batch_size
		self.embeddings = embeddings_layer.embeddings
		self.emb_size = embeddings_layer.embedding_size
		self.activation = activation or math_ops.tanh
		self.cell = rnn_cells.create_cell(rnn_cell_type, self.state_size, self.emb_size, self.scope + "__" + str(rnn_cell_type), batch_size = self.batch_size, share_params = share_params)
		self.previous_layer = previous_layer
		self.backward = backward
		
		with tf.name_scope(self.scope + self.unique_scope_addition + '__data-placeholders'):
			if previous_layer is None: 
				self.input_x = tf.placeholder(tf.int32, [self.batch_size, self.max_seq_len], name = self.scope + "__input_x")
			self.sequence_lengths = tf.placeholder(tf.float64, [self.batch_size, self.max_seq_len, self.output_size], name = self.scope + "__seq_lens")
			self.init_state = tf.placeholder(tf.float64, [self.batch_size, self.state_size], name = self.scope + "__init_state")
		
		if previous_layer is None:
			self.batch_embedded = tf.nn.embedding_lookup(self.embeddings, self.input_x)
		else:
			if backward:
				self.batch_embedded = tf.reverse(previous_layer.outputs_static, axis = [1])
			else:
				self.batch_embedded = previous_layer.outputs_static
		
		run_state = self.init_state
		for t in range(self.max_seq_len):
			output, state = self.rnn_step(t, run_state)
			run_state = state
			if t == 0:
				self.outputs_static = output
			else:
				self.outputs_static = tf.concat([self.outputs_static, output], 1)

		self.outputs_static = tf.reshape(self.outputs_static, [-1, self.max_seq_len, self.output_size])
		self.outputs = tf.multiply(self.outputs_static, self.sequence_lengths)
			
		self.l2_loss = self.cell.l2_loss
	
	def rnn_step(self, time, state):
		new_output, new_state = self.cell.add_layer(time, state, self.batch_embedded, self.activation)		
		return new_output, new_state

	def get_feed_dict(self, input_data, sequence_lengths, initial_state, initial_cell_memory = None):
		if self.previous_layer is None:
			fd_mine = { self.input_x : input_data, self.sequence_lengths : zeroout_mask_seqlen_states(sequence_lengths, self.max_seq_len, self.output_size), self.init_state : initial_state }
		else: 
			fd_mine = { self.sequence_lengths : zeroout_mask_seqlen_states(sequence_lengths, self.max_seq_len, self.output_size), self.init_state : initial_state }
		if initial_cell_memory is not None:
			fd_mine.update(self.cell.get_feed_dict(initial_cell_memory))
		return fd_mine

	def get_variable_values(self, session):
		variables = []
		store_prev = self.parent is None and self.previous_layer is not None	
		if store_prev:
			variables.append(self.previous_layer.get_variable_values(session))
		variables.append(self.cell.get_variable_values(session))
		return variables

	def set_variable_values(self, session, values):
		print("Setting variables for DynRNN " + self.scope + ", length: " + str(len(values)))
		store_prev = self.parent is None and self.previous_layer is not None	
		if store_prev:
			self.previous_layer.set_variable_values(session, values[0])
		self.cell.set_variable_values(session, values[1] if store_prev else values[0])