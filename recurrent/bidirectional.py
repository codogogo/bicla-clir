import tensorflow as tf
from recurrent import dynrnn
import numpy as np
from recurrent import rnn_cells

class BidirectionalRNN(object):
	"""
	A bidirectional dynamic RNN (two dynrnn objects combined). 
	"""

	def __init__(self, state_size, output_size, vocab_size, max_seq_len, scope = "bidirect_rnn", unique_scope_addition = "_1", parent = None):
		self.vocab_size = vocab_size
		self.max_seq_len = max_seq_len
		self.state_size = state_size
		self.output_size = output_size
		self.scope = scope
		self.unique_scope_addition = unique_scope_addition
		self.parent = parent

	def define_model(self, forward_rnn_cell, backward_rnn_cell, embeddings_layer, batch_size = 50, activation = None, previous_layer = None, share_params = False):
		self.forward_cell_type = forward_rnn_cell
		self.backward_cell_type = backward_rnn_cell	

		self.batch_size = batch_size
		self.previous_layer = previous_layer
		
		with tf.name_scope(self.scope + self.unique_scope_addition + "__data-placeholders"):
			self.sequence_lengths = tf.placeholder(tf.float64, [self.batch_size, self.max_seq_len, self.output_size], name="seq_lens")
			
		self.forward_rnn = dynrnn.DynamicRNN(self.state_size, self.output_size, self.vocab_size, self.max_seq_len, scope = self.scope + "__forward_net", unique_scope_addition = self.unique_scope_addition, parent = self)
		self.forward_rnn.define_model(forward_rnn_cell, embeddings_layer, batch_size = self.batch_size, activation = activation, previous_layer = previous_layer, backward = False, share_params = share_params)

		self.backward_rnn = dynrnn.DynamicRNN(self.state_size, self.output_size, self.vocab_size, self.max_seq_len, scope = self.scope + "__backward_net", unique_scope_addition = self.unique_scope_addition, parent = self)
		self.backward_rnn.define_model(backward_rnn_cell, embeddings_layer, batch_size = self.batch_size, activation = activation, previous_layer = previous_layer, backward = True, share_params = share_params)

		with tf.variable_scope(self.scope, reuse = share_params):
			self.W_reduce = tf.get_variable("W_reduce", shape=[2 * self.output_size, self.output_size], initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float64)

			W_reduce_tiled = tf.tile(tf.expand_dims(self.W_reduce, axis = 0), [self.batch_size, 1, 1])

			self.outputs_static = tf.matmul(tf.concat([self.forward_rnn.outputs_static, tf.reverse(self.backward_rnn.outputs_static, axis = [1])], 2), W_reduce_tiled)
			self.outputs = tf.multiply(self.outputs_static, self.sequence_lengths)
			
			self.l2_loss = self.forward_rnn.l2_loss + self.backward_rnn.l2_loss
			#if previous_layer is not None:
			#	self.l2_loss = self.l2_loss + previous_layer.l2_loss


	def get_feed_dict(self, input_data, sequence_lengths, initial_state = None, initial_cell_memory = None):
		if initial_state is None:
			initial_state = np.zeros(shape = (len(input_data), self.state_size))
		if initial_cell_memory is None and (self.forward_cell_type == rnn_cells.CellType.LSTM or self.backward_cell_type == rnn_cells.CellType.LSTM):
			initial_cell_memory = np.zeros(shape = (len(input_data), self.state_size))
		fd_mine = { self.sequence_lengths : dynrnn.zeroout_mask_seqlen_states(sequence_lengths, self.max_seq_len, self.output_size) }
		fd_mine.update(self.forward_rnn.get_feed_dict(input_data, sequence_lengths, initial_state, initial_cell_memory))
		fd_mine.update(self.backward_rnn.get_feed_dict(np.flip(input_data, axis = 1), sequence_lengths, initial_state, initial_cell_memory))
		return fd_mine

	def get_variable_values(self, session):
		variables = []
		store_prev = self.parent is None and self.previous_layer is not None	
		if store_prev:
			variables.append(self.previous_layer.get_variable_values(session))
		variables.append(self.W_reduce.eval(session = session))
		variables.append(self.forward_rnn.get_variable_values(session))
		variables.append(self.backward_rnn.get_variable_values(session))
		return variables

	def set_variable_values(self, session, values):
		print("Setting variables for the BI-RNN " + self.scope + ", length: " + str(len(values)))
		store_prev = self.parent is None and self.previous_layer is not None	
		if store_prev:
			self.previous_layer.set_variable_values(session, values[0])
		if len(values) != (3 + (1 if store_prev else 0)): 
			raise ValueError("Unexpected number of values when restoring bidirectional RNN model. A length reduction matrix and two lists containing parameters of forward and backward network are expected (len(values) should be 3).")

		session.run(self.W_reduce.assign(values[0 + (1 if store_prev else 0)]))
		self.forward_rnn.set_variable_values(session, values[1 + (1 if store_prev else 0)])
		self.backward_rnn.set_variable_values(session, values[2 + (1 if store_prev else 0)])
	
	def get_hyperparameters(self):
		return [self.state_size, self.max_seq_len, self.forward_cell_type, self.backward_cell_type]
		

		
		
		