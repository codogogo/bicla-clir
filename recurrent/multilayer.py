import tensorflow as tf
from recurrent import dynrnn
import numpy as np
from recurrent import rnn_cells
from recurrent import bidirectional as bdr
from layers import mapping_layer

class MultilayerRNN(object):
	"""
	A multilayer dynamic RNN (arbitrary number of RNN layers stacked on top of each other). 
	"""

	def __init__(self, state_size, output_size, vocab_size, max_seq_len, scope = "stacked_rnn1", embeddings = (100, None), parent = None):
		self.vocab_size = vocab_size
		self.embeddings = embeddings
		self.emb_size = embeddings[0]
		self.max_seq_len = max_seq_len
		self.state_size = state_size
		self.output_size = output_size
		self.scope = scope
		self.parent = parent

	def define_model(self, num_layers, batch_size = 50, bidirectional = True, cell_type = rnn_cells.CellType.LSTM, back_cell_type = rnn_cells.CellType.LSTM, activation = None, update_embeddings = False, previous_layer = None):
		self.batch_size = batch_size
		self.nets = []
		self.l2_loss = 0
		for i in range(num_layers):
			if bidirectional:
				net = bdr.BidirectionalRNN(self.state_size, self.output_size, self.vocab_size, self.max_seq_len, scope = "bidirect_rnn_layer_" + str(i+1), embeddings = self.embeddings, parent = self)
				net.define_model(cell_type, back_cell_type, batch_size, activation, update_embeddings, previous_layer = (self.nets[-1] if i > 0 else previous_layer))
			else:
				net = dynrnn.DynamicRNN(self.state_size, self.output_size, self.vocab_size, self.max_seq_len, scope = "rnn_layer_" + str(i+1), embeddings = self.embeddings, parent = self)
				net.define_model(cell_type, self.batch_size, activation, update_embeddings, previous_layer = (self.nets[-1] if i > 0 else previous_layer), backward = False)
			self.nets.append(net)
			self.l2_loss = self.l2_loss + net.l2_loss

		self.previous_layer = previous_layer
		if previous_layer is not None:
			self.l2_loss = self.l2_loss + previous_layer.l2_loss

		self.outputs_static = self.nets[-1].outputs_static
		self.outputs = self.nets[-1].outputs

	def get_feed_dict(self, input_data, sequence_lengths, initial_state, initial_cell_memory = None):
		fd_mine = {}
		for net in self.nets:
			fd_mine.update(net.get_feed_dict(input_data, sequence_lengths, initial_state, initial_cell_memory))
		return fd_mine

	def get_variable_values(self, session):
		variables = []
		store_prev = self.parent is None and self.previous_layer is not None	
		if store_prev:
			variables.append(self.previous_layer.get_variable_values(session))
		for i in range(len(self.nets)):
			variables.append(self.nets[i].get_variable_values(session))
		return variables
	
	def set_variable_values(self, session, values):
		store_prev = self.parent is None and self.previous_layer is not None	
		if store_prev:
			self.previous_layer.set_variable_values(session, values[0])
		if len(values) != (len(self.nets) + (1 if store_prev else 0)): 
			raise ValueError("Unexpected number of values when restoring multilayer RNN model. The length of the lists must be equal to the number of model's layers.")
		for i in range(len(values[1:] if store_prev else values)):
			self.nets[i].set_variable_values(session, (values[1:] if store_prev else values)[i])