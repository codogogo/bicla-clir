from enum import Enum
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops

def create_cell(cell_type, state_size, emb_size, scope, batch_size = 50, bias_init = 0.5, share_params = False):
	if cell_type == CellType.ELMAN:
		cell = ElmanRNNCell(state_size, emb_size, scope = scope)
	elif cell_type == CellType.LSTM:
		cell = LSTMCell(state_size, emb_size, scope = scope, batch_size = batch_size)
	elif cell_type == CellType.GRU:
		cell = GRUCell(state_size, emb_size, scope = scope)
	else:
		raise ValueError("Unknown RNN cell type!")	

	cell.define_model(bias_init = bias_init, share_params = share_params)
	return cell

class CellType(Enum):
	ELMAN = 1,
	LSTM = 2,
	GRU = 3
	
# LSTM CELL
class LSTMCell(object):
	"""
	An LSTM cell for recurrent neural networks.
	"""

	def __init__(self, state_size, emb_size, scope = "lstm_cell", batch_size = 50, unique_scope_addition = "_1"):
		self.batch_size = batch_size
		self.state_size = state_size
		self.emb_size = emb_size
		self.scope = scope
		self.unique_scope_addition = unique_scope_addition

	def define_model(self, bias_init = 0, forget_bias_init = 1.0, share_params = False):
		with tf.name_scope(self.scope + self.unique_scope_addition + '__data-placeholders'):
			self.init_memory = tf.placeholder(tf.float64, [self.batch_size, self.state_size], name = self.scope + "__init_memory")
			self.memory = None

		print("Scope is: " + self.scope)
		print("Share params is: " + str(share_params))
		with tf.variable_scope(self.scope, reuse = share_params):
			self.W_state = tf.get_variable("W_state", initializer=tf.truncated_normal([self.state_size, self.state_size], -0.1, 0.1, dtype = tf.float64), dtype = tf.float64)
			self.W_input = tf.get_variable("W_input", initializer=tf.truncated_normal([self.emb_size, self.state_size], -0.1, 0.1, dtype = tf.float64), dtype = tf.float64)
			self.bias = tf.get_variable("bias", initializer=tf.constant(bias_init, shape=[self.state_size], dtype = tf.float64), dtype = tf.float64)

			self.W_input_ingate = tf.get_variable("W_input_ingate", shape=[self.emb_size, self.state_size], initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
			self.W_state_ingate = tf.get_variable("W_state_ingate", shape=[self.state_size, self.state_size], initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
			self.bias_ingate = tf.get_variable("bias_ingate", initializer=tf.constant(bias_init, shape=[self.state_size], dtype = tf.float64), dtype = tf.float64)

			self.W_input_outgate = tf.get_variable("W_input_outgate", shape=[self.emb_size, self.state_size], initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
			self.W_state_outgate = tf.get_variable("W_state_outgate", shape=[self.state_size, self.state_size], initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
			self.bias_outgate = tf.get_variable("bias_outgate", initializer=tf.constant(bias_init, shape=[self.state_size], dtype = tf.float64), dtype = tf.float64)

			self.W_input_forgetgate = tf.get_variable("W_input_forgetgate", shape=[self.emb_size, self.state_size], initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
			self.W_state_forgetgate = tf.get_variable("W_state_forgetgate", shape=[self.state_size, self.state_size], initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
			self.bias_forgetgate = tf.get_variable("bias_forgetgate", initializer=tf.constant(forget_bias_init, shape=[self.state_size], dtype = tf.float64), dtype = tf.float64)

			# define the l2 loss of the LSTM cell 
			self.l2_loss = tf.nn.l2_loss(self.W_state) + tf.nn.l2_loss(self.W_input)
			self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.W_state_ingate) + tf.nn.l2_loss(self.W_input_ingate)
			self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.W_state_outgate) + tf.nn.l2_loss(self.W_input_outgate) 
			self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.W_state_forgetgate) + tf.nn.l2_loss(self.W_input_forgetgate) 
			self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.bias) + tf.nn.l2_loss(self.bias_ingate) + tf.nn.l2_loss(self.bias_outgate) + tf.nn.l2_loss(self.bias_forgetgate)


	def add_layer(self, index, state, input, activation):
		act_func = activation or math_ops.tanh
		input_slice = tf.reshape(tf.slice(input, [0, index, 0], [-1, 1, self.emb_size]), [-1, self.emb_size])

		z = act_func(tf.add(tf.add(tf.matmul(input_slice, self.W_input), tf.matmul(state, self.W_state)), self.bias))
		i = tf.sigmoid(tf.add(tf.add(tf.matmul(input_slice, self.W_input_ingate), tf.matmul(state, self.W_state_ingate)), self.bias_ingate))
		o = tf.sigmoid(tf.add(tf.add(tf.matmul(input_slice, self.W_input_outgate), tf.matmul(state, self.W_state_outgate)), self.bias_outgate))
		f = tf.sigmoid(tf.add(tf.add(tf.matmul(input_slice, self.W_input_forgetgate), tf.matmul(state, self.W_state_forgetgate)), self.bias_forgetgate))
		
		self.memory = tf.add(tf.multiply(f, self.init_memory if self.memory is None else self.memory), tf.multiply(i, z))
		new_state = tf.multiply(o, act_func(self.memory))

		return new_state, new_state

	def get_feed_dict(self, initial_memory):
		return { self.init_memory : initial_memory }

	def get_variable_values(self, session):
		w_state_eval = self.W_state.eval(session = session)
		w_input_eval = self.W_input.eval(session = session)
		bias_eval = self.bias.eval(session = session)
		
		w_state_ingate_eval = self.W_state_ingate.eval(session = session)
		w_input_ingate_eval = self.W_input_ingate.eval(session = session)
		bias_ingate_eval = self.bias_ingate.eval(session = session)

		w_state_outgate_eval = self.W_state_outgate.eval(session = session)
		w_input_outgate_eval = self.W_input_outgate.eval(session = session)
		bias_outgate_eval = self.bias_outgate.eval(session = session)

		w_state_forgetgate_eval = self.W_state_forgetgate.eval(session = session)
		w_input_forgetgate_eval = self.W_input_forgetgate.eval(session = session)
		bias_forgetgate_eval = self.bias_forgetgate.eval(session = session)		

		return [w_state_eval, w_input_eval, bias_eval, w_state_ingate_eval, w_input_ingate_eval, bias_ingate_eval, w_state_outgate_eval, w_input_outgate_eval, bias_outgate_eval, w_state_forgetgate_eval, w_input_forgetgate_eval, bias_forgetgate_eval]

	def set_variable_values(self, session, values):
		print("Setting LSTM values " + self.scope + ", length: " + str(len(values)))
		if len(values) != 12: 
			raise ValueError("Unexpected number of values when restoring LSTM cell of the RNN model. Expected 12 values (9 matrices and 3 vectors).")
		
		session.run(self.W_state.assign(values[0]))
		session.run(self.W_input.assign(values[1]))
		session.run(self.bias.assign(values[2]))

		session.run(self.W_state_ingate.assign(values[3]))
		session.run(self.W_input_ingate.assign(values[4]))
		session.run(self.bias_ingate.assign(values[5]))

		session.run(self.W_state_outgate.assign(values[6]))
		session.run(self.W_input_outgate.assign(values[7]))
		session.run(self.bias_outgate.assign(values[8]))

		session.run(self.W_state_forgetgate.assign(values[9]))
		session.run(self.W_input_forgetgate.assign(values[10]))
		session.run(self.bias_forgetgate.assign(values[11]))

# GRU CELL
class GRUCell(object):
	"""
	A GRU cell for recurrent neural networks.
	"""

	def __init__(self, state_size, emb_size, scope = "gru_cell"):
		self.state_size = state_size
		self.emb_size = emb_size
		self.scope = scope

	def define_model(self, bias_init = 0.5, share_params = False):
		with tf.variable_scope(self.scope, reuse = share_params):
			self.W_state = tf.get_variable("W_state", shape=[self.state_size, self.state_size], initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
			self.W_input = tf.get_variable("W_input", shape=[self.emb_size, self.state_size], initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
			self.bias = tf.get_variable("bias", initializer=tf.constant(bias_init, shape=[self.state_size], dtype = tf.float64), dtype = tf.float64)

			self.W_input_rgate = tf.get_variable("W_input_rgate", shape=[self.emb_size, self.state_size], initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
			self.W_state_rgate = tf.get_variable("W_state_rgate", shape=[self.state_size, self.state_size], initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
			self.bias_rgate = tf.get_variable("bias_rgate", initializer=tf.constant(bias_init, shape=[self.state_size], dtype = tf.float64), dtype = tf.float64)

			self.W_input_zgate = tf.get_variable("W_input_zgate", shape=[self.emb_size, self.state_size], initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
			self.W_state_zgate = tf.get_variable("W_state_zgate", shape=[self.state_size, self.state_size], initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
			self.bias_zgate = tf.get_variable("bias_zgate", initializer=tf.constant(bias_init, shape=[self.state_size], dtype = tf.float64), dtype = tf.float64)

			self.l2_loss = tf.nn.l2_loss(self.W_state) + tf.nn.l2_loss(self.W_input)
			self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.W_state_rgate) + tf.nn.l2_loss(self.W_input_rgate)
			self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.W_state_zgate) + tf.nn.l2_loss(self.W_input_zgate) 
			self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.bias) + tf.nn.l2_loss(self.bias_rgate) + tf.nn.l2_loss(self.bias_zgate)

	def add_layer(self, index, state, input, activation):
		act_func = activation or math_ops.tanh
		input_slice = tf.reshape(tf.slice(input, [0, index, 0], [-1, 1, self.emb_size]), [-1, self.emb_size])
		
		r = tf.sigmoid(tf.add(tf.add(tf.matmul(input_slice, self.W_input_rgate), tf.matmul(state, self.W_state_rgate)), self.bias_rgate))
		z = tf.sigmoid(tf.add(tf.add(tf.matmul(input_slice, self.W_input_zgate), tf.matmul(state, self.W_state_zgate)), self.bias_zgate))
		inv_state = act_func(tf.add(tf.add(tf.matmul(input_slice, self.W_input), tf.matmul(tf.multiply(r, state), self.W_state)), self.bias))
		
		new_state = tf.add(tf.multiply(z, state), tf.multiply(tf.subtract(tf.constant(1.0, shape=[self.state_size], dtype = tf.float64), z), inv_state))
		return new_state, new_state

	def get_variable_values(self, session):
		w_state_eval = self.W_state.eval(session = session)
		w_input_eval = self.W_input.eval(session = session)
		bias_eval = self.bias.eval(session = session)
		
		w_state_rgate_eval = self.W_state_rgate.eval(session = session)
		w_input_rgate_eval = self.W_input_rgate.eval(session = session)
		bias_rgate_eval = self.bias_rgate.eval(session = session)

		w_state_zgate_eval = self.W_state_zgate.eval(session = session)
		w_input_zgate_eval = self.W_input_zgate.eval(session = session)
		bias_zgate_eval = self.bias_zgate.eval(session = session)		

		return [w_state_eval, w_input_eval, bias_eval, w_state_rgate_eval, w_input_rgate_eval, bias_rgate_eval, w_state_zgate_eval, w_input_zgate_eval, bias_zgate_eval]

	def set_variable_values(self, session, values):
		if len(values) != 9: 
			raise ValueError("Unexpected number of values when restoring GRU cell of the RNN model. Expected 9 values (6 matrices and 3 vectors).")
		
		session.run(self.W_state.assign(values[0]))
		session.run(self.W_input.assign(values[1]))
		session.run(self.bias.assign(values[2]))

		session.run(self.W_state_rgate.assign(values[3]))
		session.run(self.W_input_rgate.assign(values[4]))
		session.run(self.bias_rgate.assign(values[5]))

		session.run(self.W_state_zgate.assign(values[6]))
		session.run(self.W_input_zgate.assign(values[7]))
		session.run(self.bias_zgate.assign(values[8]))

# SIMPLE ELMAN CELL
class ElmanRNNCell(object):
	"""
	A simple Elman cell for recurrent neural networks.
	"""

	def __init__(self, state_size, emb_size, scope = "elman_cell"):
		self.state_size = state_size
		self.emb_size = emb_size
		self.scope = scope

	def define_model(self, bias_init = 0.5, share_params = False):
		with tf.variable_scope(self.scope, reuse = share_params):
			self.W_state = tf.get_variable("W_state", shape=[self.state_size, self.state_size], initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
			self.W_input = tf.get_variable("W_input", shape=[self.emb_size, self.state_size], initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
			self.bias = tf.get_variable("bias", initializer=tf.constant(bias_init, shape=[self.state_size], dtype = tf.float64), dtype = tf.float64)

			self.l2_loss = tf.nn.l2_loss(self.W_state) + tf.nn.l2_loss(self.W_input) + tf.nn.l2_loss(self.bias)

	def add_layer(self, index, state, input, activation):
		act_func = activation or math_ops.tanh
		input_slice = tf.reshape(tf.slice(input, [0, index, 0], [-1, 1, self.emb_size]), [-1, self.emb_size])

		new_state = act_func(tf.add(tf.nn.xw_plus_b(input_slice, self.W_input, self.bias), tf.matmul(state, self.W_state)))
		return new_state, new_state

	def get_variable_values(self, session):
		w_state_eval = self.W_state.eval(session = session)
		w_input_eval = self.W_input.eval(session = session)
		bias_eval = self.bias.eval(session = session)
		return [w_state_eval, w_input_eval, bias_eval]

	def set_variable_values(self, session, values):
		if len(values) != 3: 
			raise ValueError("Unexpected number of values when restoring Elman cell of the RNN model. Expected two matrices and a bias vector.")

		session.run(self.W_state.assign(values[0]))
		session.run(self.W_input.assign(values[1]))
		session.run(self.bias.assign(values[2]))