import tensorflow as tf
from tensorflow.python.ops import math_ops

class MultiLayerPerceptron(object):
	"""
	A layer for mapping embeddings from different embedding spaces to the same shared embedding space. 
	"""

	def __init__(self, hidden_layer_sizes, input_size, scope = "mlp_layer1"):
		self.input_size = input_size
		self.hidden_layer_sizes = hidden_layer_sizes
		self.scope = scope

	def define_model(self, activation = None, previous_layer = None):
		self.activation = activation or math_ops.tanh
		with tf.name_scope(self.scope + '__data-placeholders'):
			if previous_layer is None: 
				self.input = tf.placeholder(tf.float64, [None, self.input_size], name = self.scope + "__input_x")
		if previous_layer is not None:
			self.input = previous_layer

		self.Ws = []
		self.biases = []
		with tf.name_scope(self.scope + "__" + 'variables'):
			for i in range(len(self.hidden_layer_sizes)): 
				self.Ws.append(tf.get_variable(self.scope + "__W_" + str(i), shape=[(self.input_size if i == 0 else self.hidden_layer_sizes[i-1]), self.hidden_layer_sizes[i]], initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float64))
				self.biases.append(tf.Variable(tf.constant(0.1, shape=[self.hidden_layer_sizes[i]], dtype = tf.float64), name = self.scope + "__bias_" + str(i)))
				
		layer_outputs = []
		with tf.name_scope(self.scope + "__" + 'computation'):
			data_runner = self.input if previous_layer is None else previous_layer.outputs
			for i in range(len(self.Ws)):
				data_runner = self.activation(tf.nn.xw_plus_b(data_runner, self.Ws[i], self.biases[i]))
				layer_outputs.append(data_runner)
		self.outputs = layer_outputs[-1]
		
		self.l2_loss = 0
		for i in range(len(self.Ws)):
			self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.Ws[i]) + tf.nn.l2_loss(self.biases[i])

	def define_loss(self, loss_function, l2_reg_factor = 0):
		self.input_y = tf.placeholder(tf.float64, [None, self.hidden_layer_sizes[-1]], name = self.scope + "__input_y")
		self.dropout = tf.placeholder(tf.float64, name="dropout")
		
		self.preds = tf.nn.dropout(self.outputs, self.dropout)
		self.pure_loss = loss_function(self.preds, self.input_y)
		self.loss = self.pure_loss + l2_reg_factor * self.l2_loss

	def define_optimization(self, learning_rate):
		self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

	def get_feed_dict(self, input_data, labels = None, dropout = 1.0):
		fd_mine = { self.input : input_data }
		if labels is not None:
			fd_mine.update({ self.input_y : labels, self.dropout : dropout })
		return fd_mine

	def get_variable_values(self, session):
		matrices = []
		biases = []
		for i in range(len(self.Ws)):
			matrices.append(self.Ws[i].eval(session = session))
			biases.append(self.biases[i].eval(session = session))
		return [matrices, biases]

	def set_variable_values(self, session, values):
		if len(values) != 2:
			raise ValueError("Two lists expected, one with values of layer matrices and another with biases")
		for i in range(len(self.Ws)):
			session.run(self.Ws[i].assign(values[0][i]))
			session.run(self.biases[i].assign(values[1][i]))	

	def get_model(self, session):
		return self.get_variable_values(session)