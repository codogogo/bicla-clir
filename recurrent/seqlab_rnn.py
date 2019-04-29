import tensorflow as tf
from helpers import io_helper
from extensions import tf_extensions

class SequenceLabelingRNN(object):
	"""
	An RNN for sequence labelling (dynrnn/birnn combined with sequence labelling objective). 
	"""

	def __init__(self, num_classes, scope = "seq_lab_rnn1"):
		self.scope = scope
		self.num_classes = num_classes

	def define_model(self, dynamic_rnn_net, l2_reg_factor = 0.1):
		self.output_size = dynamic_rnn_net.output_size
		self.max_seq_len = dynamic_rnn_net.max_seq_len
		self.net = dynamic_rnn_net

		with tf.name_scope(self.scope + "__placeholders"):
			self.input_y = tf.placeholder(tf.float64, [None, self.max_seq_len, self.num_classes], name="input_y")
			self.batch_size = tf.placeholder(tf.int32, name = "batch_size")
			self.dropout = tf.placeholder(tf.float64, name="dropout")
		
		with tf.name_scope(self.scope + "__variables"):
			self.W_softmax = tf.get_variable(self.scope + "__W_softmax", shape=[self.output_size, self.num_classes], initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
			self.b_softmax = tf.Variable(tf.constant(0.1, shape=[self.num_classes], dtype = tf.float64), name = self.scope + "__b_softmax")
		
		#self.outputs = tf.nn.dropout(self.net.outputs, self.dropout)	
		self.preds = tf.nn.dropout(tf.add(tf_extensions.broadcast_matmul(self.net.outputs, self.W_softmax, self.batch_size), self.b_softmax), self.dropout)
			
		with tf.name_scope(self.scope + "__loss"):
			self.l2_loss = dynamic_rnn_net.l2_loss + tf.nn.l2_loss(self.W_softmax) + tf.nn.l2_loss(self.b_softmax)
			self.cross_entropy_losses_all = tf.nn.softmax_cross_entropy_with_logits(logits=self.preds, labels=self.input_y)
			self.pure_loss = tf.reduce_mean(self.cross_entropy_losses_all)
			self.loss = self.pure_loss + l2_reg_factor * self.l2_loss
	
	def define_optimization(self, learning_rate = 1e-3):
		self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

	def get_feed_dict(self, labels, input_data, sequence_lengths, initial_state, dropout, initial_cell_memory = None):
		fd_mine = { self.batch_size : len(input_data), self.input_y : labels, self.dropout : dropout }
		fd_mine.update(self.net.get_feed_dict(input_data, sequence_lengths, initial_state, initial_cell_memory))
		return fd_mine

	def get_variable_values(self, session):
		w_softmax_eval = self.W_softmax.eval(session = session)
		bias_softmax_eval = self.b_softmax.eval(session = session)
		variables = [w_softmax_eval, bias_softmax_eval]
		variables.append(self.net.get_variable_values(session))
		return variables

	def set_variable_values(self, session, values):
		session.run(self.W_softmax.assign(values[0]))
		session.run(self.b_softmax.assign(values[1]))
		self.net.set_variable_values(session, values[2])		

	def get_hyperparameters(self):
		params = {"state_size" : self.net.state_size, 
				  "output_size" : self.net.single_output_size if hasattr(self.net, 'single_output_size') else self.net.output_size, 
				  "vocab_size" : self.net.vocab_size, 
				  "embedding_size" : self.net.emb_size,
				  "max_seq_len" : self.net.max_seq_len, 
				  "num_classes" : self.num_classes, 
				  "scope" : self.scope }
		return params

	def get_model(self, session):
		return [self.get_hyperparameters(), self.get_variable_values(session)]

	def serialize(self, session, path):
		variables = self.get_variable_values(session)
		to_serialize = [self.get_hyperparameters(), self.get_variable_values(session)]
		io_helper.serialize(to_serialize, path)
	
	

	

	