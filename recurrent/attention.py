import tensorflow as tf
from helpers import io_helper
from extensions import tf_extensions
from sys import stdin

class SequenceLevelAttention(object):
	"""
	The Bahdanau attention mechanism for RNNs. This attention mechanisms is sequence-level, which means that there is one weighting scheme learned per instance (for entire sequence)
    Merely learns how to combine states at different times (different positions in sequence) into a single task-dependent vector of the whole sequence. Suitable for (whole-sequence)-level classification tasks
	"""

	def __init__(self, encoder, scope = "inter_attention", unique_scope_addition = "_1"):
		self.scope = scope
		self.unique_scope_addition = unique_scope_addition
		self.net = encoder

	def define_model(self, activation = None, batch_size = 50, share_params = None):
		self.batch_size = batch_size
		self.activation = activation or tf.nn.tanh
		self.output_size = self.net.output_size
		self.max_seq_len = self.net.max_seq_len

		with tf.name_scope(self.scope + self.unique_scope_addition + "__placeholders"):
			self.zero_out_mask = tf.placeholder(tf.float64, [self.batch_size, self.max_seq_len], name = "__zero_out_mask_enc")

		with tf.variable_scope(self.scope, reuse = share_params):
			self.W_attend = tf.get_variable("W_attend", shape=[self.output_size, 1], initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
			self.b_attend = tf.get_variable("b_attend", initializer = tf.constant(0.1, shape=[1], dtype = tf.float64), dtype = tf.float64)
			
		self.attentions = tf.reshape(self.activation(tf.add(tf_extensions.broadcast_matmul(self.net.outputs, self.W_attend, none_dim_replacement = self.batch_size), self.b_attend)), [self.batch_size, self.max_seq_len])
		self.attentions_zeroed = tf.multiply(self.attentions, self.zero_out_mask)
		self.attentions_sftmx = tf_extensions.softmax_ignore_zeros(self.attentions_zeroed, none_dim_replacement = self.batch_size)

		self.attentions_zeroed_reshaped = tf.reshape(self.attentions_sftmx, [self.batch_size, self.max_seq_len, 1])
		self.attentions_tiled = tf.tile(self.attentions_zeroed_reshaped, [1, 1, self.output_size])
			
		self.outputs = tf.reduce_sum(tf.multiply(self.attentions_tiled, self.net.outputs), axis = 1)
		self.l2_loss = tf.nn.l2_loss(self.W_attend) + tf.nn.l2_loss(self.b_attend)
			
	def get_feed_dict(self, zero_out_mask):
		fd_mine = { self.zero_out_mask : zero_out_mask}
		return fd_mine

	def get_variable_values(self, session):
		w_attend_eval = self.W_attend.eval(session = session)
		bias_attend_eval = self.b_attend.eval(session = session)
		variables = [w_attend_eval, bias_attend_eval]
		return variables

	def set_variable_values(self, session, values):
		session.run(self.W_attend.assign(values[0]))
		session.run(self.b_attend.assign(values[1]))


class StateLevelAttention(object):
	"""
	The Bahdanau attention mechanism for RNNs. This attention mechanisms is state-level, i.e., a separate attention scheme is computed for every state of the sequence.  
    Learns how to combine state output vectors of one sequence for building the attended-state representation of another sequence. 
	The sequence over which we attend may be the same sequence for which we recompute the states based on attention (this would be attention for sequence labelling tasks).
	If the two sequences are different, this is attention for seq2seq (i.e., states of the encoder inform the attention for the states of the decoder)
	"""

	def __init__(self, encoder_rnn, decoder_rnn, scope = "rnn_attention_state", unique_scope_addition = "_1"):
		self.scope = scope
		self.unique_scope_addition = unique_scope_addition
		self.encoder_net = encoder_rnn
		self.decoder_net = decoder_rnn

	def define_model(self, bilinear_combination = False, activation = None, batch_size = 50, share_params = None):
		self.batch_size = batch_size
		self.activation = activation or tf.nn.tanh
		self.encoder_output_size = self.encoder_net.output_size
		self.decoder_output_size = self.decoder_net.output_size
		self.encoder_max_seq_len = self.encoder_net.max_seq_len
		self.decoder_max_seq_len = self.decoder_net.max_seq_len	
		self.bilinear_combination = bilinear_combination

		# needed if fed to another inter-attention layer, after the cross-attention
		self.max_seq_len = self.decoder_max_seq_len
		self.output_size = self.encoder_output_size
				
		with tf.name_scope(self.scope + self.unique_scope_addition + "__placeholders"):
			self.zero_out_mask_encoder = tf.placeholder(tf.float64, [self.batch_size, self.encoder_max_seq_len], name = self.scope + "__zero_out_mask_enc")
			self.zero_out_mask_decoder = tf.placeholder(tf.float64, [self.batch_size, self.decoder_max_seq_len], name = self.scope + "__zero_out_mask_dec")
		
		with tf.variable_scope(self.scope, reuse = share_params):
			if bilinear_combination:
				self.W_attend = tf.get_variable(name = "W_attend", shape=[self.decoder_output_size, self.encoder_output_size], initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
			else:
				self.W_attend = tf.get_variable(name = "W_attend", shape=[self.encoder_output_size + self.decoder_output_size, 1], initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
			self.b_attend = tf.get_variable(name = "b_attend", initializer = tf.constant(0.1, shape=[1], dtype = tf.float64), dtype = tf.float64)		

		instances_encoder = tf.unstack(self.encoder_net.outputs)
		instances_decoder = tf.unstack(self.decoder_net.outputs)
		zero_out_vectors_encoder = tf.unstack(self.zero_out_mask_encoder)
		zero_out_vectors_decoder = tf.unstack(self.zero_out_mask_decoder)

		self.attentions = []
		instances_attended = []
		for i in range(len(instances_encoder)):
			print("Instance: " + str(i) + " of " + str(len(instances_encoder)))
			if self.bilinear_combination:
				lin_part = tf.matmul(instances_decoder[i], self.W_attend)	
				self.atts_raw = self.activation(tf.add(tf.matmul(lin_part, tf.transpose(instances_encoder[i])), self.b_attend))
				self.atts_zeroed = tf.multiply(self.atts_raw, tf.tile(tf.reshape(zero_out_vectors_encoder[i], [1, self.encoder_max_seq_len]), [self.decoder_max_seq_len, 1]))
				self.atts = tf_extensions.softmax_ignore_zeros(self.atts_zeroed) 
				self.attentions.append(self.atts)
				self.atts_reshaped = tf.tile(tf.reshape(self.atts, [self.decoder_max_seq_len, self.encoder_max_seq_len, 1]), [1, 1, self.encoder_output_size])
				self.attended_decoder_matrix = tf.reduce_sum(tf.multiply(self.atts_reshaped, instances_encoder[i]), axis = 1)
				self.attended_decoder_matrix_zeroed = tf.multiply(self.attended_decoder_matrix, tf.tile(tf.transpose(tf.reshape(zero_out_vectors_decoder[i], [1, self.decoder_max_seq_len])), [1, self.encoder_output_size]))
				
				instances_attended.append(self.attended_decoder_matrix_zeroed)
			else:		
				raise NotImplementedError()
		self.outputs = tf.stack(instances_attended)
		self.l2_loss = tf.nn.l2_loss(self.W_attend) + tf.nn.l2_loss(self.b_attend)

	def define_model_iterative(self, bilinear_combination = False, activation = None, batch_size = 50, share_params = None):
		self.batch_size = batch_size
		self.activation = activation or tf.nn.tanh
		self.encoder_output_size = self.encoder_net.output_size
		self.decoder_output_size = self.decoder_net.output_size
		self.encoder_max_seq_len = self.encoder_net.max_seq_len
		self.decoder_max_seq_len = self.decoder_net.max_seq_len	
		self.bilinear_combination = bilinear_combination

		# needed if fed to another inter-attention layer, after the cross-attention
		self.max_seq_len = self.decoder_max_seq_len
		self.output_size = self.encoder_output_size
				
		with tf.name_scope(self.scope + self.unique_scope_addition + "__placeholders"):
			self.zero_out_mask_encoder = tf.placeholder(tf.float64, [self.batch_size, self.encoder_max_seq_len], name = self.scope + "__zero_out_mask_enc")
			self.zero_out_mask_decoder = tf.placeholder(tf.float64, [self.batch_size, self.decoder_max_seq_len], name = self.scope + "__zero_out_mask_dec")
		
		with tf.variable_scope(self.scope, reuse = share_params):
			if bilinear_combination:
				self.W_attend = tf.get_variable(name = "W_attend", shape=[self.decoder_output_size, self.encoder_output_size], initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
			else:
				self.W_attend = tf.get_variable(name = "W_attend", shape=[self.encoder_output_size + self.decoder_output_size, 1], initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
			self.b_attend = tf.get_variable(name = "b_attend", initializer = tf.constant(0.1, shape=[1], dtype = tf.float64), dtype = tf.float64)		
		
		self.zeroed_att_enc2dec_vectors = []
		self.sftmxd_att_enc2dec_vectors = []
		self.attended_dec_vector_state = None 	
		
		instances_encoder = tf.unstack(self.encoder_net.outputs)
		instances_decoder = tf.unstack(self.decoder_net.outputs)
		zero_out_vectors_encoder = tf.unstack(self.zero_out_mask_encoder)
		zero_out_vectors_decoder = tf.unstack(self.zero_out_mask_decoder)

		instances_attentions = []
		for i in range(len(instances_encoder)):
			print("Instance: " + str(i) + " of " + str(len(instances_encoder)))
			tokens_encoder = tf.unstack(instances_encoder[i])
			tokens_decoder = tf.unstack(instances_decoder[i])
			zeroout_vec_enc = tf.reshape(zero_out_vectors_encoder[i], [1, self.encoder_max_seq_len])
			zeroout_vec_dec = tf.tile(tf.reshape(zero_out_vectors_decoder[i], [self.decoder_max_seq_len, 1]), [1, self.encoder_output_size])

			att_vectors = []
			for tok_dec in tokens_decoder:
				print("Decoder token: " + str(tokens_decoder.index(tok_dec) + 1) + " of " + str(len(instances_encoder)))
				token_att_vector = [] 
					
				for tok_enc in tokens_encoder:
					if self.bilinear_combination:
						score = self.activation(tf.add(tf.reshape(tf.matmul(tf.matmul(tf.reshape(tok_dec, [1, self.decoder_output_size]), self.W_attend), tf.reshape(tok_enc, [self.encoder_output_size, 1])), [1]), self.b_attend))
					else:
						score = self.activation(tf.reshape(tf.add(tf.matmul(tf.reshape(tf.concat([tok_dec, tok_enc], axis = 0), [1, self.encoder_output_size + self.decoder_output_size]), self.W_attend), self.b_attend), [1]))
					token_att_vector.append(score)

				zeroed_att_vector = tf.multiply(tf.reshape(tf.stack(token_att_vector), [1, self.encoder_max_seq_len]), zeroout_vec_enc)
				if self.attended_dec_vector_state is None:
					self.zeroed_att_enc2dec_vectors.append(zeroed_att_vector)

				sftmx_att_vector = tf_extensions.softmax_ignore_zeros(zeroed_att_vector)
				if self.attended_dec_vector_state is None:
					self.sftmxd_att_enc2dec_vectors.append(sftmx_att_vector)

				sftmx_att_tiled = tf.tile(tf.transpose(sftmx_att_vector), [1, self.encoder_output_size])
				att_out_dec = tf.reduce_sum(tf.multiply(sftmx_att_tiled, instances_encoder[i]), axis = 0)
				att_vectors.append(att_out_dec)
					
				if self.attended_dec_vector_state is None:
					self.attended_dec_vector_state = att_out_dec

			dec_enc_atts_instance = tf.multiply(tf.stack(att_vectors), zeroout_vec_dec)
			instances_attentions.append(dec_enc_atts_instance)
		self.outputs = tf.stack(instances_attentions)

		self.l2_loss = tf.nn.l2_loss(self.W_attend) + tf.nn.l2_loss(self.b_attend)
							
	def get_feed_dict(self, zeroout_mask_encoder, zeroout_mask_decoder):
		fd_mine = { self.zero_out_mask_encoder : zeroout_mask_encoder, self.zero_out_mask_decoder : zeroout_mask_decoder}
		return fd_mine

	def get_variable_values(self, session):
		w_attend_eval = self.W_attend.eval(session = session)
		bias_attend_eval = self.b_attend.eval(session = session)
		variables = [w_attend_eval, bias_attend_eval]
		return variables

	def set_variable_values(self, session, values):
		session.run(self.W_attend.assign(values[0]))
		session.run(self.b_attend.assign(values[1]))		
	

	

	