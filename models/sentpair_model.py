import tensorflow as tf
from helpers import io_helper
from extensions import tf_extensions
from recurrent import attention
from ml import loss_functions
from layers import mlp_layer

class SentencePairClassifier(object):
	"""
	The model for classifying pairs of sentences, given two encoders. Optional attention and cross-lingual layer may be defined. 
	"""

	def __init__(self, first_encoder, second_encoder, scope = "sent_pair_classifier", same_encoder = False):
		self.scope = scope
		self.first_encoder = first_encoder
		self.second_encoder = second_encoder
		self.same_encoder = same_encoder
		
	def define_model(self, cross_attention = False, self_attention = False, cross_attention_bilinear = True, activation = None, batch_size = 50, bilinear_product_score = True, num_classes = None, share_cross_attention = False, share_intra_attention = False):
		self.batch_size = batch_size
		self.activation = activation or tf.nn.tanh
		self.bilinear_product_score = bilinear_product_score
		self.cross_attention = cross_attention
		self.self_attention = self_attention
		self.share_cross_attention = share_cross_attention
		self.share_intra_attention = share_intra_attention		

		self.l2_loss = self.first_encoder.l2_loss + self.second_encoder.l2_loss 
		if cross_attention:
			print("Defining cross attention first second...")
			self.cross_attention_first_second = attention.StateLevelAttention(self.first_encoder, self.second_encoder, scope = self.scope + "__ca", unique_scope_addition = "first_second")
			self.cross_attention_first_second.define_model(bilinear_combination = cross_attention_bilinear, activation = self.activation, batch_size = batch_size, share_params = None)
			self.l2_loss += self.cross_attention_first_second.l2_loss

			print("Defining cross attention second first...")
			self.cross_attention_second_first = attention.StateLevelAttention(self.second_encoder, self.first_encoder, scope = self.scope + "__ca" + ("" if share_cross_attention else "_2"), unique_scope_addition = "second_first")
			self.cross_attention_second_first.define_model(bilinear_combination = cross_attention_bilinear, activation = self.activation, batch_size = batch_size, share_params = (True if share_cross_attention else None))
			if share_cross_attention is None:
				self.l2_loss += self.cross_attention_second_first.l2_loss
	
			self.first_to_attend = self.cross_attention_second_first
			self.second_to_attend = self.cross_attention_first_second
		else:
			self.first_to_attend = self.first_encoder
			self.second_to_attend = self.second_encoder

		if self_attention:
			print("Defining intra attention first...")
			self.final_reps_first = attention.SequenceLevelAttention(self.first_to_attend, scope = self.scope + "__inter_attention")
			self.final_reps_first.define_model(activation = self.activation, batch_size = self.batch_size, share_params = None)
			self.l2_loss += self.final_reps_first.l2_loss

			print("Defining intra attention first...")
			self.final_reps_second = attention.SequenceLevelAttention(self.second_to_attend, scope = self.scope + "__inter_attention" + ("" if share_intra_attention else "_2"))
			self.final_reps_second.define_model(activation = self.activation, batch_size = self.batch_size, share_params = (True if share_intra_attention else None))
			if share_intra_attention is None:
				self.l2_loss += self.final_reps_second.l2_loss

		else:
			self.final_reps_first = tf.reduce_mean(self.first_to_attend.outputs, axis = 1)
			self.final_reps_second = tf.reduce_mean(self.second_to_attend.outputs, axis = 1)

		print("Defining (bi)linear classification layer...")
		if self.bilinear_product_score:
			# bilinear product to generate scores
			with tf.variable_scope(self.scope):
				self.bilinear_W = tf.get_variable("W_bilinear_classifier", shape=[self.final_reps_first.output_size, self.final_reps_second.output_size], initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
				#self.bilinear_W = tf.get_variable("W_bilinear_classifier", shape=[self.final_reps_first.get_shape()[1], self.final_reps_second.get_shape()[1]], initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
				self.bilinear_b = tf.get_variable("b_attend", initializer = tf.constant(0.1, shape=[1], dtype = tf.float64), dtype = tf.float64)
				
				self.lin_part = tf.matmul(self.final_reps_first.outputs, self.bilinear_W)
				self.bilin_part = tf.add(tf.reduce_sum(tf.multiply(self.lin_part, self.final_reps_second.outputs), axis = -1), self.bilinear_b)

				self.outputs = tf.nn.tanh(self.bilin_part)
				self.l2_loss += tf.nn.l2_loss(self.bilinear_W) + tf.nn.l2_loss(self.bilinear_b)
		else:
			# linear combination (concat + MLP) to generate class predictions 
			#self.final_concat = tf.concat([self.final_reps_first.outputs, self.final_reps_second.outputs], axis = 1)
			#self.num_classes = num_classes
			#self.mlp = mlp_layer.MultiLayerPerceptron([num_classes], self.final_reps_first.output_size + self.final_reps_second.output_size, scope = self.scope + "__mlp_classifier")
			#self.mlp.define_model(activation = self.activation, previous_layer = self.final_concat)
			#self.outputs = self.mlp.outputs
			
			# direct loss cosine between sentence representations
			first_norm = tf.nn.l2_normalize(self.final_reps_first.outputs, dim = [1])
			second_norm = tf.nn.l2_normalize(self.final_reps_second.outputs, dim = [1])
			self.outputs = tf.reduce_sum(tf.multiply(first_norm, second_norm), axis = 1)
			
		self.preds = self.outputs
	
	def define_optimization(self, loss_function, l2_reg_factor = 0.01, learning_rate = 1e-3, loss_function_params = None):
		print("Defining loss...")
		#if self.bilinear_product_score:
		self.input_y = tf.placeholder(tf.float64, [self.batch_size], name="input_y")
		#else:
		#	self.input_y = tf.placeholder(tf.float64, [self.batch_size, self.num_classes], name="input_y")
		if loss_function_params:
			self.pure_loss = loss_function(self.outputs, self.input_y, loss_function_params)
		else:
			self.pure_loss = loss_function(self.outputs, self.input_y)
		self.loss = self.pure_loss + l2_reg_factor * self.l2_loss

		print("Defining optimizer...")
		self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
		print("Done!...")
				
	def get_feed_dict(self, labels, zeroout_mask_first_encoder, zeroout_mask_second_encoder):
		fd = { self.input_y : labels } if labels else {}
		if self.cross_attention:
			fd.update(self.cross_attention_first_second.get_feed_dict(zeroout_mask_first_encoder, zeroout_mask_second_encoder))
			fd.update(self.cross_attention_second_first.get_feed_dict(zeroout_mask_second_encoder, zeroout_mask_first_encoder))
		fd.update(self.final_reps_first.get_feed_dict(zeroout_mask_first_encoder))
		fd.update(self.final_reps_second.get_feed_dict(zeroout_mask_second_encoder))
		return fd

	def get_variable_values(self, session):
		variables = []
		variables.append(self.first_encoder.get_variable_values(session))
		if not self.same_encoder:
			variables.append(self.second_encoder.get_variable_values(session))

		if self.cross_attention:
			variables.append(self.cross_attention_first_second.get_variable_values(session))
			if not self.share_cross_attention:
				variables.append(self.cross_attention_second_first.get_variable_values(session))
		
		if self.self_attention:
			variables.append(self.final_reps_first.get_variable_values(session))
			if not self.share_intra_attention:
				variables.append(self.final_reps_second.get_variable_values(session))

		if self.bilinear_product_score:
			w_bilinear = self.bilinear_W.eval(session = session)
			b_bilinear = self.bilinear_b.eval(session = session)
			variables.append([w_bilinear, b_bilinear])
		else:
			mlp_vars = self.mlp.get_variable_values(session)
			variables.append(mlp_vars)
		return variables

	def set_variable_values(self, session, variables):
		print("Setting variables for SP model, length: " + str(len(variables)))

		first_encoder_variables = variables.pop(0)
		self.first_encoder.set_variable_values(session, first_encoder_variables)
		if not self.same_encoder:
			second_encoder_variables = variables.pop(0)
			self.second_encoder.set_variable_values(session, second_encoder_variables)

		if self.cross_attention:
			ca_first_second_vars = variables.pop(0)
			self.cross_attention_first_second.set_variable_values(session, ca_first_second_vars)
			if not self.share_cross_attention:
				ca_second_first_vars = variables.pop(0)
				self.cross_attention_second_first.set_variable_values(session, ca_second_first_vars)
		
		if self.self_attention:
			intra_first_vars = variables.pop(0)
			self.final_reps_first.set_variable_values(session, intra_first_vars)
			if not self.share_intra_attention:
				intra_second_vars = variables.pop(0)
				self.final_reps_second.set_variable_values(session, intra_second_vars)

		if self.bilinear_product_score:
			bilin_vars = variables.pop(0)
			session.run(self.bilinear_W.assign(bilin_vars[0]))
			session.run(self.bilinear_b.assign(bilin_vars[1]))
		else:
			mlp_vars = variables.pop(0)
			self.mlp.set_variable_values(session, mlp_vars)

	def get_hyperparameters(self):
		hyp_first_enc = self.first_encoder.get_hyperparameters()
		if self.same_encoder:
			hyp_second_enc = None
		else:
			hyp_second_enc = self.second_encoder.get_hyperparameters()
		return [hyp_first_enc, hyp_second_enc, self.batch_size, self.same_encoder, self.cross_attention, self.self_attention, self.share_cross_attention, self.share_intra_attention, self.bilinear_product_score]

	def get_model(self, session):
		hyperparams = self.get_hyperparameters()
		variables = self.get_variable_values(session)
		return (hyperparams, variables)