import tensorflow as tf

class MappingLayer(object):
	"""
	A layer for mapping embeddings from different embedding spaces to the same shared embedding space. 
	"""

	def __init__(self, num_mappers, seq_len, mappers = None, embeddings = (100, None), vocab_size = None, scope = "mapper_layer1"):
		self.emb_size = embeddings[0]
		self.embs = embeddings[1]
		self.vocab_size = vocab_size if self.embs is None else self.embs.shape[0]
		self.mappers = mappers
		self.num_mappers = num_mappers
		self.seqence_length = seq_len
		self.scope = scope

	def define_model(self, activation = None, update_embeddings = False, update_mappers = False, previous_layer = None):
		with tf.name_scope(self.scope + "__" + 'data-placeholders'):
			if previous_layer is None: 
				self.input = tf.placeholder(tf.int32, [None, self.seqence_length], name = self.scope + "__input")
			self.mapper_indices = tf.placeholder(tf.int32, [None, self.seqence_length], name = self.scope + "__indices")
		
		with tf.name_scope(self.scope + "__" + 'variables'):
			if self.mappers is None:
				self.mappers = tf.get_variable(self.scope + "__mappers", shape=[self.num_mappers, self.emb_size, self.emb_size], initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
			elif update_mappers:
				self.mappers = tf.Variable(self.mappers, dtype = tf.float64, name = self.scope + "__mappers")
			else:
				self.mappers = tf.Variable(self.mappers, dtype = tf.float64, name = self.scope + "__mappers", trainable = False)

			if previous_layer is None:
				if self.embs is None:
					self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.emb_size], -1.0, 1.0), name = self.scope + "__word_embeddings")
				elif update_embeddings:
					self.embeddings = tf.Variable(self.embs, dtype = tf.float64, name = self.scope + "__word_embeddings")
				else:
					self.embeddings = tf.Variable(self.embs, dtype = tf.float64, name = self.scope + "__word_embeddings", trainable = False)

		with tf.name_scope(self.scope + "__" + 'computation'):
			if previous_layer is None: 
				batch_embedded = tf.nn.embedding_lookup(self.embeddings, self.input)
			else:
				batch_embedded = previous_layer.outputs_static

			batch_mappers = tf.nn.embedding_lookup(self.mappers, self.mapper_indices)
			batch_mapped = tf.reshape(tf.matmul(tf.reshape(batch_embedded, [-1, self.seqence_length, 1, self.emb_size]), batch_mappers), [-1, self.seqence_length, self.emb_size])
			self.outputs_static = batch_mapped if activation is None else activation(batch_mapped)

		if update_mappers:
			self.l2_loss = tf.nn.l2_loss(self.mappers)
		else:
			self.l2_loss = 0
		if previous_layer is not None:
			self.l2_loss = self.l2_loss + previous_layer.l2_loss

	def get_feed_dict(self, input_data, mapper_indices):
		fd_mine = { self.input : input_data, self.mapper_indices : mapper_indices }
		return fd_mine

	def get_variable_values(self, session):
		return [self.mappers.eval(session = session)]

	def set_variable_values(self, session, values):
		session.run(self.mappers.assign(values))
			