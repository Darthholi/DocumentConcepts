import copy

import numpy as np
import tensorflow as tf
from keras.engine import Layer
from keras.layers import Dense, Dropout


class SinCosPositionalEmbedding(Layer):
    """
    As in https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf, adapted to our needs

    If list of from_inputs_features not provided, takes a range of the input sequence length.
    If from_inputs_features is provided, takes the indices from the last dimension in from_inputs_features
     to be the positions to be embedded.

    Just to note - what we want to accomplish: for i 0...2*self.embed_dim to have these features:
    sins = tf.sin(pos /tf.pow(self.pos_divisor, 2.0 * i / self.embed_dim))
    coss = tf.cos(pos /tf.pow(self.pos_divisor, 2.0 * i / self.embed_dim))
    """
    
    def __init__(self, embed_dim, from_inputs_features=None, pos_divisor=10000, keep_ndim=True, fix_range=None,
                 embeddings=['sin', 'cos'], **kwargs):
        """
        embed_dim: Te output embedding will have embed_dim floats for sin and cos (separately).
        from_inputs_features: If not specified, will use range(of the length of the input sequence) to generate
            (integer) positions that will be embedded by sins and coss.
            If specified, it needs to be a list of coordinates to the last dimension of the input vector,
             which will be taken as inputs into the positional ebedding.
             Then the output size will be len(from_inputs_features)*embed_dim*len(embeddings)
             Has no effect when fix<-range is set.
        pos_divisor: the division constant in the calculation.
        keep_ndim: if True, the output will have all embedded features concatenated/flattened into one dimension and so
            the input dimensions number is preserved.
        fix_range: if set, will produce a sequence of a fixed range (does not read from sequence length)
            and also disables from_inputs_features.
        embeddings: a list of 'sin', 'cos', 'lin' functions to be applied
        """
        Layer.__init__(self, **kwargs)
        self.pos_divisor = pos_divisor
        self.embed_dim = embed_dim
        self.keep_ndim = keep_ndim
        self.from_inputs_features = from_inputs_features
        self.fix_range = fix_range
        self.embeddings = embeddings
    
    def get_config(self):
        config = {'pos_divisor': self.pos_divisor,
                  'embed_dim': self.embed_dim,
                  'keep_ndim': self.keep_ndim,
                  'from_inputs_features': self.from_inputs_features,
                  'fix_range': self.fix_range,
                  'embeddings': self.embeddings,
                  }
        base_config = super(SinCosPositionalEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def build(self, input_shape):
        self.built_multipliers = tf.expand_dims(tf.constant([pow(self.pos_divisor, -2.0 * i / self.embed_dim)
                                                             for i in range(self.embed_dim)]), 0)
        # ^^ [1, self.embed_dim]
        self.built = True
    
    def compute_output_shape(self, input_shape):
        # assumes input in the shape of [..., batches, sequence_len, features]
        features_to_position = len(self.from_inputs_features) if self.from_inputs_features is not None else 1
        embeddings_dims = [features_to_position, self.embed_dim, len(self.embeddings)]
        if self.keep_ndim:
            return tuple(list(input_shape[:-1]) + [np.prod(embeddings_dims)])
        else:
            return tuple(list(input_shape[:-1]) + embeddings_dims)
    
    def call(self, inputs, training=None):
        
        def broadcast_to(item, to_shape):
            try:
                # newer tensorflow has this function
                return tf.broadcast_to(item, to_shape)
            except AttributeError:
                # if we do not have it, lets use this way:
                return item + tf.zeros(dtype=item.dtype, shape=to_shape)
        
        # assumes input in the shape of [..., batches, sequence_len, features]
        shape_batches_and_sequence = tf.shape(inputs)[:-1]  # [..., batches, sequence_len]
        
        if self.from_inputs_features:
            positions = tf.gather(inputs, self.from_inputs_features, axis=-1)
        else:
            if self.fix_range is not None:
                seq_len = self.fix_range
            else:
                seq_len = tf.shape(inputs)[-2]
            positions = tf.to_float(tf.range(seq_len))
            positions = tf.expand_dims(broadcast_to(positions, shape_batches_and_sequence), -1)
        # features_to_position = len(self.from_inputs_features) if self.from_inputs_features is not None else 1
        # now positions is [..., batches, sequence_len, features_to_position]
        # now for each features_to_position, we will create positional embedding of self.embed_dim in sin and cos, so
        # totally 2*self.embed_dim
        # self.built_multipliers is [1, self.embed_dim]
        # we want to get [..., batches, sequence_len, features_to_position (, or x) self.embed_dim]
        # so we need to reshape so that we get positions into [..., batches, sequence_len, features_to_position, 1]
        
        to_mult_shape = tf.concat([shape_batches_and_sequence,
                                   # tf.shape(self.built_multipliers) ==
                                   [1, self.embed_dim]
                                   ],
                                  axis=0)
        broadcast_ = broadcast_to(self.built_multipliers, to_mult_shape)
        
        positions_divided = tf.matmul(tf.to_float(tf.expand_dims(positions, -1)), broadcast_)
        # ^^ [..., batches, sequence_len, features_to_position, self.embed_dim]
        
        list_of_embeddidngs = []  # default will use [tf.sin(positions_divided), tf.cos(positions_divided)]
        for activation_str in self.embeddings:
            act_to_tf = {'sin': tf.sin, 'cos': tf.cos, 'lin': lambda x: x}
            tf_activation = act_to_tf[activation_str]
            list_of_embeddidngs.append(tf.expand_dims(tf_activation(positions_divided), -1))
        
        positions_embedded = tf.concat(list_of_embeddidngs, axis=-1)
        # ^^ [..., batches, sequence_len, features_to_position, self.embed_dim, len(embeddings)]
        
        if self.keep_ndim:
            positions_embedded = tf.reshape(positions_embedded, tf.concat([tf.shape(inputs)[:-1], [-1]], axis=0))
            # ^^ [..., batches, sequence_len, (-1 =) features_to_position * self.embed_dim * len(embeddings)]
        
        return positions_embedded


class GatherFromIndices(Layer):
    """
    To have a graph convolution (over a fixed/fixed degree kernel) from a given sequence of nodes, we need to gather
    the data of each node's neighbours before running a simple Conv1D/conv2D,
     that would be effectively a defined convolution (or even TimeDistributed(Dense()) can be used - only
     based on data format we would output).
    This layer should do exactly that.

    Does not support non integer values, values lesser than 0 zre automatically masked.
    """
    
    def __init__(self, mask_value=0, include_self=True, flatten_indices_features=False, **kwargs):
        Layer.__init__(self, **kwargs)
        self.mask_value = mask_value
        self.include_self = include_self
        self.flatten_indices_features = flatten_indices_features
    
    def get_config(self):
        config = {'mask_value': self.mask_value,
                  'include_self': self.include_self,
                  'flatten_indices_features': self.flatten_indices_features,
                  }
        base_config = super(GatherFromIndices, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    # def build(self, input_shape):
    # self.built = True
    
    def compute_output_shape(self, input_shape):
        inp_shape, inds_shape = input_shape
        indices = inds_shape[-1]
        if self.include_self:
            indices += 1
        features = inp_shape[-1]
        if self.flatten_indices_features:
            return tuple(list(inds_shape[:-1]) + [indices * features])
        else:
            return tuple(list(inds_shape[:-1]) + [indices, features])
    
    def call(self, inputs, training=None):
        inp, inds = inputs
        # assumes input in the shape of (inp=[...,batches, sequence_len, features],
        #  inds = [...,batches,sequence_ind_len, neighbours]... indexing into inp)
        # for output we want to get  [...,batches,sequence_ind_len, indices,features]
        
        assert_shapes = tf.Assert(tf.reduce_all(tf.equal(tf.shape(inp)[:-2], tf.shape(inds)[:-2])), [inp])
        assert_positive_ins_shape = tf.Assert(tf.reduce_all(tf.greater(tf.shape(inds), 0)), [inds])
        # the shapes need to be the same (with the exception of the last dimension)
        with tf.control_dependencies([assert_shapes, assert_positive_ins_shape]):
            inp_shape = tf.shape(inp)
            inds_shape = tf.shape(inds)
            
            features_dim = -1
            # ^^ todo for future variablility of the last dimension, because maybe can be made to take not the last
            # dimension as features, but something else.
            
            inp_p = tf.reshape(inp, [-1, inp_shape[features_dim]])
            ins_p = tf.reshape(inds, [-1, inds_shape[features_dim]])
            
            # we have lost the batchdimension by reshaping, so we save it by adding the size to the respective indexes
            # we do it because we use the gather_nd as nonbatched (so we do not need to provide batch indices)
            resized_range = tf.range(tf.shape(ins_p)[0])
            different_seqs_ids_float = tf.scalar_mul(1.0 / tf.to_float(inds_shape[-2]), tf.to_float(resized_range))
            different_seqs_ids = tf.to_int32(tf.floor(different_seqs_ids_float))
            different_seqs_ids_packed = tf.scalar_mul(inp_shape[-2], different_seqs_ids)
            thseq = tf.expand_dims(different_seqs_ids_packed, -1)
            
            # in case there are negative indices, make them all be equal to -1
            #  and add masking value to the ending of inp_p - that way, everything that should be masked
            #  will get the masking value as features.
            mask = tf.greater_equal(ins_p,
                                    0)  # extract where minuses are, because the will all default to default value
            # .. before the mod operation, if provided greater id numbers, to wrap correctly small sequences
            offset_ins_p = tf.mod(ins_p, inp_shape[-2]) + thseq  # broadcast to ins_p
            minus_1 = tf.scalar_mul(tf.shape(inp_p)[0], tf.ones_like(mask, dtype=tf.int32))
            '''
            On GPU, if we use index = -1 anywhere it would throw a warning:
            OP_REQUIRES failed at gather_nd_op.cc:50 : Invalid argument:
            flat indices = [-1] does not index into param.
            Which is a warning, that there are -1s. We are using that as feature and know about that.
            '''
            offset_ins_p = tf.where(mask, offset_ins_p, minus_1)
            # also possible to do something like  tf.multiply(offset_ins_p, mask) + tf.scalar_mul(-1, mask)
            mask_value_last = tf.zeros((inp_shape[-1],))
            if self.mask_value != 0:
                mask_value_last += tf.constant(self.mask_value)  # broadcasting if needed
            inp_p = tf.concat([inp_p, tf.expand_dims(mask_value_last, 0)], axis=0)
            
            # expand dims so that it would slice n times instead having slice of length n indices
            neighb_p = tf.gather_nd(inp_p, tf.expand_dims(offset_ins_p, -1))  # [-1,indices, features]
            
            out_shape = tf.concat([inds_shape, inp_shape[features_dim:]], axis=-1)
            neighb = tf.reshape(neighb_p, out_shape)
            # ^^ [...,batches,sequence_len, indices,features]
            
            if self.include_self:  # if is set, add self at the 0th position
                self_originals = tf.expand_dims(inp, axis=features_dim - 1)
                # ^^ [...,batches,sequence_len, 1, features]
                neighb = tf.concat([neighb, self_originals], axis=features_dim - 1)
            
            if self.flatten_indices_features:
                neighb = tf.reshape(neighb, tf.concat([inds_shape[:-1], [-1]], axis=-1))
            
            return neighb


class AttentionTransformer(Layer):
    """
    Keras implementation of the multihead attention layers in tensorflow, adapted from
         https://github.com/Kyubyong/transformer

    3 inputs - queries, keys, values (in this order)
    generally: [batch size; length of sequence; features vector]
    queries: A 3d tensor with shape of [N_batches, T_q, C_q].
    keys: A 3d tensor with shape of [N_batches, T_k, C_k].
    values: A 3d tensor with shape of [N_batches, T_v, C_v].
    if called with one input, assumes keys=queries=values as in attention is all you need.
    """
    
    def __init__(self, usesoftmax=True, num_units=None, num_heads=8, dropout_rate=0, activation='relu', causality=False,
                 usequerymasks=True, **kwargs):
        self.activation = activation
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        self.usesoftmax = usesoftmax
        self.usequerymasks = usequerymasks
        Layer.__init__(self, **kwargs)
    
    def get_config(self):
        config = {'activation': self.activation,
                  'num_units': self.num_units,
                  'num_heads': self.num_heads,
                  'dropout_rate': self.dropout_rate,
                  'causality': self.causality,
                  'usesoftmax': self.usesoftmax,
                  'usequerymasks': self.usequerymasks,
                  }
        base_config = super(AttentionTransformer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def build(self, input_shape):
        (queries, keys, values) = self._care_inputs(input_shape)
        queries = list(queries)
        keys = list(keys)
        values = list(values)
        if self.num_units is None:
            self.num_units = queries[-1]
        # we will now accept inputs as sequences, so if something is not a sequence it IS a sequence of len 1
        if len(queries) <= 2:
            queries.insert(-1, 1)
        if len(keys) <= 2:
            keys.insert(-1, 1)
        if len(values) <= 2:
            values.insert(-1, 1)
        
        self.Q_dense = Dense(self.num_units, activation=self.activation, name="Q_dense")
        self.Q_dense.build(queries)
        self.K_dense = Dense(self.num_units, activation=self.activation, name="K_dense")
        self.K_dense.build(keys)
        self.V_dense = Dense(self.num_units, activation=self.activation, name="V_dense")
        self.V_dense.build(values)
        
        self.trainable_weights = self.Q_dense.trainable_weights + self.K_dense.trainable_weights + \
                                 self.V_dense.trainable_weights
        self.non_trainable_weights = self.Q_dense.non_trainable_weights + self.K_dense.non_trainable_weights + \
                                     self.V_dense.non_trainable_weights
        
        self.dropout = Dropout(rate=self.dropout_rate)
        self.built = True
    
    # a hint about the Keras implementation: it is all called in the sequence: build, compute_output_shape, call
    def _care_inputs(self, inputs):
        inputs = copy.copy(inputs)
        if (isinstance(inputs, list)):
            while (len(inputs) < 3):
                inputs.append(inputs[-1])
            inputs = inputs[0:3]
        else:
            inputs = [inputs, inputs, inputs]
        return inputs
    
    def compute_output_shape(self, input_shape):
        (queries, keys, values) = self._care_inputs(input_shape)
        # assert input_shape and len(input_shape) >= 2
        # assert input_shape[-1]
        output_shape = list(queries)
        output_shape[-1] = self.num_units  # (N, T_q, C) num units = T_q, if num units unspecified by user
        return tuple(output_shape)
    
    def call(self, inputs, training=None):
        # expects 3 inputs as merge layer https://github.com/keras-team/keras/blob/master/keras/layers/merge.py
        (queries, keys, values) = self._care_inputs(inputs)
        if self.num_units is None:  # done in build too
            self.num_units = queries.get_shape().as_list()[-1]
        # we will now accept inputs as sequences, so if something is not a sequence it IS a sequence of len 1
        if len(queries.shape) <= 2:
            queries = tf.expand_dims(queries, -2)
        if len(keys.shape) <= 2:
            keys = tf.expand_dims(keys, -2)
        if len(values.shape) <= 2:
            values = tf.expand_dims(values, -2)
        Q = self.Q_dense.call(queries)  # call is a way how to use a layer inside a layer
        K = self.K_dense.call(keys)
        V = self.V_dense.call(values)
        if len(Q.shape) <= 2:
            Q = tf.expand_dims(Q, -2)
        if len(K.shape) <= 2:
            K = tf.expand_dims(K, -2)
        if len(V.shape) <= 2:
            V = tf.expand_dims(V, -2)
        return self.multihead_attention_mechanism(Q, K, V,
                                                  queries=queries, keys=keys,
                                                  num_heads=self.num_heads,
                                                  causality=self.causality,
                                                  usequerymasks=self.usequerymasks,
                                                  scope="multihead_attention",
                                                  usesoftmax=self.usesoftmax,
                                                  reuse=None)  # todo scope & reuse?
    
    def normalize(self, inputs,
                  epsilon=1e-8,
                  scope="ln",
                  reuse=None):
        """Applies layer normalization.

        Args:
        ----
          inputs: A tensor with 2 or more dimensions, where the first dimension has
            `batch_size`.
          epsilon: A floating number. A very small number for preventing ZeroDivision Error.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns
        -------
          A tensor with the same shape and data dtype as `inputs`.
        """
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta
        return outputs
    
    def multihead_attention_mechanism(self,
                                      Qinp, Kinp, Vinp,
                                      queries, keys,
                                      num_heads=8,
                                      causality=False,
                                      usequerymasks=True,
                                      scope="multihead_attention",
                                      usesoftmax=True,
                                      reuse=None):
        """Applies multihead attention mechanism. Just the computation eithout trainable weights.

        Args:
        ----
          queries: A 3d tensor with shape of [N, T_q, C_q].
          keys: A 3d tensor with shape of [N, T_k, C_k].
          causality: Boolean. If true, units that reference the future are masked.
          num_heads: An int. Number of heads.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns
        -------
          A 3d tensor with shape of (N, T_q, C)
        """
        assert (len(Qinp.shape) + len(Kinp.shape) + len(Vinp.shape) > 3 * 2)
        with tf.variable_scope(scope, reuse=reuse):
            # Split and concat - for keras, the N dimension is HIDDEN, but in tf we see it!
            Q_ = tf.concat(tf.split(Qinp, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(Kinp, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(Vinp, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            # Multiplication                                     # T_q, T_k are the original queries and keys -
            # sequence lengths (and in the application they are the same)
            preoutputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
            # Scale
            preoutputs = preoutputs / (K_.get_shape().as_list()[-1] ** 0.5)
            # Key Masking
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)
            paddings = tf.ones_like(preoutputs) * (-2 ** 32 + 1)
            preoutputs = tf.where(tf.equal(key_masks, 0), paddings, preoutputs)  # (h*N, T_q, T_k)
            # Causality = Future blinding
            if causality:
                diag_vals = tf.ones_like(preoutputs[0, :, :])  # (T_q, T_k)
                tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()  # (T_q, T_k)
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(preoutputs)[0], 1, 1])  # (h*N, T_q, T_k)
                paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
                preoutputs = tf.where(tf.equal(masks, 0), paddings, preoutputs)  # (h*N, T_q, T_k)
            # Activation
            if (usesoftmax):
                preoutputs = tf.nn.softmax(preoutputs)  # (h*N, T_q, T_k)
            # Query Masking
            if usequerymasks:
                query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
                query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
                query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
                preoutputs *= query_masks  # broadcasting. (N, T_q, T_k)
            outputs = self.dropout.call(preoutputs)
            # Weighted sum
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)
            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)
            # Residual connection #still the same dimension
            outputs += queries
            # Normalize
            outputs = self.normalize(outputs)  # (N, T_q, C)
        return outputs
