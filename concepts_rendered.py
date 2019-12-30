from __future__ import print_function, division

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv1D
from keras.layers import Input, TimeDistributed, Dense, Concatenate, Softmax, Dropout
from keras.layers import Lambda, Conv2D
from keras.models import Model

from utils import tf_dataset_as_iterator, evaluate
from generators import RenderedConceptsPacker
from concepts_fixed import constant_testing_setting_multiclass
from utils import EvaluateFCallback
from attention import SinCosPositionalEmbedding, GatherFromIndices, AttentionTransformer
from utils import make_product_matrix


def run_keras_rendered_experiment_categorical(const_data_def=constant_testing_setting_multiclass(),
                                              validation_pages=2,
                                              n_epochs=100,
                                              verbose=2,
                                              stop_early=True,
                                              key_metric='val_categorical_accuracy',
                                              weights_best_fname='weightstmp.h5',
                                              patience=15,
                                              key_metric_mode='max',
                                              pages_per_epoch=10,
                                              batch_size=2,
                                              zero_class=(1, 0),
                                              df_proc_num=2
                                              ):
    """
    Runs a simplest model against a dataset and returns the max epoch accuracy.
    """
    
    gen = RenderedConceptsPacker(const_data_def,
                                 df_proc_num=df_proc_num,
                                 batch_size=batch_size,
                                 df_batches_to_prefetch=1, zero_class=zero_class, use_neighbours=3)
    neighbours = gen.use_neighbours
    
    features = Input(shape=gen.get_batchpadded_shapes()[0]['features'], name='features')  # per page, features
    neighbours_ids_input = Input(shape=gen.get_batchpadded_shapes()[0]['neighbours'], name='neighbours',
                                 dtype=np.int32)  # per page, features
    
    inputs_merged = GatherFromIndices(mask_value=0,
                                      include_self=True, flatten_indices_features=True) \
        ([features, neighbours_ids_input]) if neighbours > 0 else features
    
    c1 = Conv1D(filters=8, kernel_size=5, padding='same', activation='relu')(inputs_merged)
    c2 = Conv1D(filters=8, kernel_size=5, padding='same', activation='relu')(c1)
    fpb = Dense(8, activation='relu')(c2)
    outclassesperbox = Dense(gen.get_batchpadded_shapes()[1][-1], activation='sigmoid')(fpb)
    outclassesperbox = Softmax()(outclassesperbox)
    
    const_model = Model(inputs=[features, neighbours_ids_input], outputs=outclassesperbox)
    const_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[metrics.categorical_accuracy
                                                                                    ],
                        sample_weight_mode='temporal')
    const_model.summary()
    
    callbacks = {'checkpointer': ModelCheckpoint(weights_best_fname,
                                                 monitor=key_metric, save_best_only=True, mode=key_metric_mode,
                                                 verbose=verbose),
                 # 'datetime': DatetimePrinter(),
                 # 'procnum': ProcNumPrinter(),
                 # 'memory': MemoryPrinter(),
                 # 'weights_printer': PrintWeightsStats(check_on_batch=debug),
                 # 'nanterminator': TerminateOnNaNRemember(),
                 # 'sacred': SacredCallback(self.run, 'val_loss'),
                 }
    if stop_early:
        callbacks['early_stopping'] = EarlyStopping(monitor=key_metric, patience=patience,
                                                    mode=key_metric_mode, verbose=verbose)
    
    hist = const_model.fit_generator(
        tf_dataset_as_iterator(gen.get_final_tf_data_dataset(pages_per_epoch, phase='train')),
        # .make_one_shot_iterator().get_next(),
        pages_per_epoch / batch_size, n_epochs,
        verbose=verbose,
        # class_weight=class_weight,
        # keras cannot use class weight and sample weights at the same time
        validation_data=tf_dataset_as_iterator(gen.get_final_tf_data_dataset(validation_pages, phase='val')),
        # we validate on the same set btw
        validation_steps=validation_pages / batch_size,
        callbacks=[callbacks[key] for key in callbacks],
        workers=0,  # because we use dataflow
        use_multiprocessing=False
    )
    return max(hist.history['val_categorical_accuracy'])


def run_keras_rendered_experiment_binary(const_data_def=constant_testing_setting_multiclass(),
                                         validation_pages=20,
                                         n_epochs=100,
                                         verbose=2,
                                         stop_early=True,
                                         key_metric='val_loss',
                                         weights_best_fname='weightstmp.h5',
                                         patience=15,
                                         key_metric_mode='min',
                                         pages_per_epoch=100,
                                         batch_size=4,
                                         zero_class=(0, 0),
                                         df_proc_num=2,
                                         neighbours=3,
                                         count_classes=True,
                                         bin_class_weights=1.0
                                         ):
    """
    Runs a simplest model against a dataset and returns the max epoch accuracy.
    """
    
    gen = RenderedConceptsPacker(const_data_def,
                                 df_proc_num=df_proc_num,
                                 batch_size=batch_size,
                                 df_batches_to_prefetch=1, zero_class=zero_class, use_neighbours=neighbours,
                                 bin_class_weights=bin_class_weights)
    neighbours = gen.use_neighbours
    
    features = Input(shape=gen.get_batchpadded_shapes()[0]['features'], name='features')  # per page, features
    neighbours_ids_input = Input(shape=gen.get_batchpadded_shapes()[0]['neighbours'], name='neighbours',
                                 dtype=np.int32)  # per page, features
    
    inputs_merged = GatherFromIndices(mask_value=0,
                                      include_self=True, flatten_indices_features=True) \
        ([features, neighbours_ids_input]) if neighbours > 0 else features
    
    c1 = Conv1D(filters=8, kernel_size=5, padding='same', activation='relu')(inputs_merged)
    c2 = Conv1D(filters=8, kernel_size=5, padding='same', activation='relu')(c1)
    fpb = Dense(8, activation='relu')(c2)
    outclassesperbox = Dense(gen.get_batchpadded_shapes()[1][-1], activation='sigmoid')(fpb)
    
    const_model = Model(inputs=[features, neighbours_ids_input], outputs=outclassesperbox)
    const_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[metrics.binary_accuracy
                                                                               ],
                        sample_weight_mode='temporal')
    const_model.summary()
    
    eval_gen = tf_dataset_as_iterator(gen.get_final_tf_data_dataset(validation_pages, phase='val'))
    
    def report_f():
        result = evaluate(eval_gen, int(validation_pages / batch_size), const_model, None)
        return result
    
    callbacks = {'checkpointer': ModelCheckpoint(weights_best_fname,
                                                 monitor=key_metric, save_best_only=True, mode=key_metric_mode,
                                                 verbose=verbose),
                 'val_reporter': EvaluateFCallback(report_f,
                                                   monitor=key_metric,
                                                   mode=key_metric_mode
                                                   )
                 }
    
    if stop_early:
        callbacks['early_stopping'] = EarlyStopping(monitor=key_metric, patience=patience,
                                                    mode=key_metric_mode, verbose=verbose)
    
    hist = const_model.fit_generator(
        tf_dataset_as_iterator(gen.get_final_tf_data_dataset(pages_per_epoch, phase='train')),
        # .make_one_shot_iterator().get_next(),
        pages_per_epoch / batch_size, n_epochs,
        verbose=verbose,
        # class_weight=class_weight,
        # keras cannot use class weight and sample weights at the same time
        validation_data=tf_dataset_as_iterator(gen.get_final_tf_data_dataset(validation_pages, phase='val')),
        # we validate on the same set btw
        validation_steps=validation_pages / batch_size,
        callbacks=[callbacks[key] for key in callbacks],
        workers=0,  # because we use dataflow
        use_multiprocessing=False
    )
    return max(hist.history[key_metric])


def run_keras_articlemodel(const_data_def=constant_testing_setting_multiclass(),
                           validation_pages=2,
                           n_epochs=100,
                           verbose=2,
                           stop_early=True,
                           # key_metric='val_categorical_accuracy',
                           key_metric='val_loss',
                           weights_best_fname='weightstmp.h5',
                           patience=15,
                           key_metric_mode='min',
                           pages_per_epoch=10,
                           batch_size=2,
                           zero_class=(1, 0),
                           df_proc_num=2,
                           neighbours=3,
                           count_classes=True,
                           bin_class_weights=1.0,  # from 100/8000 positive class count
                           n_siz=1
                           ):
    """
    Runs a simplest model against a dataset and returns the max epoch accuracy.
    """
    
    gen = RenderedConceptsPacker(const_data_def,
                                 df_proc_num=df_proc_num,
                                 batch_size=batch_size,
                                 df_batches_to_prefetch=1, zero_class=zero_class, use_neighbours=neighbours,
                                 bin_class_weights=bin_class_weights)
    
    if count_classes:
        n_out_classes = gen.get_batchpadded_shapes()[1][-1]
        classcounts = np.zeros(n_out_classes)
        total_counts = 0
        count_classes_gen = tf_dataset_as_iterator(gen.get_final_tf_data_dataset(pages_per_epoch, phase='train'))
        for _ in range(int(pages_per_epoch / batch_size)):
            batch_data = count_classes_gen.next()
            classcounts += np.sum(batch_data[1], axis=(0, 1))
            total_counts += batch_data[1].shape[0] * batch_data[1].shape[1]
        print("class counts done {} / {}".format(classcounts, total_counts))  # cca 100/8000
    
    neighbours = gen.use_neighbours
    depth_variation = 0
    use_attention = True
    use_more_dense = True
    
    fields_input = Input(shape=gen.get_batchpadded_shapes()[0]['features'], name='features')  # per page, features
    
    neighbours_ids_input = Input(shape=gen.get_batchpadded_shapes()[0]['neighbours'], name='neighbours',
                                 dtype=np.int32)  # per page, features
    
    # merge positional embedding - from integers to sins coss
    # in the generator they are at the last 4 positions....
    positions_reading_order_embedded = SinCosPositionalEmbedding(4 * n_siz,
                                                                 from_inputs_features=[-1, -2, -3, -4],
                                                                 # embedd all 4 integers
                                                                 pos_divisor=10000,
                                                                 keep_ndim=True)(fields_input)
    # embedd lrtb
    positions_embedded = SinCosPositionalEmbedding(4 * n_siz,
                                                   from_inputs_features=[0, 1, 2, 3],
                                                   # embedd all 4 integers
                                                   pos_divisor=10000,
                                                   keep_ndim=True)(fields_input)
    
    merged_result = Concatenate()([fields_input, positions_reading_order_embedded, positions_embedded])
    
    # gather neighbours so that we will see them and can operate on them also:
    fields_input_with_neighbours = GatherFromIndices(mask_value=0,
                                                     include_self=True, flatten_indices_features=True) \
        ([merged_result, neighbours_ids_input]) if neighbours > 0 else merged_result
    
    fields = TimeDistributed(Dense(256 * n_siz, activation='relu'))(fields_input_with_neighbours)
    
    if depth_variation > 1:
        fields_input_with_neighbours = GatherFromIndices(mask_value=0,
                                                         include_self=True, flatten_indices_features=True) \
            ([fields, neighbours_ids_input]) if neighbours > 0 else fields
        
        fields = TimeDistributed(Dense(256 * n_siz, activation='relu'))(fields_input_with_neighbours)
    
    fields = Dropout(0.15)(fields)
    
    use_seq_convolution = True
    if use_seq_convolution:
        fields = Conv1D(128 * n_siz, kernel_size=5, padding='same',
                        activation='relu')(fields)  # data_format="channels_last"
    else:
        fields = fields  # Dense(128 * n_siz, activation='relu')(fields)
    
    # try lstms?
    # lstm1 = Bidirectional(LSTM(64, return_sequences=True, activation='tanh'))(merged_result)
    # lstm2 = Bidirectional(LSTM(64, return_sequences=True, activation='tanh'))(lstm1)
    # flatten_for_a = TimeDistributed(Flatten())(lstm1)
    flatten_for_a = Dense(64 * n_siz, activation='relu')(fields)
    
    if use_attention:
        tmp_a = AttentionTransformer(usesoftmax=True, usequerymasks=False, num_heads=8 * n_siz, num_units=64 * n_siz,
                                     causality=False)([flatten_for_a, flatten_for_a, flatten_for_a])
        if depth_variation > 1:
            tmp_a = AttentionTransformer(usesoftmax=True, usequerymasks=False, num_heads=8, num_units=64 * n_siz,
                                         causality=False)([tmp_a, tmp_a, tmp_a])
        # we do not want softmax for doing binary classification
        # tmp_a = AttentionTransformer(usesoftmax=True, usequerymasks=False, num_heads=8, num_units=64,
        #                             causality=False)([tmp_a, tmp_a, tmp_a])
    else:
        tmp_a = flatten_for_a
    
    # tmp_f = TimeDistributed(Flatten())(tmp_a)
    # highway = Concatenate()([flatten_for_a, tmp_f])
    
    if use_more_dense:
        bef_fork = TimeDistributed(Dense(64 * n_siz, activation='relu'))(tmp_a)
        bef_fork = Dropout(0.15)(bef_fork)
    else:
        bef_fork = tmp_a
    
    fork_node = TimeDistributed(Dense(64 * n_siz, activation='relu'))(
        bef_fork)  # from this point on, we will fork to different outputs!
    
    outclassesperbox = Dense(gen.get_batchpadded_shapes()[1][-1], activation='sigmoid')(fork_node)
    
    # outclassesperbox = Softmax()(outclassesperbox)
    
    def mean_pred(y_true, y_pred):
        return K.mean(y_pred)
    
    const_model = Model(inputs=[fields_input, neighbours_ids_input], outputs=outclassesperbox)
    const_model.compile(optimizer='adam',
                        # loss='categorical_crossentropy', metrics=[metrics.categorical_accuracy],
                        loss='binary_crossentropy', metrics=[metrics.binary_accuracy],
                        sample_weight_mode='temporal'
                        )
    const_model.summary()
    
    # val:
    eval_gen = tf_dataset_as_iterator(gen.get_final_tf_data_dataset(validation_pages, phase='val'))
    
    def report_f():
        result = evaluate(eval_gen, int(validation_pages / batch_size), const_model, None)
        return result
    
    callbacks = {'checkpointer': ModelCheckpoint(weights_best_fname,
                                                 monitor=key_metric, save_best_only=True, mode=key_metric_mode,
                                                 verbose=verbose),

                 'val_reporter': EvaluateFCallback(report_f,
                                                   monitor=key_metric,
                                                   mode=key_metric_mode
                                                   )
                 }
    
    print("Printing metrics on random model to see from where the first epoch started: ")
    callbacks['val_reporter'].on_epoch_end(0)
    
    if stop_early:
        callbacks['early_stopping'] = EarlyStopping(monitor=key_metric, patience=patience,
                                                    mode=key_metric_mode, verbose=verbose)
    
    hist = const_model.fit_generator(
        tf_dataset_as_iterator(gen.get_final_tf_data_dataset(pages_per_epoch, phase='train')),
        pages_per_epoch / batch_size, n_epochs,
        verbose=verbose,
        # class_weight=class_weight,
        # keras cannot use class weight and sample weights at the same time
        validation_data=tf_dataset_as_iterator(gen.get_final_tf_data_dataset(validation_pages, phase='val')),
        # we validate on the same set btw
        validation_steps=validation_pages / batch_size,
        callbacks=[callbacks[key] for key in callbacks],
        workers=0,  # because we use dataflow
        use_multiprocessing=False
    )
    return min(hist.history[key_metric])


def run_keras_all2all_model(const_data_def=constant_testing_setting_multiclass(),
                            validation_pages=2,
                            n_epochs=100,
                            verbose=2,
                            stop_early=True,
                            # key_metric='val_categorical_accuracy',
                            key_metric='val_loss',
                            weights_best_fname='weightstmp.h5',
                            patience=15,
                            key_metric_mode='min',
                            pages_per_epoch=10,
                            batch_size=2,
                            zero_class=(1, 0),
                            df_proc_num=2,
                            neighbours=3,
                            count_classes=True,
                            bin_class_weights=1.0,  # from 100/8000 positive class count
                            n_siz=1
                            ):
    """
    allows all fields to see all other fields.
    """
    
    gen = RenderedConceptsPacker(const_data_def,
                                 df_proc_num=df_proc_num,
                                 batch_size=batch_size,
                                 df_batches_to_prefetch=1, zero_class=zero_class, use_neighbours=neighbours,
                                 bin_class_weights=bin_class_weights)
    """
    if count_classes:
        n_out_classes = gen.get_batchpadded_shapes()[1][-1]
        classcounts = np.zeros(n_out_classes)
        total_counts = 0
        count_classes_gen = tf_dataset_as_iterator(gen.get_final_tf_data_dataset(pages_per_epoch, phase='train'))
        for _ in range(int(pages_per_epoch/batch_size)):
            batch_data = count_classes_gen.next()
            classcounts += np.sum(batch_data[1], axis=(0, 1))
            total_counts += batch_data[1].shape[0]*batch_data[1].shape[1]
        print("class counts done")  # cca 100/8000
    """
    
    neighbours = gen.use_neighbours
    depth_variation = 0
    use_attention = True
    use_more_dense = True
    
    fields_input = Input(shape=gen.get_batchpadded_shapes()[0]['features'], name='features')  # per page, features
    
    # neighbours_ids_input = Input(shape=gen.get_batchpadded_shapes()[0]['neighbours'], name='neighbours',
    #                             dtype=np.int32)  # per page, features
    
    # merge positional embedding - from integers to sins coss
    # in the generator they are at the last 4 positions....
    positions_reading_order_embedded = SinCosPositionalEmbedding(4 * n_siz,
                                                                 from_inputs_features=[-1, -2, -3, -4],
                                                                 # embedd all 4 integers
                                                                 pos_divisor=10000,
                                                                 keep_ndim=True)(fields_input)
    # embedd lrtb
    positions_embedded = SinCosPositionalEmbedding(4 * n_siz,
                                                   from_inputs_features=[0, 1, 2, 3],
                                                   # embedd all 4 integers
                                                   pos_divisor=10000,
                                                   keep_ndim=True)(fields_input)
    
    merged_result = Concatenate()([fields_input, positions_reading_order_embedded, positions_embedded])
    
    fields_r = Conv1D(128 * n_siz, kernel_size=5, padding='same', activation='relu')(merged_result)
    
    matrix_all = Lambda(make_product_matrix)(fields_r)
    
    a = Conv2D(128 * n_siz, kernel_size=(5, 5), padding='same', activation='relu')(matrix_all)
    a = Conv2D(128 * n_siz, kernel_size=(5, 5), padding='same', activation='relu', dilation_rate=5)(a)
    a = Conv2D(128 * n_siz, kernel_size=(5, 5), padding='same', activation='relu')(a)
    
    def diag2d(arr):
        # input [..., batches, N, N, features]
        # output [..., batches, N, features]
        
        arrshape = tf.shape(arr)
        assert_op = tf.Assert(tf.equal(arrshape[-2], arrshape[-3]), [arr])
        with tf.control_dependencies([assert_op]):
            # assert arrshape[-3] == arrshape[-2] that it is square before features dimension
            newshape = tf.concat([arrshape[0:-3], [-1], arrshape[-1:]], axis=-1)
            arr = tf.reshape(arr, newshape)
            
            diagind = tf.range(arrshape[-3]) * (arrshape[-2] + 1)
            return tf.gather(
                arr,
                diagind,
                axis=-2,
            )
    
    def global_reduce(arr):
        return tf.concat([diag2d(arr), tf.reduce_max(arr, axis=-2), tf.reduce_mean(arr, axis=-2),
                          tf.reduce_max(arr, axis=-3), tf.reduce_mean(arr, axis=-3)],
                         axis=-1)
    
    sequence = Lambda(global_reduce)(a)
    sequence = Concatenate(axis=-1)([sequence, fields_r])
    
    if use_more_dense:
        bef_fork = TimeDistributed(Dense(64 * n_siz, activation='relu'))(sequence)
        bef_fork = Dropout(0.15)(bef_fork)
    else:
        bef_fork = sequence
    
    fork_node = TimeDistributed(Dense(64 * n_siz, activation='relu'))(
        bef_fork)  # from this point on, we will fork to different outputs!
    
    outclassesperbox = Dense(gen.get_batchpadded_shapes()[1][-1], activation='sigmoid')(fork_node)
    
    # outclassesperbox = Softmax()(outclassesperbox)
    
    def mean_pred(y_true, y_pred):
        return K.mean(y_pred)
    
    const_model = Model(inputs=[fields_input], outputs=outclassesperbox)
    const_model.compile(optimizer='adam',
                        # loss='categorical_crossentropy', metrics=[metrics.categorical_accuracy],
                        loss='binary_crossentropy', metrics=[metrics.binary_accuracy],
                        sample_weight_mode='temporal'
                        )
    const_model.summary()
    
    # val:
    eval_gen = tf_dataset_as_iterator(gen.get_final_tf_data_dataset(validation_pages, phase='val'))
    
    def report_f():
        result = evaluate(eval_gen, int(validation_pages / batch_size), const_model, None)
        return result
    
    callbacks = {'checkpointer': ModelCheckpoint(weights_best_fname,
                                                 monitor=key_metric, save_best_only=True, mode=key_metric_mode,
                                                 verbose=verbose),
                 'val_reporter': EvaluateFCallback(report_f,
                                                   monitor=key_metric,
                                                   mode=key_metric_mode
                                                   )
                 }
    
    print("Printing metrics on random model to see from where the first epoch started: ")
    callbacks['val_reporter'].on_epoch_end(0)
    
    if stop_early:
        callbacks['early_stopping'] = EarlyStopping(monitor=key_metric, patience=patience,
                                                    mode=key_metric_mode, verbose=verbose)
    
    hist = const_model.fit_generator(
        tf_dataset_as_iterator(gen.get_final_tf_data_dataset(pages_per_epoch, phase='train')),
        # .make_one_shot_iterator().get_next(),
        pages_per_epoch / batch_size, n_epochs,
        verbose=verbose,
        # class_weight=class_weight,
        # keras cannot use class weight and sample weights at the same time
        validation_data=tf_dataset_as_iterator(gen.get_final_tf_data_dataset(validation_pages, phase='val')),
        # we validate on the same set btw
        validation_steps=validation_pages / batch_size,
        callbacks=[callbacks[key] for key in callbacks],
        workers=0,  # because we use dataflow
        use_multiprocessing=False
    )
    return min(hist.history[key_metric])
