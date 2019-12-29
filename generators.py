import numpy as np
import tensorflow as tf
from tensorpack import DataFromList, MapData, MultiProcessMapData

from boxgeometry import get_items_reading_layout, filter_seen_boxes
from concepts import ConceptsPageGen
from distributions import is_single_number

"""
Some notes:
tf data dataset and such knowhow hints:
- https://stackoverflow.com/questions/47086599/parallelising-tf-data-dataset-from-generator/47884927
- TFRecords are fast, but can be read only sequentially (for nonsequential acces use generator and/or LMDB)
- tf estimator initializes model again and again in evaluations/training (the usecase should be to train in a loop and
stop training when external process evaluates checkpointed model....)
https://medium.com/swlh/data-pipeline-using-tensorflow-dataset-api-with-keras-fit-generator-8c7a3e01c4fd
https://www.tensorflow.org/guide/datasets
https://stackoverflow.com/questions/46135499/how-to-properly-combine-tensorflows-dataset-api-and-keras
https://stackoverflow.com/questions/49840100/tf-data-dataset-padded-batch-pad-differently-each-feature
https://stackoverflow.com/questions/45955241/how-do-i-create-padded-batches-in-tensorflow-for-tf-train-sequenceexample-data-u
https://medium.com/@moritzkrger/speeding-up-keras-with-tfrecord-datasets-5464f9836c36
(keras sequence example : )
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

tf.keras and keras are different thongs so sometimes it needs this:
https://stackoverflow.com/questions/46135499/how-to-properly-combine-tensorflows-dataset-api-and-keras
"""


class PagesPacker(object):
    """
    This design is just a fast prototype, that paralelizes random page generators using dataflows (dataflow_packer
     and expand_data_call) and then batches and pads using tf data dataset.
    """
    
    def __init__(self, pagegen_obj, df_proc_num, batch_size, df_batches_to_prefetch):
        assert isinstance(pagegen_obj, ConceptsPageGen)
        self.pagegen = pagegen_obj
        self.df_proc_num = df_proc_num
        self.batch_size = batch_size
        self.df_batches_to_prefetch = df_batches_to_prefetch
    
    def expand_data_call(self, *args, **kwargs):
        return None
    
    def get_output_types(self):
        return None
    
    def get_batchpadded_shapes(self):
        return None
    
    def get_final_tf_data_dataset(self, pages_per_epoch, phase='train'):
        return self.tf_data_dataset_batcher_from_dataflow(self.dataflow_packer(pages_per_epoch, phase), phase)
    
    def dataflow_packer(self, pages_per_epoch,
                        phase='train',  # or val or predict
                        # random_state=42,
                        ):
        phase_fit = phase in ['train', 'val']  # validation happens in fitting also
        
        df = DataFromList(lst=list(range(pages_per_epoch)))  # just data indices
        if not df:
            return None
        
        buffer_size = self.batch_size * self.df_batches_to_prefetch
        orig_size = df.size()
        
        # at first, datapoint components are flat (for BatchData)
        if self.df_proc_num <= 1:
            df = MapData(df, self.expand_data_call)
        else:
            # btw in python 3 it can hang when n_proc = 1 AND buffer_size > 1 batch
            df = MultiProcessMapData(df, self.df_proc_num,
                                     self.expand_data_call,
                                     buffer_size=min(buffer_size, orig_size),
                                     strict=True)  # https://github.com/tensorpack/tensorpack/pull/794
        return df
    
    def tf_data_dataset_batcher_from_dataflow(self, ds, phase='train'):
        ds.reset_state()
        return self.tf_data_dataset_batcher_from_generator(ds.get_data, phase=phase)
    
    def tf_data_dataset_batcher_from_generator(self, gen, phase='train'):
        phase_fit = phase in ['train', 'val']  # validation happens in fitting also
        
        dataset = tf.data.Dataset.from_generator(
            gen,
            self.get_output_types(),
            output_shapes=None)
        
        '''
        if phase_fit:
            # is not needed when we generate it on the fly, but lets keep it here in case we provide fixed dataset
            #  and forget
            dataset = dataset.shuffle(1000)
        '''

        dataset = dataset.padded_batch(
            self.batch_size,
            padded_shapes=self.get_batchpadded_shapes())
        # default padding values are zeroes, ok.
        
        dataset = dataset.repeat()
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        return dataset
        

def np_samesize(arrays, arrays_defaults=0, axis=None):
    """
    All arrays will be padded to have the same size.
    If Axis is specified, concatenates the arrays along the axis, else returns list of the padded arrays.
    """
    dt_representant = arrays[0]
    if all([item.shape == dt_representant.shape for item in arrays]):
        if axis is not None:
            return np.concatenate(arrays, axis=axis)
        else:
            return arrays
    
    if is_single_number(arrays_defaults):
        arrays_defaults = [arrays_defaults] * len(arrays)
    
    assert len(arrays) == len(arrays_defaults), \
        "Provide either one default number or list of the same length as arryas"
    assert all([item.ndim == dt_representant.ndim for item in arrays]), "arrays need to be at least the same ndim"
    
    shape = [max([item.shape[i] for item in arrays])
             for i in range(dt_representant.ndim)]
    
    padded_arrays = [np.full(tuple(shape), arrays_defaults[k], dtype=array.dtype) for k, array in enumerate(arrays)]
    # Here we put all the data to the beginning and leave the padded at te end.
    # Another options are to put it at random position, end, or overflow...
    for i, item in enumerate(arrays):
        padded_arrays[i][tuple([slice(0, n) for n in item.shape])] = item
    
    if axis is not None:
        return np.concatenate(padded_arrays, axis=axis)
    else:
        return padded_arrays


class FixedNeighboursPacker(PagesPacker):
    """
    We provide the data to the network as is, so no rendering-onto-page-and-grabbing-back happens.

    What do we want to return? Various number of concepts per 'page', where each one has
     - different number of neighbours (each neigbour being a fixed size feature) <- and being and in_concept_box itself
     - its bbox (feature)
     - its output class(es) (targets)

    "Different numbers" are padded
    """
    
    def __init__(self, pagegen_obj, df_proc_num, batch_size, df_batches_to_prefetch):
        PagesPacker.__init__(self, pagegen_obj, df_proc_num, batch_size, df_batches_to_prefetch)
    
    def expand_data_call(self, *args, **kwargs):
        sample_page = self.pagegen.draw_objects(1)[0]
        # we do not do any clever logic, so each concept can have different number of neighbours,
        #  so we need to pad them over this page
        neighbours = []
        center_data = []
        output_data = []
        for concept in sample_page:
            if concept.in_concepts is not None and len(concept.in_concepts) > 0:
                neighbours.append(np.stack([inconcept.as_input_array() for inconcept in concept.in_concepts]))
            else:
                neighbours.append(np.empty([0, self.pagegen.get_in_concepts_feature_dim()]))
            center_data.append(np.concatenate([concept.bbox, concept.bbox_class]))
            output_data.append(concept.output_class)
        
        neighbours = np.stack(np_samesize(neighbours))
        center_data = np.stack(center_data)
        output_data = np.stack(output_data)
        # returned will have 3 dimensions [number on page, number neighbours, features]
        # aand [number on page, features]
        sample_weights = np.ones(center_data.shape[0])
        
        return {'in_boxes': neighbours, 'center_boxes': center_data}, output_data, sample_weights  # being keras's x,y
    
    def get_output_types(self):
        return ({'in_boxes': tf.float32, 'center_boxes': tf.float32}, tf.float32, tf.float32)
    
    def get_batchpadded_shapes(self):
        return ({'in_boxes': [None, None, self.pagegen.get_in_concepts_feature_dim()],
                 'center_boxes': [None, self.pagegen.get_center_data_feature_dim()]},
                [None, self.pagegen.get_output_class_feature_dim()],
                [None])


class FixedNeighboursAllPacker(PagesPacker):
    """
    We provide the data to the network as is, so no rendering-onto-page-and-grabbing-back happens.

    What do we want to return? Various number of concepts per 'page', where each one has
     - different number of neighbours (each neigbour being a fixed size feature) <- and being and in_concept_box itself
     - its bbox (feature)
     - its output class(es) (targets)

    "Different numbers" are padded
    """
    
    def __init__(self, pagegen_obj, df_proc_num, batch_size, df_batches_to_prefetch, zero_class, shuffle_neighbours):
        PagesPacker.__init__(self, pagegen_obj, df_proc_num, batch_size, df_batches_to_prefetch)
        assert zero_class is not None
        
        assert len(zero_class) == self.pagegen.get_output_class_feature_dim(), \
            "we insist, that the 'zero-class' is defined to be the same dimensionality as other classes in the " \
            "concepts"
        self.zero_class = zero_class
        # the zero class is applied on all unclassified boxes.
        
        assert self.pagegen.get_in_concepts_feature_dim() == self.pagegen.get_center_data_feature_dim(), \
            "All boxes should have the same feature-dimensionality"
        
        self.shuffle_neighbours = shuffle_neighbours
    
    def expand_data_call(self, *args, **kwargs):
        sample_page = self.pagegen.draw_objects(1)[0]
        # we do not do any clever logic, so each concept can have different number of neighbours,
        #  so we need to pad them over this page
        neighbours = []
        center_data = []
        output_data = []
        for concept in sample_page:
            center = np.concatenate([concept.bbox, concept.bbox_class])
            this_neighbours = np.stack([inconcept.as_input_array() for inconcept in concept.in_concepts])
            # the center:
            if concept.in_concepts is not None and len(concept.in_concepts) > 0:
                neighbours.append(this_neighbours)
            else:
                neighbours.append(np.empty([0, self.pagegen.get_in_concepts_feature_dim()]))
            center_data.append(center)
            output_data.append(concept.output_class)
            
            # all other views:
            for i, inconcept in enumerate(concept.in_concepts):
                neighb_replace = np.copy(this_neighbours)
                neighb_replace[i] = center
                neighbours.append(neighb_replace)
                
                center_data.append(inconcept.as_input_array())
                output_data.append(self.zero_class)
        
        if self.shuffle_neighbours:
            for n_i in neighbours:
                np.random.shuffle(n_i)
        
        neighbours = np.stack(np_samesize(neighbours))
        center_data = np.stack(center_data)
        output_data = np.stack(output_data)
        # returned will have 3 dimensions [number on page, number neighbours, features]
        # aand [number on page, features]
        assert center_data.shape[0] == output_data.shape[0]
        assert center_data.shape[0] == neighbours.shape[0]
        sample_weights = np.ones((center_data.shape[0],))
        
        return ({'in_boxes': neighbours, 'center_boxes': center_data}, output_data,
                sample_weights)
    
    def get_output_types(self):
        return ({'in_boxes': tf.float32, 'center_boxes': tf.float32}, tf.float32, tf.float32)
    
    def get_batchpadded_shapes(self):
        return ({'in_boxes': [None, None, self.pagegen.get_in_concepts_feature_dim()],
                 'center_boxes': [None, self.pagegen.get_center_data_feature_dim()]},
                [None, self.pagegen.get_output_class_feature_dim()],
                [None])


class RenderedConceptsPacker(PagesPacker):
    """
    We provide the data to the network after they are rendered on a page and then read back.

    Should be harder by two factors:
    - all the boxes are to be classified (not only the center ones)
    - they are read by a specific algorithm that tries to give all the inromation but is not an oracle.

    # todo note some difference, we do not make a box, that has nonzero output class also to be a necessary input for
    other class
    # is that okay bias? Lets hope so by our intuition

    """
    
    def __init__(self, pagegen_obj, df_proc_num, batch_size, df_batches_to_prefetch,
                 zero_class, use_neighbours=3,
                 bin_class_weights=(1.0, 1.0)):
        PagesPacker.__init__(self, pagegen_obj, df_proc_num, batch_size, df_batches_to_prefetch)
        assert zero_class is not None
        
        assert len(zero_class) == self.pagegen.get_output_class_feature_dim(), \
            "we insist, that the 'zero-class' is defined to be the same dimensionality as otjher classes in the " \
            "concepts"
        self.zero_class = zero_class
        # the zero class is applied on all unclassified boxes.
        
        assert self.pagegen.get_in_concepts_feature_dim() == self.pagegen.get_center_data_feature_dim(), \
            "All boxes should have the same feature-dimensionality"
        
        self.use_neighbours = use_neighbours
        
        self.bin_class_weights = bin_class_weights  # later can be made for each class differently...
    
    def expand_data_call(self, *args, **kwargs):
        sample_page = self.pagegen.draw_objects(1)[0]
        # we do not do any clever logic, so each concept can have different number of neighbours,
        #  so we need to pad them over this page
        
        all_boxes = []
        
        def add_box(box_data, box_class):
            all_boxes.append((box_data, box_class))
        
        for concept in sample_page:
            for inconcept in concept.in_concepts:
                add_box(inconcept.as_input_array(), self.zero_class)
            add_box(np.concatenate([concept.bbox, concept.bbox_class]), concept.output_class)
        
        read_data = self.bboxes_reading_algorithm(all_boxes)
        return read_data
        # we will return array of boxes, their in-classes and array of the resulting lasses
        # return {'in_boxes': neighbours, 'center_boxes': center_data}, output_data  # being keras's x,y
    
    def bboxes_reading_algorithm(self, all_boxes):
        reading_percents = [0.5, 0.5]
        
        all_boxes_features = [{'pos': box[0][0:4], 'in_class': box[0][4:], 'out_class': box[1]} for box in all_boxes]
        
        boxdata_to_lrtb = lambda item: item['pos']
        
        # first lets order all the boxes by the reading layout
        all_boxes_features = sum(get_items_reading_layout(all_boxes_features, key_f=boxdata_to_lrtb,
                                                          percent_thr_inline=reading_percents[0]), [])
        
        # then lets assign coordinates based on the reding order (and also rotated in y dir)
        for i, group in enumerate(get_items_reading_layout(all_boxes_features, key_f=boxdata_to_lrtb,
                                                           percent_thr_inline=reading_percents[0])):
            for j, field_feature in enumerate(group):
                field_feature['row_readings_pos'] = (i, j)
        for i, group in enumerate(
                get_items_reading_layout(all_boxes_features, key_f=boxdata_to_lrtb, row_by_row=False,
                                         percent_thr_inline=reading_percents[1])):
            for j, field_feature in enumerate(group):
                field_feature['col_readings_pos'] = (i, j)
        
        if self.use_neighbours > 0:
            # each field has its own data in the whole sequence AND the datas of 4*use_neighbours neighbours
            
            all_seen_data = []
            for ith, boxf in enumerate(all_boxes_features):
                all_seen_data.append((boxf, filter_seen_boxes(ith, list(range(len(all_boxes_features))),
                                                              key_f=lambda i: boxdata_to_lrtb(all_boxes_features[i]),
                                                              ortho_filter=False, max_sqr_dist=1000.0 * 1000.0,
                                                              order_by_dist=True, max_slots=self.use_neighbours)))
                # filter_seen returns {edge_id: (item, edge_id, dist)}
            defaulted_neigh_ids = []
            use_self = False  # will index to self data , false will index to minus
            for tid, seens in enumerate(all_seen_data):
                np_neigh_ids = np.full((4 * self.use_neighbours,), tid if use_self else -1, dtype=np.int32)
                # ^^ a default value for a neighbour, that is not present will be -1
                # (the index of exactly this field can also be used, but then the default batcher cannot do things
                # easily)
                # todo remember to give it the defaults and channel for dataflowing...
                for edge_n in range(4):
                    for nid, seen in enumerate(seens[1][edge_n]):  # max count of seens is known to be 'use_neighbours'
                        np_neigh_ids[edge_n * self.use_neighbours + nid] = seen[0]
                        # from (item, edge_id, dist) we take item which is id
                defaulted_neigh_ids.append(np_neigh_ids)
            
            f_neigh_ids = np.array(defaulted_neigh_ids, dtype=np.int32)
            # todo check if is produced an array of a len,4 size
        else:
            f_neigh_ids = None
        
        box_data = np.stack(
            [np.concatenate([box['pos'], box['in_class'], box['row_readings_pos'], box['col_readings_pos']])
             for box in all_boxes_features])
        box_out = np.array([box['out_class'] for box in all_boxes_features])
        
        # so the output is :
        # features - [num bboxes, features]
        # neighbours - [num bboxes, num neighbours]
        # outputs: - [num bboxes, out class dimensionality]
        # sample weights [num boxes, 1]
        
        # sample_weights = np.sum(box_out, axis=-1) * self.bin_class_weights[0] + \
        #                 np.sum(1 - box_out, axis=-1) * self.bin_class_weights[1]
        sample_weights = np.max(box_out, axis=-1) * self.bin_class_weights[0] + \
                         (1 - np.max(box_out, axis=-1)) * self.bin_class_weights[1]
        
        return ({'features': box_data, 'neighbours': f_neigh_ids},
                box_out,
                sample_weights)
    
    def get_output_types(self):
        return ({'features': tf.float32, 'neighbours': tf.int32},
                tf.float32,
                tf.float32)
    
    def get_batchpadded_shapes(self):
        return ({'features': [None, self.pagegen.get_in_concepts_feature_dim() + 4 + 4],  # for col/row_reading_pos
                 'neighbours': [None, 4 * self.use_neighbours]},
                [None, self.pagegen.get_output_class_feature_dim()],  # targets
                [None])  # weights