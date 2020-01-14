import numpy as np
import scipy.stats as st
from keras import Model, Input, metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, Concatenate, Dense, Softmax

from concepts import InputBoxRuleScorable, ConceptRuleDefinition, ConceptsPageGen
from distributions import conditioned_continuous, clipnorm, StochasticScorableWrapper, FixdimDistribution, \
    MultiDistribution, DistributionAsRadial, is_np_array, is_iterable
from generators import FixedNeighboursPacker, FixedNeighboursAllPacker
from utils import EvaluateFCallback
from utils import GlobalMaxPooling1DFrom4D, tf_dataset_as_iterator, evaluate


def constant_testing_setting():
    """
    One specific fixed combination resulting in class = 1
    """
    inputbs = InputBoxRuleScorable(input_classes=[0, 1],
                                   # positions=[0.1, 0.2],  # of the center of the box
                                   # sizes=[0.001, 0.002],
                                   positions=[0.1, 0.0],  # of the center of the box
                                   sizes=[0.01, 0.00],
                                   is_relative_position=True)
    
    concept = ConceptRuleDefinition(num_per_page=3,
                                    present=1.0,
                                    # center_position=[0.5, 0.51],
                                    # center_size=[0.01, 0.02],
                                    center_position=[0.1, 0.2],
                                    center_size=[0.1, 0.4],
                                    center_class=[0, 0],
                                    output_class=1,
                                    boxes_star_graph=[inputbs])
    
    page_c = ConceptsPageGen(num_different_rules_per_page=1,
                             num_noise_bboxes=0,
                             concept_rules=[concept],
                             noise_bbox_rule=None)
    return page_c


def small_testing_setting():
    """
        One specific fixed combination resulting in class = 1
        """
    inputbs = InputBoxRuleScorable(input_classes=[0, 1],
                                   # is domewhere around the center box
                                   positions=[StochasticScorableWrapper(st.uniform(loc=0.0, scale=0.5)),
                                              StochasticScorableWrapper(st.norm(loc=0.0, scale=0.5))],
                                   sizes=[StochasticScorableWrapper(
                                       clipnorm(loc=0.001, scale=0.0005, val_min=0.00001, val_max=1.0)),
                                       StochasticScorableWrapper(
                                           clipnorm(loc=0.0008, scale=0.0005, val_min=0.00001, val_max=1.0))],
                                   is_relative_position=True)
    
    concept = ConceptRuleDefinition(num_per_page=3,
                                    present=1.0,
                                    # center_position=[0.5, 0.51],
                                    # center_size=[0.01, 0.02],
                                    center_position=[0.1, 0.2],
                                    center_size=[0.1, 0.4],
                                    center_class=[0, 0],
                                    output_class=1,
                                    boxes_star_graph=[inputbs])
    
    page_c = ConceptsPageGen(num_different_rules_per_page=1,
                             num_noise_bboxes=0,
                             concept_rules=[concept],
                             noise_bbox_rule=None)
    return page_c


def small_testing_setting_fixdim_class():
    """
        One specific fixed combination resulting in class = 1
        """
    inputbs = InputBoxRuleScorable(input_classes=[0, 1],
                                   # is domewhere around the center box
                                   positions=[FixdimDistribution(st.uniform(loc=0.0, scale=0.5), 1),
                                              FixdimDistribution(st.norm(loc=0.0, scale=0.5), 1)],
                                   sizes=[StochasticScorableWrapper(
                                       clipnorm(loc=0.001, scale=0.0005, val_min=0.00001, val_max=1.0)),
                                       FixdimDistribution(
                                           clipnorm(loc=0.0008, scale=0.0005, val_min=0.00001, val_max=1.0), 1)],
                                   is_relative_position=True)
    
    concept = ConceptRuleDefinition(num_per_page=3,
                                    present=1.0,
                                    # center_position=[0.5, 0.51],
                                    # center_size=[0.01, 0.02],
                                    center_position=FixdimDistribution(st.uniform(loc=0, scale=0.01), 2),
                                    center_size=[0.1, 0.4],
                                    center_class=[0, 0],
                                    output_class=1,
                                    boxes_star_graph=[inputbs])
    
    page_c = ConceptsPageGen(num_different_rules_per_page=1,
                             num_noise_bboxes=0,
                             concept_rules=[concept],
                             noise_bbox_rule=None)
    return page_c


def constant_testing_setting_more_inboxes():
    """
    One specific fixed combination resulting in class = 1
    """
    inputbs = InputBoxRuleScorable(input_classes=[0, 1],
                                   # positions=[0.1, 0.2],  # of the center of the box
                                   # sizes=[0.001, 0.002],
                                   positions=[0.1, 0.0],  # of the center of the box
                                   sizes=[0.01, 0.00],
                                   is_relative_position=True)
    
    concept = ConceptRuleDefinition(num_per_page=3,
                                    present=1.0,
                                    # center_position=[0.5, 0.51],
                                    # center_size=[0.01, 0.02],
                                    center_position=[0.1, 0.2],
                                    center_size=[0.1, 0.4],
                                    center_class=[0, 0],
                                    output_class=1,
                                    boxes_star_graph=[inputbs, inputbs])
    
    page_c = ConceptsPageGen(num_different_rules_per_page=1,
                             num_noise_bboxes=0,
                             concept_rules=[concept],
                             noise_bbox_rule=None)
    return page_c


def constant_testing_setting_multiclass():
    """
    One specific fixed combination resulting in class = 1
    """
    inputbs = InputBoxRuleScorable(input_classes=[0, 1],
                                   # positions=[0.1, 0.2],  # of the center of the box
                                   # sizes=[0.001, 0.002],
                                   positions=[0.1, 0.0],  # of the center of the box
                                   sizes=[0.01, 0.00],
                                   is_relative_position=True)
    
    concept = ConceptRuleDefinition(num_per_page=3,
                                    present=1.0,
                                    # center_position=[0.5, 0.51],
                                    # center_size=[0.01, 0.02],
                                    center_position=[0.1, 0.2],
                                    center_size=[0.1, 0.4],
                                    center_class=[0, 0],
                                    output_class=[0, 1],
                                    boxes_star_graph=[inputbs])
    
    page_c = ConceptsPageGen(num_different_rules_per_page=1,
                             num_noise_bboxes=0,
                             concept_rules=[concept],
                             noise_bbox_rule=None)
    return page_c


def constant_testing_setting_2pg():
    """
    Two specific fixed combinations resulting in zero and 1 class.
    Differs only by positional and size information.
    """
    inputbs = InputBoxRuleScorable(input_classes=[0, 1],
                                   # positions=[0.1, 0.2],  # of the center of the box
                                   # sizes=[0.001, 0.002],
                                   positions=[0.1, 0.0],  # of the center of the box
                                   sizes=[0.01, 0.00],
                                   is_relative_position=True)
    
    concept = ConceptRuleDefinition(num_per_page=3,
                                    present=1.0,
                                    # center_position=[0.5, 0.51],
                                    # center_size=[0.01, 0.02],
                                    center_position=[0.1, 0.2],
                                    center_size=[0.1, 0.4],
                                    center_class=0,
                                    output_class=1,
                                    boxes_star_graph=[inputbs])
    
    # and zero class:
    inputbs0 = InputBoxRuleScorable(input_classes=[0, 1],
                                    # positions=[0.1, 0.2],  # of the center of the box
                                    # sizes=[0.001, 0.002],
                                    positions=[0.0, 0.1],  # of the center of the box
                                    sizes=[0.00, 0.01],
                                    is_relative_position=True)
    
    concept0 = ConceptRuleDefinition(num_per_page=2,
                                     present=1.0,
                                     # center_position=[0.5, 0.51],
                                     # center_size=[0.01, 0.02],
                                     center_position=[0.2, 0.1],
                                     center_size=[0.4, 0.1],
                                     center_class=0,
                                     output_class=0,
                                     boxes_star_graph=[inputbs0])
    
    page_c = ConceptsPageGen(num_different_rules_per_page=1,
                             num_noise_bboxes=0,
                             concept_rules=[concept, concept0],
                             noise_bbox_rule=None)
    return page_c


def constant_wrong_testing_setting_2pg():
    """
    A specifically designed unsatisfiable task (cinflicting data). Accuracy should never go above 0.7143
    """
    inputbs = InputBoxRuleScorable(input_classes=[0, 1],
                                   # positions=[0.1, 0.2],  # of the center of the box
                                   # sizes=[0.001, 0.002],
                                   positions=[0.1, 0.0],  # of the center of the box
                                   sizes=[0.01, 0.00],
                                   is_relative_position=True)
    
    concept = ConceptRuleDefinition(num_per_page=3,
                                    present=1.0,
                                    # center_position=[0.5, 0.51],
                                    # center_size=[0.01, 0.02],
                                    center_position=[0.1, 0.2],
                                    center_size=[0.1, 0.4],
                                    center_class=0,
                                    output_class=1,
                                    boxes_star_graph=[inputbs])
    
    # and zero class:
    inputbs0 = InputBoxRuleScorable(input_classes=[0, 1],
                                    # positions=[0.1, 0.2],  # of the center of the box
                                    # sizes=[0.001, 0.002],
                                    positions=[0.0, 0.1],  # of the center of the box
                                    sizes=[0.00, 0.01],
                                    is_relative_position=True)
    
    concept0 = ConceptRuleDefinition(num_per_page=2,
                                     present=1.0,
                                     # center_position=[0.5, 0.51],
                                     # center_size=[0.01, 0.02],
                                     center_position=[0.2, 0.1],
                                     center_size=[0.4, 0.1],
                                     center_class=0,
                                     output_class=0,
                                     boxes_star_graph=[inputbs0])
    
    # and wrong class:
    
    concept0w = ConceptRuleDefinition(num_per_page=2,
                                      present=1.0,
                                      # center_position=[0.5, 0.51],
                                      # center_size=[0.01, 0.02],
                                      center_position=[0.2, 0.1],
                                      center_size=[0.4, 0.1],
                                      center_class=0,
                                      output_class=1,
                                      boxes_star_graph=[inputbs0])
    
    page_c = ConceptsPageGen(num_different_rules_per_page=3,
                             num_noise_bboxes=0,
                             concept_rules=[concept, concept0, concept0w],
                             noise_bbox_rule=None)
    return page_c


def random_testing_setting():
    """
    Two specific fixed combinations resulting in zero and 1 class.
    Differs only by positional and size information (of the center) and number of neighbours.
    """
    inputbs = InputBoxRuleScorable(input_classes=[0, 1],
                                   # is domewhere around right from the center box
                                   positions=[StochasticScorableWrapper(st.uniform(loc=0.0, scale=0.5)),
                                              StochasticScorableWrapper(st.norm(loc=0.0, scale=0.5))],
                                   sizes=[StochasticScorableWrapper(
                                       clipnorm(loc=0.001, scale=0.0005, val_min=0.00001, val_max=1.0)),
                                       StochasticScorableWrapper(
                                           clipnorm(loc=0.0008, scale=0.0005, val_min=0.00001, val_max=1.0))],
                                   is_relative_position=True)
    
    concept1 = ConceptRuleDefinition(num_per_page=StochasticScorableWrapper(st.randint(high=3, low=1)),
                                     present=StochasticScorableWrapper(st.uniform(loc=0.9, scale=0.05)),
                                     # is mostly in the top part
                                     center_position=[StochasticScorableWrapper(st.uniform(loc=0.0, scale=1.0)),
                                                      StochasticScorableWrapper(st.uniform(loc=-0.5, scale=0.5))],
                                     center_size=[StochasticScorableWrapper(
                                         clipnorm(loc=0.001, scale=0.0005, val_min=0.00001, val_max=1.0)),
                                         StochasticScorableWrapper(
                                             clipnorm(loc=0.0008, scale=0.0005, val_min=0.00001,
                                                      val_max=1.0))],
                                     center_class=[0, 1],
                                     output_class=[0, 1, 0, 0],
                                     boxes_star_graph=[inputbs, inputbs])
    
    # concept 2 - at the bottom
    concept2 = ConceptRuleDefinition(num_per_page=StochasticScorableWrapper(st.randint(high=3, low=1)),
                                     present=StochasticScorableWrapper(st.uniform(loc=0.9, scale=0.05)),
                                     # is mostly in the bottom  right part
                                     center_position=[StochasticScorableWrapper(st.uniform(loc=0.0, scale=1.0)),
                                                      StochasticScorableWrapper(st.uniform(loc=0.5, scale=0.5))],
                                     center_size=[StochasticScorableWrapper(
                                         clipnorm(loc=0.001, scale=0.0005, val_min=0.00001, val_max=1.0)),
                                         StochasticScorableWrapper(
                                             clipnorm(loc=0.0008, scale=0.0005, val_min=0.00001,
                                                      val_max=1.0))],
                                     center_class=[0, 1],
                                     output_class=[0, 0, 1, 0],
                                     boxes_star_graph=[inputbs, inputbs])
    
    # concept 3 - anywhere AND only 1 neighbour
    concept3 = ConceptRuleDefinition(num_per_page=StochasticScorableWrapper(st.randint(high=3, low=1)),
                                     present=StochasticScorableWrapper(st.uniform(loc=0.9, scale=0.05)),
                                     # is mostly in the bottom part
                                     center_position=[StochasticScorableWrapper(st.uniform(loc=0.0, scale=1.0)),
                                                      StochasticScorableWrapper(st.uniform(loc=0.0, scale=1.0))],
                                     center_size=[StochasticScorableWrapper(
                                         clipnorm(loc=0.001, scale=0.0005, val_min=0.00001, val_max=1.0)),
                                         StochasticScorableWrapper(
                                             clipnorm(loc=0.0008, scale=0.0005, val_min=0.00001,
                                                      val_max=1.0))],
                                     center_class=[0, 1],
                                     output_class=[0, 0, 0, 1],
                                     boxes_star_graph=[inputbs])
    
    # concept 4 - nearly anywhere botright AND no neighbour
    concept4 = ConceptRuleDefinition(num_per_page=StochasticScorableWrapper(st.randint(high=3, low=1)),
                                     present=StochasticScorableWrapper(st.uniform(loc=0.9, scale=0.05)),
                                     # is mostly in the bottom part
                                     center_position=[StochasticScorableWrapper(st.uniform(loc=0.0, scale=0.9)),
                                                      StochasticScorableWrapper(st.uniform(loc=0.0, scale=0.9))],
                                     center_size=[StochasticScorableWrapper(
                                         clipnorm(loc=0.001, scale=0.0005, val_min=0.00001, val_max=1.0)),
                                         StochasticScorableWrapper(
                                             clipnorm(loc=0.0008, scale=0.0002, val_min=0.00001,
                                                      val_max=1.0))],
                                     center_class=[0, 1],
                                     output_class=[1, 0, 0, 0],
                                     boxes_star_graph=[])
    
    page_c = ConceptsPageGen(num_different_rules_per_page=StochasticScorableWrapper(st.randint(high=4, low=1)),
                             num_noise_bboxes=0,
                             concept_rules=[concept1, concept2, concept3, concept4],
                             noise_bbox_rule=None)
    return page_c


def random_testing_setting_distances():
    """
    4 classes differ by position of boxes AND their distance.
    Also the clipping might be an issue because we generate everything all around the page!
    """
    inputright = InputBoxRuleScorable(input_classes=[0, 1],
                                      # is domewhere around the center box
                                      positions=[StochasticScorableWrapper(st.uniform(loc=0.1, scale=0.04)),
                                                 StochasticScorableWrapper(st.norm(loc=0.0, scale=0.01))],
                                      sizes=[StochasticScorableWrapper(
                                          clipnorm(loc=0.001, scale=0.0005, val_min=0.0002, val_max=0.1)),
                                          StochasticScorableWrapper(
                                              clipnorm(loc=0.0008, scale=0.0005, val_min=0.0002, val_max=0.1))],
                                      is_relative_position=True)
    
    inputfarright = InputBoxRuleScorable(input_classes=[0, 1],
                                         # is domewhere around the center box
                                         positions=[StochasticScorableWrapper(st.uniform(loc=0.2, scale=0.14)),
                                                    StochasticScorableWrapper(st.norm(loc=0.0, scale=0.01))],
                                         sizes=[StochasticScorableWrapper(
                                             clipnorm(loc=0.001, scale=0.0005, val_min=0.0002, val_max=0.1)),
                                             StochasticScorableWrapper(
                                                 clipnorm(loc=0.0008, scale=0.0005, val_min=0.0002, val_max=0.1))],
                                         is_relative_position=True)
    
    inputtop = InputBoxRuleScorable(input_classes=[0, 1],
                                    # is domewhere around the center box
                                    positions=[StochasticScorableWrapper(st.uniform(loc=0.0, scale=0.01)),
                                               StochasticScorableWrapper(st.norm(loc=-0.3, scale=0.08))],
                                    sizes=[StochasticScorableWrapper(
                                        clipnorm(loc=0.001, scale=0.0005, val_min=0.0002, val_max=0.1)),
                                        StochasticScorableWrapper(
                                            clipnorm(loc=0.0008, scale=0.0005, val_min=0.0002, val_max=0.1))],
                                    is_relative_position=True)
    
    inputbottom = InputBoxRuleScorable(input_classes=[0, 1],
                                       # is domewhere around the center box
                                       positions=[StochasticScorableWrapper(st.uniform(loc=0.0, scale=0.01)),
                                                  StochasticScorableWrapper(st.norm(loc=0.4, scale=0.15))],
                                       sizes=[StochasticScorableWrapper(
                                           clipnorm(loc=0.001, scale=0.0005, val_min=0.0002, val_max=0.1)),
                                           StochasticScorableWrapper(
                                               clipnorm(loc=0.0008, scale=0.0005, val_min=0.0002, val_max=0.1))],
                                       is_relative_position=True)
    
    concept1 = ConceptRuleDefinition(num_per_page=StochasticScorableWrapper(st.randint(high=3, low=1)),
                                     present=StochasticScorableWrapper(st.uniform(loc=0.9, scale=0.05)),
                                     # is mostly in the bottom part
                                     center_position=[StochasticScorableWrapper(st.uniform(loc=0.0, scale=1.0)),
                                                      StochasticScorableWrapper(st.uniform(loc=0.0, scale=1.0))],
                                     center_size=[StochasticScorableWrapper(
                                         clipnorm(loc=0.001, scale=0.0005, val_min=0.0002, val_max=0.1)),
                                         StochasticScorableWrapper(
                                             clipnorm(loc=0.0008, scale=0.0005, val_min=0.0002, val_max=0.1))],
                                     center_class=[0, 1],
                                     output_class=[1, 0, 0, 0],
                                     boxes_star_graph=[inputright])
    
    # concept 2 - anywhere
    concept2 = ConceptRuleDefinition(num_per_page=StochasticScorableWrapper(st.randint(high=3, low=1)),
                                     present=StochasticScorableWrapper(st.uniform(loc=0.9, scale=0.05)),
                                     # is mostly in the bottom right part
                                     center_position=[StochasticScorableWrapper(st.uniform(loc=0.0, scale=1.0)),
                                                      StochasticScorableWrapper(st.uniform(loc=0.0, scale=1.0))],
                                     center_size=[StochasticScorableWrapper(
                                         clipnorm(loc=0.001, scale=0.0005, val_min=0.0002, val_max=0.1)),
                                         StochasticScorableWrapper(
                                             clipnorm(loc=0.0008, scale=0.0005, val_min=0.0002, val_max=0.1))],
                                     center_class=[0, 1],
                                     output_class=[0, 1, 0, 0],
                                     boxes_star_graph=[inputfarright])
    
    # concept 3 - anywhere
    concept3 = ConceptRuleDefinition(num_per_page=StochasticScorableWrapper(st.randint(high=3, low=1)),
                                     present=StochasticScorableWrapper(st.uniform(loc=0.9, scale=0.05)),
                                     # is mostly in the bottom right part
                                     center_position=[StochasticScorableWrapper(st.uniform(loc=0.0, scale=1.0)),
                                                      StochasticScorableWrapper(st.uniform(loc=0.0, scale=1.0))],
                                     center_size=[StochasticScorableWrapper(
                                         clipnorm(loc=0.001, scale=0.0005, val_min=0.0002, val_max=0.1)),
                                         StochasticScorableWrapper(
                                             clipnorm(loc=0.0008, scale=0.0005, val_min=0.0002, val_max=0.1))],
                                     center_class=[0, 1],
                                     output_class=[0, 0, 1, 0],
                                     boxes_star_graph=[inputtop, inputbottom])
    
    # concept 4 - anywhere
    concept4 = ConceptRuleDefinition(num_per_page=StochasticScorableWrapper(st.randint(high=3, low=1)),
                                     present=StochasticScorableWrapper(st.uniform(loc=0.9, scale=0.05)),
                                     # is mostly in the bottom  right part
                                     center_position=[StochasticScorableWrapper(st.uniform(loc=0.0, scale=1.0)),
                                                      StochasticScorableWrapper(st.uniform(loc=0.0, scale=1.0))],
                                     center_size=[StochasticScorableWrapper(
                                         clipnorm(loc=0.001, scale=0.0005, val_min=0.0002, val_max=0.1)),
                                         StochasticScorableWrapper(
                                             clipnorm(loc=0.0008, scale=0.0002, val_min=0.0002, val_max=0.1))],
                                     center_class=[0, 1],
                                     output_class=[0, 0, 0, 1],
                                     boxes_star_graph=[inputtop, inputtop])
    
    page_c = ConceptsPageGen(num_different_rules_per_page=StochasticScorableWrapper(st.randint(high=4, low=1)),
                             num_noise_bboxes=0,
                             concept_rules=[concept1, concept2, concept3, concept4],
                             noise_bbox_rule=None)
    return page_c


def deep_eps_compare(a, b, eps=0.0000001):
    if not (type(a) is type(b)):
        return False
    if isinstance(a, dict):
        try:
            keys = list(a.keys()) + list(b.keys())
            a = [a[key] for key in keys]
            b = [b[key] for key in keys]
        except:
            return False
    
    if is_np_array(a):
        return ((np.asarray(a) - np.asarray(b)) < eps).all()
    
    if is_iterable(a):
        if len(a) != len(b):
            return False
        for ita, itb in zip(a, b):
            if not deep_eps_compare(ita, itb, eps):
                return False
        return True
    return False


def fixed_experiment_binary(const_data_def=constant_testing_setting_2pg(),
                            validation_pages=6,
                            n_epochs=20,
                            verbose=2,
                            stop_early=True,
                            key_metric='val_loss',
                            weights_best_fname='weightstmp.h5',
                            patience=4,
                            key_metric_mode='min',
                            pages_per_epoch=2,
                            batch_size=2,
                            df_proc_num=2,
                            zero_class=0,
                            predict_all_boxes=False,
                            shuffle_bboxes=False
                            ):
    """
    Runs a simplest model against a (mostly) fixed dataset and returns the last epoch accuracy.
    """
    if predict_all_boxes:
        gen = FixedNeighboursAllPacker(const_data_def,
                                       df_proc_num=df_proc_num,
                                       batch_size=batch_size,
                                       df_batches_to_prefetch=1, zero_class=zero_class,
                                       shuffle_neighbours=shuffle_bboxes)
    else:
        gen = FixedNeighboursPacker(const_data_def,
                                    df_proc_num=df_proc_num,
                                    batch_size=batch_size,
                                    df_batches_to_prefetch=1)
    
    in_boxes = Input(shape=gen.get_batchpadded_shapes()[0]['in_boxes'],
                     name='in_boxes')  # per page, neighbours, features
    center_boxes = Input(shape=gen.get_batchpadded_shapes()[0]['center_boxes'],
                         name='center_boxes')  # per page, features
    
    c1 = Conv2D(filters=8, kernel_size=(1, 5), padding='same', activation='relu')(in_boxes)
    c2 = Conv2D(filters=8, kernel_size=(1, 5), padding='same', activation='relu')(c1)
    gcb = GlobalMaxPooling1DFrom4D()(c2)
    fpb = Concatenate(axis=-1)([center_boxes, gcb])
    fpb = Dense(8, activation='relu')(fpb)
    outclassesperbox = Dense(gen.get_batchpadded_shapes()[1][-1], activation='sigmoid')(fpb)
    
    const_model = Model(inputs=[in_boxes, center_boxes], outputs=outclassesperbox)
    const_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[metrics.binary_accuracy,
                                                                               ],
                        sample_weight_mode="temporal")
    const_model.summary()
    
    # val:
    eval_gen = tf_dataset_as_iterator(gen.get_final_tf_data_dataset(validation_pages, phase='val'))
    
    def report_f():
        result = evaluate(eval_gen, int(validation_pages / batch_size), const_model, None)
        return result
    
    callbacks = {'checkpointer': ModelCheckpoint(weights_best_fname,
                                                 monitor=key_metric, save_best_only=True, mode=key_metric_mode,
                                                 verbose=verbose),
    
                 # 'datetime': DatetimePrinter(),
                 # 'procnum': ProcNumPrinter(),
                 # 'memory': MemoryPrinter(),
                 # 'weights_printer': PrintWeightsStats(check_on_batch=debug),
                 # 'nanterminator': TerminateOnNaNRemember(),
                 # 'sacred': SacredCallback(self.run, 'val_loss'),
                 'val_reporter': EvaluateFCallback(report_f,
                                                   monitor=key_metric,
                                                   mode=key_metric_mode
                                                   )
                 }
    if stop_early:
        callbacks['early_stopping'] = EarlyStopping(monitor=key_metric, patience=patience,
                                                    mode=key_metric_mode, verbose=verbose)
    
    validation_steps = validation_pages / batch_size
    hist = const_model.fit_generator(
        tf_dataset_as_iterator(gen.get_final_tf_data_dataset(pages_per_epoch, phase='train')),
        # .make_one_shot_iterator().get_next(),
        pages_per_epoch / batch_size, n_epochs,
        verbose=verbose,
        # class_weight=class_weight,
        # keras cannot use class weight and sample weights at the same time
        validation_data=tf_dataset_as_iterator(gen.get_final_tf_data_dataset(validation_pages, phase='val')),
        # we validate on the same set because it is a random generator, so the data are never the same
        validation_steps=validation_steps,
        callbacks=[callbacks[key] for key in callbacks],
        workers=0,  # because we use dataflow
        use_multiprocessing=False,
    )
    return max(hist.history['val_binary_accuracy'])


def run_keras_fixed_experiment_binary_bigger(const_data_def=constant_testing_setting_2pg(),
                                             validation_pages=6,
                                             n_epochs=20,
                                             verbose=2,
                                             stop_early=True,
                                             key_metric='val_loss',
                                             weights_best_fname='weightstmp.h5',
                                             patience=4,
                                             key_metric_mode='min',
                                             pages_per_epoch=2,
                                             batch_size=2,
                                             df_proc_num=2,
                                             zero_class=0,
                                             predict_all_boxes=False,
                                             shuffle_bboxes=False
                                             ):
    """
    Runs a simplest model against a (mostly) fixed dataset and returns the last epoch accuracy.
    """
    if predict_all_boxes:
        gen = FixedNeighboursAllPacker(const_data_def,
                                       df_proc_num=df_proc_num,
                                       batch_size=batch_size,
                                       df_batches_to_prefetch=1, zero_class=zero_class,
                                       shuffle_neighbours=shuffle_bboxes)
    else:
        gen = FixedNeighboursPacker(const_data_def,
                                    df_proc_num=df_proc_num,
                                    batch_size=batch_size,
                                    df_batches_to_prefetch=1)
    
    in_boxes = Input(shape=gen.get_batchpadded_shapes()[0]['in_boxes'],
                     name='in_boxes')  # per page, neighbours, features
    center_boxes = Input(shape=gen.get_batchpadded_shapes()[0]['center_boxes'],
                         name='center_boxes')  # per page, features
    
    c1 = Conv2D(filters=64, kernel_size=(1, 5), padding='same', activation='relu')(in_boxes)
    c2 = Conv2D(filters=64, kernel_size=(1, 5), padding='same', activation='relu')(c1)
    c3 = Conv2D(filters=64, kernel_size=(1, 5), padding='same', activation='relu')(c2)
    gcb = GlobalMaxPooling1DFrom4D()(c3)
    fpb = Concatenate(axis=-1)([center_boxes, gcb])
    fpb = Dense(64, activation='relu')(fpb)
    fpb = Dense(64, activation='relu')(fpb)
    outclassesperbox = Dense(gen.get_batchpadded_shapes()[1][-1], activation='sigmoid')(fpb)
    
    const_model = Model(inputs=[in_boxes, center_boxes], outputs=outclassesperbox)
    const_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[metrics.binary_accuracy,
                                                                               ],
                        sample_weight_mode="temporal")
    const_model.summary()
    
    # val:
    eval_gen = tf_dataset_as_iterator(gen.get_final_tf_data_dataset(validation_pages, phase='val'))
    
    def report_f():
        result = evaluate(eval_gen, int(validation_pages / batch_size), const_model, None)
        return result
    
    callbacks = {'checkpointer': ModelCheckpoint(weights_best_fname,
                                                 monitor=key_metric, save_best_only=True, mode=key_metric_mode,
                                                 verbose=verbose),
    
                 # 'datetime': DatetimePrinter(),
                 # 'procnum': ProcNumPrinter(),
                 # 'memory': MemoryPrinter(),
                 # 'weights_printer': PrintWeightsStats(check_on_batch=debug),
                 # 'nanterminator': TerminateOnNaNRemember(),
                 # 'sacred': SacredCallback(self.run, 'val_loss'),
                 'val_reporter': EvaluateFCallback(report_f,
                                                   monitor=key_metric,
                                                   mode=key_metric_mode
                                                   )
                 }
    if stop_early:
        callbacks['early_stopping'] = EarlyStopping(monitor=key_metric, patience=patience,
                                                    mode=key_metric_mode, verbose=verbose)
    
    validation_steps = validation_pages / batch_size
    hist = const_model.fit_generator(
        tf_dataset_as_iterator(gen.get_final_tf_data_dataset(pages_per_epoch, phase='train')),
        # .make_one_shot_iterator().get_next(),
        pages_per_epoch / batch_size, n_epochs,
        verbose=verbose,
        # class_weight=class_weight,
        # keras cannot use class weight and sample weights at the same time
        validation_data=tf_dataset_as_iterator(gen.get_final_tf_data_dataset(validation_pages, phase='val')),
        # we validate on the same set because it is a random generator, so the data are never the same
        validation_steps=validation_steps,
        callbacks=[callbacks[key] for key in callbacks],
        workers=0,  # because we use dataflow
        use_multiprocessing=False,
    )
    return max(hist.history['val_binary_accuracy'])


def run_keras_fixed_experiment_categorical(const_data_def=constant_testing_setting_2pg(),
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
                                           df_proc_num=2
                                           ):
    """
    Runs a simplest model against a (mostly) fixed dataset and returns the last epoch accuracy.
    """
    
    gen = FixedNeighboursPacker(const_data_def,
                                df_proc_num=df_proc_num,
                                batch_size=batch_size,
                                df_batches_to_prefetch=1)
    
    in_boxes = Input(shape=gen.get_batchpadded_shapes()[0]['in_boxes'],
                     name='in_boxes')  # per page, neighbours, features
    center_boxes = Input(shape=gen.get_batchpadded_shapes()[0]['center_boxes'],
                         name='center_boxes')  # per page, features
    
    c1 = Conv2D(filters=8, kernel_size=(1, 5), padding='same', activation='relu')(in_boxes)
    c2 = Conv2D(filters=8, kernel_size=(1, 5), padding='same', activation='relu')(c1)
    gcb = GlobalMaxPooling1DFrom4D()(c2)
    fpb = Concatenate(axis=-1)([center_boxes, gcb])
    fpb = Dense(8, activation='relu')(fpb)
    outclassesperbox = Dense(gen.get_batchpadded_shapes()[1][-1], activation='sigmoid')(fpb)
    outclassesperbox = Softmax()(outclassesperbox)
    
    const_model = Model(inputs=[in_boxes, center_boxes], outputs=outclassesperbox)
    const_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[metrics.binary_accuracy,
                                                                                    metrics.categorical_accuracy
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
    
    validation_steps = int(validation_pages / batch_size)
    hist = const_model.fit_generator(
        tf_dataset_as_iterator(gen.get_final_tf_data_dataset(pages_per_epoch, phase='train')),
        # .make_one_shot_iterator().get_next(),
        pages_per_epoch, n_epochs,
        verbose=verbose,
        # class_weight=class_weight,
        # keras cannot use class weight and sample weights at the same time
        validation_data=tf_dataset_as_iterator(gen.get_final_tf_data_dataset(validation_steps, phase='val')),
        # we validate on the same set because it is a random generator, so the data are never the same
        validation_steps=validation_steps,
        callbacks=[callbacks[key] for key in callbacks],
        workers=0,  # because we use dataflow
        use_multiprocessing=False
    )
    return max(hist.history['val_categorical_accuracy'])


def realistic_setting(tot_classes=18, num_pp_high=13, num_pp_low=7, keep_near=False):
    """
    - it does not look like an invoice on first sight, it has much less noise!
    - it should have common only the basic 'bias' that decisions are done based on 'concepts'
    - this setting has some named concepts
      - then we have added 'mass' different concepts, because our setting is sparsely multiclass
      (look at presets at labels.py)

    - class counts are cca 100 positive / per 8000 total
    """
    
    def s_uniform(loc, scale):
        if isinstance(loc, list):
            return st.uniform(loc=np.asarray(loc) - np.asarray(scale), scale=2 * np.asarray(scale))
        else:
            return st.uniform(loc=loc - scale, scale=2 * scale)
    
    def anywhere(s=1.0):
        return [StochasticScorableWrapper(s_uniform(loc=0.0, scale=2.0 * s)),
                StochasticScorableWrapper(s_uniform(loc=0.0, scale=1.0 * s))]
    
    def circular_around(distance=2 * 0.16):
        phir = MultiDistribution([StochasticScorableWrapper(st.norm(loc=distance, scale=distance * 0.3)),
                                  StochasticScorableWrapper(st.norm(loc=0, scale=2))])
        return DistributionAsRadial(phir)
    
    def corners_peaks_around(distance=0.16):
        return MultiDistribution(
            [StochasticScorableWrapper(conditioned_continuous(st.rv_discrete(values=([0, 1], [0.5, 0.5])),
                                                              [st.norm(loc=-distance, scale=distance * 0.5),
                                                               st.norm(loc=distance, scale=distance * 0.5)])),
             StochasticScorableWrapper(
                 conditioned_continuous(st.rv_discrete(values=([0, 1], [0.5, 0.5])),
                                        [st.norm(loc=-distance, scale=distance * 0.5),
                                         st.norm(loc=distance, scale=distance * 0.5)]))])
    
    def top_and_bot_around(distance=0.16):
        return MultiDistribution([StochasticScorableWrapper(s_uniform(loc=0, scale=distance * 2)),
                                  StochasticScorableWrapper(
                                      conditioned_continuous(st.rv_discrete(values=([0, 1], [0.5, 0.5])),
                                                             [st.norm(loc=-distance, scale=distance * 0.5),
                                                              st.norm(loc=distance, scale=distance * 0.5)]))])
    
    def left_and_right_around(distance=0.16):
        return MultiDistribution([
            StochasticScorableWrapper(
                conditioned_continuous(st.rv_discrete(values=([0, 1], [0.5, 0.5])),
                                       [st.norm(loc=-distance, scale=distance * 0.5),
                                        st.norm(loc=distance, scale=distance * 0.5)])),
            StochasticScorableWrapper(s_uniform(loc=0, scale=distance * 2))])
    
    def generally_this_position(dist):
        dist_x, dist_y = dist
        return [StochasticScorableWrapper(s_uniform(loc=dist_x,
                                                    scale=0.5 * word_size[0])),
                StochasticScorableWrapper(st.norm(loc=dist_y, scale=0.5 * word_size[1]))]
    
    def generally_this_way(phi):
        phir = MultiDistribution([StochasticScorableWrapper(s_uniform(loc=phi, scale=np.pi * 0.3)),
                                  StochasticScorableWrapper(s_uniform(loc=0.5, scale=0.5))])
        return DistributionAsRadial(phir)
    
    def out_class(i, tot_c=tot_classes):
        ret = np.zeros(tot_c)
        if i >= 0:
            ret[i] = 1
        return ret
    
    # lets say 13 x 18 words per page (250 words pper page distributed evenly on A4 dimensions)
    # which is for viewport -1->+1 : 0.1538, 0.1111
    word_size = (0.1538, 0.1111)
    
    def word_sized(scales=(1, 1)):
        # lets say 13 x 18 words per page (250 words pper page distributed evenly on A4 dimensions)
        # which is for viewport -1->+1 : 0.1538, 0.1111
        return [StochasticScorableWrapper(s_uniform(loc=word_size[0], scale=0.5 * word_size[0])),
                StochasticScorableWrapper(
                    clipnorm(loc=word_size[1], scale=0.3 * word_size[1], val_min=0.05, val_max=0.3))]
    
    def pseudo_word_embedding_total():
        return FixdimDistribution(st.norm(loc=[0.5, 0.2, 0, 0.9],
                                          scale=[0.11, 0.09, 0.1, 0.02]), item_dimension=4)
    
    def pseudo_word_embedding_page():
        return FixdimDistribution(s_uniform(loc=[0.1, 0.1, 0.7, 0.5],
                                            scale=[0.21, 0.39, 0.01, 0.2]), item_dimension=4)
    
    def pseudo_number_embedding():
        return FixdimDistribution(st.norm(loc=[0.3, 0.2, 0.9, 0.1],
                                          scale=[0.1, 0.4, 0.2, 0.1]), item_dimension=4)
    
    def pseudo_word_embedding_tax():
        return FixdimDistribution(st.norm(loc=[0.7, 0.3, 0.4, 0.5],
                                          scale=[0.21, 0.05, 0.6, 0.04]), item_dimension=4)
    
    def random_embbedding():
        return FixdimDistribution(s_uniform(loc=[0.5, 0.5, 0.5, 0.5],
                                            scale=[0.5, 0.5, 0.5, 0.5]), item_dimension=4)
    
    def grid_point(i, tot, dim):
        # total grid points  is t^dim
        assert i < tot ** dim
        d = 1.0 / tot
        loc = d * np.asarray([(i % ((j + 1) * tot)) for j in range(dim)])
        return loc, d
    
    def grid_point_dist(i, t=4):
        loc, d = grid_point(i, t, 4)
        return FixdimDistribution(st.norm(loc=loc,
                                          scale=np.full_like(loc, d)), item_dimension=4)
    
    def generated_inp(index):
        
        if keep_near:
            tot_distances = 4
            distances = [0.5 * (i + 0.5) * word_size[0] for i in range(tot_distances)]
        else:
            tot_distances = 4
            distances = [(i + 1.5) * word_size[0] for i in range(tot_distances)]
        tot_positions = 4 ** 2
        positions = [0.4 * grid_point(i, 4, 2)[0] for i in range(tot_positions)]
        tot_dirs = 8
        directions = [2 * np.pi * (i / tot_dirs) for i in range(tot_dirs)]
        if keep_near:
            poss_opt = [(circular_around, distances), (corners_peaks_around, distances),
                        (top_and_bot_around, distances), (left_and_right_around, distances),
                        ]
        else:
            poss_opt = [(circular_around, distances), (corners_peaks_around, distances),
                        (top_and_bot_around, distances), (left_and_right_around, distances),
                        (generally_this_position, positions),
                        (generally_this_way, directions),
                        (anywhere, [0.5])]
        p_i = index % len(poss_opt)
        r_i = int(index / len(poss_opt))
        pos_sel_o = poss_opt[p_i][0](poss_opt[p_i][1][r_i % len(poss_opt[p_i][1])])
        return InputBoxRuleScorable(input_classes=grid_point_dist(index),
                                    positions=pos_sel_o,
                                    sizes=word_sized(),
                                    is_relative_position=True,
                                    name="{}".format(index))
    
    noise_bbox = InputBoxRuleScorable(input_classes=random_embbedding(),
                                      # is domewhere around the center box
                                      positions=anywhere(0.7),
                                      sizes=word_sized(),
                                      is_relative_position=False,
                                      name="noise")
    
    inputleft_tax = InputBoxRuleScorable(input_classes=pseudo_word_embedding_tax(),
                                         # is domewhere around the center box
                                         positions=[StochasticScorableWrapper(s_uniform(loc=-1.7 * word_size[0],
                                                                                        scale=0.5 * word_size[0])),
                                                    StochasticScorableWrapper(
                                                        st.norm(loc=0.0, scale=0.1 * word_size[1]))],
                                         sizes=word_sized(),
                                         is_relative_position=True,
                                         name="tax")
    
    inputleft_total = InputBoxRuleScorable(input_classes=pseudo_word_embedding_total(),
                                           # issomewhere around the center box
                                           positions=[StochasticScorableWrapper(s_uniform(loc=-1.7 * word_size[0],
                                                                                          scale=0.2 * word_size[0])),
                                                      StochasticScorableWrapper(
                                                          st.norm(loc=0.0, scale=0.1 * word_size[1]))],
                                           sizes=word_sized(),
                                           is_relative_position=True,
                                           name="total")
    
    inputtop_total = InputBoxRuleScorable(input_classes=pseudo_word_embedding_total(),
                                          # is domewhere around the center box
                                          positions=[StochasticScorableWrapper(s_uniform(loc=0.0,
                                                                                         scale=0.1 * word_size[0])),
                                                     StochasticScorableWrapper(
                                                         st.norm(loc=-1.5 * word_size[1], scale=0.5 * word_size[1]))],
                                          sizes=word_sized(),
                                          is_relative_position=True,
                                          name="total")
    
    inputsome_page = InputBoxRuleScorable(input_classes=pseudo_word_embedding_page(),
                                          # is domewhere around the center box
                                          positions=circular_around(),
                                          sizes=word_sized(),
                                          is_relative_position=True,
                                          name="page")
    
    concept_tax_and_value = ConceptRuleDefinition(num_per_page=StochasticScorableWrapper(st.randint(high=2, low=1)),
                                                  present=StochasticScorableWrapper(s_uniform(loc=0.9, scale=0.05)),
    
                                                  center_position=anywhere(0.9),
                                                  center_size=word_sized(),
                                                  center_class=pseudo_number_embedding(),
                                                  output_class=out_class(0),
                                                  boxes_star_graph=[inputleft_tax],
                                                  name="tax-value")
    
    concept_total_and_value = ConceptRuleDefinition(num_per_page=StochasticScorableWrapper(st.randint(high=2, low=1)),
                                                    present=StochasticScorableWrapper(s_uniform(loc=0.9, scale=0.05)),
    
                                                    center_position=anywhere(0.9),
                                                    center_size=word_sized(),
                                                    center_class=pseudo_number_embedding(),
                                                    output_class=out_class(1),
                                                    boxes_star_graph=[inputleft_total],
                                                    name="total-value")
    
    concept_totaltax_and_value = ConceptRuleDefinition(
        num_per_page=StochasticScorableWrapper(st.randint(high=2, low=1)),
        present=StochasticScorableWrapper(s_uniform(loc=0.9, scale=0.05)),
        
        center_position=anywhere(0.9),
        center_size=word_sized(),
        center_class=pseudo_number_embedding(),
        output_class=out_class(0),
        boxes_star_graph=[inputtop_total, inputleft_tax],
        name="(total)-tax-value")
    
    concept_page_and_value = ConceptRuleDefinition(num_per_page=StochasticScorableWrapper(st.randint(high=2, low=1)),
                                                   present=StochasticScorableWrapper(s_uniform(loc=0.9, scale=0.05)),
    
                                                   center_position=anywhere(0.9),
                                                   center_size=word_sized(),
                                                   center_class=pseudo_number_embedding(),
                                                   output_class=out_class(2),
                                                   boxes_star_graph=[inputsome_page],
                                                   name="page-value")
    
    concept_auto_gen = []
    xaccum = 0
    for i in range(tot_classes - 3):
        inp_boxes = [generated_inp(xaccum + j) for j in range(i % 3 + 1)]
        xaccum += len(inp_boxes)
        concept_new_a = ConceptRuleDefinition(num_per_page=StochasticScorableWrapper(st.randint(high=2, low=1)),
                                              present=StochasticScorableWrapper(s_uniform(loc=0.9, scale=0.05)),
        
                                              center_position=anywhere(0.9),
                                              center_size=word_sized(),
                                              center_class=grid_point_dist(i),
                                              output_class=out_class(3 + i),
                                              boxes_star_graph=inp_boxes,
                                              name="gen-{}".format(i))
        concept_auto_gen.append(concept_new_a)
    
    if num_pp_high == num_pp_low:
        diff_rules = num_pp_high
    else:
        diff_rules = StochasticScorableWrapper(st.randint(high=num_pp_high, low=num_pp_low))
    
    page_c = ConceptsPageGen(num_different_rules_per_page=diff_rules,
                             num_noise_bboxes=StochasticScorableWrapper(st.randint(high=2, low=0)),
                             noise_bbox_rule=noise_bbox,
                             concept_rules=[concept_page_and_value, concept_tax_and_value, concept_total_and_value,
                                            concept_totaltax_and_value] + concept_auto_gen,
                             )
    return page_c
