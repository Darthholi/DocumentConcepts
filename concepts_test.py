#
# COPYRIGHT Martin Holecek 2019
#

import numpy as np
import pytest
import tensorflow as tf
from scipy import stats as st

from concepts import DistributionOfDistributions
from concepts_fixed import constant_testing_setting, deep_eps_compare, fixed_experiment_binary, \
    constant_testing_setting_2pg, constant_wrong_testing_setting_2pg, run_keras_fixed_experiment_categorical, \
    random_testing_setting, random_testing_setting_distances, constant_testing_setting_multiclass, \
    constant_testing_setting_more_inboxes, small_testing_setting, small_testing_setting_fixdim_class
from concepts_rendered import run_keras_rendered_experiment_categorical
from distributions import gaussian_smoothed_discrete, jensen_snannon_divergence_monte_carlo, Determined, \
    StochasticScorableWrapper, FixdimDistribution
from generators import FixedNeighboursPacker, RenderedConceptsPacker
from utils import StructIndexer


def test_struct_indexer():
    """
    Dataflow objects use an object, that allows us to index to deeper structures.
    """
    struct_orig = {"a": [0, 1], "b": {"c": None}, "d": [[None], [None]]}
    str_o = StructIndexer.from_example(struct_orig)
    indexed = str_o.unpack_from(struct_orig)
    struct_created = str_o.pack_from(indexed)
    assert struct_created == struct_orig


def test_concepts_page_gen():
    """
    Test that we can draw objects from a testing setting.
    todo add test condition that all concepts are the same for the constant one...
    """
    page_c = constant_testing_setting()
    drawn = page_c.draw_objects(1)
    assert drawn is not None
    
    page_c = small_testing_setting()
    drawn = page_c.draw_objects(1)
    assert drawn is not None
    
    page_c = small_testing_setting_fixdim_class()
    drawn = page_c.draw_objects(1)
    assert drawn is not None


def test_concepts_tf_gen():
    """
    Test dataflow and tf.data.dataset data generators.
    """
    test_df_data = ({'in_boxes': np.array([[[0.145, 0., 0.155, 0., 0., 1.]],
    
                                           [[0.145, 0., 0.155, 0., 0., 1.]],
    
                                           [[0.145, 0., 0.155, 0., 0., 1.]]]),
                     'center_boxes': np.array([[0.05, 0., 0.15, 0.4, 0., 0.],
                                               [0.05, 0., 0.15, 0.4, 0., 0.],
                                               [0.05, 0., 0.15, 0.4, 0., 0.]])}, np.array([[1.],
                                                                                           [1.],
                                                                                           [1.]]),
                    np.array([1., 1., 1.]))
    
    pagegen_obj = constant_testing_setting()
    dataset = FixedNeighboursPacker(pagegen_obj,
                                    df_proc_num=1,
                                    batch_size=2,
                                    df_batches_to_prefetch=1)
    
    # test the dataflow part
    df = dataset.dataflow_packer(pages_per_epoch=2, phase='train')
    df.reset_state()
    for x in df.get_data():
        assert deep_eps_compare(x, test_df_data)
    
    def test_df_data_generator():
        while True:
            yield test_df_data
    
    # test the tensorflow generators:
    test_tfds = dataset.tf_data_dataset_batcher_from_generator(test_df_data_generator)
    test_iter = test_tfds.make_one_shot_iterator()
    xti, yti, wti = test_iter.get_next()
    with tf.Session() as sess:
        x, y, w = sess.run([xti, yti, wti])
        assert x['in_boxes'].shape == (2, 3, 1, 6) and x['center_boxes'].shape == (2, 3, 6)
        assert y.shape == (2, 3, 1)
        assert w.shape == (2, 3)
    
    # test compatibility of two parts
    train_set = dataset.get_final_tf_data_dataset(pages_per_epoch=2, )
    
    iterator = train_set.make_one_shot_iterator()
    
    xti, yti, wti = iterator.get_next()
    with tf.Session() as sess:
        x, y, w = sess.run([xti, yti, wti])
        assert x['in_boxes'].shape == (2, 3, 1, 6) and x['center_boxes'].shape == (2, 3, 6)
        assert y.shape == (2, 3, 1)
        assert w.shape == (2, 3)


@pytest.mark.parametrize('data_setting, accept_range', [
    (constant_testing_setting(), [0.9, 1.0]),
    (constant_testing_setting_2pg(), [0.9, 1.0]),
    (constant_wrong_testing_setting_2pg(), [0.6, 0.8]),
])
def test_constant_models_binary(data_setting, accept_range):
    """
    Test that the fixed binary experiment can easily figure the constant testing settings
    (and fail when we provide misleading data).
    """
    val_metric = fixed_experiment_binary(data_setting, df_proc_num=1)
    assert val_metric >= accept_range[0] and val_metric <= accept_range[1]


@pytest.mark.parametrize('data_setting, accept_range', [
    (random_testing_setting(), [0.9, 1.0]),  # test_well_separated_random
    (random_testing_setting_distances(), [0.3, 1.0]),  # test_distancedifferent_clipping
])
def test_constant_models_categorical(data_setting, accept_range):
    """
    Test that the fixed categorical experiment can easily figure the constant testing settings
    Sometimes it can be as high as 1.00, but sometimes it can fail because of the clipping.
    Can we make it work everytime?
    Yea by using bigger pages number per epoch the randomness will probably go away, but from the point of the test
    Lets set it to 30% to not fail randomly
    """
    val_metric = run_keras_fixed_experiment_categorical(data_setting, df_proc_num=1)
    assert val_metric >= accept_range[0] and val_metric <= accept_range[1]


@pytest.mark.parametrize('data_setting, zero_class, accept_range', [
    (constant_testing_setting_multiclass(), [1, 0], [0.9, 1.0]),
])
def test_rendered_models(data_setting, zero_class, accept_range):
    """
    Test that the rendered categorical model is able to figure out multiclass setting.
    """
    val_metric = run_keras_rendered_experiment_categorical(data_setting, zero_class=zero_class)
    assert val_metric >= accept_range[0] and val_metric <= accept_range[1]


def test_rendered_concepts_tf_gen():
    """
    Test data generators for rendered experiment setting.
    """
    test_df_data = ({'features': np.array([[0.145, 0., 0.155, 0., 0., 1., 0., 0., 0.,
                                            3.],
                                           [0.05, 0., 0.15, 0.4, 0., 0., 1., 0., 0.,
                                            0.],
                                           [0.145, 0., 0.155, 0., 0., 1., 2., 0., 0.,
                                            4.],
                                           [0.05, 0., 0.15, 0.4, 0., 0., 3., 0., 0.,
                                            1.],
                                           [0.145, 0., 0.155, 0., 0., 1., 4., 0., 0.,
                                            5.],
                                           [0.05, 0., 0.15, 0.4, 0., 0., 5., 0., 0.,
                                            2.]]),
                     'neighbours': np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 3, 5],
                                             [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 2, 4],
                                             [-1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 3, 5],
                                             [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 2, 4],
                                             [-1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 3, 5],
                                             [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 2, 4]], dtype=np.int32)},
                    np.array([[0.],
                              [1.],
                              [0.],
                              [1.],
                              [0.],
                              [1.]]), np.array([1., 1., 1., 1., 1., 1.]))
    
    pagegen_obj = constant_testing_setting()
    dataset = RenderedConceptsPacker(pagegen_obj,
                                     df_proc_num=1,
                                     batch_size=2,
                                     df_batches_to_prefetch=1, zero_class=[0])
    
    # test the dataflow part
    df = dataset.dataflow_packer(pages_per_epoch=2, phase='train')
    df.reset_state()
    for x in df.get_data():
        assert deep_eps_compare(x, test_df_data)
    
    def test_df_data_generator():
        while True:
            yield test_df_data
    
    # test the tensorflow generators:
    test_tfds = dataset.tf_data_dataset_batcher_from_generator(test_df_data_generator)
    test_iter = test_tfds.make_one_shot_iterator()
    xti, yti, wti = test_iter.get_next()
    with tf.Session() as sess:
        x, y, w = sess.run([xti, yti, wti])
        assert x['features'].shape == (2, 6, 14) and x['neighbours'].shape == (2, 6, 12)
        assert y.shape == (2, 6, 1)
        assert w.shape == (2, 6)
    
    # test compatibility of two parts
    train_set = dataset.get_final_tf_data_dataset(pages_per_epoch=1)
    
    iterator = train_set.make_one_shot_iterator()
    
    xti, yti, wti = iterator.get_next()
    with tf.Session() as sess:
        x, y, w = sess.run([xti, yti, wti])
        assert x['features'].shape == (1, 6, 14) and x['neighbours'].shape == (1, 6, 12)
        assert y.shape == (1, 6, 1)
        assert w.shape == (1, 6)


def test_scoring_samples():
    """
    tests that the drawn samples can be scored (at least for the constant distributions for now ...)
    """
    scorable = constant_testing_setting().concept_rules[0].concept_scorable
    samples = scorable.draw_samples(1)
    scores = scorable.score_samples(samples)
    assert scores == 1.0
    
    scorable = constant_testing_setting_more_inboxes().concept_rules[0].concept_scorable
    samples = scorable.draw_samples(1)
    scores = scorable.score_samples(samples)
    assert scores == 1.0
    
    samples = scorable.draw_samples(2)
    samples[1][1, 0, 0] = 0.0
    scores = scorable.score_samples(samples)
    assert np.all(scores == [1.0, 0.0])


def test_dist_of_dists():
    """
    Test that we can draw from distribution of distributions.
    """
    uniforms = DistributionOfDistributions(st.uniform, {'loc': FixdimDistribution(st.uniform(loc=0.0, scale=0.5), 1),
                                                        'scale': Determined(1.0)})
    samples = uniforms.draw_samples(1)
    score = uniforms.score_samples(samples)
    assert score is not None
    objects = uniforms.generated_to_objects(samples)
    assert objects is not None
    sampled_numbers = objects[0].draw_samples(1)
    assert sampled_numbers is not None


def test_dist_of_dists_nonfix():
    """
    Test that we can draw from distribution of distributions that is non fixed.
    """
    uniforms = DistributionOfDistributions(st.uniform,
                                           {'loc': StochasticScorableWrapper(st.uniform(loc=0.0, scale=0.5)),
                                            'scale': Determined(1.0)})
    samples = uniforms.draw_samples(1)
    score = uniforms.score_samples(samples)
    assert score is not None
    objects = uniforms.generated_to_objects(samples)
    assert objects is not None
    sampled_numbers = objects[0].draw_samples(1)
    assert sampled_numbers is not None


def test_jensen_shannon():
    """
    Test properties of jensen-shannon divergence numerical computation.
    """
    assert jensen_snannon_divergence_monte_carlo(st.norm(loc=10000), st.norm(loc=0)) >= 0.99
    assert jensen_snannon_divergence_monte_carlo(st.norm(loc=0), st.norm(loc=0)) <= 0.01
    assert jensen_snannon_divergence_monte_carlo(st.norm(loc=0, scale=1.0),
                                                 gaussian_smoothed_discrete(st.rv_discrete(values=
                                                                                           ([0], [1.0])
                                                                                           ), smooth_scale=1.0)) <= 0.01
    assert jensen_snannon_divergence_monte_carlo(st.norm(loc=0, scale=1.0),
                                                 gaussian_smoothed_discrete(st.rv_discrete(values=
                                                                                           ([0, 0], [0.2, 0.8])
                                                                                           ), smooth_scale=1.0)) <= 0.1
