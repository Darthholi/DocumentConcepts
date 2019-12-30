#!/usr/bin/python
# -*- coding: utf8 -*-
"""

"""

from __future__ import print_function, division

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from copy import copy
from sacred import Experiment
from sacred.initialize import Scaffold
from sacred.utils import apply_backspaces_and_linefeeds

from distributions import conditioned_continuous, StochasticScorableWrapper, FixdimDistribution, MultiDistribution, \
    DistributionAsRadial
from concepts_fixed import run_keras_fixed_experiment_categorical, random_testing_setting_distances, realistic_setting, constant_testing_setting_multiclass, run_keras_fixed_experiment_binary
from concepts_rendered import run_keras_articlemodel, run_keras_rendered_experiment_binary, run_keras_rendered_experiment_categorical
from concepts_test import test_jensen_shannon


# % matplotlib inline
def noop(item):
    pass


# https://gab41.lab41.org/effectively-running-thousands-of-experiments-hyperopt-with-sacred-dfa53b50f1ec
Scaffold._warn_about_suspicious_changes = noop


def deepdictify(config):
    ret = dict()
    for item in config:
        value = config[item]
        if isinstance(value, dict):
            ret[item] = deepdictify(value)
        elif isinstance(value, list):
            ret[item] = list(value)
        else:
            ret[item] = copy(value)
    return ret


concept_experiments = Experiment('concepts_experiments')
concept_experiments.captured_out_filter = apply_backspaces_and_linefeeds


def draw_lrtg_bbox(bbox, **kwargs):
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], **kwargs)
    return rect


def draw_texted_bbox(ax, bbox, text, **kwargs):
    ax.add_patch(draw_lrtg_bbox(bbox, **kwargs))
    cx, cy = lrtb_center(bbox)
    ax.annotate(text, (cx, cy), color='w', weight='bold',
                fontsize=6, ha='center', va='center')


def lrtb_center(bbox):
    cx = (bbox[0] + bbox[2]) * 0.5
    cy = (bbox[1] + bbox[3]) * 0.5
    return cx, cy


@concept_experiments.command
def sample_concepts_example():
    """
    Visual inspection for concepts randomly generated from setting.
    """
    page_c = realistic_setting()
    
    for i in range(10):
        drawn_page = page_c.draw_objects(1)[0]
        lims = (-1.0, 1.0)
        fig1 = plt.figure(figsize=(10, 10))
        ax1 = fig1.add_subplot(111, aspect='equal')
        for concept in drawn_page:
            c_ce = lrtb_center(concept.bbox)
            txt = concept.params["name"] if "name" in concept.params else type(concept).__name__
            draw_texted_bbox(ax1, concept.bbox, txt, color='blue')
            for in_concept in concept.in_concepts:
                txt = in_concept.params["name"] if "name" in in_concept.params else type(concept).__name__
                draw_texted_bbox(ax1, in_concept.bbox, txt, color='red')
                i_ce = lrtb_center(in_concept.bbox)
                
                ax1.plot([c_ce[0], i_ce[0]], [c_ce[1], i_ce[1]], color='green', marker='o', linestyle='dashed',
                         linewidth=2, markersize=12)
        plt.ylim(lims)
        plt.xlim(lims)
        # fig1.show()
        fig1.savefig("plot{}.png".format(i))
        print("saved figure")


@concept_experiments.command
def paint_distribution():
    """
    Visual int
    """
    """
    gaussian_smoothed_discrete(st.rv_discrete(values=
                                              ([0], [1.0])
                                              ), smooth_scale=1.0)gaussian_smoothed_discrete(st.rv_discrete(values=
                                                                                           ([0], [1.0])
                                                                                           ), smooth_scale=1.0)
    """
    #  dist = st.norm(loc=[0], scale=[1])
    # dist = st.gamma(loc=[0], scale=[1], a=1.)
    
    dist = conditioned_continuous(st.rv_discrete(values=([0, 1], [0.5, 0.5])),
                                  [st.norm(loc=-2, scale=1), st.norm(loc=2, scale=1)])
    
    fig1 = plt.figure()  # figsize=(10, 10))
    ax1 = fig1.add_subplot(111, aspect='equal')
    
    data = dist.rvs(size=(500,))
    ax1.hist(data, density=True, bins=100)
    
    X = np.linspace(-5.0, 5.0, 100)
    ax1.plot(X, dist.pdf(X), label='PDF')
    
    # ax1.plot([c_ce[0], i_ce[0]], [c_ce[1], i_ce[1]], color='green', marker='o', linestyle='dashed',
    #         linewidth=2, markersize=12)
    plt.ylim((-1, 1))
    plt.xlim((-5, 5))
    # fig1.show()
    fig1.savefig("plotdist.png")
    print("saved figure")


@concept_experiments.command
def paint_2ddistribution():
    dist = MultiDistribution([StochasticScorableWrapper(st.norm(loc=0.3, scale=0.05)),
                              StochasticScorableWrapper(st.norm(loc=0, scale=2))])
    dist = DistributionAsRadial(dist)
    """
    dist = MultiDistribution([StochasticScorableWrapper(conditioned_continuous(st.rv_discrete(values=([0, 1], [0.5, 0.5])),
                                                                               [st.norm(loc=-2, scale=1), st.norm(loc=2, scale=1)])),
                                                        StochasticScorableWrapper(
                                                            conditioned_continuous(st.rv_discrete(values=([0, 1], [0.5, 0.5])),
                                                     [st.norm(loc=-2, scale=1), st.norm(loc=2, scale=1)]))])
    """
    
    fig1 = plt.figure()  # figsize=(10, 10))
    ax1 = fig1.add_subplot(111, aspect='equal')
    
    data = dist.draw_samples((500,))
    
    ax1.plot(data[..., 0], data[..., 1], 'bo')
    
    # ax1.plot([c_ce[0], i_ce[0]], [c_ce[1], i_ce[1]], color='green', marker='o', linestyle='dashed',
    #         linewidth=2, markersize=12)
    # plt.ylim((-5, 5))
    # plt.xlim((-5, 5))
    # fig1.show()
    fig1.savefig("plot2dist.png")
    print("saved figure")


@concept_experiments.command
def fixed_known_borders():
    """
    #todo delete this one and
    with known borders of concepts: 'apriori info'
    """
    max_acc = run_keras_fixed_experiment_binary(realistic_setting(tot_classes=4, num_pp_high=2, num_pp_low=1),
                                      validation_pages=100,
                                      n_epochs=100,
                                      verbose=2,
                                      stop_early=True,
                                      key_metric='val_loss',
                                      weights_best_fname='weightstmp.h5',
                                      patience=20,
                                      key_metric_mode='min',
                                      pages_per_epoch=200,
                                      batch_size=8,
                                      df_proc_num=1,
                                      )
    print(max_acc)
    # realistic_setting(tot_classes=4, num_pp_high=2, num_pp_low=1)
    # gets to 0.97 bin acc, ourf1nonbg: 0.83 all f1micro 0.96
    

@concept_experiments.command
def realistic_experiment():
    """
    #with known borders of concepts: 'apriori info'
    max_acc = run_keras_fixed_experiment_binary(realistic_setting(tot_classes=4, num_pp_high=2, num_pp_low=1),
                                      validation_pages=100,
                                      n_epochs=100,
                                      verbose=2,
                                      stop_early=True,
                                      key_metric='val_loss',
                                      weights_best_fname='weightstmp.h5',
                                      patience=20,
                                      key_metric_mode='min',
                                      pages_per_epoch=200,
                                      batch_size=8,
                                      df_proc_num=2,
                                      )
    print(max_acc)
    # realistic_setting(tot_classes=4, num_pp_high=2, num_pp_low=1)
    # gets to 0.97 bin acc, ourf1nonbg: 0.83 all f1micro 0.96
    """
    
    """
    # witth known borderss of concepts: 'apriori info' but bigger
    max_acc = run_keras_fixed_experiment_binary_bigger(realistic_setting(tot_classes=18, num_pp_high=13, num_pp_low=7),
                                                validation_pages=100,
                                                n_epochs=100,
                                                verbose=2,
                                                stop_early=True,
                                                key_metric='val_loss',
                                                weights_best_fname='weightstmp.h5',
                                                patience=20,
                                                key_metric_mode='min',
                                                pages_per_epoch=200,
                                                batch_size=8,
                                                df_proc_num=2,
                                                )
    print(max_acc)
    # realistic_setting(tot_classes=4, num_pp_high=2, num_pp_low=1)
    # gets to 0.97 bin acc, ourf1nonbg: 0.94 all f1micro 0.98
    """
    
    """
    # witth known borderss of concepts: AND AALL BOXES
    max_acc = run_keras_fixed_experiment_binary(realistic_setting(tot_classes=4, num_pp_high=2, num_pp_low=1),
                                                zero_class=[0]*4,
                                                validation_pages=100,
                                                n_epochs=100,
                                                verbose=1,
                                                stop_early=True,
                                                key_metric='val_loss',
                                                weights_best_fname='weightstmp.h5',
                                                patience=20,
                                                key_metric_mode='min',
                                                pages_per_epoch=200,
                                                batch_size=8,
                                                df_proc_num=2,
                                                predict_all_boxes=True,
                                                shuffle_bboxes=False
                                                )
    print(max_acc)
    # realistic_setting(tot_classes=4, num_pp_high=2, num_pp_low=1), predict all, shuffle false
    # gets again to accuracy 0.96 and nonbg f1 0.76
    # realistic_setting(tot_classes=18, num_pp_high=13, num_pp_low=7)
    """
    
    """
    # witth known borderss of concepts: AND AALL BOXES and bigger
    max_acc = run_keras_fixed_experiment_binary_bigger(realistic_setting(tot_classes=18, num_pp_high=13, num_pp_low=7),
                                                zero_class=[0] * 18,
                                                validation_pages=100,
                                                n_epochs=100,
                                                verbose=2,
                                                stop_early=True,
                                                key_metric='val_loss',
                                                weights_best_fname='weightstmp.h5',
                                                patience=20,
                                                key_metric_mode='min',
                                                pages_per_epoch=200,
                                                batch_size=8,
                                                df_proc_num=2,
                                                predict_all_boxes=True,
                                                shuffle_bboxes=False
                                                )
    print(max_acc)
    # realistic_setting(tot_classes=4, num_pp_high=2, num_pp_low=1), predict all, shuffle false
    # goes to nonbg micro f1 0.87 (all micro f1 0.98)
    # realistic_setting(tot_classes=18, num_pp_high=13, num_pp_low=7),
    # nonbg f1 0.91
    """
    
    """
    # witth known borderss of concepts: AND AALL BOXES and bigger AND SHUFFLING
    max_acc = run_keras_fixed_experiment_binary_bigger(realistic_setting(tot_classes=18, num_pp_high=13, num_pp_low=7),
                                                       zero_class=[0] * 18,
                                                       validation_pages=100,
                                                       n_epochs=100,
                                                       verbose=2,
                                                       stop_early=True,
                                                       key_metric='val_loss',
                                                       weights_best_fname='weightstmp.h5',
                                                       patience=20,
                                                       key_metric_mode='min',
                                                       pages_per_epoch=200,
                                                       batch_size=8,
                                                       df_proc_num=2,
                                                       predict_all_boxes=True,
                                                       shuffle_bboxes=True
                                                       )
    print(max_acc)
    # realistic_setting(tot_classes=18, num_pp_high=13, num_pp_low=7),
    # # goes to nonbg micro f1 0.88 (all micro f1 0.99)
    """
    
    """
    # baseline
    run_keras_rendered_experiment_binary(realistic_setting(tot_classes=4, num_pp_high=2, num_pp_low=1), zero_class=[0]*4,
                                      validation_pages=100,
                                      n_epochs=100,
                                      verbose=2,
                                      stop_early=True,
                                      key_metric='val_loss',
                                      weights_best_fname='weightstmp.h5',
                                      patience=20,
                                      key_metric_mode='min',
                                      pages_per_epoch=200,
                                      batch_size=8,
                                      df_proc_num=2,
                                      neighbours=3,
                                      bin_class_weights=(80.0, 1.0),
                                         )
    # realistic_setting(tot_classes=4, num_pp_high=2, num_pp_low=1)
    # goes to 0.59 nonbg micro
    with
    bin_class_weights=(4.0, 1.0) goes to 0.80 nonbg micro f1
    """
    
    """
    maxscore = run_keras_articlemodel(realistic_setting(tot_classes=4, num_pp_high=1, num_pp_low=1), zero_class=[0]*4,
                                      validation_pages=100,
                                      n_epochs=100,
                                      verbose=2,
                                      stop_early=True,
                                      key_metric='val_loss',
                                      weights_best_fname='weightstmp.h5',
                                      patience=20,
                                      key_metric_mode='min',
                                      pages_per_epoch=200,
                                      batch_size=8,
                                      df_proc_num=2,
                                      neighbours=3,
                                      # bin_class_weights=(80.0, 1.0),  # from 100/8000 positive class count for tot_classes=18, num_pp_high=13, num_pp_low=7
                                      bin_class_weights=(4.0, 1.0), # from 80/700 positive class count for tot_classes=4, num_pp_high=1, num_pp_low=1
                                      n_siz=2
                                      )
    print(maxscore)
    """
    
    # articlemodel:
    # tot_classes=18, num_pp_high=13, num_pp_low=7:
    # gets to nonbg micro f1 to >> 0.35 << in 90 epochs with bin_class_weights=(800.0, 1.0)
    # same for bin_class_weights=(80.0, 1.0)
    # so to see the results, it needed to see like 90*200 pages
    
    # (tot_classes=10, num_pp_high=7, num_pp_low=4:
    # 0.36, bin_class_weights=(80.0, 1.0)
    
    # tot_classes=8, num_pp_high=5, num_pp_low=2
    # 0.40  # was on nu, neiighbours = 3, 5 ddoes not help, 7 does not help
    # tot_classes=8, num_pp_high=5, num_pp_low=2 & neighbours = 1 helps
    # - 0.42
    
    # tot_classes = 4, num_pp_high = 2, num_pp_low = 1
    # 0.674
    
    # ok lets see hhow it can do it with apriori info ('run_keras_fixed_experiment_binary') and only with center bboxes
    # gets to 0.97 binary acc but ourf1nonbg: 0.83, all f1micro 0.96
    # 1) so first we need a bigger network? ('run_keras_fixed_experiment_binary_bigger')
    # ok when making the network bigger it is able to - gets to 0.97 bin acc, ourf1nonbg: 0.94 all f1micro 0.98
    # funny when  realistic_setting(tot_classes=18, num_pp_high=13, num_pp_low=7):
    # gets to acc: 0.99, nonbg f1: 0.98, micto fr: 0.99
    
    # 2) now more realistic baseline - trying to predict all boxes (based on all others)
    # realistic_setting(tot_classes=4, num_pp_high=2, num_pp_low=1), predict all, shuffle false
    # goes to nonbg micro f1 0.87 (all micro f1 0.98)
    # realistic_setting(tot_classes=18, num_pp_high=13, num_pp_low=7),
    # nonbg f1 0.91
    # predicting all boxes is not a problem
    # 3) Trying to predict all boxes and shuffled! Ok the same results nearly as 2) (1% higher even) so it is not memorizing anything
    # shuffling is not the problem
    # 4) So, the problem is either in the rendering-derendering.
    
    # lets either -- make classifier of noncenter bboxes  but WITH apriori info!
    # (or eval only the center ones)
    
    # when tried prediccting all bboxes ( predict_all_boxes=True,), we got to 0.97 again.
    # ... even in the case of tot_classes=18, num_pp_high=13, num_pp_low=7
    
    # so when we  run the article model  on 1 concept per page  it should be also high....?
    # scores are nonbg f1s:
    # realistic_setting(tot_classes=18, num_pp_high=1, num_pp_low=1), neighbours = 1
    # 0.47
    # realistic_setting(tot_classes=10, num_pp_high=1, num_pp_low=1), neighbours = 3
    # 0.40
    # realistic_setting(tot_classes=4, num_pp_high=2, num_pp_low=1), neighbours = 3. n_siz=1
    # 0.63 (nonarticle just baseline gets to 0.59)
    # realistic_setting(tot_classes=4, num_pp_high=2, num_pp_low=1), neighbours = 3. n_siz=2
    # 0.43
    # realistic_setting(tot_classes=4, num_pp_high=1, num_pp_low=1), neighbours = 3. n_siz=2, bin_class_weights=(4.0, 1.0)
    # 0.67
    
    # so we have 2 outcomes:
    # it helps to let the network see ALL boxes to ALL boxes in this case and not only using attention
    # the problem is different - in invoices there is some information we thought would be redundant when modeling with concepts.
    # but so far when rendering-derendering only concepts, the network is unable to continue with a good precision.
    
    """
    class counts are cca 100 positive / per 8000 total

    has effectivity of a random guessing, needs class weights:
        def multiclass_temoral_class_weights(self, targets, class_weights):
        s_weights = np.ones((targets.shape[0],))
        # if we are counting the classes, the weights do not exist yet!
        if class_weights is not None:
            for i in range(len(s_weights)):
                weight = 0.0
                for itarget, target in enumerate(targets[i]):
                    weight += class_weights[itarget][int(round(target))]
                s_weights[i] = weight
        return s_weights

    """
    
    """
    # since we believe that it is now a harder task, lets try a model that sees everything without attention:
    maxscore = run_keras_all2all_model(realistic_setting(tot_classes=4, num_pp_high=1, num_pp_low=1), zero_class=[0] * 4,
                                      validation_pages=100,
                                      n_epochs=100,
                                      verbose=2,
                                      stop_early=True,
                                      key_metric='val_loss',
                                      weights_best_fname='weightstmp.h5',
                                      patience=20,
                                      key_metric_mode='min',
                                      pages_per_epoch=200,
                                      batch_size=4,
                                      df_proc_num=2,
                                      neighbours=3,
                                      bin_class_weights=(1.0, 1.0),
                                      #bin_class_weights=(80.0, 1.0),  # from 100/8000 positive class count
                                      n_siz=1
                                      )
    print(maxscore)
    # 0.79 nonbg f1 tot classes = 4, weights  1,1 nsiz2
    # 0.65 nonbg f1 tot classes = 8, weights  1,1 nsiz=1
    """
    
    # now we have inspected the data and found out that the boundingboxes are more local.
    # will that change our success?
    # run_keras_articlemodel did not get better
    #
    """
    maxscore = run_keras_articlemodel(realistic_setting(tot_classes=4, num_pp_high=1, num_pp_low=1, keep_local=True),
                                      zero_class=[0] * 4,
                                      validation_pages=100,
                                      n_epochs=100,
                                      verbose=2,
                                      stop_early=True,
                                      key_metric='val_loss',
                                      weights_best_fname='weightstmp.h5',
                                      patience=20,
                                      key_metric_mode='min',
                                      pages_per_epoch=200,
                                      batch_size=8,
                                      df_proc_num=2,
                                      neighbours=3,
                                      # bin_class_weights=(80.0, 1.0),  # from 100/8000 positive class count for
                                      # tot_classes=18, num_pp_high=13, num_pp_low=7
                                      bin_class_weights=(4.0, 1.0),
                                      # from 80/700 positive class count for tot_classes=4, num_pp_high=1, num_pp_low=1
                                      n_siz=2
                                      )
    print(maxscore)
    """
    run_keras_rendered_experiment_binary(
        realistic_setting(tot_classes=4, num_pp_high=2, num_pp_low=1, keep_local=False),
        zero_class=[0] * 4,
        validation_pages=100,
        n_epochs=100,
        verbose=2,
        stop_early=True,
        key_metric='val_loss',
        weights_best_fname='weightstmp.h5',
        patience=20,
        key_metric_mode='min',
        pages_per_epoch=200,
        batch_size=8,
        df_proc_num=2,
        neighbours=3,
        bin_class_weights=(4.0, 1.0),
    )
    # realistic_setting(tot_classes=4, num_pp_high=2, num_pp_low=1, keep_local=False) with
    # bin_class_weights=(4.0, 1.0) goes to 0.86 nonbg micro f1


@concept_experiments.command
def constant_experiment():
    run_keras_fixed_experiment_categorical(random_testing_setting_distances())
    # test_constant_models()


@concept_experiments.command
def rendered_experiment():
    # test_rendered_concepts_tf_gen()
    '''
    OK
    run_keras_rendered_experiment_categorical(constant_testing_setting_multiclass(), zero_class=[1, 0],
                                              validation_pages=100,
                                              n_epochs=100,
                                              verbose=1,
                                              stop_early=True,
                                              key_metric='val_categorical_accuracy',
                                              weights_best_fname='weightstmp.h5',
                                              patience=15,
                                              key_metric_mode='max',
                                              pages_per_epoch=200,
                                              batch_size=2,
                                              df_proc_num=2
                                              )
                                              # should be > 0.9 (we got to 1.0)
    '''
    '''
    maxscore = run_keras_rendered_experiment_categorical(random_testing_setting(), zero_class=[1, 0, 0, 0],
                                              validation_pages=100,
                                              n_epochs=100,
                                              verbose=2,
                                              stop_early=True,
                                              key_metric='val_categorical_accuracy',
                                              weights_best_fname='weightstmp.h5',
                                              patience=15,
                                              key_metric_mode='max',
                                              pages_per_epoch=200,
                                              batch_size=4,
                                              df_proc_num=2
                                              )
    print(maxscore)  # 0.8151
    #  why - because the example has only two ways of differentiating between classes - center position AND number of
    # neighbors in a concept. which we cannot know for sure. OK.
    '''
    
    '''
    random_testing_setting_distances - same classes of nearly everything but depends on directions and distances
    sxpscore = run_keras_rendered_experiment_binary(random_testing_setting_distances(), zero_class=[0, 0, 0, 0],
                                         validation_pages=20,
                                         n_epochs=100,
                                         verbose=2,
                                         stop_early=True,
                                         key_metric='val_loss',
                                         weights_best_fname='weightstmp.h5',
                                         patience=20,
                                         key_metric_mode='min',
                                         pages_per_epoch=100,
                                         batch_size=4,
                                         df_proc_num=2
                                         )
    okay the base model with our problem definition and its nonbg f1 metric fails misreably.
    '''
    
    '''
    ok this one also displays the f1 nonbg score and it gets only to like 0.43
    (actually jumps 0.40-0.45)
    Is not 100% because the discretization happens only based on 1) center positions (okay) 2) distances, which we cannot know for sure,
    because we do not know to which concept they belong now.
    maxscore = run_keras_articlemodel(random_testing_setting_distances(), zero_class=[0, 0, 0, 0],
                                                         validation_pages=100,
                                                         n_epochs=100,
                                                         verbose=2,
                                                         stop_early=True,
                                                         key_metric='val_loss',
                                                         weights_best_fname='weightstmp.h5',
                                                         patience=20,
                                                         key_metric_mode='min',
                                                         pages_per_epoch=200,
                                                         batch_size=8,
                                                         df_proc_num=2
                                                         )
    print(maxscore)
    ... has, of course, bigger succes (even in keras metric only) when we approach it as categorical problem.
    But by definition our problems are binary classifications.
    '''
    
    # so far the results are:
    # - finished rendered/derenderer
    # - articlemodel able to tackle hard things at least somehow
    # - better then the simplest model
    # - dropped score for obvious hard cases
    # we want: random rule definition to test the average cases
    # and do the analysis of the networks capacity based on hyperparameters
    # we can: find a better architecture then the one from article
    #  ... the generator needs to work for true label reuse
    
    # lets see, make the epochs bigger, but keep binary problem
    maxscore = run_keras_articlemodel(random_testing_setting_distances(), zero_class=[0, 0, 0, 0],
                                      validation_pages=100,
                                      n_epochs=100,
                                      verbose=2,
                                      stop_early=True,
                                      key_metric='val_loss',
                                      weights_best_fname='weightstmp.h5',
                                      patience=20,
                                      key_metric_mode='min',
                                      pages_per_epoch=200,
                                      batch_size=8,
                                      df_proc_num=2,
                                      neighbours=3
                                      )
    print(maxscore)


@concept_experiments.command
def try_tf():
    # to gather the diagnal
    import tensorflow as tf
    def diag2d(arr):
        """
        input [..., batches, N, N, features]
        output [..., batches, N, features]
        """
        arrshape = tf.shape(arr)
        # assert arrshape[-3] == arrshape[-2] that it is square before features dimension
        newshape = tf.concat([arrshape[0:-3], [-1], arrshape[-1:]], axis=-1)
        arr = tf.reshape(arr, newshape)
        
        diagind = tf.range(arrshape[-3]) * (arrshape[-2] + 1)
        return tf.gather(
            arr,
            diagind,
            axis=-2,
        )
    
    inp = tf.constant([[[1], [4]],
                       [[5], [2]]])
    outp = diag2d(inp)
    
    inp2 = tf.constant([[[1, 0], [4, 5], [0, 0]],
                        [[5, 6], [2, 2], [1, 1]],
                        [[4, 4], [3, 1], [3, 100]]])
    outp2 = diag2d(inp2)
    
    with tf.Session() as sess:
        # Run the initializer on `w`.
        # sess.run(tf.initialize_all_variables())
        got_result = sess.run([outp, outp2])  # or a list of things.
        print(got_result)


@concept_experiments.command
def try_scipy():
    import scipy.stats as st
    import numpy as np
    dist_try = st.uniform(loc=0, scale=10.0)  # st.norm(loc=0, scale=1)#
    points = dist_try.rvs(size=2)
    print(points)
    scores = dist_try.pdf(points)
    print(scores)
    
    dist_try = st.uniform(loc=[0, 5], scale=[1.0, 1.0])  # st.norm(loc=0, scale=1)#
    points = dist_try.rvs(size=(100, 2))
    print(points)
    scores = dist_try.pdf(points)
    print(scores)
    
    dist_try = FixdimDistribution(st.uniform, item_dimension=2, loc=0.5, scale=0.1)
    points = dist_try.rvs(size=(100, 2))
    print(points)
    
    # still then we need to compute things like https://stackoverflow.com/questions/26079881/kl-divergence-of-two-gmms
    
    def distributions_js(distribution_p, distribution_q, n_samples=10 ** 5):
        # jensen shannon divergence. (Jensen shannon distance is the square root of the divergence)
        # all the logarithms are defined as log2 (because of information entrophy)
        X = distribution_p.rvs(size=n_samples)
        p_X = distribution_p.pdf(X)
        q_X = distribution_q.pdf(X)
        log_mix_X = np.log2(p_X + q_X)
        
        Y = distribution_q.rvs(size=n_samples)
        p_Y = distribution_p.pdf(Y)
        q_Y = distribution_q.pdf(Y)
        log_mix_Y = np.log2(p_Y + q_Y)
        
        jsp_m = np.log2(p_X).mean() - (log_mix_X.mean() - np.log2(2))
        jsq_m = np.log2(q_Y).mean() - (log_mix_Y.mean() - np.log2(2))
        
        return (jsp_m + jsq_m) / 2
    
    def distributions_discrete_js(distribution_p, distribution_q, n_samples=10 ** 5):
        # jensen shannon divergence. (Jensen shannon distance is the square root of the divergence)
        # all the logarithms are defined as log2 (because of information entrophy)
        X = distribution_p.rvs(size=n_samples)
        p_X = distribution_p.pmf(X)
        q_X = distribution_q.pmf(X)
        log_mix_X = np.log2(p_X + q_X)
        
        Y = distribution_q.rvs(size=n_samples)
        p_Y = distribution_p.pmf(Y)
        q_Y = distribution_q.pmf(Y)
        log_mix_Y = np.log2(p_Y + q_Y)
        
        return (np.log2(p_X).mean() - (log_mix_X.mean() - np.log2(2))
                + np.log2(q_Y).mean() - (log_mix_Y.mean() - np.log2(2))) / 2
    
    def distributions_discrete_cont_js(distribution_p, distribution_q, n_samples=10 ** 5):
        # jensen shannon divergence. (Jensen shannon distance is the square root of the divergence)
        # all the logarithms are defined as log2 (because of information entrophy)
        X = distribution_p.rvs(size=n_samples)
        p_X = distribution_p.pmf(X)
        q_X = distribution_q.pdf(X)
        log_mix_X = np.log2(p_X + q_X)
        
        Y = distribution_q.rvs(size=n_samples)
        p_Y = distribution_p.pmf(Y)
        q_Y = distribution_q.pdf(Y)
        log_mix_Y = np.log2(p_Y + q_Y)
        
        jsp_m = np.log2(p_X).mean() - (log_mix_X.mean() - np.log2(2))
        jsq_m = np.log2(q_Y).mean() - (log_mix_Y.mean() - np.log2(2))
        
        return (jsp_m + jsq_m) / 2
    
    print("experiment small and norrmal (distance)")
    print(np.sqrt(distributions_js(st.norm(loc=0), st.norm(loc=0, scale=0.01))))
    
    print("experiment uniforms: (distance)")
    print(np.sqrt(distributions_js(st.uniform(loc=0), st.uniform(loc=0, scale=2))))
    
    print("experiment:")
    p = st.rv_discrete(values=([0, 1, 2], [0.25, 0.5, 0.25]))
    print(distributions_discrete_js(p, p))
    
    print("experiment:")
    print(distributions_discrete_cont_js(p, st.uniform(loc=0.5)))
    print(distributions_discrete_cont_js(p, st.uniform(loc=0.5), n_samples=1000000))
    
    # xnorm = st.norm(loc=1) + st.norm(loc=0)
    
    # print(xnorm)
    # print(distributions_discrete_cont_js(p, st.uniform(loc=0.5), n_samples=10000000))


if __name__ == '__main__':
    concept_experiments.run_commandline()
