# -*- coding: utf8 -*-
from __future__ import division

import copy
from collections import Counter

import numpy as np

from distributions import StochasticScorable, FixdimDistribution, _squeeze_last, single_input_as_bool_param_s, \
    single_number_param_s, tuple_param_s


class InputBoxRuleScorable(StochasticScorable):
    def __init__(self,
                 input_classes,
                 positions,  # of the center of the box
                 sizes,
                 is_relative_position=True,
                 rotations=None,  # todo
                 info_is_noise=False,  # todo
                 **kwargs
                 ):
        self.input_classes = tuple_param_s(input_classes)  # todo decide if is onehot encoded
        self.positions = tuple_param_s(positions, 2)  # of the center of the box
        self.sizes = tuple_param_s(sizes, 2)
        self.is_relative_position = single_input_as_bool_param_s(is_relative_position)
        
        self.kwargs = kwargs
    
    def dim(self):
        return 5 + self.input_classes.dim()
    
    def score_samples(self, samples):
        pos_s = self.positions.score_samples(samples[:, 0:2])  # todo
        siz_s = self.sizes.score_samples(samples[:, 2:4])  # todo
        is_r = self.is_relative_position.score_samples(samples[:, 4:5])  # todo
        cl_s = self.input_classes.score_samples(samples[:, 5:])  # todo
        return pos_s * siz_s * is_r * cl_s
    
    def draw_samples(self, size, random_state=None):
        positions = self.positions.draw_samples(size, random_state=random_state)
        sizes = self.sizes.draw_samples(size, random_state=random_state)
        is_relative = self.is_relative_position.draw_samples(size, random_state=random_state)
        input_classes = self.input_classes.draw_samples(size, random_state=random_state)
        # if there happens to ever be a need to say 'i produce no sample', return some flag
        # (because we need to work with this as with numpy array so no variable lengths)
        
        return np.concatenate((positions, sizes, np.expand_dims(is_relative, -1), input_classes), axis=-1)
    
    def interpret_sampled(self, item):
        return ConceptItem(bbox=positions_sizes_to_ltrb(item[0:2], item[2:4]),
                           is_relative=item[4] >= 1.0,  # todo check
                           classes=item[5:], **self.kwargs)
    
    def draw_objects(self, size, random_state=None):
        assert isinstance(size, int)
        return [self.interpret_sampled(item) for item in self.draw_samples(size, random_state)]


class ConceptRuleBaseScorable(StochasticScorable):
    RELATIVE_IN_BOXES = False
    """
    A rule:
    - distribution of how many concepts will appear on a page (given that the concept appears on the page)
    - the probability of this concept being present (needs to be then normalized with other rules of course)
    - the resulting output class of the center rectangle (can be 'zero'! that means valid non-concept!)
    - for each box from star graph:
      - its input class (onehot; later we can decide what an input class means, if specific text or image)
      - its distribution of valid position
      - (its distribution of box sizes)
      - is its position relative (to the center box) or absolute (on the page)?
      - (can be rotated and still be valid?)
    """
    
    def __init__(self,
                 center_position,
                 center_size,
                 center_class,
                 output_class,
                 boxes_star_graph,
                 clipping_area=(-1.0, -1.0, 1.0, 1.0),
                 clipping_option='clip_coords',
                 # todo clip method - reject / move / clip coords
                 # lets say that reject just sets the output class to zero for example...
                 **kwargs
                 ):
        self.center_position = tuple_param_s(center_position, 2)
        self.center_size = tuple_param_s(center_size, 2)
        self.center_class = tuple_param_s(center_class)
        
        self.output_class = tuple_param_s(output_class)
        self.in_boxes = boxes_star_graph
        assert all([isinstance(inp, InputBoxRuleScorable) for inp in self.in_boxes])
        assert all([inp.dim() == self.in_boxes[0].dim() for inp in self.in_boxes]), \
            "all input boxes must have the same (feature) dimensionality ina concept rule"
        
        self.clipping_area = clipping_area
        self.clipping_option = clipping_option
        
        self.kwargs = kwargs
    
    def get_in_concepts_feature_dim(self):
        # self.in_boxes[0].dim() is 5 + ... because it counts the relative parameter. We do not account for it.
        return 4 + self.in_boxes[0].input_classes.dim() if self.in_boxes else None  # we dont know in this case
    
    def get_center_data_feature_dim(self):
        return 4 + self.center_class.dim()
    
    def get_output_class_feature_dim(self):
        return self.output_class.dim()
    
    def dim(self):
        return 4 + self.center_class.dim() + self.output_class.dim(), sum(inbox.dim() for inbox in self.in_boxes)
    
    def draw_samples(self, size, random_state=None):
        # self.num_per_page.draw_sample(random_state)
        center_position = self.center_position.draw_samples(size, random_state=random_state)
        center_size = self.center_size.draw_samples(size, random_state=random_state)
        center_class = self.center_class.draw_samples(size, random_state=random_state)
        out_class = self.output_class.draw_samples(size, random_state=random_state)
        # inboxes = np.asarray([inbox.draw_samples(size, random_state=random_state) for inbox in self.in_boxes])
        # inboxes are [number_per_sample, #n samples]
        
        if self.in_boxes:
            inboxes_to_stack = [inbox.draw_samples(size, random_state=random_state) for inbox in self.in_boxes]
            inboxes = np.stack(inboxes_to_stack,
                               axis=1)  # todo check
        else:
            inboxes = np.full(size, None, object)
        # inboxes are [#n samples, number_per_sample,(params per inbox)]
        
        #  ^ if there happens to ever be a need to say 'i produce no sample', return some flag
        # (because we need to work with this as with numpy array so no variable lengths)
        
        return np.concatenate([center_position, center_size, np.reshape(center_class, (size, -1)),
                               np.reshape(out_class, (size, -1))], axis=-1), inboxes
    
    def score_samples(self, samples):
        """
        Since the input rules are interchangeable, we need to query all-to-all

        samples: (pos, size, classes, (boxes))

        """
        samples, queryboxes = samples
        pos_s = self.center_position.score_samples(samples[:, 0:2])
        siz_s = self.center_size.score_samples(samples[:, 2:4])
        incls_s = self.center_class.score_samples(samples[:, 4:4 + self.center_class.dim()])
        cls_s = self.output_class.score_samples(samples[:, 4 + self.center_class.dim():])
        # todo check ofrmat of queryboxes
        #   ... assert False, "changed the inboxes order in draw samples, change here accordingly"
        # queryboxes: [number of queryboxes, number of samples, array of features per inbox]
        # we need to score all to all to get: [number of our inboxes, number of queryboxes, number of samples]
        # and then lets say that we do not pair them, but use the best score for each box per sample:
        # [number of queryboxes, number of samples]
        # and then multiply because it needs to be generated at once:
        # because we want an array of size [number of samples]
        all2all_scores = np.zeros((len(self.in_boxes), queryboxes.shape[1], queryboxes.shape[0]))
        for i, our_inbox in enumerate(self.in_boxes):
            for j in range(queryboxes.shape[1]):
                all2all_scores[i, j, :] = our_inbox.score_samples(queryboxes[:, j, :])
        best_case = np.max(all2all_scores, axis=0)
        sample_score = np.prod(best_case, axis=0)
        
        return pos_s * siz_s * incls_s * cls_s * sample_score
    
    def interpret_sampled(self, item):
        center_box_data, in_boxes_data = item
        # inboxes are [number_per_sample,(params per inbox)]
        if in_boxes_data is not None and len(in_boxes_data) > 0:
            assert len(in_boxes_data) == len(
                self.in_boxes), "the sampled arrays must have the same length as our inbox definitions," \
                                "i.e. must come from us"
            interpret_inboxes = [inbox_object.interpret_sampled(code_inbox)
                                 for inbox_object, code_inbox in zip(self.in_boxes, in_boxes_data) if
                                 code_inbox is not None]
        else:
            interpret_inboxes = []
        
        concept = Concept(bbox=positions_sizes_to_ltrb(center_box_data[0:2], center_box_data[2:4]),
                          bbox_class=center_box_data[4:4 + self.center_class.dim()],
                          output_class=center_box_data[4 + self.center_class.dim():], in_concepts=interpret_inboxes,
                          relative_force=self.RELATIVE_IN_BOXES, **self.kwargs)
        concept.apply_clipping_option(self.clipping_area, self.clipping_option)
        return concept
    
    def draw_objects(self, size, random_state=None):
        samples, inboxes_samples = self.draw_samples(size, random_state)
        return [self.interpret_sampled((sample, inboxes)) for sample, inboxes in zip(samples, inboxes_samples)]


def positions_sizes_to_ltrb(position, size):
    return [position[0] - 0.5 * size[0], position[1] - 0.5 * size[1],
            position[0] + 0.5 * size[0], position[1] + 0.5 * size[1]]


def absolute_to_relative(bbox, center_bbox):
    # returns relative position of lrtb bbox to center_bbox; we do relative vs topright corner
    bbox = list(bbox)
    bbox[0] = bbox[0] - center_bbox[0]
    bbox[1] = bbox[1] - center_bbox[1]
    bbox[2] = bbox[2] - center_bbox[0]
    bbox[3] = bbox[3] - center_bbox[1]
    return bbox


def relative_to_absolute(bbox, center_bbox):
    # returns absolute position of lrtb bbox to center_bbox; we do relative vs topright corner
    bbox = list(bbox)
    bbox[0] = bbox[0] + center_bbox[0]
    bbox[1] = bbox[1] + center_bbox[1]
    bbox[2] = bbox[2] + center_bbox[0]
    bbox[3] = bbox[3] + center_bbox[1]
    return bbox


class ConceptItem(object):
    def __init__(self, bbox, classes, is_relative, **kwargs):
        self.bbox = bbox
        assert bbox[0] <= bbox[2], "change generators to produce nonnegative bbox size (truncnorm / clipnorm)"
        assert bbox[1] <= bbox[3], "change generators to produce nonnegative bbox size (truncnorm / clipnorm)"
        self.classes = classes
        self.is_relative = is_relative
        self.params = kwargs
    
    def __str__(self):
        return "({}, {})".format(self.bbox, self.classes)
    
    def as_input_array(self):
        return np.concatenate([self.bbox, self.classes])


def clip_move_bbox_bounds(bbox, clip_minmax):
    """check lrtb bbox against [lmin, tmin, rmax, bmax] bounds and if outside, moves it to be inside but retains the
    size."""
    bbox = copy.copy(bbox)
    if bbox[0] < clip_minmax[0]:
        bbox[2] += clip_minmax[0] - bbox[0]
        bbox[0] = clip_minmax[0]
    
    if bbox[1] < clip_minmax[1]:
        bbox[3] += clip_minmax[1] - bbox[1]
        bbox[1] = clip_minmax[1]
    
    if bbox[2] > clip_minmax[2]:
        bbox[0] -= bbox[2] - clip_minmax[2]
        bbox[2] = clip_minmax[2]
    
    if bbox[3] > clip_minmax[3]:
        bbox[1] -= bbox[3] - clip_minmax[3]
        bbox[3] = clip_minmax[3]
    
    return bbox


class Concept(object):
    def __init__(self, bbox, bbox_class,
                 output_class, in_concepts, relative_force=None, **kwargs):
        self.bbox = bbox  # input
        self.bbox_class = bbox_class  # input
        self.output_class = output_class  # output
        self.relative_force = relative_force
        
        # center_point = ((center_bbox[0] + center_bbox[2]) * 0.5, (center_bbox[1] + center_bbox[3]) * 0.5)
        self.in_concepts = copy.copy(in_concepts)
        self._sanitize_in_concepts(self.in_concepts)
        
        self.params = kwargs
    
    def _sanitize_in_concepts(self, in_concepts):
        for in_concept in in_concepts:
            if self.relative_force == True and not in_concept.is_relative:
                in_concept.bbox = absolute_to_relative(in_concept.bbox, self.bbox)
                in_concept.is_relative = self.relative_force
            if self.relative_force == False and in_concept.is_relative:
                in_concept.bbox = relative_to_absolute(in_concept.bbox, self.bbox)
                in_concept.is_relative = self.relative_force
    
    def add_input_bboxes(self, in_concepts):
        in_concepts = copy.copy(in_concepts)
        self._sanitize_in_concepts(in_concepts)
        self.in_concepts.extend(in_concepts)
    
    def apply_clipping_option(self, clipping_area, clipping_option):
        if clipping_option == 'clip_coords':
            self.bbox = clip_move_bbox_bounds(self.bbox, clipping_area)
            for in_concept in self.in_concepts:
                if in_concept.is_relative:
                    # if that concept is stored in relative coordinates, we need to move clipping area accordingly
                    clipping_area_case = relative_to_absolute(clipping_area, self.bbox)
                else:
                    # else clipping area is never stored relatively
                    clipping_area_case = clipping_area
                in_concept.bbox = clip_move_bbox_bounds(in_concept.bbox, clipping_area_case)
        
        else:
            raise ValueError("use valid clipping option")
    
    def __str__(self):
        return "({}, {}, {}) <- {}".format(self.bbox, self.bbox_class, self.output_class,
                                           [str(concept) for concept in self.in_concepts])


class ConceptRuleDefinition(object):
    """
    Rule with probability of being generated. But the probability is checked at the level of ConceptsPage only.

    Not scorable anymore, because can produce different number of items.
    """
    
    def __init__(self,
                 num_per_page,
                 present,
                 center_position,
                 center_size,
                 center_class,
                 output_class,
                 boxes_star_graph,
                 **kwargs
                 ):
        self.concept_scorable = ConceptRuleBaseScorable(center_position, center_size, center_class, output_class,
                                                        boxes_star_graph, **kwargs)
        self.num_per_page = single_number_param_s(num_per_page)
        self.present = single_number_param_s(present)  # todo is this the right implementation?
        # (assert present in 0.0-1.0)
    
    def get_in_concepts_feature_dim(self):
        return self.concept_scorable.get_in_concepts_feature_dim()
    
    def get_center_data_feature_dim(self):
        return self.concept_scorable.get_center_data_feature_dim()
    
    def get_output_class_feature_dim(self):
        return self.concept_scorable.get_output_class_feature_dim()


def all_possibly_same(inp):
    different = Counter(inp)
    return len([key for key in different.keys() if key is not None]) == 1


class ConceptsPageGen(object):
    '''
    so far not rendered, so they live in some 'well separated space'...
    - num_noise_bboxes are added to the concepts
    '''
    
    def __init__(self,
                 num_different_rules_per_page,
                 num_noise_bboxes,  # still being only random - uniform
                 concept_rules,
                 noise_bbox_rule,
                 ):
        for concept in concept_rules:
            assert isinstance(concept, ConceptRuleDefinition)
        assert isinstance(concept_rules, list)
        
        all_f_dims = [concept_rule.get_output_class_feature_dim() for concept_rule in concept_rules]
        all_i_dims = [concept_rule.get_in_concepts_feature_dim() for concept_rule in concept_rules]
        all_c_dims = [concept_rule.get_center_data_feature_dim() for concept_rule in concept_rules]
        assert all_possibly_same(all_c_dims)
        assert all_possibly_same(all_i_dims)
        assert all_possibly_same(all_f_dims)
        self.concept_rules = concept_rules
        self.num_different_rules_per_page = single_number_param_s(num_different_rules_per_page)
        self.num_noise_bboxes = single_number_param_s(num_noise_bboxes)
        self.noise_bbox_rule = noise_bbox_rule
        assert self.noise_bbox_rule is None or isinstance(self.noise_bbox_rule, InputBoxRuleScorable)
    
    def get_in_concepts_feature_dim(self):
        return self.concept_rules[0].get_in_concepts_feature_dim()
    
    def get_center_data_feature_dim(self):
        return self.concept_rules[0].get_center_data_feature_dim()
    
    def get_output_class_feature_dim(self):
        return self.concept_rules[0].get_output_class_feature_dim()
    
    def draw_objects(self, size, random_state=None):
        different_rules = self.num_different_rules_per_page.draw_samples(size, random_state=random_state)
        concept_realizations = [(realization,
                                 realization.present.draw_samples(size, random_state=random_state),
                                 realization.num_per_page.draw_samples(size, random_state=random_state))
                                for realization in self.concept_rules]
        # [realization, info_id, sample_id]
        
        ret = np.empty((size), object)
        for i in range(size):
            this_realizations = [(realization[0],
                                  realization[1][i],
                                  realization[2][i],
                                  ) for realization in concept_realizations]
            this_realizations.sort(key=lambda realization: realization[1])  # todo is desc?
            
            picked_realizations = this_realizations[0:different_rules[i]]
            
            concepts = []
            
            for realization in picked_realizations:
                rule = realization[0]
                number = realization[2]
                realization_concepts = rule.concept_scorable.draw_objects(number, random_state=random_state)
                
                for concept in realization_concepts:
                    if self.noise_bbox_rule and self.num_noise_bboxes:
                        num_noise = self.num_noise_bboxes.draw_samples(1, random_state=random_state)[0]
                        if num_noise > 0:
                            noise_bboxes = self.noise_bbox_rule.draw_objects(num_noise, random_state=random_state)
                            concept.add_input_bboxes(noise_bboxes)
                    concepts.append(concept)
            # todo maybe add a noise rule for the whole page? This way we are noising only the concepts
            ret[i] = concepts
        
        return ret


class DistributionOfDistributions(StochasticScorable):
    def __init__(self, rv_class, class_params_distributions, **kwargs):
        assert isinstance(class_params_distributions, dict)
        
        for key in class_params_distributions:
            assert isinstance(class_params_distributions[key], StochasticScorable)
        
        dim = max(item.dim() for item in class_params_distributions.values())
        for key in class_params_distributions:
            assert class_params_distributions[key].dim() in [dim, 1], "All dimensions of all parameters must be the" \
                                                                      " same (or one for broadcast)"
        
        self._dim = dim
        self.rv_class = rv_class
        self.class_params_distributions = class_params_distributions
        self.kwargs = kwargs
    
    def dim(self):
        return self._dim
    
    def draw_samples(self, size, random_state=None):
        init_params = {param: _squeeze_last(self.class_params_distributions[param].draw_samples(size=size,
                                                                                                random_state=random_state))
                       for param in self.class_params_distributions}
        # assert all have the same dimensionality...
        return init_params
    
    def draw_objects(self, size, random_state=None):
        return self.generated_to_objects(self.draw_samples(size, random_state))
    
    def generated_to_objects(self, init_dict):
        size = init_dict[list(init_dict.keys())[0]].shape
        # assert all same sizes
        objects = np.empty(size, object)
        for i in np.ndindex(size):
            init = {param: init_dict[param][i] for param in self.class_params_distributions}
            objects[i] = FixdimDistribution(self.rv_class(**init), item_dimension=self.dim(), **self.kwargs)
        return objects
    
    def score_samples(self, samples):
        assert isinstance(samples, dict)
        cls_pdfs = [self.class_params_distributions[param].score_samples(samples[param]) for param in samples.keys()]
        return np.prod(cls_pdfs)


# and now it needs also a mechanism of distribution of distribution of positions and sizes
# i.e. if we select the center (of the positions) to be X, then scale (of the positions) must be clipped, such that
# it will never generate anything outside [-1, -1] x [+1, +1]
# ... for the sizes we will not be doing such restrictions for now.
# until implemented, there is only a clipping option when creating new items

class ConceptRulesDistribution(object):
    # todo make stochasticScorable or even distribtuionofdistribtuions...
    # todo make then maybe some initializer that would initialize it (wisely) only given dimensions of the space...
    def __init__(self,
                 # the concept:
                 num_per_page,
                 present,
                 center_position,
                 center_size,
                 center_class,
                 output_class,
                 # the concepts boxes:
                 num_boxes_star_graph,
                 input_classes,
                 positions,  # of the center of the box
                 sizes,
                 is_relative_position=True,
                 ):
        assert isinstance(num_per_page, DistributionOfDistributions)
        assert isinstance(present, DistributionOfDistributions)
        assert isinstance(center_position, DistributionOfDistributions)
        assert isinstance(center_size, DistributionOfDistributions)
        assert isinstance(output_class, DistributionOfDistributions)
        assert isinstance(center_class, DistributionOfDistributions)
        
        assert isinstance(num_boxes_star_graph, StochasticScorable), \
            "The number of generated boxes needs to be a simple distribution"
        
        assert isinstance(input_classes, DistributionOfDistributions)
        assert isinstance(positions, DistributionOfDistributions)
        assert isinstance(sizes, DistributionOfDistributions)
        assert isinstance(is_relative_position, bool), "positioning is required to be strictly set to True/False"
        self.num_per_page = num_per_page
        self.present = present
        self.center_position = center_position
        self.center_size = center_size
        self.center_class = center_class
        self.output_class = output_class
        
        self.num_boxes_star_graph = num_boxes_star_graph
        
        self.input_classes = input_classes
        self.positions = positions
        self.sizes = sizes
        self.is_relative_position = is_relative_position
    
    def draw_object(self):
        # todo maybe add size ... resp generate all by sizes and then create classes by iterating over np.ndindex
        
        inboxes_num = self.num_boxes_star_graph.draw_samples(1)
        
        iboxes = [InputBoxRuleScorable(input_classes=self.input_classes.draw_samples(1),
                                       positions=self.positions.draw_samples(1),
                                       sizes=self.sizes.draw_samples(1),
                                       is_relative_position=self.is_relative_position)
                  for i in range(inboxes_num)]
        
        concept = ConceptRuleDefinition(num_per_page=self.num_per_page.draw_samples(1),
                                        present=self.present.draw_samples(1),
                                        center_position=self.center_position.draw_samples(1),
                                        center_size=self.center_size.draw_samples(1),
                                        center_class=self.center_class.draw_samples(1),
                                        output_class=self.output_class.draw_samples(1),
                                        boxes_star_graph=iboxes)
        return concept


"""
class ConceptsRuleGenerator(object):
    '''
    generate rules:
    Rules generating process: (because for later for label reuse we need to be able to generate not only concepts,
    but also new rules!)
    - how many (of already generated rules) to check for uniqueness (of the newly created rule)
      - (we will check by some similarity metric of two rules against a threshold)
    - allow overlapping classboxes in a rule?
    - distribution of 'zero rules' (a rule that just confirms the output class to be background)
    - distribution of 'misleading/similar up to 1 specific property' rules (constructed to deliberately confuse/make
    the problem hard)
    - distribution of distributions for per-rule parameters ('Make it gaussians but sometimes normal distribution',
    for example).
    '''
    def __init__(self,
                 check_uniqueness_num,
                 check_uniqueness_threshold,
                 allow_overlapping_input_bboxes,
                 generate_zero_rules,
                 generate_misleading_rules,

                 num_per_page,
                 present,
                 center_position,
                 center_size,
                 output_class,
                 boxes_star_graph,

                 input_classes,
                 positions,  # of the center of the box
                 sizes,
                 is_relative_position=True,
                 ):
        pass
"""

"""
... here we paused the distribution of distributions.

ConceptRulesDistribution(num_per_page=DistributionOfDistributions(st.randint,
                                                                  {'low': Determined(1),
                                                                   'high': StochasticScorableWrapper(st.randint(low=1, high=5)),
                                                                    }),
             present=DistributionOfDistributions(st.norm,
                                                                  {'loc': StochasticScorableWrapper(st.norm(loc=0.5, scale=0.5)),
                                                                   'scale': StochasticScorableWrapper(st.uniform(loc=0.2, scale=0.1)),
                                                                    }),
             # setting: each box has its own normal distribution that defines it.
             center_position=DistributionOfDistributions(st.uniform,
                                                                  {'loc': FixdimDistribution(st.uniform(loc=0.0, scale=0.1), 2),
                                                                   'scale': FixdimDistribution(st.uniform(loc=0.8, scale=0.1), 2),
                                                                    }),

             center_size=DistributionOfDistributions(st.norm,
                                                                  {'loc': FixdimDistribution(st.uniform(loc=0.0, scale=0.9), 2),
                                                                   'scale': FixdimDistribution(st.uniform(loc=0.2, scale=0.1), 2),
                                                                    }),
             center_class,
             output_class,
              # the concepts boxes:
             num_boxes_star_graph,
             input_classes,
             positions,  # of the center of the box
             sizes,
             is_relative_position=True,)
uniforms = DistributionOfDistributions(st.uniform, {'loc': FixdimDistribution(st.uniform(loc=0.0, scale=0.5), 1),
                                                    'scale': Determined(1.0)})
samples = uniforms.draw_samples(1)
score = uniforms.score_samples(samples)
assert score is not None
objects = uniforms.generated_to_objects(samples)
assert objects is not None
sampled_numbers = objects[0].draw_samples(1)
assert sampled_numbers is not None
"""