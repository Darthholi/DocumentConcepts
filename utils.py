from __future__ import division, print_function

import tempfile

import numpy as np
import sklearn.metrics as skmetrics
import tensorflow as tf
from keras.callbacks import Callback


class tempmap(np.memmap):
    def __new__(subtype, dtype=np.uint8, mode='w+', offset=0,
                shape=None, order='C'):
        ntf = tempfile.NamedTemporaryFile()
        self = np.memmap.__new__(subtype, ntf, dtype, mode, offset, shape, order)
        self.temp_file_obj = ntf
        return self
    
    def __del__(self):
        if hasattr(self, 'temp_file_obj') and self.temp_file_obj is not None:
            self.temp_file_obj.close()
            del self.temp_file_obj


def np_as_tmp_map(nparray):
    tmpmap = tempmap(dtype=nparray.dtype, mode='w+', shape=nparray.shape)
    tmpmap[...] = nparray
    return tmpmap


def equal_ifarray(a, b):
    if isinstance(a, np.ndarray):
        return all(a == b)
    else:
        return a == b


def array_all_classification_metrics(y_pred_percent, y_real_classes, classnames=None, as_binary_problems=False):
    if (as_binary_problems):
        xret = []
        classes = y_pred_percent.shape[-1]
        for c in range(classes):
            print("Evaluation scores for " + str(c) + "-th class--------------------")
            if (classnames is not None):
                print(classnames[c])
            print("auc scores:")
            auc_scores = None
            try:
                auc_scores = skmetrics.roc_auc_score(y_real_classes[:, c], y_pred_percent[:, c], average=None,
                                                     sample_weight=None)
                print(auc_scores)
            except:
                print("cannot compute auc scores this time.")
            
            print("confusion matrices: (from arrays of size {})".format(y_pred_percent.shape))
            y_round = np.round(y_pred_percent[:, c])
            # df_confusion = pd.crosstab(y_real_classes[:, c], y_round, rownames=['Actual'], colnames=['Predicted'],
            #                            margins=True)
            df_confusion = skmetrics.confusion_matrix(y_real_classes[:, c], y_round)
            if df_confusion.shape == (1, 1):
                real_confusion = np.zeros((2, 2), dtype=int)
                r_pos = int(y_real_classes[:, c][0])
                real_confusion[r_pos, r_pos] = df_confusion[0, 0]
                df_confusion = real_confusion
            
            print(df_confusion)
            print("Accuracy: {}".format(float(sum([df_confusion[i, i] for i in range(len(df_confusion))], 0.0))
                                        / float(sum(df_confusion.flatten(), 0.0))))
            clsreport = skmetrics.classification_report(y_real_classes[:, c], y_round)  # target_names!=classnames
            print(clsreport)
            
            metricsreport = skmetrics.precision_recall_fscore_support(y_real_classes[:, c], y_round)
            
            xret.append({"confusion": df_confusion, "classfication_report": clsreport, "metricsreport": metricsreport})
        return xret
    else:
        if (len(y_pred_percent.shape) <= 1):
            xarrlen = y_pred_percent.shape[0]
            y_pred = np.zeros(xarrlen)
            for i in range(xarrlen):
                if (y_pred_percent[i] >= 0.5):
                    y_pred[i] = 1
                else:
                    y_pred[i] = 0
        else:
            y_pred = np.argmax(y_pred_percent, axis=1)
        
        if (len(y_real_classes.shape) <= 1):
            y_real = y_real_classes
        else:
            y_real = np.argmax(y_real_classes, axis=1)
        
        print("auc scores:")
        auc_scores = None
        try:
            assert y_real_classes.ndim > 1 and y_real_classes.shape[-1] > 1
            auc_scores = skmetrics.roc_auc_score(y_real_classes, y_pred_percent, average=None, sample_weight=None)
            print(auc_scores)
        except:
            print("cannot compute auc scores this time.")
        
        print("confusion matrices:")
        # df_confusion = pd.crosstab(y_real, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
        df_confusion = skmetrics.confusion_matrix(y_real, y_pred)
        print(df_confusion)
        print("Accuracy: {}".format(float(sum([df_confusion[i, i] for i in range(len(df_confusion))], 0.0))
                                    / float(sum(df_confusion.flatten(), 0.0))))
        clsreport = skmetrics.classification_report(y_real, y_pred, target_names=classnames)
        print(clsreport)
        
        metricsreport = skmetrics.precision_recall_fscore_support(y_real, y_pred)
        
        return {"confusion": df_confusion, "classfication_report": clsreport, "metricsreport": metricsreport}


class EvaluateFCallback(Callback):
    """
    Calls evaluate function each time a key metric improves (or after each epoch if key metric not provided).
    """
    
    def __init__(self, evaluation_f, validation_gen, validation_steps, config, monitor=None, mode=None, min_delta=0):
        super(EvaluateFCallback, self).__init__()
        self.evaluation_f = evaluation_f
        self.validation_data = validation_gen
        self.validation_steps = validation_steps
        self.config = config


def make_product_matrix(vect_inp):
    '''
    vect_inp: [batches, Nfields, Cfeatures]
    should return [batches, Nfields, Nfields, 2*Cfeatures]
    [b,i,j,...] should be (vect_inp[b,i,:], vect_inp[b,j,:])

    example:
    data = [[0 1],[2 3],[4 5]]  # shape (1, 3, 2),
    [[[0 1],[2 3],[4 5]]]  # (1,1,3,2)
    [[[0 1]],[[2 3]],[[4 5]]]  # (1,3,1,2)
    after repets:
    [[[0 1],[2 3],[4 5]], [[0 1],[2 3],[4 5]], [[0 1],[2 3],[4 5]] ]  # (1,3,3,2)
    [[[0 1], [0 1], [0 1]],[[2 3], [2 3], [2 3]],[[4 5], [4 5], [4 5]]]  # (1,3,3,2)
    and then it could be just concatenated
    '''
    Nfields = tf.shape(vect_inp)[-2]
    v = tf.expand_dims(vect_inp, -3)
    v_t = tf.expand_dims(vect_inp, -2)
    v = tf.tile(v, [1, Nfields, 1, 1])
    v_t = tf.tile(v_t, [1, 1, Nfields, 1])
    ret = tf.concat([v, v_t], -1)
    
    assert_op = tf.Assert(tf.equal(tf.shape(ret)[-2], tf.shape(ret)[-3]), [ret])
    with tf.control_dependencies([assert_op]):
        return ret
