import collections
from typing import List, Optional

import tensorflow as tf


class InputFeatures(object):
    def __init__(self,
                 sentences: List[str],
                 tokens: List[str],
                 input_ids: List[int],
                 input_mask: List[int],
                 segment_ids: List[int],
                 cls_indices: List[int],
                 cls_mask: List[int],
                 cls_outputs: Optional[List[int]] = None):
        self.sentences = sentences
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_indices = cls_indices
        self.cls_mask = cls_mask
        self.cls_outputs = cls_outputs

    def serialize_to_string(self):
        def create_int_feature(values):
            feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return feature

        def create_float_feature(values):
            feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
            return feature

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(self.input_ids)
        features["input_mask"] = create_int_feature(self.input_mask)
        features["segment_ids"] = create_int_feature(self.segment_ids)
        features["cls_indices"] = create_int_feature(self.cls_indices)
        features["cls_mask"] = create_int_feature(self.cls_mask)

        if self.cls_outputs:
            features["cls_outputs"] = create_float_feature(self.cls_outputs)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        return tf_example.SerializeToString()

    def to_prediction_input(self):
        x = {
            'input_ids': tf.convert_to_tensor([self.input_ids]),
            'input_mask': tf.convert_to_tensor([self.input_mask]),
            'segment_ids': tf.convert_to_tensor([self.segment_ids]),
            'cls_indices': tf.convert_to_tensor([self.cls_indices]),
            'cls_mask': tf.convert_to_tensor([self.cls_mask])
        }

        y = {}
        if self.cls_outputs:
            y['cls_outputs'] = tf.convert_to_tensor([self.cls_outputs])

        return x, y
