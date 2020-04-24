from typing import Dict

import tensorflow as tf


def decode_tf_record(record, input_len: int, n_sentences: int, is_training: bool) -> Dict[str, any]:
    read_features = {
        "input_ids": tf.io.FixedLenFeature([input_len], tf.int64),
        "input_mask": tf.io.FixedLenFeature([input_len], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([input_len], tf.int64),
        "cls_indices": tf.io.FixedLenFeature([n_sentences], tf.int64),
        "cls_mask": tf.io.FixedLenFeature([n_sentences], tf.int64),
    }

    if is_training:
        read_features["cls_outputs"] = tf.io.FixedLenFeature([n_sentences], tf.float32)

    example = tf.io.parse_single_example(record, read_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in example.keys():
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
        example[name] = t

    return example
