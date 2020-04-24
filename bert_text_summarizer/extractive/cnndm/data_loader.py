from typing import Generator, Optional

import tensorflow_datasets as tfds
from official.nlp.bert.tokenization import FullTokenizer

from bert_text_summarizer.extractive.input_features import InputFeatures
from bert_text_summarizer.extractive.input_preprocessing import raw_input_to_features


class ExtractiveDataLoader:
    @staticmethod
    def load_cnndm(tokenizer: FullTokenizer,
                   split="train",
                   max_examples: int = 0) -> Generator[InputFeatures, None, None]:
        ds, info = tfds.load('cnn_dailymail', split=split, with_info=True, shuffle_files=False)

        if max_examples > 0:
            print(f'Loading {max_examples} examples from CNN/DM')
            ds = ds.take(max_examples)

        ds = tfds.as_numpy(ds)

        for example in ds:
            article = example['article'].decode('utf-8')
            highlights = example['highlights'].decode('utf-8')

            features = raw_input_to_features(tokenizer=tokenizer,
                                             article=article,
                                             summary=highlights,
                                             max_seq_len=512,
                                             max_sentences=20,
                                             is_training=True)
            for feature in features:
                yield feature
