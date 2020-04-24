from typing import List, Optional

import numpy as np
from nltk.tokenize import word_tokenize
from official.nlp.bert.tokenization import FullTokenizer

from bert_text_summarizer.extractive.article_chunk import ArticleChunk
from bert_text_summarizer.extractive.input_features import InputFeatures
from bert_text_summarizer.metrics.rouge import rouge_n


def raw_input_to_features(tokenizer: FullTokenizer,
                          article: str,
                          max_seq_len: int,
                          max_sentences: int,
                          is_training: bool,
                          summary: Optional[str] = None) -> List[InputFeatures]:
    if is_training:
        assert summary is not None
        summary_words = word_tokenize(summary)
    else:
        summary_words = None
    article_chunks = ArticleChunk.chunk_article(article=article,
                                                tokenizer=tokenizer,
                                                max_len=max_seq_len,
                                                max_sentences=max_sentences)

    result = []

    for chunk in article_chunks:
        cls_outputs = None
        if is_training:
            cls_outputs = []

            for s in chunk.sentences:
                sentence_words = word_tokenize(s)
                rouge_1 = rouge_n(summary_words, sentence_words, n=1)
                rouge_2 = rouge_n(summary_words, sentence_words, n=2)
                cls_outputs.append(np.mean([rouge_1, rouge_2]))

            while len(cls_outputs) < max_sentences:
                cls_outputs.append(0)

        input_ids = chunk.input_ids
        input_mask = chunk.input_mask
        segment_ids = chunk.segment_ids

        assert len(input_ids) == len(input_mask) == len(segment_ids) == max_seq_len

        cls_indices = chunk.cls_indices
        cls_mask = chunk.cls_mask

        assert len(cls_indices) == len(cls_mask) == max_sentences

        if is_training:
            assert len(cls_outputs) == max_sentences
        else:
            assert cls_outputs is None

        features = InputFeatures(sentences=chunk.sentences,
                                 tokens=chunk.all_tokens,
                                 input_ids=input_ids,
                                 input_mask=input_mask,
                                 segment_ids=segment_ids,
                                 cls_indices=cls_indices,
                                 cls_outputs=cls_outputs,
                                 cls_mask=cls_mask)
        result.append(features)

    return result
