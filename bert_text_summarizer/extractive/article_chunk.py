import logging
from typing import List

from nltk.tokenize import sent_tokenize
from official.nlp.bert.tokenization import FullTokenizer


class ArticleChunk:
    def __init__(self,
                 tokenizer: FullTokenizer,
                 sentences: List[str],
                 sentence_tokens: List[List[str]],
                 total_tokens: int,
                 max_len: int,
                 max_sentences: int):
        assert total_tokens == sum([len(s) for s in sentence_tokens])
        self.sentences = sentences
        self.num_sentences = len(sentences)
        self.sentence_tokens = sentence_tokens
        self.sentence_tokens[-1].append('[SEP]')
        self.total_tokens = total_tokens + 1

        self.all_tokens = []
        self.cls_indices = []
        self.segment_ids = []

        for i, st in enumerate(self.sentence_tokens):
            self.cls_indices.append(len(self.all_tokens))
            self.all_tokens += st
            self.segment_ids += [i % 2] * len(st)

        self.input_ids = tokenizer.convert_tokens_to_ids(self.all_tokens)
        self.input_mask = [1] * len(self.all_tokens)
        self.cls_mask = [1] * len(self.cls_indices)

        while len(self.input_ids) < max_len:
            self.input_ids.append(0)
            self.input_mask.append(0)
            self.segment_ids.append(0)

        while len(self.cls_indices) < max_sentences:
            self.cls_indices.append(0)
            self.cls_mask.append(0)

    @staticmethod
    def chunk_article(article: str,
                      tokenizer: FullTokenizer,
                      max_len: int,
                      max_sentences: int) -> List[any]:
        max_len = max_len - 1  # subtract one to account for '[SEP]' token
        result = []
        sentences = sent_tokenize(article)

        chunk_sentences = []
        chunk_sentence_tokens = []
        chunk_token_len = 0
        for s in sentences:
            tokens = ['[CLS]'] + tokenizer.tokenize(s)
            if len(tokens) > max_len:
                logging.warning('Excessively long sentence detected. Truncating.')
                tokens = tokens[:max_len]

            if chunk_token_len + len(tokens) > max_len or \
                    len(chunk_sentences) + 1 > max_sentences:
                # Take current chunk and start a new one
                chunk = ArticleChunk(tokenizer=tokenizer,
                                     sentences=chunk_sentences,
                                     sentence_tokens=chunk_sentence_tokens,
                                     total_tokens=chunk_token_len,
                                     max_len=max_len + 1,
                                     max_sentences=max_sentences)
                result.append(chunk)

                # Start new chunk
                chunk_sentences = [s]
                chunk_sentence_tokens = [tokens]
                chunk_token_len = len(tokens)
            else:
                # Extend the current chunk
                chunk_sentences.append(s)
                chunk_sentence_tokens.append(tokens)
                chunk_token_len += len(tokens)

        # Add left over chunk
        if chunk_token_len > 0:
            assert chunk_token_len <= max_len
            chunk = ArticleChunk(tokenizer=tokenizer,
                                 sentences=chunk_sentences,
                                 sentence_tokens=chunk_sentence_tokens,
                                 total_tokens=chunk_token_len,
                                 max_len=max_len + 1,
                                 max_sentences=max_sentences)
            result.append(chunk)

        return result
