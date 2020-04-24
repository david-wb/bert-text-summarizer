# A BERT-based Text Summarizer

Currently, only **extractive** summarization is supported.

This repo is TensorFlow centric (apologies to the PyTorch people.)

Using a word limit of 200, this model achieves approximately the following ROUGE scores on the CNN/DM validation set.

```buildoutcfg
ROUGE-1: 64.0
ROUGE-2: 30.0
```
## How does it work?

During preprocessing, the input text is divided into chunks up to 512 tokens long. Each sentence is
 tokenized using the bert official tokenizer and a special `[CLS]` is placed 
 at the begging of each sentence. The ROUGE-1 and ROUGE-2 scores of each sentence with 
 respect to the example summary are calculated. The model ouputs a single value corresponding to each `[CLS]` token and is
 trained to directly predict the mean of the ROUGE-1 and 2 scores. 
 
 During post-processing, the sentences are ranked according to their
 predicted ROUGE score. Finally, the top sentences are selected until the 
 word limit is reached and resorted according to their positions within the text.
 
## Install
```buildoutcfg
pip install -U bert-text-summarizer
```

## Usage

### Get training data

```buildoutcfg
bert-text-summarizer get-cnndm-train --max-examples=10000
```

This outputs a tf-record file named `cnndm_train.tfrec` by default.

Leaving out `--max-examples` it will process the entire CNN/DM training set which may take >1 hours to complete.

### Train the model

```buildoutcfg
bert-text-summarizer train-ext-summarizer \
  --saved-model-dir=bert_ext_summ_model \
  --train-data-path=cnndm_train.tfrec \
  --epochs=10
```

### Get Summary

```buildoutcfg
bert-text-summarizer get-summary \
  --saved-model-dir=bert_ext_summ_model \
  --article-file=article.txt \
  --max-words=150
```

### Evaluate on the CNN/DM validation set

```
bert-text-summarizer eval-ext-summarizer \
  --saved-model-dir=bert_ext_summ_model
```
