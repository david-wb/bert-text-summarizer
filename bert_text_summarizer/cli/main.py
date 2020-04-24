import os

import click
import tensorflow as tf
import tensorflow_hub as hub
from nltk import word_tokenize
from official.nlp.bert import tokenization
from tqdm import tqdm
import tensorflow_datasets as tfds

from bert_text_summarizer.extractive.cnndm.data_loader import ExtractiveDataLoader
from bert_text_summarizer.extractive.model import ExtractiveSummarizer
from bert_text_summarizer.metrics.rouge import rouge_n


@click.group()
def cli():
    pass


@cli.command()
@click.option('--output-path', type=click.Path(), default='cnndm_train.tfrec')
@click.option('--max-examples', type=int, default=0, help='Number of CNN/DM examples to process. 0 means all.')
@click.option('--tfhub-model-url', type=str, default="https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1")
def get_cnndm_train(output_path: str, max_examples: int, tfhub_model_url: str):
    bert_layer = hub.KerasLayer(tfhub_model_url, trainable=True)
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

    features = ExtractiveDataLoader.load_cnndm(tokenizer, max_examples=max_examples)

    print(f'Loading CNN/DM training set. Writing TF records to {os.path.basename(output_path)}')
    writer = tf.io.TFRecordWriter(output_path)
    for i, input_feature in enumerate(tqdm(features)):
        writer.write(input_feature.serialize_to_string())
    writer.close()


@cli.command()
@click.option('--output-path', type=click.Path(), default='cnndm_eval.tfrec')
@click.option('--tfhub-model-url', type=str, default="https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1")
def get_cnndm_eval(output_path: str, tfhub_model_url: str):
    bert_layer = hub.KerasLayer(tfhub_model_url, trainable=True)
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

    features = ExtractiveDataLoader.load_cnndm(tokenizer, split='validation')

    print(f'Loading CNN/DM eval set. Writing TF records to {os.path.basename(output_path)}')
    writer = tf.io.TFRecordWriter(output_path)
    for i, input_feature in enumerate(tqdm(features)):
        writer.write(input_feature.serialize_to_string())
    writer.close()


@cli.command()
@click.option('--saved-model-dir', type=click.Path(exists=False), help="Directory in which to save model checkpoints")
@click.option('--train-data-path', type=click.Path(exists=True), help="Path to the tf-record training set file")
@click.option('--batch-size', type=int, default=1)
@click.option('--steps-per-epoch', type=int, default=500)
@click.option('--epochs', type=int, default=500)
def train_ext_summarizer(
        saved_model_dir: str,
        train_data_path: str,
        batch_size: int,
        steps_per_epoch: int,
        epochs: int):
    training_set = tf.data.TFRecordDataset([train_data_path])

    if os.path.exists(saved_model_dir):
        print('Loading saved model')
        model = tf.keras.models.load_model(saved_model_dir)
    else:
        model = ExtractiveSummarizer.build_model()

    ExtractiveSummarizer.compile_model(model)
    ExtractiveSummarizer.train(model=model,
                               ds=training_set,
                               batch_size=batch_size,
                               steps_per_epoch=steps_per_epoch,
                               epochs=epochs,
                               saved_model_path=saved_model_dir)


@cli.command()
@click.option('--article-file', type=click.Path(exists=True))
@click.option('--saved-model-dir', type=click.Path(exists=True))
@click.option('--max-words', type=int, default=200)
@click.argument('tfhub-model-url', type=str, default="https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1")
def get_summary(article_file: str, saved_model_dir: str, max_words: int, tfhub_model_url: str):
    bert_layer = hub.KerasLayer(tfhub_model_url, trainable=False)
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
    predictor = ExtractiveSummarizer(tokenizer=tokenizer, saved_model_dir=saved_model_dir)

    article = open(article_file, 'r').read().strip()
    summary = predictor.get_summary(text=article, max_words=max_words)
    click.echo(summary)


@cli.command()
@click.option('--saved-model-dir', type=click.Path(exists=False), help="Directory in which to save model checkpoints")
@click.option('--eval-data-path', type=click.Path(exists=True), help="Path to the tf-record eval set file")
@click.option('--max-words', type=int, default=200)
@click.argument('tfhub-model-url', type=str, default="https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1")
def eval_ext_summarizer(saved_model_dir: str, eval_data_path: str, max_words: int, tfhub_model_url: str):
    bert_layer = hub.KerasLayer(tfhub_model_url, trainable=False)
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
    predictor = ExtractiveSummarizer(tokenizer=tokenizer, saved_model_dir=saved_model_dir)

    val_set, info = tfds.load('cnn_dailymail', split='validation', with_info=True, shuffle_files=False)
    val_set = tfds.as_numpy(val_set)

    rouge1_mean = 0
    rouge2_mean = 0

    for i, example in enumerate(tqdm(val_set)):
        article = example['article'].decode('utf-8')
        highlights = example['highlights'].decode('utf-8')
        summary = predictor.get_summary(text=article, max_words=max_words)

        ref_words = word_tokenize(highlights)
        sys_words = word_tokenize(summary)

        rouge1 = rouge_n(ref_words, sys_words, n=1)
        rouge2 = rouge_n(ref_words, sys_words, n=2)

        rouge1_mean = rouge1_mean + (rouge1 - rouge1_mean) / (i + 1)
        rouge2_mean = rouge2_mean + (rouge2 - rouge2_mean) / (i + 1)

    click.echo(f'ROUGE-1: {rouge1_mean * 100}')
    click.echo(f'ROUGE-2: {rouge2_mean * 100}')


def main():
    cli()


if __name__ == '__main__':
    cli()
