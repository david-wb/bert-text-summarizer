#!/bin/bash

set -eo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/..

bert-text-summarizer train-ext-summarizer \
  --saved-model-dir=bert_ext_summ_model \
  --train-data-path=cnndm_train.tfrec