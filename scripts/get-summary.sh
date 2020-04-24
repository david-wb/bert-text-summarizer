#!/bin/bash

set -eo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/..

article=$1

bert-text-summarizer get-summary \
  --saved-model-dir=bert_ext_summ_model \
  --article-file=${article}
