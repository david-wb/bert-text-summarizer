#!/bin/bash

set -eo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/..

bert-text-summarizer eval-ext-summarizer \
  --saved-model-dir=bert_ext_summ_model