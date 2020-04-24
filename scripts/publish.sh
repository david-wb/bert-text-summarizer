#!/bin/bash

set -eo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/..

rm -rf dist/
rm -rf build/
python setup.py sdist bdist_wheel
twine upload --skip-existing dist/*
