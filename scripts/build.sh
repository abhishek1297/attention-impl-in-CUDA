#! /bin/bash

set -e

# From root of the repository
rm -rf build
cmake -S . -B build
make -C build