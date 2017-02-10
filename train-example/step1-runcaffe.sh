#!/usr/bin/env sh

set -e
~/caffe/build/tools/caffe train \
	 --solver=./solver.prototxt
