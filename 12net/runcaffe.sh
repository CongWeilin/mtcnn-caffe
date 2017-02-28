#!/usr/bin/env sh
export PYTHONPATH=$PYTHONPATH:/home/cmcc/caffe-master/examples/mtcnn-caffe/12net

set -e
~/caffe-master/build/tools/caffe train \
	 --solver=./solver.prototxt \
