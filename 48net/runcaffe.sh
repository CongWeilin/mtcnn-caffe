#!/usr/bin/env sh
export PYTHONPATH=$PYTHONPATH:/home/cmcc/caffe-master/examples/mtcnn-caffe/48net
set -e
~/caffe-master/build/tools/caffe train \
	 --solver=./solver.prototxt \
  	 #--weights=./48net-only-cls.caffemodel
