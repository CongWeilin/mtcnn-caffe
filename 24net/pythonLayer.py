import sys
sys.path.append('/home/congweilin/caffe/python')
sys.path.append('/usr/lib/python2.7/dist-packages')
import cv2
import caffe
import numpy as np
import random
import time
################################################################################
#########################Data Layer By Python###################################
################################################################################
class Data_Layer_train(caffe.Layer):
    def setup(self, bottom, top):
        self.batch_size = 64
	cls_list = './cls.txt'
	roi_list = './roi.txt'
	net_side = 24
	cls_root = "./cls_images/"
	roi_root = "./roi_images/"
        self.batch_loader = BatchLoader(cls_list,roi_list,net_side,cls_root,roi_root)
        top[0].reshape(self.batch_size, 3, net_side, net_side)
        top[1].reshape(self.batch_size, 1)
	top[2].reshape(self.batch_size, 1)
	self.loss_task = -1
    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
	self.loss_task += 1
        for itt in range(self.batch_size):
            im, label, roi= self.batch_loader.load_next_image(self.loss_task)
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = label
	    top[2].data[itt, ...] = roi
    def backward(self, top, propagate_down, bottom):
        pass

class BatchLoader(object):
    def __init__(self,cls_list,roi_list,net_side,cls_root,roi_root):
	self.mean = 128
        self.im_shape = net_side
        self.cls_root = cls_root
	self.roi_root = roi_root
	self.roi_list = []
	self.cls_list = []
	print "Start Reading Data into Memory..."
	fid = open(cls_list,'r')
        lines = fid.readlines()
	fid.close()
	for line in lines:
	    words = line.split()
	    image_file_name = self.cls_root + words[0]
	    im = cv2.imread(image_file_name)
	    h,w,ch = im.shape
	    if h!=self.im_shape or w!=self.im_shape:
	        im = cv2.resize(im,(int(self.im_shape),int(self.im_shape)))
	    im = np.swapaxes(im, 0, 2)
	    im -= self.mean
            label    = int(words[1])
            roi      = int(words[2])
	    self.cls_list.append([im,label,roi])
	random.shuffle(self.cls_list)
        self.cls_cur = 0
	print str(len(self.cls_list))," Classify Data have been read into Memory..."

	fid = open(roi_list,'r')
        lines = fid.readlines()
	fid.close()
	for line in lines:
	    words = line.split()
	    image_file_name = self.roi_root + words[0]
	    im = cv2.imread(image_file_name)
	    h,w,ch = im.shape
	    if h!=self.im_shape or w!=self.im_shape:
	        im = cv2.resize(im,(int(self.im_shape),int(self.im_shape)))
	    im = np.swapaxes(im, 0, 2)
	    im -= self.mean
            label    = int(words[1])
            roi      = int(words[2])
	    self.roi_list.append([im,label,roi])
	random.shuffle(self.roi_list)
	self.roi_cur = 0 
	print str(len(self.roi_list))," Regression Data have been read into Memory..."

    def load_next_image(self,loss_task): 
        if self.cls_cur == len(self.cls_list):
            self.cls_cur = 0
            random.shuffle(self.cls_list)
	if loss_task % 2 == 0:
            cur_data = self.cls_list[self.cls_cur]  # Get the image index
	    im	     = cur_data[0]
            label    = cur_data[1]
            roi      = cur_data[2]
            self.cls_cur += 1
            return im, label, roi
	
	if self.roi_cur == len(self.roi_list):
            self.roi_cur = 0
            random.shuffle(self.roi_list)
	if loss_task % 2 == 1:
	    cur_data = self.roi_list[self.roi_cur]  # Get the image index
	    im	     = cur_data[0]
            label    = cur_data[1]
            roi      = cur_data[2]
            self.roi_cur += 1
            return im, label, roi

################################################################################
#########################ROI Loss Layer By Python###############################
################################################################################
class roi_Layer(caffe.Layer):
    def setup(self,bottom,top):
	if len(bottom) != 2:
	    raise Exception("Need 2 Inputs")
    def reshape(self,bottom,top):
	label = bottom[1].data
	self.valid_index = np.where(label != -1)[0]
	self.count = len(self.valid_index)
	#bottom[0].data = (batchsize,27)
	#bottom[1].data = (batchsize,1)
	top[0].reshape(len(bottom[1].data), 27,1,1)
	top[1].reshape(len(bottom[1].data), 1)
    def forward(self,bottom,top):
	top[0].data[...][...]=0
	top[1].data[...][...]=0
	top[0].data[0:self.count] = bottom[0].data[self.valid_index]
	top[1].data[0:self.count] = bottom[1].data[self.valid_index]
    def backward(self,top,propagate_down,bottom):
	if propagate_down[0] and self.count!=0:
	    #print "cls0 backward propagate"
	    bottom[0].diff[...]=0
	    bottom[0].diff[self.valid_index]=top[0].diff[...]
	    #print bottom[0].diff[...]
	if propagate_down[1] and self.count!=0:
	    #print "cls1 backward propagate"
	    bottom[1].diff[...]=0
	    bottom[1].diff[self.valid_index]=top[1].diff[...]
class roi_Layer24(caffe.Layer):
    def setup(self,bottom,top):
	if len(bottom) != 2:
	    raise Exception("Need 2 Inputs")
    def reshape(self,bottom,top):
	label = bottom[1].data
	self.valid_index = np.where(label != -1)[0]
	self.count = len(self.valid_index)
	#bottom[0].data = (batchsize,27)
	#bottom[1].data = (batchsize,1)
	top[0].reshape(len(bottom[1].data), 27)
	top[1].reshape(len(bottom[1].data), 1)
    def forward(self,bottom,top):
	top[0].data[...][...]=0
	top[1].data[...][...]=0
	top[0].data[0:self.count] = bottom[0].data[self.valid_index]
	top[1].data[0:self.count] = bottom[1].data[self.valid_index]
    def backward(self,top,propagate_down,bottom):
	if propagate_down[0] and self.count!=0:
	    #print "roi0 backward propagate"
	    bottom[0].diff[...]=0
	    bottom[0].diff[self.valid_index]=top[0].diff[...]
	    #print bottom[0].diff[...]
	if propagate_down[1] and self.count!=0:
	    #print "roi1 backward propagate"
	    bottom[1].diff[...]=0
	    bottom[1].diff[self.valid_index]=top[1].diff[...]
################################################################################
#############################Classify Layer By Python###########################
################################################################################
class cls_Layer(caffe.Layer):
    def setup(self,bottom,top):
	if len(bottom) != 2:
	    raise Exception("Need 2 Inputs")
    def reshape(self,bottom,top):
	label = bottom[1].data
	self.valid_index = np.where(label != -1)[0]
	self.count = len(self.valid_index)
	#bottom[0].data = (batchsize,2)
	#bottom[1].data = (batchsize,1)
	top[0].reshape(len(bottom[1].data), 2,1,1)
	top[1].reshape(len(bottom[1].data), 1)
    def forward(self,bottom,top):
	top[0].data[...][...]=0
	top[1].data[...][...]=0
	top[0].data[0:self.count] = bottom[0].data[self.valid_index]
	top[1].data[0:self.count] = bottom[1].data[self.valid_index]
    def backward(self,top,propagate_down,bottom):
	if propagate_down[0] and self.count!=0:
	    #print "cls0 backward propagate"
	    bottom[0].diff[...]=0
	    bottom[0].diff[self.valid_index]=top[0].diff[...]
	    #print bottom[0].diff[...]
	if propagate_down[1] and self.count!=0:
	    #print "cls1 backward propagate"
	    bottom[1].diff[...]=0
	    bottom[1].diff[self.valid_index]=top[1].diff[...]
class cls_Layer24(caffe.Layer):
    def setup(self,bottom,top):
	if len(bottom) != 2:
	    raise Exception("Need 2 Inputs")
    def reshape(self,bottom,top):
	label = bottom[1].data
	self.valid_index = np.where(label != -1)[0]
	self.count = len(self.valid_index)
	#bottom[0].data = (batchsize,2)
	#bottom[1].data = (batchsize,1)
	top[0].reshape(len(bottom[1].data), 2)
	top[1].reshape(len(bottom[1].data), 1)
    def forward(self,bottom,top):
	top[0].data[...][...]=0
	top[1].data[...][...]=0
	top[0].data[0:self.count] = bottom[0].data[self.valid_index]
	top[1].data[0:self.count] = bottom[1].data[self.valid_index]
    def backward(self,top,propagate_down,bottom):
	if propagate_down[0] and self.count!=0:
	    #print "cls0 backward propagate"
	    bottom[0].diff[...]=0
	    bottom[0].diff[self.valid_index]=top[0].diff[...]
	    #print bottom[0].diff[...]
	if propagate_down[1] and self.count!=0:
	    #print "cls1 backward propagate"
	    bottom[1].diff[...]=0
	    bottom[1].diff[self.valid_index]=top[1].diff[...]
