import numpy as np
import numpy.random as npr
size = 12
net = str(size)
with open('%s/pos_%s.txt'%(net, size), 'r') as f:
    pos = f.readlines()

with open('%s/neg_%s.txt'%(net, size), 'r') as f:
    neg = f.readlines()

with open('%s/part_%s.txt'%(net, size), 'r') as f:
    part = f.readlines()
    
def view_bar(num, total):
    rate = float(num) / total
    rate_num = int(rate * 100)+1
    r = '\r[%s%s]%d%%' % ("#"*rate_num, " "*(100-rate_num), rate_num, )
    sys.stdout.write(r)
    sys.stdout.flush()
    
import sys
import cv2
import os
import numpy as np

cls_list = []
print '\n'+'positive-12'
cur_ = 0
sum_ = len(pos)
for line in pos:
    view_bar(cur_,sum_)
    cur_ += 1
    words = line.split()
    image_file_name = words[0]+'.jpg'
    im = cv2.imread(image_file_name)
    h,w,ch = im.shape
    if h!=12 or w!=12:
        im = cv2.resize(im,(12,12))
    im = np.swapaxes(im, 0, 2)
    im = (im - 127.5)/127.5
    label    = 1
    roi      = [-1,-1,-1,-1]
    pts	     = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    cls_list.append([im,label,roi])
print '\n'+'negative-12'
cur_ = 0
neg_keep = npr.choice(len(neg), size=600000, replace=False)
sum_ = len(neg_keep)
for i in neg_keep:
    line = neg[i]
    view_bar(cur_,sum_)
    cur_ += 1
    words = line.split()
    image_file_name = words[0]+'.jpg'
    im = cv2.imread(image_file_name)
    h,w,ch = im.shape
    if h!=12 or w!=12:
        im = cv2.resize(im,(12,12))
    im = np.swapaxes(im, 0, 2)
    im = (im - 127.5)/127.5
    label    = 0
    roi      = [-1,-1,-1,-1]
    pts	     = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    cls_list.append([im,label,roi]) 
import cPickle as pickle
fid = open("12/cls.imdb",'w')
pickle.dump(cls_list, fid)
fid.close()

roi_list = []
print '\n'+'part-12'
cur_ = 0
part_keep = npr.choice(len(part), size=300000, replace=False)
sum_ = len(part_keep)
for i in part_keep:
    line = part[i]
    view_bar(cur_,sum_)
    cur_ += 1
    words = line.split()
    image_file_name = words[0]+'.jpg'
    im = cv2.imread(image_file_name)
    h,w,ch = im.shape
    if h!=12 or w!=12:
        im = cv2.resize(im,(12,12))
    im = np.swapaxes(im, 0, 2)
    im -= 128
    label    = -1
    roi      = [float(words[2]),float(words[3]),float(words[4]),float(words[5])]
    pts	     = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    roi_list.append([im,label,roi])
print '\n'+'positive-12'
cur_ = 0
sum_ = len(pos)
for line in pos:
    view_bar(cur_,sum_)
    cur_ += 1
    words = line.split()
    image_file_name = words[0]+'.jpg'
    im = cv2.imread(image_file_name)
    h,w,ch = im.shape
    if h!=12 or w!=12:
        im = cv2.resize(im,(12,12))
    im = np.swapaxes(im, 0, 2)
    im = (im - 127.5)/127.5
    label    = -1
    roi      = [float(words[2]),float(words[3]),float(words[4]),float(words[5])]
    pts	     = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    roi_list.append([im,label,roi])
import cPickle as pickle
fid = open("12/roi.imdb",'w')
pickle.dump(roi_list, fid)
fid.close()
