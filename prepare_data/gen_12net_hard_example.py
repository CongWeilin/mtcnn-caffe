import sys
sys.path.append('../demo/')
sys.path.append('.')
sys.path.append('/home/cmcc/caffe-master/python')
import tools
import caffe
import cv2
import numpy as np
import os
from utils import *
deploy = '12net.prototxt'
caffemodel = '12net.caffemodel'
net_12 = caffe.Net(deploy,caffemodel,caffe.TEST)
def view_bar(num, total):
    rate = float(num) / total
    rate_num = int(rate * 100)
    r = '\r[%s%s]%d%%  (%d/%d)' % ("#"*rate_num, " "*(100-rate_num), rate_num, num, total)
    sys.stdout.write(r)
    sys.stdout.flush()
def detectFace(img_path,threshold):
    img = cv2.imread(img_path)
    caffe_img = img.copy()-128
    origin_h,origin_w,ch = caffe_img.shape
    scales = tools.calculateScales(img)
    out = []
    for scale in scales:
        hs = int(origin_h*scale)
        ws = int(origin_w*scale)
        scale_img = cv2.resize(caffe_img,(ws,hs))
        scale_img = np.swapaxes(scale_img, 0, 2)
        net_12.blobs['data'].reshape(1,3,ws,hs)
        net_12.blobs['data'].data[...]=scale_img
	caffe.set_device(0)
	caffe.set_mode_gpu()
	out_ = net_12.forward()
        out.append(out_)
    image_num = len(scales)
    rectangles = []
    for i in range(image_num):    
        cls_prob = out[i]['cls_score'][0][1]
        roi      = out[i]['conv4-2'][0]
        out_h,out_w = cls_prob.shape
        out_side = max(out_h,out_w)
        rectangle = tools.detect_face_12net(cls_prob,roi,out_side,1/scales[i],origin_w,origin_h,threshold[0])
        rectangles.extend(rectangle)
    return rectangles
anno_file = 'wider_face_train.txt'
im_dir = "WIDER_train/images/"
neg_save_dir  = "24/negative"
pos_save_dir  = "24/positive"
part_save_dir = "24/part"
image_size = 24
f1 = open('24/pos_24.txt', 'w')
f2 = open('24/neg_24.txt', 'w')
f3 = open('24/part_24.txt', 'w')
threshold = [0.6,0.6,0.7]
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
print "%d pics in total" % num

p_idx = 0 # positive
n_idx = 0 # negative
d_idx = 0 # dont care
image_idx = 0

for annotation in annotations:
    annotation = annotation.strip().split(' ')
    bbox = map(float, annotation[1:])
    gts = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    img_path = im_dir + annotation[0] + '.jpg'
    rectangles = detectFace(img_path,threshold)
    img = cv2.imread(img_path)
    image_idx += 1
    view_bar(image_idx,num)
    for box in rectangles:
        x_left, y_top, x_right, y_bottom, _ = box
        crop_w = x_right - x_left + 1
        crop_h = y_bottom - y_top + 1
        # ignore box that is too small or beyond image border
        if crop_w < image_size or crop_h < image_size :
            continue

        # compute intersection over union(IoU) between current box and all gt boxes
        Iou = IoU(box, gts)
        cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1]
        resized_im = cv2.resize(cropped_im, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

        # save negative images and write label
        if np.max(Iou) < 0.3:
            # Iou with all gts must below 0.3
            save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
            f2.write("%s/negative/%s"%(image_size, n_idx) + ' 0\n')
            cv2.imwrite(save_file, resized_im)
            n_idx += 1
        else:
            # find gt_box with the highest iou
            idx = np.argmax(Iou)
            assigned_gt = gts[idx]
            x1, y1, x2, y2 = assigned_gt

            # compute bbox reg label
            offset_x1 = (x1 - x_left)   / float(crop_w)
            offset_y1 = (y1 - y_top)    / float(crop_h)
            offset_x2 = (x2 - x_right)  / float(crop_w)
            offset_y2 = (y2 - y_bottom )/ float(crop_h)

            # save positive and part-face images and write labels
            if np.max(Iou) >= 0.65:
                save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
                f1.write("%s/positive/%s"%(image_size, p_idx) + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                p_idx += 1

            elif np.max(Iou) >= 0.4:
                save_file = os.path.join(part_save_dir, "%s.jpg"%d_idx)
                f3.write("%s/part/%s"%(image_size, d_idx)     + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                d_idx += 1

f1.close()
f2.close()
f3.close()
