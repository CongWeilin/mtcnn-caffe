import sys
from operator import itemgetter
import numpy as np
import cv2
'''
Function:
	calculate Intersect of Union
Input: 
	rect_1: 1st rectangle
	rect_2: 2nd rectangle
Output:
	IoU
'''
def IoU(rect_1, rect_2):
    x11 = rect_1[0]    # first rectangle top left x
    y11 = rect_1[1]    # first rectangle top left y
    x12 = rect_1[2]    # first rectangle bottom right x
    y12 = rect_1[3]    # first rectangle bottom right y
    x21 = rect_2[0]    # second rectangle top left x
    y21 = rect_2[1]    # second rectangle top left y
    x22 = rect_2[2]    # second rectangle bottom right x
    y22 = rect_2[3]    # second rectangle bottom right y
    x_overlap = max(0, min(x12,x22) -max(x11,x21))
    y_overlap = max(0, min(y12,y22) -max(y11,y21))
    intersection = x_overlap * y_overlap
    union = (x12-x11) * (y12-y11) + (x22-x21) * (y22-y21) - intersection
    if union == 0:
	return 0
    else:
        return float(intersection) / union
'''
Function:
	calculate Intersect of Min area
Input: 
	rect_1: 1st rectangle
	rect_2: 2nd rectangle
Output:
	IoM
'''
def IoM(rect_1, rect_2):
    x11 = rect_1[0]    # first rectangle top left x
    y11 = rect_1[1]    # first rectangle top left y
    x12 = rect_1[2]    # first rectangle bottom right x
    y12 = rect_1[3]    # first rectangle bottom right y
    x21 = rect_2[0]    # second rectangle top left x
    y21 = rect_2[1]    # second rectangle top left y
    x22 = rect_2[2]    # second rectangle bottom right x
    y22 = rect_2[3]    # second rectangle bottom right y
    x_overlap = max(0, min(x12,x22) -max(x11,x21))
    y_overlap = max(0, min(y12,y22) -max(y11,y21))
    intersection = x_overlap * y_overlap
    rect1_area = (y12 - y11) * (x12 - x11)
    rect2_area = (y22 - y21) * (x22 - x21)
    min_area = min(rect1_area, rect2_area)
    return float(intersection) / min_area
'''
Function:
	apply NMS(non-maximum suppression) on ROIs in same scale
Input:
	rectangles: rectangles[i][0:3] is the position, rectangles[i][4] is scale, rectangles[i][5] is score
Output:
	rectangles: same as input
'''
def NMS(rectangles,threshold,type):
    sorted(rectangles,key=itemgetter(4),reverse=True)
    result_rectangles = rectangles
    number_of_rects = len(result_rectangles)
    cur_rect = 0
    while cur_rect < number_of_rects : 
        rects_to_compare = number_of_rects - cur_rect - 1 
        cur_rect_to_compare = cur_rect + 1 
        while rects_to_compare > 0:
	    score = 0
	    if type == 'iou':
		score =  IoU(result_rectangles[cur_rect], result_rectangles[cur_rect_to_compare])
	    else:
		score =  IoM(result_rectangles[cur_rect], result_rectangles[cur_rect_to_compare])
            if score >= threshold:
                del result_rectangles[cur_rect_to_compare]      # delete the rectangle
                number_of_rects -= 1
            else:
                cur_rect_to_compare += 1    # skip to next rectangle            
            rects_to_compare -= 1
        cur_rect += 1   # finished comparing for current rectangle
    return result_rectangles
'''
Function:
	Detect face position and calibrate bounding box on 12net feature map
Input:
	cls_prob : softmax feature map for face classify
	roi      : feature map for regression
	out_side : feature map's largest size
	scale    : current input image scale in multi-scales
	width    : image's origin width
	height   : image's origin height
	threshold: 0.6 can have 99% recall rate
'''
def detect_face_12net(cls_prob,roi,out_side,scale,width,height,threshold):
    in_side = 2*out_side+11
    stride = 0
    if out_side != 1:
        stride = float(in_side-12)/(out_side-1)
    boundingBox = []

    for (x,y), prob in np.ndenumerate(cls_prob):
        if(prob >= threshold):
            original_x1 = int((stride*x + 1)*scale)
            original_y1 = int((stride*y + 1)*scale)
            original_w  = int((12.0 -1)*scale)
            original_h  = int((12.0 -1)*scale)
            original_x2 = original_x1 + original_w
            original_y2 = original_y1 + original_h
            rect = []
            x1 = int(round(max(0     , original_x1 + original_w * roi[0][x][y])))
            y1 = int(round(max(0     , original_y1 + original_h * roi[1][x][y])))
            x2 = int(round(min(width , original_x2 + original_w * roi[2][x][y])))
            y2 = int(round(min(height, original_y2 + original_h * roi[3][x][y])))
	    if x2>x1 and y2>y1:
                rect = [x1,y1,x2,y2,prob]
                boundingBox.append(rect)
    return NMS(boundingBox,0.5,'iou')
'''
Function:
	Filter face position and calibrate bounding box on 12net's output
Input:
	cls_prob  : softmax feature map for face classify
	roi_prob  : feature map for regression
	rectangles: 12net's predict
	width     : image's origin width
	height    : image's origin height
	threshold : 0.6 can have 97% recall rate
Output:
	rectangles: possible face positions
'''
def filter_face_24net(cls_prob,roi,rectangles,width,height,threshold):
    boundingBox = []
    rect_num = len(rectangles)
    for i in range(rect_num):
	if cls_prob[i][1]>threshold:
	    original_w = rectangles[i][2]-rectangles[i][0]+1
	    original_h = rectangles[i][3]-rectangles[i][1]+1
	    x1 = int(round(max(0     , rectangles[i][0] + original_w * roi[i][0])))
            y1 = int(round(max(0     , rectangles[i][1] + original_h * roi[i][1])))
            x2 = int(round(min(width , rectangles[i][2] + original_w * roi[i][2])))
            y2 = int(round(min(height, rectangles[i][3] + original_h * roi[i][3])))
	    if x2>x1 and y2>y1:
	        rect = [x1,y1,x2,y2,cls_prob[i][1]]
	        boundingBox.append(rect)
    return NMS(boundingBox,0.7,'iou')
'''
Function:
	Filter face position and calibrate bounding box on 12net's output
Input:
	cls_prob  : cls_prob[1] is face possibility
	roi       : roi offset
	pts       : 5 landmark
	rectangles: 12net's predict, rectangles[i][0:3] is the position, rectangles[i][4] is score
	width     : image's origin width
	height    : image's origin height
	threshold : 0.7 can have 94% recall rate on CelebA-database
Output:
	rectangles: face positions and landmarks
'''
def filter_face_48net(cls_prob,roi,pts,rectangles,width,height,threshold):
    boundingBox = []
    rect_num = len(rectangles)
    for i in range(rect_num):
	if cls_prob[i][1]>threshold:
	    rect = [rectangles[i][0],rectangles[i][1],rectangles[i][2],rectangles[i][3],cls_prob[i][1],
		   roi[i][0],roi[i][1],roi[i][2],roi[i][3],
		   pts[i][0],pts[i][5],pts[i][1],pts[i][6],pts[i][2],pts[i][7],pts[i][3],pts[i][8],pts[i][4],pts[i][9]]
	    boundingBox.append(rect)
    rectangles = NMS(boundingBox,0.7,'iom')
    rect = []
    
    for rectangle in rectangles:
	roi_w = rectangle[2]-rectangle[0]+1
	roi_h = rectangle[3]-rectangle[1]+1

  	x1 = round(max(0     , rectangle[0]+rectangle[5]*roi_w))
        y1 = round(max(0     , rectangle[1]+rectangle[6]*roi_h))
        x2 = round(min(width , rectangle[2]+rectangle[7]*roi_w))
        y2 = round(min(height, rectangle[3]+rectangle[8]*roi_h))
	pt0 = rectangle[ 9]*roi_w + rectangle[0] -1
	pt1 = rectangle[10]*roi_h + rectangle[1] -1
	pt2 = rectangle[11]*roi_w + rectangle[0] -1
	pt3 = rectangle[12]*roi_h + rectangle[1] -1
	pt4 = rectangle[13]*roi_w + rectangle[0] -1
	pt5 = rectangle[14]*roi_h + rectangle[1] -1
	pt6 = rectangle[15]*roi_w + rectangle[0] -1
	pt7 = rectangle[16]*roi_h + rectangle[1] -1
	pt8 = rectangle[17]*roi_w + rectangle[0] -1
	pt9 = rectangle[18]*roi_h + rectangle[1] -1
	score = rectangle[4]
	rect_ = np.round([x1,y1,x2,y2,pt0,pt1,pt2,pt3,pt4,pt5,pt6,pt7,pt8,pt9]).astype(int)
	rect_= np.append(rect_,score)
	rect.append(rect_)
    return rect
'''
Function:
	calculate multi-scale and limit the maxinum side to 1000 
Input: 
	img: original image
Output:
	pr_scale: limit the maxinum side to 1000, < 1.0
	scales  : Multi-scale
'''
def calculateScales(img):
    caffe_img = img.copy()
    h,w,ch = caffe_img.shape
    pr_scale = 1000.0/max(h,w)
    w = int(w*pr_scale)
    h = int(h*pr_scale)

    #multi-scale
    scales = []
    factor = 0.7937
    factor_count = 0
    minl = min(h,w)
    while minl >= 12:
        scales.append(pr_scale*pow(factor, factor_count))
        minl *= factor
        factor_count += 1
    return scales
