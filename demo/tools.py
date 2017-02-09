import sys
sys.path.append('/usr/lib/python2.7/dist-packages')
from operator import itemgetter
import numpy as np
import cv2
'''
Function:
	This trans_map is used for bounding box calibration on 12net and 24net
'''
s_change = [ 0.83, 1.0, 1.21]
x_change = [-0.17, 0.0, 0.17]
y_change = [-0.17, 0.0, 0.17]
trans_map = np.zeros((27,3))
label = 0
for s in s_change:
    for x in x_change:
        for y in y_change:
            trans_map[label][0] = s
            trans_map[label][1] = x
            trans_map[label][2] = y
            label += 1
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
	roi_prob : feature map for regression
	out_side : feature map's largest size
	scale    : current input image scale in multi-scales
	width    : image's origin width
	height   : image's origin height
	threshold: 0.6 can have 99% recall rate on CelebA-database
Output:
	rectangles: possible face positions
'''
def detect_face_12net(cls_prob,roi_prob,out_side,scale,width,height,threshold):
    in_side = 2*out_side+11
    stride = 0
    if out_side != 1:
        stride = float(in_side-12)/(out_side-1)
    boundingBox = []
    for (x,y), prob in np.ndenumerate(cls_prob):
        if(prob >= threshold):
            original_x1 = int(stride * x * scale)
            original_y1 = int(stride * y * scale)
            original_w  = int(12.0 * scale)
            original_h  = int(12.0 * scale)
            s_change = 0
            x_change = 0
            y_change = 0
	    change_number = 0
            for label in range(0,27):
		if roi_prob[label][x][y]>0.2:
            	    #give Rect maltiple change
		    change_number += 1
            	    s_change += trans_map[label][0]
            	    x_change += trans_map[label][1]
           	    y_change += trans_map[label][2]
	    if change_number == 0:
		roi = [original_x1,original_y1,original_x1+original_w,original_y1+original_h,prob]
                boundingBox.append(roi)
		continue
	    s_change = s_change/change_number
	    x_change = x_change/change_number
	    y_change = y_change/change_number
            #put the change into image
            x1 = int(max(0, original_x1 + original_w * x_change))
            y1 = int(max(0, original_y1 + original_h * y_change))
            x2 = int(min(width , x1 + original_w * s_change))
            y2 = int(min(height, y1 + original_h * s_change))
            roi = [x1,y1,x2,y2,prob]
            boundingBox.append(roi)
    return boundingBox
'''
Function:
	Filter face position and calibrate bounding box on 12net's output
Input:
	cls_prob  : softmax feature map for face classify
	roi_prob  : feature map for regression
	rectangles: 12net's predict, rectangles[i][0:3] is the position, rectangles[i][4] is score, rectangles[i][5] is scale
	width     : image's origin width
	height    : image's origin height
	threshold : 0.6 can have 97% recall rate on CelebA-database
Output:
	rectangles: possible face positions
'''
def filter_face_24net(cls_prob,roi_prob,rectangles,width,height,threshold):
    boundingBox = []
    rect_num = len(rectangles)
    for i in range(rect_num):
	if cls_prob[i][1]>threshold:
	    rectangles[i][4] = cls_prob[i][1] # update score
	    indices = np.nonzero(roi_prob[i]>0.2)[0]
	    number_of_cals = len(indices)
	    if number_of_cals == 0:
		boundingBox.append(rectangles[i])
		continue
	    
	    s_change = 0
            x_change = 0
            y_change = 0
            for label in indices:
                #give Rect maltiple change
                s_change += trans_map[label][0]
                x_change += trans_map[label][1]
                y_change += trans_map[label][2]
            s_change = s_change/number_of_cals
            x_change = x_change/number_of_cals
            y_change = y_change/number_of_cals
	    original_x1 = int(rectangles[i][0])
            original_y1 = int(rectangles[i][1])
            original_x2 = int(rectangles[i][2])
            original_y2 = int(rectangles[i][3])
            original_w = original_x2 - original_x1
            original_h = original_y2 - original_y1
	    rectangles[i][0] = int(max(0, original_x1 + original_w * x_change))
            rectangles[i][1] = int(max(0, original_y1 + original_h * y_change))
            rectangles[i][2] = int(min(width , rectangles[i][0] + original_w * s_change))
            rectangles[i][3] = int(min(height, rectangles[i][1] + original_h * s_change))
	    boundingBox.append(rectangles[i])

    return NMS(boundingBox,0.3,'iou')
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
	    rect = [rectangles[i][0],rectangles[i][1],rectangles[i][2],rectangles[i][3],rectangles[i][4],
		   roi[i][0],roi[i][1],roi[i][2],roi[i][3],
		   pts[i][0],pts[i][1],pts[i][2],pts[i][3],pts[i][4],pts[i][5],pts[i][6],pts[i][7],pts[i][8],pts[i][9],cls_prob[i][1]]

	    boundingBox.append(rect)
    rectangles = NMS(boundingBox,0.3,'iom')
    rect = []
    for rectangle in rectangles:
	roi_w = rectangle[2]-rectangle[0]
	roi_h = rectangle[3]-rectangle[1]
	x1 = rectangle[0]-rectangle[5]*roi_w
	y1 = rectangle[1]-rectangle[6]*roi_h
	x2 = rectangle[2]-rectangle[7]*roi_w
	y2 = rectangle[3]-rectangle[8]*roi_h
	pt0 = rectangle[ 9]*roi_w + rectangle[0]
	pt1 = rectangle[10]*roi_h + rectangle[1]
	pt2 = rectangle[11]*roi_w + rectangle[0]
	pt3 = rectangle[12]*roi_h + rectangle[1]
	pt4 = rectangle[13]*roi_w + rectangle[0]
	pt5 = rectangle[14]*roi_h + rectangle[1]
	pt6 = rectangle[15]*roi_w + rectangle[0]
	pt7 = rectangle[16]*roi_h + rectangle[1]
	pt8 = rectangle[17]*roi_w + rectangle[0]
	pt9 = rectangle[18]*roi_h + rectangle[1]
	score = rectangle[19]
	rect.append([x1,y1,x2,y2,pt0,pt1,pt2,pt3,pt4,pt5,pt6,pt7,pt8,pt9,score])

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
    pr_scale = 1.0
    h,w,ch = caffe_img.shape
    if w > 1000 or h > 1000: #max side <= 1000
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
