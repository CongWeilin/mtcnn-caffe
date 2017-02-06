fid = open("roi.txt",'r')
lines = fid.readlines()
fid.close()
fid = open("roi1.txt",'w')
import sys
sys.path.append('/usr/lib/python2.7/dist-packages')
import cv2
for line in lines:
	img = "roi_images/"+line.split()[0]
	im = cv2.imread(img)
	h,w,ch = im.shape
	if h >= 24 and w >= 24:
		im = cv2.resize(im,(24,24))
		cv2.imwrite(img,im)
		fid.write(line)
	else:
		print line
