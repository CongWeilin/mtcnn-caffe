# mtcnn-caffe
Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Neural Networks.

### Requirement
0. ubuntu
1. caffe && pycaffe: [https://github.com/BVLC/caffe](https://github.com/BVLC/caffe)
2. opencv && cv2 && numpy

### Train Data
The sample train data is upload to [Baidu Drive](https://pan.baidu.com/s/1kVNVGfd), code is 'ujuv'
The training data generate process can refer to [Seanlinx/mtcnn](https://github.com/Seanlinx/mtcnn)

### Net
The main idea is block backward propagation for different task

12net
![12net](https://github.com/CongWeilin/mtcnn-caffe/blob/master/12net/train12.png)
24net
![24net](https://github.com/CongWeilin/mtcnn-caffe/blob/master/24net/train24.png)
48net
![48net](https://github.com/CongWeilin/mtcnn-caffe/blob/master/48net/train48.png)

### Questions
The Q&A bellow can solve most of your problem.
Q1: What data base do you use?
A1: Similar to official paper, Wider Face for detection and CelebA for alignment.
Q2: What is "12(24/48)net-only-cls.caffemodel" file for?
A2: Provide a initial weigh to train. Since caffe's initial weigh is random, a bad initial weigh may take a long ran to converge even might overfit before that.
Q3: Why preprocess images by minus 128?
A3: Separating data from (0,+) to (-,+), can make converge faster and more accurate. Refer to [Batch Normalization Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
Q4: Do you implement OHEM(Online-Hard-Example-Mining)?
A4: No. OHEM is used when training data is not that much big. Refer to [faster-rcnn's writer RBG's paper](https://arxiv.org/pdf/1604.03540.pdf)
Q5: Ratio positive/negative samples for 12net?
A5: This caffemodel used neg:pos=3:1. Because 12net's function is to eliminate negative answers, similar to exclusive method, we should learn more about negative elininate the wrong answer.
Q6: Why your stride is different to official?
A6: If you input a X*X image, the output Y = (X-11)/2. Every point on output represent a ROI on input. The ROI's left side moving range = (0, X-12) on input, and (0, Y-1) on output. So that stride = (X-12)/(Y-1) ≈≈ 2 in this net

