# My-YOLO
What is YOLO Object Detection?

YOLO (“You Only Look Once”) is an effective real-time object recognition algorithm, first described in the seminal 2015 paper by Joseph Redmon et al. In this article we introduce the concept of object detection, the YOLO algorithm itself, and one of the algorithm’s open source implementations: Darknet. To learn more about PP-YOLO (or PaddlePaddle YOLO), which is an improvement on YOLOv4, read our explanation of why PP-YOLO is faster than YOLOv4. 

Image classification is one of the many exciting applications of convolutional neural networks. Aside from simple image classification, there are plenty of fascinating problems in computer vision, with object detection being one of the most interesting. It is commonly associated with self-driving cars where systems blend computer vision, LIDAR and other technologies to generate a multidimensional representation of the road with all its participants. Object detection is also commonly used in video surveillance, especially in crowd monitoring to prevent terrorist attacks, count people for general statistics or analyze customer experience with walking paths within shopping centers.
Object Detection Overview

YOLO Algorithm

There are a few different algorithms for object detection and they can be split into two groups:

    Algorithms based on classification. They are implemented in two stages. First, they select regions of interest in an image. Second, they classify these regions using convolutional neural networks. This solution can be slow because we have to run predictions for every selected region. A widely known example of this type of algorithm is the Region-based convolutional neural network (RCNN) and its cousins Fast-RCNN, Faster-RCNN and the latest addition to the family: Mask-RCNN. Another example is RetinaNet.
    Algorithms based on regression – instead of selecting interesting parts of an image, they predict classes and bounding boxes for the whole image in one run of the algorithm. The two best known examples from this group are the YOLO (You Only Look Once) family algorithms and SSD (Single Shot Multibox Detector). They are commonly used for  real-time object detection as, in general, they trade a bit of accuracy for large improvements in speed.

To understand the YOLO algorithm, it is necessary to establish what is actually being predicted. Ultimately, we aim to predict a class of an object and the bounding box specifying object location. Each bounding box can be described using four descriptors:

    center of a bounding box (x,y)
    width (wt)
    height (ht)
    value ct is corresponding to a class of an object (such as: car, traffic lights, etc.).

In addition, we have to predict the pc value, which is the probability that there is an object in the bounding box.

