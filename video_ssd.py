import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2
import sys

slim = tf.contrib.slim

sys.path.append('./')

from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing


object_type = {0:"none", 1:"aeroplane", 2:"bicycle", 3:"bird", 4:"boat", 5:"bottle",
               6:"bus", 7:"car", 8:"cat", 9:"chair", 10:"cow", 11:"diningtable", 12:"dog",
               13:"horse", 14:"motorbike", 15:"person",16:"pottedplant", 17:"sheep",
               18:"sofa", 19:"train", 20:"tvmonitor"}


# Main image processing routine.
def process_image(sess, img, ssd_anchors,
                  select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = sess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes


def cv2_bboxes(image, classes, scores, bboxes, linewidth=1.5):
    colors = dict()
    height, width = image.shape[:2]
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = scores[i]
            ymin = int(bboxes[i, 0] * height)
            xmin = int(bboxes[i, 1] * width)
            ymax = int(bboxes[i, 2] * height)
            xmax = int(bboxes[i, 3] * width)
            class_name = str(cls_id)
            image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            image = cv2.putText(image, '{:s} | {:.3f}'.format(object_type[cls_id], score),
                                (xmin, ymin-2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return image


# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = '/Users/hdx/code/python3/SSD-Tensorflow/checkpoints/ssd_300_vgg.ckpt'

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_filename)

    # SSD default anchor boxes.
    ssd_anchors = ssd_net.anchors(net_shape)

    # # read images
    # img = cv2.imread("/Users/hdx/Desktop/IMG_1531.jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # rclasses, rscores, rbboxes = process_image(sess, img, ssd_anchors)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # img = cv2_bboxes(img, rclasses, rscores, rbboxes)
    # cv2.imshow("demo", img)
    # cv2.waitKey(0)

    # cap = cv2.VideoCapture(0)
    # while (cap.isOpened()):
    #     # Read the frame
    #     ret, img = cap.read()
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     rclasses, rscores, rbboxes = process_image(sess, img, ssd_anchors)
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     img = cv2_bboxes(img, rclasses, rscores, rbboxes)
    #     cv2.imshow("demo", img)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    # cap.release()

    cap = cv2.VideoCapture("/Users/hdx/Desktop/1562551881608587.mp4")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('/Users/hdx/Desktop/1562551881608587_output.mp4', fourcc, fps, (height, width))

    success, img = cap.read()
    while success:
        # there is a pit, image will be rotate 90
        img = cv2.transpose(img)
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rclasses, rscores, rbboxes = process_image(sess, img, ssd_anchors)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2_bboxes(img, rclasses, rscores, rbboxes)
        out.write(img)
        # cv2.imshow("demo", img)
        # cv2.waitKey(1)
        success, img = cap.read()

    out.release()
    cap.release()

