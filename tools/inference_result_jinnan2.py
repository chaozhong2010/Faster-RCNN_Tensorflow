# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os, sys
import tensorflow as tf
import time
import cv2
import argparse
import numpy as np
import json
sys.path.append("../")

from data.io.image_preprocess import short_side_resize_for_inference_data
from libs.configs import cfgs
from libs.networks import build_whole_network
from libs.box_utils import draw_box_in_img
from help_utils import tools


def detect(det_net, inference_save_path, real_test_imgname_list):

    # 1. preprocess img
    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])  # is RGB. not GBR
    img_batch = tf.cast(img_plac, tf.float32)
    img_batch = short_side_resize_for_inference_data(img_tensor=img_batch,
                                                     target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                                                     length_limitation=cfgs.IMG_MAX_LENGTH)
    img_batch = img_batch - tf.constant(cfgs.PIXEL_MEAN)
    img_batch = tf.expand_dims(img_batch, axis=0) # [1, None, None, 3]

    detection_boxes, detection_scores, detection_category = det_net.build_whole_detection_network(
        input_img_batch=img_batch,
        gtboxes_batch=None)

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
################################################################
###根据checkpoint恢复最新的
    # restorer, restore_ckpt = det_net.get_restorer()

###恢复指定的
    restore_ckpt = os.path.join(cfgs.TRAINED_CKPT, cfgs.VERSION) + '/voc_16000model.ckpt'
    restorer = tf.train.Saver()
    print("model restore from :", restore_ckpt)
    print(20 * "****")
################################################################
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if not restorer is None:
            restorer.restore(sess, restore_ckpt)
            print('restore model')
        all_images_detect_result = []
        for i, a_img_name in enumerate(real_test_imgname_list):

            raw_img = cv2.imread(a_img_name)
            raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]
            start = time.time()
            resized_img, detected_boxes, detected_scores, detected_categories = \
                sess.run(
                    [img_batch, detection_boxes, detection_scores, detection_category],
                    feed_dict={img_plac: raw_img[:, :, ::-1]}  # cv is BGR. But need RGB
                )
            end = time.time()
            # print("{} cost time : {} ".format(img_name, (end - start)))

            show_indices = detected_scores >= cfgs.SHOW_SCORE_THRSHOLD
            show_scores = detected_scores[show_indices]
            show_boxes = detected_boxes[show_indices]
            show_categories = detected_categories[show_indices]
            final_detections = draw_box_in_img.draw_boxes_with_label_and_scores(np.squeeze(resized_img, 0),
                                                                                boxes=show_boxes,
                                                                                labels=show_categories,
                                                                                scores=show_scores)
            nake_name = os.path.split(a_img_name)[1]
            print (inference_save_path + '/' + nake_name)

            # cv2.imwrite(inference_save_path + '/' + nake_name,
            #             final_detections[:, :, ::-1])



            xmin, ymin, xmax, ymax = show_boxes[:, 0], show_boxes[:, 1], \
                                     show_boxes[:, 2], show_boxes[:, 3]
            resized_h, resized_w = resized_img.shape[1], resized_img.shape[2]

            xmin = xmin * raw_w / resized_w
            xmax = xmax * raw_w / resized_w

            ymin = ymin * raw_h / resized_h
            ymax = ymax * raw_h / resized_h
            a_img_detect_result = []
            for idex in range(len(show_scores)):
                # label, score, bbox = a_det[0], a_det[1], a_det[2:]
                det_object = {"xmin": int(xmin[idex]),
                              "xmax": int(xmax[idex]),
                              "ymin": int(ymin[idex]),
                              "ymax": int(ymax[idex]),
                              "label": int(show_categories[idex]),
                              "confidence": float(show_scores[idex])}
                # print (det_object)

                a_img_detect_result.append(det_object)
            image_result = {"filename": nake_name,
                            "rects": a_img_detect_result}
            all_images_detect_result.append(image_result)
            all_images_result_dict = {"results": all_images_detect_result}

            f = open( 'result_jinlian20190314.json', 'w')
            json.dump(all_images_result_dict, f)  # , indent=4
            f.close()
            tools.view_bar('{} image cost {}s'.format(a_img_name, (end - start)), i + 1, len(real_test_imgname_list))

def inference(test_dir, inference_save_path):

    test_imgname_list = [os.path.join(test_dir, img_name) for img_name in os.listdir(test_dir)
                                                          if img_name.endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))]
    assert len(test_imgname_list) != 0, 'test_dir has no imgs there.' \
                                        ' Note that, we only support img format of (.jpg, .png, and .tiff) '

    faster_rcnn = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                       is_training=False)
    detect(det_net=faster_rcnn, inference_save_path=inference_save_path, real_test_imgname_list=test_imgname_list)


def parse_args():
    """
    Parse input arguments
    """
    # parser = argparse.ArgumentParser(description='TestImgs...U need provide the test dir')
    parser = argparse.ArgumentParser('TestImgs...U need provide the test dir')
    parser.add_argument('--data_dir', dest='data_dir',
                        help='data path',
                        default='./test_images', type=str)
    parser.add_argument('--save_dir', dest='save_dir',
                        help='demo imgs to save',
                        default='./test_result', type=str)
    parser.add_argument('--GPU', dest='GPU',
                        help='gpu id ',
                        default='0', type=str)

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()

    return args
if __name__ == '__main__':

    args = parse_args()
    print('Called with args:')
    print(20 * "--")
    print(args)
    print(20 * "--")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    inference(args.data_dir,
              inference_save_path=args.save_dir)
















