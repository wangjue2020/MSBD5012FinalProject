#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp.build import get_exp
from yolox.utils import boxes, fuse_model, get_model_info, postprocess, vis
import json
from yolox.utils import bboxes_iou
import numpy as np

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    # parser.add_argument(
    #     "demo", default="image", help="demo type, eg. image, video and webcam"
    # )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="gpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res

############################# Read Json to Get Annotations ################################
def getAnnotations():
    with open("./datasets/cancer/annotations/test_data.json") as f:
        json_data = json.load(f)
        annotations = json_data["annotations"]
        infos = json_data["images"]
    # Get size info
    size_list = []
    for info in infos:
        img_id = info["image_id"]
        size = (info["width"], info["height"])
        size_list.append(size)
    # Get box info
    ann_dict = {}
    for ann in annotations:
        img_id = ann["image_id"]
        # ann["bbox"][0] *= size_list[img_id][0]
        # ann["bbox"][2] *= size_list[img_id][0]
        # ann["bbox"][1] *= size_list[img_id][1]
        # ann["bbox"][3] *= size_list[img_id][1]
        ann["bbox"][2] += ann["bbox"][0] / 2
        ann["bbox"][3] += ann["bbox"][1] / 2
        if img_id in ann_dict.keys():
            ann_dict[img_id]["bbox"].append(ann["bbox"])
            ann_dict[img_id]["class"].append(ann["category_id"])
        else:
            ann_dict[img_id] = {}
            ann_dict[img_id]["bbox"] = [ann["bbox"]]
            ann_dict[img_id]["class"] = [ann["category_id"]]
    return ann_dict

# def getClosestId(bbox, tru):
    
def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    ann_dict = getAnnotations()
    wrong_cnt = 0
    correct_cnt = 0
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        ####################################### EVALUATE #############################
        id = img_info['id'] + 1
        output = outputs[0]
        # Get ground-truth number of labels
        if id not in ann_dict.keys():
            true_label_num = 0
        else:
            true_label_num = len(ann_dict[id]["bbox"])
        # Get predicted number of labels
        if output == None:
            wrong_cnt += true_label_num
            correct_cnt += 1 if true_label_num == 0 else 0
        else:
            pred_bboxes = output[:,0:4].cpu()
            pred_classes = output[:,6].cpu().numpy()
            true_bboxes = torch.tensor(np.array(ann_dict[id]["bbox"]))
            true_classes = ann_dict[id]["class"]
            true_num = len(true_classes)
            ious = bboxes_iou(pred_bboxes, true_bboxes)
            idxes = []
            i = 0
            for iou in ious:
                if iou[0] > 0 and iou[1] > 0:
                   correct_cnt += 1
                   idxes.append(i)
                i += 1
            
            pred_num = len(idxes)
            loop = pred_num if pred_num <= true_num else true_num

            for j in range(loop):
                if pred_classes[j] == true_classes[j]:
                    correct_cnt += 1
                else:
                    wrong_cnt += 1
            wrong_cnt += abs(pred_num - true_num)
        ####################################### EVALUATE #############################
                
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
    print("Accuracy is:", correct_cnt / (correct_cnt + wrong_cnt))

def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    
    args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(
        model, exp, COCO_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )
    current_time = time.localtime()
    # IMAGE ONLY
    image_demo(predictor, vis_folder, args.path, current_time, args.save_result)


if __name__ == "__main__":
    args = make_parser().parse_args()
    args.name = "yolox-s"
    args.ckpt = "./weight/cancer_detector/last_epoch_ckpt.pth"
    args.path = "./datasets/cancer/val2017/"
    args.nms = 0.45
    args.conf = 0.25
    args.tsize = 640
    args.save_result = True
    args.exp_file = "./exps/cancer.py"
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
