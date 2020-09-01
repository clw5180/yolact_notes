import os
from data import COCODetection, MEANS, COLORS, COCO_CLASSES
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
import pycocotools

from data import cfg, set_cfg, set_dataset

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle
import json
import os
from pathlib import Path
from collections import OrderedDict
from PIL import Image
import cv2
from tqdm import tqdm

import matplotlib.pyplot as plt
import time

import numpy as np
import numba
from numba import jit


# 传入jit，numba装饰器中的一种
@jit(nopython=True)
def get_outline_from_mask(pear_mask_np, w, h):
    pear_outline = np.zeros_like(pear_mask_np)
    for i, line in enumerate(pear_mask_np):
        for j, point in enumerate(line):
            if i != 0 and i != h - 1 and j != 0 and j != w - 1:  # 图像边缘位置不做判断，这样也能适应比如梨只有一部分在图片内的情形；
                if point == 1 and (line[j - 1] == 0 or line[j + 1] == 0 or line[i - 1] == 0 or line[i + 1] == 0):  # 边缘
                    pear_outline[i][j] = 1
    return pear_outline

# 传入jit，numba装饰器中的一种
#@jit(nopython=True)
def compute_roundness(pear_outline):
    x, y = np.where(pear_outline == 1)
    #x_ctr = x.mean()  # 正常应该0~360度每个角度取一个点，否则比如有些地方锯齿比较多，那么点就比较多，计算中心点时权重就会往这边偏，导致结果不准确
    #y_ctr2 = y.mean()
    x_ctr = (x.max() + x.min()) / 2  # 算下来其实和直接取均值差不多
    y_ctr = (y.max() + y.min()) / 2
    distance = np.sqrt((x - x_ctr) * (x - x_ctr) + (y - y_ctr) * (y - y_ctr))
    # 测量圆度方法1：（不好）
    # dis_min = distance.min()
    # dis_max = distance.max()
    # print('圆度:', dis_max - dis_min)

    # 测量圆度方法2： Σr/N·R计算求出.R为该颗粒轮廓内最大半径
    print('圆度:', distance.mean() / distance.max() )
    print('end')


# run your code
def detect(img_path, save_path):
    net.detect.cross_class_nms = True
    net.detect.use_fast_nms = True
    cfg.mask_proto_debug = False

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    img_names = [name for name in os.listdir(img_path) if name.endswith('.jpg') or name.endswith('.png')]
    #for img_name in tqdm(img_names):
    for img_name in img_names:
        img = cv2.imread(os.path.join(img_path, img_name))
        img = torch.from_numpy(img).cuda().float()
        img = FastBaseTransform()(img.unsqueeze(0))
        start = time.time()
        preds = net(img)
        print('clw: image_name: %s, inference time use %.3fs' % (img_name, time.time() - start))  # inference time use 0.023s, 550x550

        # start = time.time()
        h, w = img.shape[2:]
        result = postprocess(preds, w, h, crop_masks=True, score_threshold=0.3)  # classes, scores, boxes, masks 按照score排序
        # top_k = 10
        # classes, scores, boxes, masks = [x[:top_k].cpu().numpy() for x in result]  # clw note TODO: 是否有必要只取top_k个？
        # print('clw: postprocess time use %.3fs' % (time.time() - start))  # 0.001s


        ### 顺序遍历result[0]，找到第一个是0的值，也就是梨，也就拿到了相应的mask
        # start = time.time()
        bFindPear = False
        for i, cls_id in enumerate(result[0]):
            if cls_id == 0 and not bFindPear:
                pear_mask = result[3][i].cpu().numpy()
                bFindPear = True

        # 从梨的mask中提取轮廓
        pear_outline = get_outline_from_mask(pear_mask, w, h)
        # print('pear_mask.sum:', pear_mask.sum())     # 124250.0
        # print('pear_outline.sum:', pear_outline.sum())  # 34335.0
        # print('clw: outline extract time use %.3fs' % (time.time() - start))  # 0.001s
        roundness = compute_roundness(pear_outline)
        ###



if __name__ == '__main__':
    set_cfg('pear_config')
    with torch.no_grad():
        torch.cuda.set_device(0)

        ######
        # If the input image size is constant, this make things faster (hence why we can use it in a video setting).
        # cudnn.benchmark = True
        # cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        ######

        net = Yolact()
        net.load_weights('./weights/yolact_darknet53_1176_20000.pth')
        net.eval()
        net = net.cuda()
        print('model loaded...')
        detect('/home/user/dataset/pear/train/JPEGImages', '/home/user/pear_output')



print('end!')