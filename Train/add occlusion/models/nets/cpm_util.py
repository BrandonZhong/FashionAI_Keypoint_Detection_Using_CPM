# -*- coding: utf-8 -*-
from __future__ import print_function

import os, cv2
import random, math
import numpy as np
import argparse
import matplotlib.pyplot as plt

from PIL import Image, ImageEnhance, ImageFilter
from skimage import transform
from pycocotools.coco import COCO

import scipy.io
#data = scipy.io.loadmat('examples.mat')

import sys
sys.path.append('../../aframe')

from aimage import ImageUtil


def _reshape_coco(joints_flat, num_joint):
    """ reshape coco annotation in flat list format
        AND Caculate the Neck joint position
    Args:
        joints_flat: flat format list of joints info directly from COCO
    Return:
        Numpy format matrix of shape of (num_joint, 2)
    """
    joints = np.zeros((num_joint, 3), np.float)
    weights = np.ones((num_joint), np.int8)
    joints[:-1, :] = np.array(joints_flat).reshape((-1, 3))
    joints[np.where(joints==0)] = -1

    # neck is uncalculatable if any side of shoulder is invisible
    if (joints[5]==-1).any() or (joints[6]==-1).any():
        joints[-1, :] = np.array([-1, -1, -1])
    else:
        joints[-1, :] = (joints[5] + joints[6]) / 2

    for i in range(joints.shape[0]):
        if (joints[i]==-1).any():
            weights[i] = 0

    return joints, weights

def _color_augment(img):
    image = Image.fromarray(img)
    
    # 亮度增强
    if random.choice([0, 1]):
        enh_bri = ImageEnhance.Brightness(image)
        brightness = random.choice([0.6,0.8,1.2,1.4])
        image = enh_bri.enhance(brightness)
        # image.show()

    # 色度增强
    if random.choice([0, 1]):
        enh_col = ImageEnhance.Color(image)
        color = random.choice([0.6,0.8,1.2,1.4])
        image = enh_col.enhance(color)
        # image.show()

    # 对比度增强
    if random.choice([0, 1]):
        enh_con = ImageEnhance.Contrast(image)
        contrast = random.choice([0.6,0.8,1.2,1.4])
        image = enh_con.enhance(contrast)
        # image.show()

    # 锐度增强
    if random.choice([0, 1]):
        enh_sha = ImageEnhance.Sharpness(image)
        sharpness = random.choice([0.6,0.8,1.2,1.4])
        image = enh_sha.enhance(sharpness)
        # image.show()

    # mo hu
    if random.choice([0, 1]):
        image = image.filter(ImageFilter.BLUR)

    img = np.asarray(image)
    return img

def _angle_augment(ori_img, img, hm, max_rotation=30):
    if random.choice([0, 1]):
        r_angle = np.random.randint(-1 * max_rotation, max_rotation)
        ori_img = transform.rotate(ori_img, r_angle, preserve_range=True)
        img = transform.rotate(img, r_angle, preserve_range=True)
        hm = transform.rotate(hm, r_angle)
    return ori_img, img, hm

def make_gaussian(size, radius=3, center=None):
    """ Make a square gaussian kernel.
    size: the length of a side of the square
    radius: full-width-half-maximum
    """
    x, y = np.meshgrid(range(size), range(size))
    cx, cy = (size // 2, size // 2) if center is None else center[:2]

    gaussian_map = np.exp(-4 * np.log(2) * ((x - cx) ** 2 + (y - cy) ** 2) / radius ** 2)
    return gaussian_map

def norm_image(img, input_size):
    ori_img = cv2.imread(img) if isinstance(img, str) else img

    scale = float(input_size) / max(ori_img.shape[:2])
    image = cv2.resize(ori_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

    output_img = np.ones((input_size, input_size, 3), dtype=np.uint8) * 128

    img_h, img_w = image.shape[:2]

    offset = dict(l=0, t=0, r=0, b=0)
    if img_w < img_h:
        offset['l'] = int(input_size/2-img_w/2)
        offset['r'] = input_size - img_w - offset['l']
    if img_w > img_h:
        offset['t'] = int(input_size/2-img_h/2)
        offset['b'] = input_size - img_h - offset['t']

    output_img[offset['t']:offset['t']+img_h, offset['l']:offset['l']+img_w, :] = image

    return (output_img, offset, scale)

def display_image(img, heatmap):
    fig = plt.figure()

    rgb_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    
    target_size = (heatmap.shape[1], heatmap.shape[0])
    resized_img = cv2.resize(rgb_img, target_size, interpolation=cv2.INTER_AREA)
    
    a = fig.add_subplot(1, 2, 1)
    a.set_title('Image')
    plt.imshow(rgb_img)

    a = fig.add_subplot(1, 2, 2)
    a.set_title('Heatmap')
    plt.imshow(resized_img, alpha=0.5)
    tmp = np.amax(heatmap[:,:,:-1], axis=2)
    plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
    #plt.colorbar()

    plt.show()

def render_image(img, joint_list, limb_pair, joint_desc=None):
    num_joint = len(joint_list)
    indice = np.linspace(1, 255, num_joint).astype(int)
    color = (plt.cm.brg(indice)*255)[:,:3].astype(int).tolist()
    joint_desc = joint_desc or [''] * num_joint

    # Plot limb colors
    second_axis = int(max(3, 1e-2 * img.shape[0]))
    for joint1, joint2 in limb_pair:
        x1, y1, c1 = map(int, joint_list[joint1, :])
        x2, y2, c2 = map(int, joint_list[joint2, :])
        if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
            #cv2.line(img, (x1, y1), (x2, y2), color[joint1], 3)
            deg = math.degrees(math.atan2(y1 - y2, x1 - x2))
            length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            polygon = cv2.ellipse2Poly(
                (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                (int(length / 2), second_axis), int(deg), 0, 360, 1)
            cv2.fillConvexPoly(img, polygon, color=color[joint1])

    # Plot joint colors
    font_scale = 1e-3 * img.shape[0]
    thickness = int(max(1, 2e-3 * img.shape[0]))
    for i in range(num_joint):
        if joint_list[i, 0] > 0 and joint_list[i, 1] > 0:
            cv2.circle(img, center=(int(joint_list[i, 0]), int(joint_list[i, 1])), 
                radius=3, color=color[i], thickness=thickness)

            display_txt = '{:.2f} {}'.format(joint_list[i, 2], joint_desc[i])
            text_size = cv2.getTextSize(display_txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            width, height = text_size[0][0], text_size[0][1]
            x1, y1 = int(joint_list[i, 0]+5), int(joint_list[i, 1])
            _y = max(y1, height+6)
            region = np.array([
                [x1-3, _y+3], [x1-3, _y-height-6],
                [x1+width+3, _y-height-6], [x1+width+3, _y+3]], dtype='int32')
            #cv2.fillPoly(img=img, pts=[region], color=color[i])

            #cv2.putText(img, display_txt, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, np.array([255,255,255])-color[i], thickness)


def cpm_generator(anno_path, img_dir, num_joint, input_size, radius, limb_pair, 
    cropped=False, debug=False):
    indices = np.linspace(1, 255, num_joint).astype(int)
    color = (plt.cm.brg(indices)*255)[:,:3].astype(int).tolist()

    scale_pool = 8

    coco_anno = COCO(anno_path)
    cate_ids = coco_anno.getCatIds(catNms=['person'])
    #imgIds = coco_anno.getImgIds(catIds=cate_ids)
    anno_ids = coco_anno.getAnnIds(catIds=cate_ids, iscrowd=0)
    anno_list = coco_anno.loadAnns(anno_ids)

    while True:
        random.shuffle(anno_list)
        for num, anno in enumerate(anno_list):
            if anno['num_keypoints'] < 8 or anno["area"] < 32 * 32: continue

            dic = {}
            dic['keypoints'], dic['weights'] = _reshape_coco(anno['keypoints'], num_joint)
            dic['bbox'] = anno['bbox']
            img = coco_anno.loadImgs(anno['image_id'])[0]
            dic['file_name'] = img['file_name']
            dic['height'] = img['height']
            dic['width'] = img['width']

            img_path = os.path.join(img_dir, dic['file_name'])
            cur_img = cv2.imread(img_path)
            assert cur_img.shape[:2] == (dic['height'], dic['width'])

            if cropped:
                _x, _y, _w, _h = map(int, dic['bbox'])

                p = 0.15
                _x = max(0, int(_x - _w * p))
                _y = max(0, int(_y - _h * p))
                _w = int(_w + _w * p * 2)
                _h = int(_h + _h * p * 2)
                _w = _w - (max(dic['width'], _w+_x) - dic['width'])
                _h = _h - (max(dic['height'], _h+_y) - dic['height'])

                cur_img = cur_img[_y:_y+_h, _x:_x+_w, :]
            else:
                _x, _y, _w, _h = 0, 0, dic['width'], dic['height']

            #cur_img = _color_augment(cur_img)

            output_image, offset, scale = norm_image(cur_img, input_size)
            output_image = output_image / 255.0

            # Relocalize points
            joint_list = dic['keypoints']
            for i in range(num_joint):
                if (joint_list[i, 0] < _x or joint_list[i, 0] > _x + _w or
                    joint_list[i, 1] < _y or joint_list[i, 1] > _y + _h):
                    joint_list[i, :] = [-1, -1, -1]
                else:
                    joint_list[i, 0] = (joint_list[i, 0] - _x) * scale + offset['l']
                    joint_list[i, 1] = (joint_list[i, 1] - _y) * scale + offset['t']

            # set headmap
            heapmap_size = input_size / scale_pool
            output_heatmap = np.zeros((heapmap_size, heapmap_size, num_joint))
            for i in range(num_joint):
                if joint_list[i, 0] > 0 and joint_list[i, 1] > 0:
                    output_heatmap[:, :, i] = make_gaussian(heapmap_size, radius,
                        [joint_list[i, 0]/scale_pool, joint_list[i, 1]/scale_pool])

            img = (output_image * 255.0).astype(np.uint8)
            if debug:
                render_image(img, joint_list, limb_pair)

            #img, output_image, output_heatmap = _angle_augment(img, output_image, output_heatmap)

            # Create background map
            output_background_map = np.ones((heapmap_size, heapmap_size))
            output_background_map -= np.amax(output_heatmap, axis=2)
            output_heatmap = np.concatenate((output_heatmap, 
                output_background_map.reshape((heapmap_size, heapmap_size, 1))), axis=2)

            if debug:
                display_image(img, output_heatmap)

            yield output_image, output_heatmap


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate ground-truth file')
    parser.add_argument('--generator', action='store_true', default=False, help='generate')
    parser.add_argument('--anno_path', type=str, default='/home/common/coco2017/annotations/person_keypoints_val2017.json', help='anno_path')
    parser.add_argument('--img_dir', type=str, default='/home/common/coco2017/val2017', help='img_dir')
    parser.add_argument('--num_joint', type=int, default=18, help='num_joint')
    parser.add_argument('--input_size', type=int, default=368, help='input_size')
    parser.add_argument('--radius', type=int, default=3, help='radius')
    parser.add_argument('--cropped', action='store_true', default=False, help='cropped')
    parser.add_argument('--gaussian', action='store_true', default=False, help='gaussian')
    parser.add_argument('--image', action='store_true', default=False, help='read image')
    args = parser.parse_args()

    if args.generator:
        limb_pair = [
            [0, 1],  [0, 2],   [1, 3],   [2, 4],   #[3, 5],   [4, 6],  
            [5, 7],  [7, 9],   [6, 8],   [8, 10],  [0, 17],  [5, 17],
            [6, 17], [17, 11], [11, 13], [13, 15], [17, 12], [12, 14], [14, 16],
        ]
        for output_image, output_heatmap in cpm_generator(
            args.anno_path, args.img_dir, args.num_joint, args.input_size, args.radius, 
                limb_pair, cropped=args.cropped, debug=True): pass
    elif args.gaussian:
        gaussian_map = make_gaussian(46, 2, (13,23))
        ImageUtil.plot_image([(gaussian_map, 'gaussian', 1)])
    elif args.image:
        ImageUtil.show_image(norm_image('data/longhand.jpg', 128)[0])
        ImageUtil.show_image(norm_image('data/roger.png', 128)[0])
