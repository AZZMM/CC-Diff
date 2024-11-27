import os
import json
import collections
import xml.etree.ElementTree as ET # for parsing the annotations in DIOR (.XML format)

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torchvision.utils import save_image, make_grid

from segment_anything import sam_model_registry, SamPredictor

from pdb import set_trace as ST


def _load_dota_txt(txtfile):
    """Load DOTA's txt annotation.

    Args:
        txtfile (str): Filename of single txt annotation.

    Returns:
        dict: Annotation of single image.
    """
    gsd, bboxes, labels, diffs = None, [], [], []
    if txtfile is None:
        pass
    elif not os.path.isfile(txtfile):
        print(f"Can't find {txtfile}, treated as empty txtfile")
    else:
        with open(txtfile, 'r') as f:
            for line in f:
                if line.startswith('gsd'):
                    num = line.split(':')[-1]
                    try:
                        gsd = float(num)
                    except ValueError:
                        gsd = None
                    continue

                items = line.split(' ')
                if len(items) >= 9:
                    bboxes.append([float(i) for i in items[:8]])
                    labels.append(items[8])
                    diffs.append(int(items[9]) if len(items) == 10 else 0)

    bboxes = np.array(bboxes, dtype=np.float32) if bboxes else \
        np.zeros((0, 8), dtype=np.float32)
    diffs = np.array(diffs, dtype=np.int64) if diffs else \
        np.zeros((0,), dtype=np.int64)
    ann = dict(bboxes=bboxes, labels=labels, diffs=diffs)
    return dict(gsd=gsd, ann=ann)

def poly2obb_np_le90(poly):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    bboxps = np.array(poly).reshape((4, 2))
    rbbox = cv2.minAreaRect(bboxps)
    x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[
        2]
    if w < 1 or h < 1:
        return
    a = a / 180 * np.pi
    if w < h:
        w, h = h, w
        a += np.pi / 2
    while not np.pi / 2 > a >= -np.pi / 2:
        if a >= np.pi / 2:
            a -= np.pi
        else:
            a += np.pi
    assert np.pi / 2 > a >= -np.pi / 2
    return x, y, w, h, a


def poly2hbb(polys):
    """Convert polygons to horizontal bboxes.

    Args:
        polys (np.array): Polygons with shape (N, 8)

    Returns:
        np.array: Horizontal bboxes.
    """
    shape = polys.shape
    polys = polys.reshape(*shape[:-1], shape[-1] // 2, 2)
    lt_point = np.min(polys, axis=-2)
    rb_point = np.max(polys, axis=-2)
    return np.concatenate([lt_point, rb_point], axis=-1)


if __name__ == '__main__':
    # config dataset info
    dataset_name = 'DOTA'
    img_dir  = 'path_to_data/DOTA/train/images'
    anno_dir = 'path_to_data/DOTA/train/labelTxt'
    save_dir = 'path_to_data/DOTA/results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # setup SAM
    ckpt_path  = "./checkpoint/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device     = "cuda"
    SAM = sam_model_registry[model_type](checkpoint=ckpt_path)
    SAM.to(device=device)
    
    predictor = SamPredictor(SAM)
    # go through the dataset
    for img_name, anno_name in zip(sorted(os.listdir(img_dir)), sorted(os.listdir(anno_dir))):
        print(img_name)

        # read in image and parse the annotation
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        anno_path = os.path.join(anno_dir, anno_name)
        content = _load_dota_txt(anno_path)['ann']
        # obj_name_list, bbox_list = anno_parser(anno_path, dataset_name)
        obj_name_list = []
        bbox_list = []
        for label, poly in zip(content['labels'], content['bboxes']):
            obbox = poly2obb_np_le90(poly)
            if obbox == None:
                continue
            obj_name_list.append(label)
            bbox_list.append(poly2hbb(poly)[None, ...])
        if len(obj_name_list) == 0:
            continue
        # batched segmentation with SAM
        predictor.set_image(img)
        input_bboxes = torch.tensor(np.concatenate(bbox_list, axis=0), device=predictor.device)
        transformed_boxes = predictor.transform.apply_boxes_torch(input_bboxes, img.shape[:2])
        mask_list, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        # crop out the foreground object
        copy_cnt = 1
        for obj_name, bbox, mask in zip(obj_name_list, bbox_list, mask_list):
            x_min, y_min, x_max, y_max = bbox[0].astype(int)
            h, w, c = img.shape

            mask = mask.float().cpu().numpy().reshape(h, w, 1)
            img_masked = img * mask
            obj_patch = img_masked[y_min:y_max, x_min:x_max]

            ###########################
            # save the cropping result
            ###########################
            # save the foreground
            save_folder = os.path.join(save_dir, 'foreground', obj_name)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            
            save_path_probe = os.path.join(save_folder, f'{img_name[:-4]}_{copy_cnt}.jpg') # a probing path for saving the cropping results
            if os.path.exists(save_path_probe):
                # the current file name is occupied (multiple instance of the same class in one image)
                copy_cnt += 1
            else:
                # first instance of an object class in the current image
                copy_cnt = 1
            
            save_path = os.path.join(save_folder, f'{img_name[:-4]}_{copy_cnt}.jpg')
            cv2.imwrite(save_path, obj_patch)

            # save the entire foreground
            save_folder = os.path.join(save_dir, 'img_patch', obj_name)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_path = os.path.join(save_folder, f'{img_name[:-4]}_{copy_cnt}.jpg')
            cv2.imwrite(save_path, img[y_min:y_max, x_min:x_max])

            # save the binary mask
            save_folder = os.path.join(save_dir, 'mask', obj_name)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_path = os.path.join(save_folder, f'{img_name[:-4]}_{copy_cnt}.jpg')
            cv2.imwrite(save_path, mask[y_min:y_max, x_min:x_max] * 255)
