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


# object classes with no clear foreground/backgroud seperation
no_seg_class = {
    'DIOR': ['golffield', ]
}

def anno_parser(anno_path, dataset_anme):
    if dataset_name == 'DIOR':
        root = ET.parse(anno_path).getroot()

        name_list = []
        bbox_list = []
        for node in root.findall('object'):
            name = node.find('name').text
            name_list.append(name)

            bbox_node = node.find('bndbox')
            bbox = [int(child.text) for child in bbox_node]
            bbox_list.append(bbox)
        
        return name_list, bbox_list
    else:
        return None, None


if __name__ == '__main__':
    # config dataset info
    dataset_name = 'DIOR'
    img_dir  = os.path.join('dataset', dataset_name, 'img')
    anno_dir = os.path.join('dataset', dataset_name, 'anno')
    save_dir = os.path.join('results', dataset_name)
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
        obj_name_list, bbox_list = anno_parser(anno_path, dataset_name)

        # batched segmentation with SAM
        predictor.set_image(img)
        input_bboxes = torch.tensor(bbox_list, device=predictor.device)
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
            x_min, y_min, x_max, y_max = bbox
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
