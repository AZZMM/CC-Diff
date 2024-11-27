import os
import json
import warnings
import xml.etree.ElementTree as ET

import cv2
import numpy as np

anno_path = 'path_to_data/DIOR/Annotations'        # DIOR-RSVG dataset annotations
poly_anno_path = 'path_to_data/DIOR/Annotations/Oriented_Bounding_Boxes'  # DIOR-R dataset annotations

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

def create_layout2img_jsonl(mode: str, thr: int = 6, dior_caption_path: str = None):
    
    with open('path_to_data/DIOR/' + mode + '.txt', 'r') as f:   # DIOR-RSVG annotations
        trt_images = list(int(id) for id in f)
    with open(dior_caption_path, 'r') as f:
        dior_caption = json.load(f)
    
    total_image_num = 0
    with open('metadata.jsonl', 'w') as writer:
        for anno in sorted(os.listdir(anno_path)):
            
            root = ET.parse(os.path.join(anno_path, anno)).getroot()
            poly_root = ET.parse(os.path.join(poly_anno_path, anno)).getroot()
            poly_list = list(poly_root.findall('object'))
            
            image_id = int(anno.split('.')[0])  
            if image_id not in trt_images:
                continue
            file_name = anno.split('.')[0] + '.jpg'
            caption = [dior_caption[str(image_id)]]
            
            width = int(root.find('size').find('width').text)
            height = int(root.find('size').find('height').text)
            
            # drop off image with more than `thr` objects
            # if len(root.findall('object')) > thr:
            #     continue
            
            categories = []
            bndboxes   = []
            obboxes    = []
            for node in root.findall('object'):
                name = node.find('name').text
                categories.append(name)
                
                bndbox_node = node.find('bndbox')
                x1, y1, x2, y2 = (int(child.text) for child in bndbox_node)
                bndbox = [x1/width, y1/height, x2/width, y2/height]
                # bndbox = [x1, y1, x2, y2]
                bndboxes.append(bndbox)
                
                for poly_node in poly_list:
                    aux_name = poly_node.find('name').text
                    if name == aux_name:
                        # angle = poly_node.find('angle').text
                        robndbox = [int(child.text) for child in poly_node.find('robndbox')]
                        x, y, w, h, a = poly2obb_np_le90(robndbox)
                        # obbox = [x/width, y/height, w/width, h/height, a]
                        obbox = [coord/width if i % 2 == 0 else coord/height for i, coord in enumerate(robndbox)]
                        obbox = [round(f, 5) for f in obbox]
                        # obbox = [x, y, w, h, a]
                        poly_list.remove(poly_node)
                        break
                if name != aux_name:
                    warnings.warn(f"new instance: {name} in file_{file_name}")
                    angle = 0
                    xc = (bndbox[0] + bndbox[2]) / 2
                    yc = (bndbox[1] + bndbox[3]) / 2
                    w = bndbox[2] - bndbox[0] # width of bbox
                    h = bndbox[3] - bndbox[1] # height of bbox
                    if w >= h:
                        angle = 0
                    else:
                        w, h = h, w
                        angle = round(-np.pi / 2, 5)
                    # obbox = [xc, yc, w, h, angle]
                    obbox = [bndbox[0], bndbox[1], bndbox[2], bndbox[1], bndbox[2], bndbox[3], bndbox[0], bndbox[3]]
                    # obbox = [coord/width if i % 2 == 0 else coord/height for i, coord in enumerate(obbox)]
                    obbox = [round(f, 5) for f in obbox]
                # obbox = [int(f) for f in obbox]
                # x, y, w, h, a = obbox
                # obbox = [x / width * 512, y / height * 512, w / width * 512, h / height * 512, a / np.pi * 180]
                # obbox = [int(f) for f in obbox]
                obboxes.append(obbox)
                
            assert len(categories) == len(bndboxes) == len(obboxes)
            if len(bndboxes) > thr:
                categories = categories[:thr]
                bndboxes = bndboxes[:thr]
                obboxes = obboxes[:thr]
            for i in range(thr - len(categories)):
                categories.append("")
                bndboxes.append([0, 0, 0, 0])
                obboxes.append([0, 0, 0, 0, 0, 0, 0, 0])
            
            caption.extend(categories)
            # Both HBB and OBB normalized
            example = {"file_name": file_name, "caption": caption, "bndboxes": bndboxes, "obboxes": obboxes}
            total_image_num += 1
            print(example)
            # writer.write(json.dumps(example))
            print(json.dumps(example), file=writer)

    print(f"total number of images: {total_image_num}")
        
if __name__ == "__main__":
    create_layout2img_jsonl("val", dior_caption_path="./dior_caption.json")
    # create_layout2img_jsonl("train", dior_caption_path="./dior_caption.json")