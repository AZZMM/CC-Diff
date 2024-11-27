import os
import json
import warnings
import xml.etree.ElementTree as ET

import cv2
import numpy as np


anno_path = 'path_to_data/DOTA/train/labelTxt'

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

def create_layout2img_jsonl(mode: str, thr: int = 6, dota_caption_path: str = None):
    
    with open(dota_caption_path, 'r') as f:
        dota_caption = json.load(f)
    
    total_image_num = 0
    with open('train_metadata.jsonl', 'w') as writer:
        
        for txtfile in os.listdir(anno_path):
        
            filename = txtfile.split('.')[0]
            file_name = filename + '.png'
            content = _load_dota_txt(os.path.join(anno_path, txtfile))['ann']
            caption = [dota_caption[filename]]
            labels = content['labels'].copy()
            bndboxes   = []
            obboxes    = []
            for poly, label in zip(content['bboxes'], content['labels']):
                obbox = poly2obb_np_le90(poly)
                if obbox == None:
                    labels.remove(label)
                    continue
                obbox = (poly / 512).tolist()
                bndbox = (poly2hbb(poly) / 512).tolist()
                obboxes.append(obbox)
                bndboxes.append(bndbox)               
                
            assert len(labels) == len(bndboxes) == len(obboxes)
            if len(labels) == 0:
                print(file_name)
                continue
            for i in range(thr - len(labels)):
                labels.append("")
                bndboxes.append([0, 0, 0, 0])
                obboxes.append([0, 0, 0, 0, 0, 0, 0, 0])
            
            caption.extend(labels)
            # Both HBB and OBB normalized
            example = {"file_name": file_name, "caption": caption, "bndboxes": bndboxes, "obboxes": obboxes}
            total_image_num += 1
            print(example)
            # writer.write(json.dumps(example))
            print(json.dumps(example), file=writer)

    print(f"total number of images: {total_image_num}")
        
if __name__ == "__main__":
    create_layout2img_jsonl("train", dota_caption_path="./dota_train_caption.json")