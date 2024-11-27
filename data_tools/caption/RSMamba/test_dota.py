import json, os, pickle
import xml.etree.ElementTree as ET # for parsing the annotations in DIOR (.XML format)

from mmengine.fileio import dump
from rich import print_json

from mmpretrain.apis import ImageClassificationInferencer

from tqdm import tqdm

from pdb import set_trace as ST

category = ("plane","ship","storage-tank","baseball-diamond","tennis-court",
            "basketball-court","ground-track-field","harbor","bridge","large-vehicle",
            "small-vehicle","helicopter","roundabout","soccer-ball-field","swimming-pool")
img_dir  = 'path_to_data/DOTA/val/images'
anno_dir = 'path_to_data/DOTA/val/labelTxt'

if __name__ == '__main__':
    img_list = os.listdir(img_dir)
    img_list.sort()
    anno_list = os.listdir(anno_dir)
    anno_list.sort()
    assert len(img_list) == len(anno_list), 'Number of images and annotations does not match!'

    # load RSMamba
    inferencer = ImageClassificationInferencer(model='./configs/rsmamba/rsmamba_nwpu_h.py', 
                                               pretrained='./checkpoints/RSMamba-h_NWPU.pth', 
                                               device='cuda')

    img_caption_dict = {}
    for img_name, anno_name in zip(img_list, anno_list):
        scene_caption = None
        
        # using RSMamba for scene classification if needed
        if scene_caption is None:
            result, _ = inferencer(inputs=os.path.join(img_dir, img_name), show_dir=None)
            pred = result[0]['pred_class']
            pred = pred.replace('_', ' ')
            scene_caption = f' of {pred}'

        scene_caption = 'This is an aerial image' + scene_caption + '. '
        img_caption_dict[img_name[:-4]] = scene_caption

    # dump the caption
    save_path = 'path_to_data/DOTA/val_scene_caption.json'
    with open(save_path, 'w') as fp:
        json.dump(img_caption_dict, fp)

