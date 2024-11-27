import json, os, pickle
import xml.etree.ElementTree as ET # for parsing the annotations in DIOR (.XML format)

from mmengine.fileio import dump
from rich import print_json

from mmpretrain.apis import ImageClassificationInferencer

from tqdm import tqdm

from pdb import set_trace as ST

category = ("vehicle", "baseballfield", "groundtrackfield", "windmill", "bridge", \
            "overpass", "ship", "airplane", "tenniscourt", "airport", \
            "Expressway-Service-area", "basketballcourt", "stadium", "storagetank", "chimney", \
            "dam", "Expressway-toll-station", "golffield", "trainstation", "harbor")
img_dir  = './dataset/DIOR/img'
anno_dir = './dataset/DIOR/anno'

if __name__ == '__main__':
    img_list = os.listdir(img_dir)
    img_list.sort()
    anno_list = os.listdir(anno_dir)
    anno_list.sort()
    assert len(img_list) == len(anno_list), 'Number of images and annotations does not match!'

    # load RSMamba
    inferencer = ImageClassificationInferencer(model='./configs/rsmamba/rsmamba_nwpu_h.py', 
                                               pretrained='./checkpoints/pretrained/RSMamba-h_NWPU.pth', 
                                               device='cuda')

    img_caption_dict = {}
    for img_name, anno_name in zip(img_list, anno_list):
        # read in the name of all objects
        root = ET.parse(os.path.join(anno_dir, anno_name)).getroot()
        obj_cnt_dict = {}
        for node in root.findall('object'):
            name = node.find('name').text
            bndbox_node = node.find('bndbox')
            bndbox = [int(child.text) for child in bndbox_node]
            obj_cnt_dict[name] = len(bndbox)


        scene_caption = None
        # process the category name, and determine the name of scene if possible
        for obj_name, obj_cnt in obj_cnt_dict.items():
            if obj_name in ['golffield', 'harbor', 'airport']:
                scene_caption = f' of {obj_name}'
            elif obj_name in ['Expressway-Service-area', "Expressway-toll-station"]:
                scene_caption = ' of freeway'
        
        # using RSMamba for scene classification if needed
        if scene_caption is None:
            result, _ = inferencer(inputs=os.path.join(img_dir, img_name), show_dir=None)
            pred = result[0]['pred_class']
            pred = pred.replace('_', ' ')
            scene_caption = f' of {pred}'

        scene_caption = 'This is an aerial image' + scene_caption + '. '
        img_caption_dict[img_name[:-4]] = scene_caption

    # dump the caption
    save_path = 'path_to_data/DIOR/scene_caption.json'
    with open(save_path, 'w') as fp:
        json.dump(img_caption_dict, fp)

