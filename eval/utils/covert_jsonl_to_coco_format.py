import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm

CLASSES = ('vehicle','chimney','golffield','Expressway-toll-station','stadium',
               'groundtrackfield','windmill','trainstation','harbor','overpass',
               'baseballfield','tenniscourt','bridge','basketballcourt','airplane',
               'ship','storagetank','Expressway-Service-area','airport','dam')

def get_image_info(file_path):
    with Image.open(file_path) as img:
        width, height = img.size
    return width, height

def convert_jsonl_to_coco(meta_path, img_dir, output_json):
    categories = []
    # category_set = {}
    image_set = {}
    annotations = []
    annotation_id = 1

    # Initialize COCO structure
    coco_output = {
        "info": {
            "description": "DIOR to COCO Dataset",
            "version": "1.0",
            "year": 2024,
            "contributor": "",
            "date_created": "2024-11-09"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Parse XML files
    # for xml_file in os.listdir(xml_dir):
    data = []
    with open(meta_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
        
    for i, sample in enumerate(tqdm(data)):
        
        filename = sample['file_name']
        image_id = i+1
        width, height = get_image_info(os.path.join(img_dir, filename))
        coco_output["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        })
        for label, bbox in zip(sample['caption'][1:], sample['bndboxes']):
            if label == '':
                continue
            x1, y1, x2, y2 = bbox
            xmin = int(x1 * width)
            ymin = int(y1 * height)
            xmax = int(x2 * width)
            ymax = int(y2 * height)
            o_width = xmax - xmin
            o_height = ymax - ymin
            
            try:
                category_id = CLASSES.index(label) + 1
            except:
                print(label)
                continue
            
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [xmin, ymin, o_width, o_height],
                "area": o_width * o_height,
                "segmentation": [],
                "iscrowd": 0
            })
            annotation_id += 1

    for i, category in enumerate(CLASSES):
        categories.append({
                            "id": i+1,
                            "name": category,
                            "supercategory": "none"
                        })
    
    coco_output["categories"] = categories
    coco_output["annotations"] = annotations

    # Write to output JSON file
    with open(output_json, 'w') as json_file:
        json.dump(coco_output, json_file, indent=4)

if __name__ == '__main__':
    meta_path = 'path_to_data/metadata.jsonl'
    img_dir = 'path_to_data/train/'
    output_json = 'path_to_result/train_annotations.json'
    convert_jsonl_to_coco(meta_path, img_dir, output_json)

