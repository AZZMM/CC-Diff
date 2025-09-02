import os
import shutil
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

def update_coco_with_new_jsonl_and_images(existing_coco_json, meta_path, img_dir, new_img_dir, output_json, combined_img_dir):
    # Load existing COCO JSON
    with open(existing_coco_json, 'r') as f:
        coco_data = json.load(f)
    
    # Prepare to add new images and annotations
    image_set = {image['file_name']: image['id'] for image in coco_data['images']}
    max_image_id = max(image_set.values()) if image_set else 0
    max_annotation_id = max(ann['id'] for ann in coco_data['annotations']) if coco_data['annotations'] else 0

    annotations = coco_data['annotations']

    # Ensure the combined image directory exists
    os.makedirs(combined_img_dir, exist_ok=True)

    # Copy existing images to the combined image directory
    for image_info in coco_data['images']:
        src_img_path = os.path.join(img_dir, image_info['file_name'])
        dst_img_path = os.path.join(combined_img_dir, image_info['file_name'])
        if not os.path.exists(dst_img_path):
            shutil.copy(src_img_path, dst_img_path)
    
    data = []
    with open(meta_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
        
    for i, sample in enumerate(tqdm(data)):
        filename = sample['file_name']
        img_path = os.path.join(new_img_dir, filename)
        new_filename = sample['file_name'].split('.')[0] + '_1.' + sample['file_name'].split('.')[1]
        combined_img_path = os.path.join(combined_img_dir, new_filename)
        image_id = i+1+max_image_id
        width, height = get_image_info(os.path.join(new_img_dir, filename))
        coco_data["images"].append({
            "id": image_id,
            "file_name": new_filename,
            "width": width,
            "height": height
        })
        max_image_id += 1
        shutil.copy(img_path, combined_img_path)
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
            
            category_id = CLASSES.index(label) + 1
            max_annotation_id += 1
            annotations.append({
                "id": max_annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [xmin, ymin, o_width, o_height],
                "area": o_width * o_height,
                "segmentation": [],
                "iscrowd": 0
            })

    coco_data["annotations"] = annotations

    # Write updated COCO JSON to output file
    with open(output_json, 'w') as json_file:
        json.dump(coco_data, json_file, indent=4)

if __name__ == '__main__':
    existing_coco_json = 'path_to_data/dior_train_annotations.json'     # dataset json converted by eval/utils/covert_jsonl_to_coco_format.py
    meta_path = 'path_to_data/metadata.jsonl'                           # dataset metadata
    img_dir = 'path_to_data/train/'                                     # dataset images path
    new_img_dir = 'path_to_data/syn_train/'                             # synthesized images path
    output_json = 'path_to_data/combined_train_annotations.json'        # combined json save path
    combined_img_dir = 'path_to_data/combined_train'                    # combined images path (original & synthesized images)
    update_coco_with_new_jsonl_and_images(existing_coco_json, meta_path, img_dir, new_img_dir, output_json, combined_img_dir)
