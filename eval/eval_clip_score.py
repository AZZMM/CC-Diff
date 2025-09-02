import os 
import PIL
import json
import torch 
import argparse
import open_clip
import numpy as np
from tqdm import tqdm
from PIL import Image
# from pycocotools.coco import COCO
import torch.nn.functional as F
from transformers import CLIPModel,CLIPTokenizer,CLIPFeatureExtractor

def get_args_parser():
    parser = argparse.ArgumentParser('Eval script', add_help=False)
    parser.add_argument("--job_index", type=int, default=0, help="")
    parser.add_argument("--num_jobs", type=int, default=1, help="")
    # args = parser.parse_args()
    return parser

def clip_score(text:str, image:PIL.Image, args:argparse.Namespace):

    with torch.no_grad():
        if use_open_clip:
            text_token = tokenizer(text).cuda()
            txt_features = model.encode_text(text_token)

            image = preprocess(image).unsqueeze(0).cuda()
            img_features = model.encode_image(image)

        else:
            inputs = tokenizer(text,
                            max_length=tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt")
            inputs["input_ids"] = inputs["input_ids"].cuda()
            txt_features = model.get_text_features(inputs["input_ids"])

            inputs = feature_extractor(image)
            inputs['pixel_values'] = torch.tensor(inputs['pixel_values'][0][None]).cuda()
            img_features = model.get_image_features(inputs['pixel_values'])

        img_features = F.normalize(img_features, dim=-1)
        txt_features = F.normalize(txt_features, dim=-1)
        clip_score = (img_features * txt_features).sum(dim=-1).item()
    
    return clip_score


def convert_coco_box(bbox, img_info):
    x0 = bbox[0]/img_info['width']
    y0 = bbox[1]/img_info['height']
    x1 = (bbox[0]+bbox[2])/img_info['width']
    y1 = (bbox[1]+bbox[3])/img_info['height']
    return [x0, y0, x1, y1]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="path_to_data/", help="")
    parser.add_argument("--ann", type=str, default="path_to_data/metadata.jsonl", help="")

    args = parser.parse_args()

    ann_file = args.ann
    max_objs = 6
    meta_dict_list = []
    with open(ann_file, 'r') as j:
        for line in j:
            meta_dict_list.append(json.loads(line))
    
    use_open_clip = True

    if not use_open_clip:
        version = "openai/clip-vit-large-patch14"
        tokenizer = CLIPTokenizer.from_pretrained(version)
        model = CLIPModel.from_pretrained(version).cuda()
        feature_extractor = CLIPFeatureExtractor.from_pretrained(version)
    else:
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b-s32b-b82k')
        tokenizer = open_clip.get_tokenizer('ViT-L-14')

        model = model.cuda()

    # compute clip scores
    local_clip_score_list = []
    global_clip_score_list = []

    folder = args.folder

    pbar = tqdm(meta_dict_list)
    for meta_dict in pbar:
        file_name = meta_dict['file_name']
        locations = meta_dict['bndboxes']
        global_caption = meta_dict['caption'][0]
        phrases = meta_dict['caption'][1:]
        num_object = len([phrase for phrase in phrases if phrase != ""])
        locations = locations[:num_object]
        phrases = phrases[:num_object]
        
        image_path = os.path.join(folder, file_name)

        # read images
        image = Image.open(image_path).convert("RGB")
        
        try:
            global_clip_score = clip_score(global_caption, image, args)
        except:
            print("error: ", file_name)
            continue
        global_clip_score_list.append(global_clip_score)
        
        # crop images using bounding boxes
        images_cropped = []
        for location in locations:
            x0, y0, x1, y1 = location
            image_cropped = image.crop((x0*image.width, y0*image.height, x1*image.width, y1*image.height))
            images_cropped.append(image_cropped)

        # measure clip scores for each cropped image (instance)
        clip_score_single_image = []

        for text, image in zip(phrases, images_cropped):
            try:
                score_single_instance = clip_score(text, image, args)
                clip_score_single_image.append(score_single_instance)
            except:
                print("error: ", file_name)
                continue
        
        
        if len(clip_score_single_image) != 0:
            local_clip_score_list.append(np.mean(clip_score_single_image))
        
        pbar.set_postfix({'local_similarity': np.mean(local_clip_score_list), 'global_similarity': np.mean(global_clip_score_list)})

    print(f"local_similarity: {np.mean(local_clip_score_list)}, 'global_similarity': {np.mean(global_clip_score_list)}")