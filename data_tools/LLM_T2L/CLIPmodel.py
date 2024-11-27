import os, json, random
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPModel, AutoTokenizer, AutoProcessor
from PIL import Image

from pdb import set_trace as ST


class myCLIPEnc(nn.Module):
    # def __init__(self, model_config='openai/clip-vit-large-patch14', device='cuda'):
    def __init__(self, model_config='openai/clip-vit-large-patch14', 
                 device='cuda'):
        super().__init__()
        self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_config)
        self.processor = AutoProcessor.from_pretrained(model_config)
        self.model = CLIPModel.from_pretrained(model_config).to(device)
        self.model.eval()

    def forward(self, caption=None, img=None):
        if caption is not None:
            txt_inp  = self.tokenizer(caption, padding=True, truncation=True, return_tensors="pt").to(self.device) # pad and truncate to the max_length
            txt_feat = self.model.get_text_features(**txt_inp)
            txt_feat = F.normalize(txt_feat, dim=-1).detach().cpu()
        else:
            txt_feat = None

        if img is not None:
            img_inp  = self.processor(images=img, return_tensors="pt").to(self.device)
            img_feat = self.model.get_image_features(**img_inp)
            img_feat = F.normalize(img_feat, dim=-1).detach().cpu()
        else:
            img_feat = None
        
        return txt_feat, img_feat
    

if __name__ == '__main__':
    cache_name = 'dior_emb.pt'
    if not os.path.exists(cache_name):
        myCLIP = myCLIPEnc()

        data = []
        with open(os.path.join('path_to_data/DIOR/train/metadata.jsonl'), 'r') as f:
            for line in f:
                data.append(json.loads(line))
        emb_dict = {}
        sample_data = random.sample(data, 5000)
        for sample in tqdm(sample_data):
            img_name = sample['file_name']
            emb_dict[img_name] = {}

            caption = sample['caption'][0]
            img = Image.open(os.path.join('path_to_data/DIOR/train/', img_name)).convert('RGB')
            txt_emb, img_emb = myCLIP(caption=caption, img=img)
            
            emb_dict[img_name]['txt_emb'] = txt_emb.detach().cpu()
            emb_dict[img_name]['img_emb'] = img_emb.detach().cpu()
            # torch.cuda.empty_cache()

        torch.save(emb_dict, cache_name)