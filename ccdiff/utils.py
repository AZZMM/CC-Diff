import functools
import os
import random
import numpy as np
import imagesize
import torch
import torch.nn.functional as F
from PIL import Image

ref_img_path = 'path_to_data/DIOR/foreground'
dict_of_images = {}
list_of_name = ("vehicle", "baseballfield", "groundtrackfield", "windmill", "bridge", \
                "overpass", "ship", "airplane", "tenniscourt", "airport", \
                "Expressway-Service-area", "basketballcourt", "stadium", "storagetank", "chimney", \
                "dam", "Expressway-toll-station", "golffield", "trainstation", "harbor")
for name in list_of_name:
    name_of_dir = os.path.join(ref_img_path, name)
    list_of_image = os.listdir(name_of_dir)
    list_of_image = sorted(list_of_image, 
                           key = lambda img: functools.reduce(lambda x, y: x*y, imagesize.get(os.path.join(name_of_dir, img))), 
                           reverse=True)
    dict_of_images[name] = {img: functools.reduce(lambda x, y: x/y, imagesize.get(os.path.join(name_of_dir, img))) 
                            for img in list_of_image[:200]}

def seed_everything(seed):
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array/value - 1)).argmin()
        return idx

def get_sup_mask(mask_list):
    or_mask = np.zeros_like(mask_list[0])
    for mask in mask_list:
        or_mask += mask
    or_mask[or_mask >= 1] = 1
    sup_mask = 1 - or_mask
    return sup_mask
 
data_emb_dict = None
if os.path.exists('data_emb.pt'):
    data_emb_dict = torch.load('data_emb.pt')
    
def get_similar_examplers(query_img_name, prompt_emb, topk=5, sim_mode='both'):
    prompt_emb = F.normalize(prompt_emb, dim=-1).detach().cpu()

    # go through the embeddings and get the most similar topk examples
    img_name_list = []
    sim_val_list = []
    for img_name, data_emb in data_emb_dict.items():
        img_name_list.append(img_name)
        txt_emb = data_emb['txt_emb']
        img_emb = data_emb['img_emb']

        if sim_mode == 'text2text':
            sim_val = (prompt_emb * txt_emb).sum(dim=-1)
        elif sim_mode == 'text2img':
            sim_val = (prompt_emb * img_emb).sum(dim=-1)
        elif sim_mode == 'both':
            txt_sim_val = (prompt_emb * txt_emb).sum(dim=-1)
            img_sim_val = (prompt_emb * img_emb).sum(dim=-1)
            sim_val = (txt_sim_val + img_sim_val) * 0.5
        else:
            raise ValueError('Invalid mode for similarity computation! (text2text | text2img | both)')
    
        sim_val_list.append(sim_val.item())

    # sort the similarity values and obtain the topk one
    sim_val_list, img_name_list = zip(*sorted(zip(sim_val_list, img_name_list)))
    sim_val_list = list(sim_val_list)
    img_name_list = list(img_name_list)

    img_emb_list = [data_emb_dict[img_name]['img_emb'] for img_name in img_name_list[-topk:]]

    return img_name_list[-topk:]