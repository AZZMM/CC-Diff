import os, json
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import random
import numpy as np
from openai import OpenAI
from PIL import Image
from tqdm import tqdm
import torch

from data_tools.LLM_T2L.utils import visualize_examplars, visualize_layout, resolve_overlap_and_boundary, poly2hbb, poly2obb_np_le90, obb2poly_np_le90

from pdb import set_trace as ST


'''
Define global variables
'''
from CLIPmodel import myCLIPEnc
myCLIP = myCLIPEnc()

# Setup the OpenAI API
client = OpenAI(
    api_key="key",
    base_url="url"
)

# Load the cache file of caption and image embeddings
data_emb_dict = None
if os.path.exists('dior_emb.pt'):
    data_emb_dict = torch.load('dior_emb.pt')


'''
Helper function for creating GPT prompts
'''
def get_GPT_instruction(use_GPT_prompt=True):
    if use_GPT_prompt: # use GPT-modified prompt
        Instruction = """
[Instruction]:

You are an intelligent research assistant. I will provide the caption of an aerial image captured by a satellite. Your task is to:

1. Identify the object categories mentioned in the caption.
2. Count the number of instances for each category.
3. Generate an oriented bounding box (OBB) for each instance in the format: 
(object name, [center x, center y, width, height, rotation angle]).

Constraints:
- Image size is 512x512, with the top-left at [0, 0] and the bottom-right at [512, 512].
- The width must always be greater than the height.
- The rotation angle is the angle between the longer edge (width) and the positive x-axis, measured in degrees within [-90, 90].
- Bounding boxes must stay entirely within the image boundaries.
- Do not include objects not mentioned in the caption.

Validate that all bounding boxes meet the width > height and boundary conditions. If necessary, make reasonable assumptions for object layout based on common aerial imagery.

Please refer to the example below for the desired output format.

"""
    else: # prompt commonly used in previous studies
        # Universal prefix
        Instruction = '[Instruction]: \nYou are an intelligent research assistant. '

        # Task description
        Instruction += 'I will provide you with the caption of an aerial image captured by a satellite. '
        Instruction += 'Your task is to identify the object categories, count the number of instances for each category, and generate the oriented bounding box for each instance mentioned in the caption. '
        
        # Canvas description
        Instruction += 'The images are of size 512 by 512. The top-left corner has coordinate [0, 0], and the bottom-right corner has coordinate [512, 512]. '
        Instruction += 'The generated oriented bounding boxes should locate within the image boundaries and must not go beyond. '
        
        # Bounding box format description
        Instruction += 'Each oriented bounding box should be in the format of (object name, [center x coordinate, center y coordinate, box width, box height, rotation angle). '
        Instruction += 'Keep in mind that the value of *box width* is defined to be always greater than that of *box height* (*box width* > *box height*). '
        Instruction += 'Also note that the value of *rotation angle* measures the angle between the direction of the longer edge of the bounding box (i.e., *box width*) and the positive x-axis of the image coordinate, and is measured in degrees (within the interval [-90, 90]). '
        Instruction += 'Each oriented bounding box should include exactly one object, and do not include objects not mentioned in the caption. When necessary, you can make reasonable and educated guesses based on your knowledge of plausible layout of objects in aerial images. '
        
        # Finish-up instructions
        Instruction += 'Please refer to the example below for the desired output format.\n\n'

    return Instruction


def get_GPT_examplars(caption_list, obj_names_list, bboxes_list):
    # Universal prefix
    In_Context_Example = '[In-context Examples]:'

    for ex_ind, (caption, obj_names, bboxes) in enumerate(zip(caption_list, obj_names_list, bboxes_list)):
        In_Context_Example += f'\nExample #{ex_ind+1}:\n'

        # Input caption
        In_Context_Example += f'<Input Caption>\n'
        In_Context_Example += f'{caption}\n'

        # Output bboxes
        In_Context_Example += f'<Output Bounding Boxes>\n'
        for obj_name, bbox in zip(obj_names, bboxes):
            In_Context_Example += f'{obj_name}: {str(bbox)}\n'

    # print(In_Context_Example)
    return In_Context_Example


'''
Helper functions
'''
def skip_overlap_check(obj_name_list):
    skip_check = False
    if 'stadium' in obj_name_list and 'groundtrackfield' in obj_name_list:
        print('Skipping for overlap check for \'stadium\' and \'ground track field\'...')
        skip_check = True

    '''
    TODO: Include any special case where the overlap check should be omitted.
    '''
        
    return skip_check


def get_similar_examplars(query_img_name, prompt, topk=5, sim_mode='both'):
    prompt_emb, _ = myCLIP(prompt)

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

    # exclude the query image
    # query_ind = img_name_list.index(query_img_name)
    # sim_val_list.pop(query_ind)
    # img_name_list.pop(query_ind)

    return img_name_list[-topk:], sim_val_list[-topk:]


def parse_GPT_response(gpt_response):
    lines = gpt_response.split('\n')[1:] # discard the response prefix

    obj_name_list = []
    bbox_list = []
    for line in lines:
        obj_name, bbox = line.split(': ')
        bbox = bbox.strip()
        bbox = bbox[1:-1].split(', ')

        obj_name_list.append(obj_name)
        bbox_list.append(list(map(int, bbox)))

    return obj_name_list, bbox_list
            

if __name__ == '__main__':
    do_vis_gt = False

    # load the metadata of images
    full_data = []
    with open(os.path.join('train_metadata.jsonl'), 'r') as f:
        for line in f:
            full_data.append(json.loads(line))
    data = []
    with open(os.path.join('val_metadata.jsonl'), 'r') as f:
        for line in f:
            data.append(json.loads(line))
    w_filenames = []     
    # go over the test images
    for sample in tqdm(data):
        img_name = sample['file_name']
        # if img_name in complete_files:
        #     continue
        img_path  = os.path.join('./images', img_name)
        caption   = sample['caption'][0]
        obj_names, bboxes = [], []
        for obj_name, obbox in zip(sample['caption'][1:], sample['obboxes']):
            if obj_name == '':
                continue
            obj_names.append(obj_name)
            poly = (np.array(obbox) * 512).astype(int)
            bboxes.append(poly2obb_np_le90(poly))

        '''
        Visualize the layout of gt images if necessary
        '''
        if do_vis_gt:
            gt_save_dir = os.path.join('./results_dior', 'gt_layout_vis')
            if not os.path.exists(gt_save_dir):
                os.makedirs(gt_save_dir)
            gt_save_path = os.path.join(gt_save_dir, img_name)
            visualize_layout(img_path, obj_names, bboxes, gt_save_path)

        '''
        Construct the instruction prompt
        '''
        GPT_instruction = get_GPT_instruction()

        '''
        Construct the prompt for in-context examplar learning
        '''        
        # Find the examplars
        sim_mode = 'both'
        ex_name_list, _ = get_similar_examplars(img_name, caption, sim_mode=sim_mode)

        # visualize the retrieved similar examplars
        vis_save_dir = os.path.join('./results_dior', 'examplar_vis', sim_mode)
        if not os.path.exists(vis_save_dir):
            os.makedirs(vis_save_dir)
        vis_save_path = os.path.join(vis_save_dir, img_name)
        visualize_examplars([img_name, *ex_name_list], None, vis_save_path)

        # obtain the GPT prompt of in-context examplars
        caption_list  = []
        obj_names_list = []
        bboxes_list   = []
        for ex_name in ex_name_list:
            ex_sample = [d for d in full_data if d['file_name'] == ex_name][0]
            caption_list.append(ex_sample['caption'][0])
            obj_names, bboxes = [], []
            for obj_name, obbox in zip(ex_sample['caption'][1:], ex_sample['obboxes']):
                if obj_name == '':
                    continue
                poly = (np.array(obbox) * 512).astype(int)
                poly = np.array(poly2obb_np_le90(poly))
                if len(poly.shape) == 0:
                    continue
                bboxes.append(poly.astype(int).tolist())
                obj_names.append(obj_name)
            obj_names_list.append(obj_names)
            bboxes_list.append(bboxes)
        GPT_examplars = get_GPT_examplars(caption_list, obj_names_list, bboxes_list)
            
        '''
        Start conversation with GPT
        '''
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": GPT_instruction + GPT_examplars},
                {"role": "user", "content": f"<Input Caption>\n{caption}"}
                ],
            stream=False,
            timeout=10
        )
        gpt_response = completion.choices[0].message.content


        '''
        Parse and visualize the results obtained from GPT
        '''
        # Parse the response obtained from GPT
        try:
            obj_names_GPT, bboxes_GPT = parse_GPT_response(gpt_response)
        except:
            # print(f'{img_name} not parsed successfully!')
            w_filenames.append(img_name)
            continue
        overlap_detected = False
        if not skip_overlap_check(obj_names_GPT):
            overlap_detected, bboxes_GPT_proc = resolve_overlap_and_boundary(bboxes_GPT)

        '''
        Save the post-processed results into disk
        '''
        # visualize the bboxes returned by GPT
        res_save_dir = os.path.join('./results_dior', 'GPT_layout_vis', sim_mode)
        if not os.path.exists(res_save_dir):
            os.makedirs(res_save_dir)
        res_save_path = os.path.join(res_save_dir, img_name)
        visualize_layout(img_path, obj_names_GPT, bboxes_GPT, res_save_path)

        if overlap_detected:
            res_save_dir = os.path.join('./results_dior', 'GPT_proc_layout_vis', sim_mode)
            if not os.path.exists(res_save_dir):
                os.makedirs(res_save_dir)
            res_save_path = os.path.join(res_save_dir, img_name)
            visualize_layout(img_path, obj_names_GPT, bboxes_GPT_proc, res_save_path)

            # ST()


        # print the layout obtained by GPT
        # for obj_name_GPT, bbox_GPT in zip(obj_names_GPT, bboxes_GPT):
        #     print(f'{obj_name_GPT}: {bbox_GPT}')

        '''
        gt_img = draw_box_desc(gt_img, bboxes[0], prompt[0][1:])
        image = draw_box_desc(image, s['layout'], objs_list)
        result = Image.new('RGB', (gt_img.width + image.width, image.height))
        result.paste(gt_img, (0, 0))
        result.paste(image, (image.width, 0))
        result.save(f'saved/output_{file_name}') 
        image.save(f'saved/anno_output_{file_name}')
        image.save(f'saved/anno_output_{file_name}')
        '''
        obboxes = bboxes_GPT_proc if overlap_detected else bboxes_GPT
        example = {"file_name": img_name, "caption": [caption] + obj_names_GPT, "obboxes": obboxes}
        # print(example)
            # writer.write(json.dumps(example))
            
    
        thr = 6       
        filename = example['file_name']
        caption = example['caption'][0]
        labels = example['caption'][1:]
        obboxes = example['obboxes']
        bndboxes = []
        polys = []
        for obbox in obboxes:
            obbox[-1] = obbox[-1] / 180 * np.pi
            poly = obb2poly_np_le90(obbox)
            bndbox = poly2hbb(poly)
            polys.append((poly / 512).round(4).tolist())
            bndboxes.append((bndbox / 512).round(4).tolist())
        assert len(labels) == len(bndboxes) == len(polys)
        if len(labels) < thr:
            for i in range(thr-len(labels)):
                labels.append('')
                bndboxes.append([0., 0., 0., 0.])
                polys.append([0., 0., 0., 0., 0., 0., 0., 0.])   
        example = {'file_name': filename, 'caption': [caption] + labels, 'bndboxes': bndboxes, 'obboxes': polys} 
        with open('gpt_metadata.jsonl', 'a') as writer: 
            print(json.dumps(example), file=writer)
    print(w_filenames)
            