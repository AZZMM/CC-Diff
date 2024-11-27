import json, os
import cv2
import numpy as np

from pdb import set_trace as ST

anno_path = 'path_to_data/DOTA/train/labelTxt'
N = 2 # the selected category number in the edge area
size = (512, 512)

# image grid configuration
x_grid = range(0, size[0]+1, size[0]//4)
y_grid = range(0, size[1]+1, size[1]//4)


category = ("plane","ship","storage-tank","baseball-diamond","tennis-court",
            "basketball-court","ground-track-field","harbor","bridge","large-vehicle",
            "small-vehicle","helicopter","roundabout","soccer-ball-field","swimming-pool")
category_name_map = {"plane": ["plane", "planes"], 
                "ship" : ["ship", "ships"], 
                "storage-tank": ["storage tank", "storage tanks"],
                "baseball-diamond": ["baseball diamond", "baseball diamonds"],
                "tennis-court": ["tennis court", "tennis courts"],
                "basketball-court": ["basketball court", "basketball courts"],
                "ground-track-field": ["ground track field", "ground track fields"], 
                "harbor": ["harbor", "harbors"],
                "bridge": ["bridge", "bridges"], 
                "large-vehicle": ["large vehicle", "large vehicles"],
                "small-vehicle": ["small vehicle", "small vehicles"],
                "helicopter": ["helicopter", "helicopters"],
                "roundabout": ["roundabout", "roundabouts"],
                "soccer-ball-field": ["soccer ball field", "soccer ball fields"],
                "swimming-pool": ["swimming pool", "swimming pools"]
                }
num_map = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
           6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
position = ("center", "upper-left corner", "top edge", "upper-right corner", \
            "left edge", "right edge", "lower-left corner", "lower edge", "lower-right corner")
edge_position = position[1:]
direction = ("northeast-southwest", "north-south", "northwest-southeast", "east-west")

# function for determining the position class of the object
def pos_judge(obj: dict) -> str:
    obj_center = (int(obj["obbox"][0]), 
                  int(obj["obbox"][1]))
    
    # obj_center is in center four patches (total 16 patches)
    if ( obj_center[0] in range(x_grid[1], x_grid[3]) ) and ( obj_center[1] in range(y_grid[1], y_grid[3]) ):
        return "center"
    
    elif ( obj_center[0] in range(x_grid[0], x_grid[1]) ) and ( obj_center[1] in range(y_grid[0], y_grid[1]) ):
        return "upper-left corner"
    
    elif ( obj_center[0] in range(x_grid[1], x_grid[3]) ) and ( obj_center[1] in range(y_grid[0], y_grid[1]) ):
        return "top edge"
    
    elif ( obj_center[0] in range(x_grid[3], x_grid[-1]) ) and ( obj_center[1] in range(y_grid[0], y_grid[1]) ):
        return "upper-right corner"
    
    elif ( obj_center[0] in range(x_grid[0], x_grid[1]) ) and ( obj_center[1] in range(y_grid[1], y_grid[3]) ):
        return "left edge"
    
    elif ( obj_center[0] in range(x_grid[3], x_grid[-1]) ) and ( obj_center[1] in range(y_grid[1], y_grid[3]) ):
        return "right edge"
    
    elif ( obj_center[0] in range(x_grid[0], x_grid[1]) ) and ( obj_center[1] in range(y_grid[3], y_grid[-1]) ):
        return "lower-left corner"
    
    elif ( obj_center[0] in range(x_grid[1], x_grid[3]) ) and ( obj_center[1] in range(y_grid[3], y_grid[-1]) ):
        return "lower edge"
    
    elif ( obj_center[0] in range(x_grid[3], x_grid[-1]) ) and ( obj_center[1] in range(y_grid[3], y_grid[-1]) ):
        return "lower-right corner"

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

def dirc_judge(obj: dict) -> str:
    # w = obj["bndbox"][2] - obj["bndbox"][0] 
    # h = obj["bndbox"][3] - obj["bndbox"][1]
    theta = float(obj['obbox'][-1])
    # if w < h:   # theta is determined by w, convert to the theta of long edge
    #     theta = theta - 90 if theta >= 0 else theta + 90
    angle_scope = np.arange(8) / 8 * 180 - 90
    angle_scope = angle_scope / 180 * np.pi
    
    if angle_scope[1] <= theta < angle_scope[3]:
        return "northeast-southwest"
    
    elif angle_scope[3] <= theta < angle_scope[5]:
        return "east-west"
    
    elif angle_scope[5] <= theta < angle_scope[7]:
        return "northwest-southeast"
    
    else:
        return "north-south"

def gen_dirc_prompt(
    data: np.array,
    category_index: int,
    position_index: int,
):
    dirc_prompt = ""
    dirc_array = data[category_index, position_index]
    n_valid = np.count_nonzero(dirc_array)
    assert n_valid > 0
    sorted_dirc_indices = np.argsort(dirc_array)[::-1]
    
    if n_valid > 1: # more than one direction of this class in this position
        for i_d in sorted_dirc_indices[:n_valid]:
            num  = num_map[dirc_array[i_d]]
            dirc = direction[i_d]
            dirc_prompt += f", {num} towards the {dirc} direction"
    else:
        i_d  = sorted_dirc_indices[0]
        # num  = num_map[dirc_array[i_d]]
        dirc = direction[i_d]
        dirc_prompt += f" towards the {dirc} direction"
    return dirc_prompt

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

if __name__ == '__main__':
    TEXT = {}

    with open('path_to_data/DOTA/scene_caption.json', 'r') as f:
        # scene_caption.json obtained from RSMamba
        scene_captions = json.load(f)    

    n_c, n_p, n_d = len(category), len(position), len(direction)
    
    for txtfile in os.listdir(anno_path):
        
        filename = txtfile.split('.')[0]
        content = _load_dota_txt(os.path.join(anno_path, txtfile))['ann']
        objs = []
        for poly, label in zip(content['bboxes'], content['labels']):
            obbox = poly2obb_np_le90(poly)
            if obbox == None:
                continue
            name = label
            obj = {"name": name, "obbox": obbox}
            obj["pos"] = pos_judge(obj)
            obj["dirc"] = dirc_judge(obj)
            
            objs.append(obj) # list of object dicts: <name, obbox, pos, direction>      
        
        # assert len(objs) > 0, "at least one instance in the image"
        if len(objs) == 0:
            print(filename)
            continue
        
        # abstract objs into 3-dim array
        data = np.zeros((n_c, n_p, n_d), dtype=int)
        for obj in objs:
            
            i_c = category.index(obj["name"])
            i_p = position.index(obj["pos"])
            i_d = direction.index(obj["dirc"])
            
            data[i_c][i_p][i_d] += 1  
        
        # describe the image at the aspect of position, first from `center`
        prompt = ""
        name_pos_array = np.sum(data, axis=2)
        pos_array = np.sum(name_pos_array, axis=0)
        
        # print the prompt for center objects
        name_center_array = name_pos_array[:, 0]
        if pos_array[0] == 0: # if there is nothing in the center of the image
            prompt += "There is no salient visual object in the center of the image. "
        else:
            prompt += "There" 
            n_class = np.count_nonzero(name_center_array)
            sorted_center_indices = np.argsort(name_center_array)[::-1]
            if n_class > 1: # more than one class in the center
                for i, i_c in enumerate(sorted_center_indices[:n_class]):                   
                    n_obj = name_center_array[i_c]
                    if i == 0:    
                        if n_obj > 1: # more than one instance of this class
                            verb = "are"
                            num  = num_map[n_obj]
                            noun = category_name_map[category[i_c]][1] # plural        
                        else:
                            verb = "is"
                            num  = num_map[n_obj]
                            noun = category_name_map[category[i_c]][0] # single
                        prompt += f" {verb} {num} {noun}"     
                    else:
                        if n_obj > 1:
                            num  = num_map[n_obj]
                            noun = category_name_map[category[i_c]][1] # plural
                        else:
                            num  = num_map[n_obj]
                            noun = category_name_map[category[i_c]][0] # single
                        prompt += f", {num} {noun}"
                    aux = gen_dirc_prompt(data, category_index=i_c, position_index=0)
                    prompt += aux
                prompt += " in the center of the image. "               
            else: # only one class in the center
                i_c = sorted_center_indices[0]
                n_obj = name_center_array[i_c]
                if n_obj > 1:
                    verb = "are"
                    num  = num_map[n_obj]
                    noun = category_name_map[category[i_c]][1] # plural
                else:
                    verb = "is"
                    num  = num_map[n_obj]
                    noun = category_name_map[category[i_c]][0] # single
                prompt += f" {verb} {num} {noun}"
                aux = gen_dirc_prompt(data, category_index=i_c, position_index=0)
                prompt += (aux + " in the center of the image. ")
        
        # print the prompt for edge objects, describe the image at the aspect of category
        name_edge_array = name_pos_array[:, 1:]
        sorted_edge_indices = np.argsort(np.sum(name_edge_array, axis=1))[::-1]  
        n_class = np.count_nonzero(np.sum(name_edge_array, axis=1))
        for i_c in sorted_edge_indices[:n_class]:
            n_pos = np.count_nonzero(name_edge_array[i_c]) # number of locations for the current object
            sorted_edge_pos_incides = np.argsort(name_edge_array[i_c])[::-1]
            if n_pos > 1: # appears at multiple different locations
                # count the total amount
                cnt = np.sum(name_edge_array, axis=1)[i_c]

                # specify the total amount and the location for each
                prompt += f"There are {num_map[cnt]} {category_name_map[category[i_c]][1]}"
                for i_p in sorted_edge_pos_incides[:n_pos]:
                    n_obj = name_edge_array[i_c, i_p]
                    num  = num_map[n_obj]
                    prompt += f", {num} in the {position[i_p+1]}"
                    aux = gen_dirc_prompt(data, category_index=i_c, position_index=i_p+1)
                    prompt += aux
            else: # appear only at one location
                prompt += "There"
                i_p = sorted_edge_pos_incides[0]
                n_obj = name_edge_array[i_c, i_p]
                if n_obj > 1:
                    verb = "are"
                    num  = num_map[n_obj]
                    noun = category_name_map[category[i_c]][1] # plural
                else:
                    verb = "is"
                    num  = num_map[n_obj]
                    noun = category_name_map[category[i_c]][0] # single
                prompt += f" {verb} {num} {noun}"
                aux = gen_dirc_prompt(data, category_index=i_c, position_index=i_p+1)
                prompt += (aux + f" in the {position[i_p+1]} of the image")
            
            prompt += ". "
        
        # print(prompt)

        if filename in scene_captions.keys():
            TEXT[filename] = scene_captions[filename] + prompt.rstrip()
        
    # dump the caption and structured data to disk
    with open('./dota_train_caption.json', 'w') as f:
        json.dump(TEXT, f)