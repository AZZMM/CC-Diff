# Dataset Processing
(We recommend to create a new conda environment to process these scripts.)
## Caption Construction
**Step 1: Generate a general description of the RS image.**
* Clone [RSMamba](https://github.com/KyanChen/RSMamba.git) repo, and mv [test_DIOR.py](./caption/RSMamba/test_DIOR.py) to the folder.
* Generate descriptions and save to ```scene_caption.json```.

**Step 2: Descript objects with relative position and direction.**
* Use conversion script [dior_img_to_caption.py](./caption/dior_img_to_caption.py) and save to ```dior_caption.json```.

**Step 3: Merge both mentioned above**
* Use script [dior_txt_to_metajsonl.py](./caption/dior_txt_to_metajsonl.py) to get the final annotation ```metadata.jsonl```.

## Retrieval Base preparation
**CLIP embedding calculation**

Calculate text embs and img embs for reference dataset and save as dior_emb.pt, Relating script is [CLIPmodel.py](./LLM_T2L/CLIPmodel.py).

**Obtain segmented instances**

We select [SAM](https://github.com/facebookresearch/segment-anything.git) as our segmentation tool, implementation in [dior_run_sam.py](./SAM/dior_run_sam.py). Results are saved in folder ```results```.

## Text to Layout with GPT
We use **GPT-4o** as the LLM for layout planning based on the constructed text prompt. The prompt consists of two main sections: Instruction and Context Examples.
The script can refer to [LLM_Text2Layout.py](./LLM_T2L/LLM_Text2Layout.py).

## Datasets
In this work, we conduct experiments on [DIOR-RSVG](https://github.com/ZhanYang-nwpu/RSVG-pytorch.git) and [DOTAv1.0](https://captain-whu.github.io/DOTA/dataset.html).
For DIOR-RSVG, we use oriented bounding box annotations from [DIOR](https://arxiv.org/abs/1909.00133) in addition. For DOTA, we split the original image with 512×512 resolution and 100 gap, the processing steps are the same as DIOR-RSVG.
