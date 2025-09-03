from diffusers import EulerDiscreteScheduler
from diffusers.loaders import AttnProcsLayers
import os, json, random
from PIL import Image
import torch
from safetensors.torch import load_file
from diffusers.utils import logging
from ccdiff.attention_processor import set_processors
from ccdiff.utils import seed_everything
from infer_dior import StableDiffusionMIPipeline
from thop import profile, clever_format
logger = logging.get_logger(__name__)


if __name__ == '__main__':
    sd1x_path = 'stable-diffusion-v1-4'
    save_path = 'checkpoint-dior'

    pipe = StableDiffusionMIPipeline.from_pretrained(
        sd1x_path)
    
    set_processors(pipe.unet)
    custom_layers = AttnProcsLayers(pipe.unet.attn_processors)
    state_dict = {k: v for k, v in load_file(os.path.join(save_path, 'unet/diffusion_pytorch_model.safetensors')).items() if '.processor' in k or '.self_attn' in k}
    custom_layers.load_state_dict(state_dict)
    pipe.image_proj_model.load_state_dict(torch.load(os.path.join(save_path, 'ImageProjModel.pth')))
    pipe = pipe.to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    
    # 1. all learnable parameters
    total_learnable_params = sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad)
    
    # 2. GFLOPs
    def calc_unet_flops(pipe, latent_model_input, t, prompt_embeds, cross_attention_kwargs):
        pipe.unet.eval()
        with torch.no_grad():
            flops, params = profile(
                pipe.unet,
                inputs=(latent_model_input, t,  prompt_embeds, None,None,None, cross_attention_kwargs),
                custom_ops={},
                verbose=False
            )
            flops, _ = clever_format([flops, params], '%.3f')
            print(flops)
        print(f"FLOPs: {flops/1e9:.2f} GFLOPs")
    
    # Adapt dummy inputs according to model setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_model_input = torch.randn(2, 4, 64, 64).to(device)  # batch=2, c=4, h=64, w=64
    t = torch.tensor([1, 1]).to(device)
    prompt_embeds = torch.randn(2, 77, 768).to(device)  # batch=2, seq=77, dim=768
    cross_attention_kwargs = {'prompt_nums': [1], 'bboxes': [[[0.1,0.1,0.5,0.5]]], 'ith': 0, 'embeds_pooler': torch.randn(2,1,768).to(device), 'timestep': t, 'height': 512, 'width': 512, 'MIGCsteps':20, 'NaiveFuserSteps':-1, 'ca_scale':None, 'ea_scale':None, 'sac_scale':None}
    calc_unet_flops(pipe, latent_model_input, t, prompt_embeds, cross_attention_kwargs)
