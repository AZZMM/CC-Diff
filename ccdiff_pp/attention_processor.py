import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention

from .layers import CoFusion
from .utils import get_sup_mask

class AttnProcessor(nn.Module):
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class MaskedProcessor(nn.Module):
    def __init__(self, hidden_size, cross_attention_dim=None,
                 use_ea_attn=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.use_ea_attn = use_ea_attn
        self.fusion = CoFusion(hidden_size, context_dim=cross_attention_dim) if use_ea_attn else None
        
    def __call__(
            self,
            attn: Attention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            bboxes=[],
            obboxes=[],
            embeds_pooler=None,
            height=512,
            width=512,
            ref_features=None,
            do_classifier_free_guidance=False,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        # assert(batch_size == 2, "We currently only implement sampling with batch_size=1, \
        #        and we will implement sampling with batch_size=N as soon as possible.")
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        
        instance_num = len(obboxes[0])
        
        if not self.use_ea_attn:
            encoder_hidden_states = encoder_hidden_states[:2, ...] if do_classifier_free_guidance else encoder_hidden_states[:1, ...]
        
        if self.use_ea_attn:
            if do_classifier_free_guidance:
                hidden_states_uncond = hidden_states[[0], ...]
                hidden_states_cond = hidden_states[[1], ...].repeat(instance_num + 1, 1, 1)
                hidden_states = torch.cat([hidden_states_uncond, hidden_states_cond])
            else:
                hidden_states_cond = hidden_states.repeat(instance_num + 1, 1, 1)
                hidden_states = hidden_states_cond
        
        # QKV Operation of Vanilla Self-Attention or Cross-Attention
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask)  # 48 4096 77
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        if not self.use_ea_attn:
            return hidden_states
        
        if do_classifier_free_guidance:
            hidden_states_uncond = hidden_states[[0], ...]  # torch.Size([1, HW, C])
            ca_output = hidden_states[1: , ...].unsqueeze(0)  # torch.Size([1, 1+instance_num, 5, 64, 1280])
        else:
            ca_output = hidden_states.unsqueeze(0)
        # ca_output = ca_output.unsqueeze(0)
        
        guidance_masks = []
        in_box = []
        # Construct Instance Guidance Mask
        guidance_masks = []
        for obbox in obboxes[0]:  
            guidance_mask = np.zeros((height, width))
            if np.count_nonzero(obbox):
                pts = np.array(obbox).reshape(-1, 1, 2)
                pts[..., 0] = pts[..., 0] * width
                pts[..., 1] = pts[..., 1] * height
                pts = np.int32(pts)
                guidance_masks.append(cv2.fillPoly(guidance_mask, [pts], 1)[None, ...])
            else:
                guidance_masks.append(guidance_mask[None, ...])
            in_box.append([obbox[0], obbox[2], obbox[4], obbox[6], obbox[1], obbox[3], obbox[5], obbox[7]])
        
        # Construct Background Guidance Mask
        sup_mask = get_sup_mask(guidance_masks)
        supplement_mask = torch.from_numpy(sup_mask[None, ...])
        supplement_mask = F.interpolate(supplement_mask, (height//8, width//8), mode='bilinear').float()
        supplement_mask = supplement_mask.to(hidden_states.device)  # (1, 1, H, W)
        
        guidance_masks = np.concatenate(guidance_masks, axis=0)
        guidance_masks = guidance_masks[None, ...]
        guidance_masks = torch.from_numpy(guidance_masks).float().to(ca_output.device)
        guidance_masks = F.interpolate(guidance_masks, (height//8, width//8), mode='bilinear')  # (1, instance_num, H, W)

        sigmoid_values = []
        for w_min, h_min, w_max, h_max in bboxes[0]:
            H, W = height//8, width // 8
            x = torch.linspace(0, W - 1, W)
            y = torch.linspace(0, H - 1, H)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            xx, yy = xx / H, yy / W
            mu1 = (w_min + w_max) / 2
            mu2 = (h_min + h_max) / 2
            sigma1 = ((w_max - w_min) ** 2) / 4
            sigma2 = ((h_max - h_min) ** 2) / 4
            if sigma1 == 0 or sigma2 == 0:
                sigmoid_values.append(torch.zeros_like(xx))
                continue
            exponent = -10 * (1 - ((xx - mu1) ** 2) / sigma1 - ((yy - mu2) ** 2) / sigma2)
            sigmoid_value = 1 / (1 + torch.exp(exponent))
            sigmoid_values.append(sigmoid_value)
        sigmoid_values = torch.stack(sigmoid_values, dim=0)[None, ...].to(ca_output.device)
        

        in_box = torch.from_numpy(np.array(in_box))[None, ...].float().to(ca_output.device)  # (1, instance_num, 4)

        other_info = {}
        other_info['image_token'] = hidden_states_cond[None, ...]
        other_info['context'] = encoder_hidden_states[1:, ...] if do_classifier_free_guidance else encoder_hidden_states
        other_info['box'] = in_box
        other_info['context_pooler'] = embeds_pooler  # (instance_num, 1, 768)
        other_info['supplement_mask'] = supplement_mask
        other_info['height'] = height
        other_info['width'] = width
        other_info['ref_features'] = ref_features
        other_info['sigmoid_values'] = sigmoid_values

        hidden_states_cond = self.fusion(ca_output,
                                    guidance_masks,
                                    other_info=other_info)
        
        # hidden_states = hidden_states.squeeze(1)
        hidden_states = torch.cat([hidden_states_uncond, hidden_states_cond]) if do_classifier_free_guidance else hidden_states_cond
        return hidden_states
    
def set_processors(unet, **kwargs):
    attn_processors = {}
    for name, _ in unet.attn_processors.items():
        use_ea_attn = False
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
            use_ea_attn = True         
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            if block_id == 1:
                use_ea_attn = True                    
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]                
        if cross_attention_dim is not None:    
            attn_processors[name] = MaskedProcessor(hidden_size=hidden_size, 
                                                    cross_attention_dim=cross_attention_dim,
                                                    use_ea_attn=use_ea_attn,
                                                    **kwargs)
        else:
            attn_processors[name] = AttnProcessor()            
    unet.set_attn_processor(attn_processors)