from typing import Any, Callable, Dict, List, Optional, Union
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline, AutoencoderKL
from diffusers.loaders import AttnProcsLayers
import os, json, random
from PIL import Image
import torch
from safetensors.torch import load_file
from diffusers.loaders import TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipeline,
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import logging
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    CLIPImageProcessor, 
    CLIPTextModel, 
    CLIPTokenizer, 
    CLIPVisionModelWithProjection, 
    CLIPTextModelWithProjection, 
    AutoProcessor,
)
import inspect
from ccdiff.attention_processor import set_processors
from ccdiff.utils import dict_of_images, find_nearest, get_similar_examplers, seed_everything
from ccdiff.projection import Resampler, SerialSampler
from ccdiff.modules import FrozenDinoV2Encoder

logger = logging.get_logger(__name__)

class StableDiffusionMIPipeline(StableDiffusionPipeline):
    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler: KarrasDiffusionSchedulers,
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPImageProcessor,
            image_encoder: CLIPVisionModelWithProjection = None,
            requires_safety_checker: bool = True,
    ):
        # Get the parameter signature of the parent class constructor
        parent_init_signature = inspect.signature(super().__init__)
        parent_init_params = parent_init_signature.parameters 
        self.text_projector = CLIPTextModelWithProjection.from_pretrained('openai/clip-vit-large-patch14')
        self.image_encoder = FrozenDinoV2Encoder()
        self.image_proj_model = SerialSampler(
            dim=1280,
            depth=4,
            dim_head=64,
            # heads=20,
            num_queries=[16, 8, 8],
            embedding_dim=self.image_encoder.model.embed_dim,
            output_dim=unet.config.cross_attention_dim,
            ff_mult=4,
        )
        
        # Dynamically build a parameter dictionary based on the parameters of the parent class constructor
        init_kwargs = {
            # "image_encoder": image_encoder,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "unet": unet,
            "scheduler": scheduler,
            "safety_checker": safety_checker,
            "feature_extractor": feature_extractor,
            "requires_safety_checker": requires_safety_checker
        }
        if 'image_encoder' in parent_init_params.items():
            init_kwargs['image_encoder'] = image_encoder
        super().__init__(**init_kwargs)

    def _encode_prompt(
            self,
            prompts,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompts is not None and isinstance(prompts, str):
            batch_size = 1
        elif prompts is not None and isinstance(prompts, list):
            batch_size = len(prompts)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_embeds_none_flag = (prompt_embeds is None)
        prompt_embeds_list = []
        embeds_pooler_list = []
        text_embeds_list = []
        for prompt in prompts:
            if prompt_embeds_none_flag:
                # textual inversion: procecss multi-vector tokens if necessary
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

                text_inputs = self.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                untruncated_ids = self.tokenizer(
                    prompt, padding="longest", return_tensors="pt"
                ).input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[
                    -1
                ] and not torch.equal(text_input_ids, untruncated_ids):
                    removed_text = self.tokenizer.batch_decode(
                        untruncated_ids[:, self.tokenizer.model_max_length - 1: -1]
                    )
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                    )

                if (
                        hasattr(self.text_encoder.config, "use_attention_mask")
                        and self.text_encoder.config.use_attention_mask
                ):
                    attention_mask = text_inputs.attention_mask.to(device)
                else:
                    attention_mask = None

                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device),
                    attention_mask=attention_mask,
                )
                embeds_pooler = prompt_embeds.pooler_output
                prompt_embeds = prompt_embeds[0]
                text_embeds = self.text_projector(text_input_ids.to(device)).text_embeds

            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
            embeds_pooler = embeds_pooler.to(dtype=self.text_encoder.dtype, device=device)
            text_embeds = text_embeds.to(dtype=self.text_encoder.dtype, device=device)

            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            embeds_pooler = embeds_pooler.repeat(1, num_images_per_prompt)
            prompt_embeds = prompt_embeds.view(
                bs_embed * num_images_per_prompt, seq_len, -1
            )
            embeds_pooler = embeds_pooler.view(
                bs_embed * num_images_per_prompt, -1
            )
            prompt_embeds_list.append(prompt_embeds)
            embeds_pooler_list.append(embeds_pooler)
            text_embeds_list.append(text_embeds)
        prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
        embeds_pooler = torch.cat(embeds_pooler_list, dim=0)
        text_embeds = torch.cat(text_embeds_list, dim=0)
        # negative_prompt_embeds: (prompt_nums[0]+prompt_nums[1]+...prompt_nums[n], token_num, token_channel), <class 'torch.Tensor'>

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                negative_prompt = "worst quality, low quality, bad anatomy"
            uncond_tokens = [negative_prompt] * batch_size

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                    hasattr(self.text_encoder.config, "use_attention_mask")
                    and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=self.text_encoder.dtype, device=device
            )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )
            # negative_prompt_embeds: (len(prompt_nums), token_num, token_channel), <class 'torch.Tensor'>

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            final_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return final_prompt_embeds, prompt_embeds, embeds_pooler[:, None, :], text_embeds


    @staticmethod
    def draw_box_desc(pil_img: Image, bboxes: List[List[float]], prompt: List[str]) -> Image:
        """Utility function to draw bbox on the image"""
        color_list = ['red', 'blue', 'yellow', 'purple', 'green', 'black', 'brown', 'orange', 'white', 'gray']
        width, height = pil_img.size
        draw = ImageDraw.Draw(pil_img)
        font_folder = os.path.dirname(os.path.dirname(__file__))
        font_path = os.path.join('./Rainbow-Party-2.ttf')
        font = ImageFont.truetype(font_path, 30)

        for box_id in range(len(bboxes)):
            obj_box = bboxes[box_id]
            text = prompt[box_id]
            fill = 'black'
            for color in prompt[box_id].split(' '):
                if color in color_list:
                    fill = color
            text = text.split(',')[0]
            # x_min, y_min, x_max, y_max = (
            #     obj_box[0] * width,
            #     obj_box[1] * height,
            #     obj_box[2] * width,
            #     obj_box[3] * height,
            # )
            # draw.rectangle(
            #     [int(x_min), int(y_min), int(x_max), int(y_max)],
            #     outline=fill,
            #     width=4,
            # )
            x_min, y_min = (
                obj_box[0] * width,
                obj_box[1] * height,
            )
            draw.polygon(
                [(x*width, y*height) for x, y in zip(obj_box[::2], obj_box[1::2])],
                outline=fill,
                width=4,
            )
            draw.text((int(x_min), int(y_min)), text, fill=fill, font=font)

        return pil_img


    @torch.no_grad()
    def __call__(
            self,
            prompt: List[List[str]] = None,
            obboxes: List[List[List[float]]] = None,
            bboxes: List[List[List[float]]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            GUI_progress=None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            token_indices (Union[List[List[List[int]]], List[List[int]]], optional):
                The list of the indexes in the prompt to layout. Defaults to None.
            bboxes (Union[List[List[List[float]]], List[List[float]]], optional):
                The bounding boxes of the indexes to maintain layout in the image. Defaults to None.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            max_guidance_iter (`int`, *optional*, defaults to `10`):
                The maximum number of iterations for the layout guidance on attention maps in diffusion mode.
            max_guidance_iter_per_step (`int`, *optional*, defaults to `5`):
                The maximum number of iterations to run during each time step for layout guidance.
            scale_factor (`int`, *optional*, defaults to `50`):
                The scale factor used to update the latents during optimization.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        
        
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_nums = [0] * len(prompt)
        for i, _ in enumerate(prompt):
            prompt_nums[i] = len(_)

        device = self._execution_device
        self.image_encoder.to(device)
        self.text_projector.to(device)
        self.image_proj_model.to(device)
        # self.bg_proj_model.to(device)
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, cond_prompt_embeds, embeds_pooler, text_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        # print(prompt_embeds.shape)  3 77 768

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        
        
        ref_imgs = []
        for index, (caption, bndboxes) in enumerate(zip(prompt, bboxes)):
            categories = caption[1:]
            instances = []
            instance_imgs = []
            for name,bbox in zip(categories, bndboxes):
                if name == '':
                    instances.append(torch.zeros([3, 224, 224]))
                else:                   
                    value = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])    # foreground size based on HBB
                    chosen_file = list(dict_of_images[name].keys())[find_nearest(list(dict_of_images[name].values()), value)]                    
                    img = Image.open(os.path.join('path_to_data/DIOR/results/foreground', name, chosen_file))   
                    instance_imgs.append(img)                
                    img = self.feature_extractor(images=img, return_tensors="pt")['pixel_values'].squeeze(0)
                    instances.append(img)
            ref_imgs.append(torch.stack([instance for instance in instances])) 
           
        ref_imgs = torch.stack([img for img in ref_imgs]).view(len(obboxes) * len(obboxes[0]), 3, 224, 224).to(device)
        with torch.no_grad():
            img_features = self.image_encoder(ref_imgs)
            bg_img = get_similar_examplers(None, text_embeds[0], topk=1, sim_mode='text2img')
            bg_img = Image.open(os.path.join('path_to_data/DIOR/train', bg_img[0])).convert('RGB')
            bg_features = self.feature_extractor(images=bg_img, return_tensors="pt")['pixel_values'].to(img_features.device)
            bg_features = self.image_encoder(bg_features)
        img_features, bg_features = self.image_proj_model(img_features, obboxes, bg_features)
        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if GUI_progress is not None:
                    GUI_progress[0] = int((i + 1) / len(timesteps) * 100)
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )
                
                # predict the noise residual
                cross_attention_kwargs = {'bboxes': bboxes,
                                          'obboxes': obboxes,                                         
                                          'embeds_pooler': embeds_pooler,
                                          'height': height,
                                          'width': width,
                                          'ref_features': (img_features, bg_features),
                                          'do_classifier_free_guidance': do_classifier_free_guidance,
                                          }
                self.unet.eval()
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                    )

                step_output = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                )
                latents = step_output.prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if output_type == "latent":
            image = latents
        elif output_type == "pil":
            # 8. Post-processing
            image = self.decode_latents(latents)
            image = self.numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, None)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=None
        )

if __name__ == '__main__':
    sd1x_path = 'stable-diffusion-v1-4'
    save_path = 'checkpoint-dior'
    data_path = 'path_to_data/DIOR/val'

    pipe = StableDiffusionMIPipeline.from_pretrained(
        sd1x_path)
    
    set_processors(pipe.unet)
    custom_layers = AttnProcsLayers(pipe.unet.attn_processors)
    state_dict = {k: v for k, v in load_file(os.path.join(save_path, 'unet/diffusion_pytorch_model.safetensors')).items() if '.processor' in k or '.self_attn' in k}
    custom_layers.load_state_dict(state_dict)
    pipe.image_proj_model.load_state_dict(torch.load(os.path.join(save_path, 'ImageProjModel.pth')))
    pipe = pipe.to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    
    data = []
    with open(os.path.join(data_path, 'metadata.jsonl'), 'r') as f:
        for line in f:
            data.append(json.loads(line))
        
    negative_prompt = 'worst quality, low quality, bad anatomy, watermark, text, blurry'
    seed = 7351007268695528845
    seed_everything(seed)
    
    sample_num = 100
    for i in range(sample_num):
        sample = random.choice(data)
        file_name = sample['file_name']
        gt_img = Image.open(os.path.join(data_path, file_name)).convert('RGB')
        prompt = [sample['caption']]
        obboxes = [sample['obboxes']]
        bboxes = [sample['bndboxes']]
        
        image = pipe(prompt, obboxes, bboxes, num_inference_steps=50, guidance_scale=7.5, negative_prompt=negative_prompt).images[0]
        gt_img = gt_img.resize(image.size)
        result = Image.new('RGB', (gt_img.width + image.width, image.height))
        result.paste(gt_img, (0, 0))
        result.paste(image, (image.width, 0))
        result.save(f'saved/output_{file_name}')
        
        image = pipe.draw_box_desc(image, obboxes[0], prompt[0][1:])
        image.save(f'saved/anno_output_{file_name}')
    
