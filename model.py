import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import torch
from packaging import version
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipeline,
    StableDiffusionPipelineOutput,
)

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
    rescale_noise_cfg,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers import DDIMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput



logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class DDIMCustomScheduler(DDIMScheduler):
    def get_alpha_beta(self, timestep):
        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod # 1

        beta_prod_t = 1 - alpha_prod_t
        return alpha_prod_t, alpha_prod_t_prev, beta_prod_t

    def get_original_sample(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        # eta: float = 0.0,
        # use_clipped_model_output: bool = False,
        # generator=None,
        # variance_noise: Optional[torch.FloatTensor] = None,
        # return_dict: bool = True,
    ):
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            # eta (`float`):
            #     The weight of noise for added noise in diffusion step.
            # use_clipped_model_output (`bool`, defaults to `False`):
            #     If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary
            #     because predicted original sample is clipped to [-1, 1] when `self.config.clip_sample` is `True`. If no
            #     clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
            #     `use_clipped_model_output` has no effect.
            # generator (`torch.Generator`, *optional*):
            #     A random number generator.
            # variance_noise (`torch.FloatTensor`):
            #     Alternative to generating noise with `generator` by directly providing the noise for the variance
            #     itself. Useful for methods such as [`CycleDiffusion`].
            # return_dict (`bool`, *optional*, defaults to `True`):
            #     Whether or not to return a [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`.

        Returns:
            original_sample (`torch.FloatTensor`)
            # [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
            #     If return_dict is `True`, [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] is returned, otherwise a
            #     tuple is returned where the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        alpha_prod_t, alpha_prod_t_prev, beta_prod_t = self.get_alpha_beta(timestep)
        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )
        # # 4. Clip or threshold "predicted x_0"
        # if self.config.thresholding:
        #     pred_original_sample = self._threshold_sample(pred_original_sample)
        # elif self.config.clip_sample:
        #     pred_original_sample = pred_original_sample.clamp(
        #         -self.config.clip_sample_range, self.config.clip_sample_range
        #     )
        return pred_original_sample

    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        branch_size: int = 1,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            eta (`float`):
                The weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`, defaults to `False`):
                If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary
                because predicted original sample is clipped to [-1, 1] when `self.config.clip_sample` is `True`. If no
                clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
                `use_clipped_model_output` has no effect.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            variance_noise (`torch.Tensor`):
                Alternative to generating noise with `generator` by directly providing the noise for the variance
                itself. Useful for methods such as [`CycleDiffusion`].
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # 4. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5)

        if use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        prev_sample_mean = prev_sample.detach().clone()

        if eta > 0:
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )

            if variance_noise is None:
                ## branch out here
                prev_sample = torch.repeat_interleave(prev_sample, repeats=branch_size, dim=0)  # [branch_size * batch_size, *image_shape]

                # variance_noise = randn_tensor(
                #     model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
                # )

                variance_noise = randn_tensor(
                    prev_sample.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
                )

            variance = std_dev_t * variance_noise

            prev_sample = prev_sample + variance

        if not return_dict:
            return (
                prev_sample,
                pred_original_sample,
                prev_sample_mean,
                std_dev_t,
            )

        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)


class TreeSearchStableDiffusionPipeline(
        StableDiffusionPipeline,
        DiffusionPipeline,
        StableDiffusionMixin,
        TextualInversionLoaderMixin,
        StableDiffusionLoraLoaderMixin,
        IPAdapterMixin,
        FromSingleFileMixin,
    ):
    """
    A pipeline for text-to-image generation using Stable Diffusion with tree search sampling.

    This pipeline inherits from `StableDiffusionPipeline` and adds support for tree search sampling.
    It allows for generating images based on text prompts using a tree search strategy.

    """

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
        
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker,
        )

        self.scheduler = DDIMCustomScheduler.from_config(self.scheduler.config)

   
    @torch.no_grad()
    def __call__(
        self,
        search_method: str = "TreeG-SC",
        active_size: int = 1,
        branch_size: int = 1,
        t_start: int = 0,
        t_end: int = 50,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        if not height or not width:
            height = (
                self.unet.config.sample_size
                if self._is_unet_config_sample_size_int
                else self.unet.config.sample_size[0]
            )
            width = (
                self.unet.config.sample_size
                if self._is_unet_config_sample_size_int
                else self.unet.config.sample_size[1]
            )
            height, width = height * self.vae_scale_factor, width * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]


        ## Tree Search: set active size
        sample_size = batch_size * num_images_per_prompt
        # batch_size = batch_size * active_size
        num_images_per_prompt *= active_size


        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

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

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)


        #### get repeated prompt_embeds
        prompt_embeds_for_tree_search = torch.repeat_interleave(prompt_embeds, repeats=branch_size, dim=0)


        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # print(f"latent_model_input shape: {latent_model_input.shape}")
                # print(f"timestep_cond shape: {timestep_cond.shape if timestep_cond is not None else None}")
                # print(f"prompt_embeds shape: {prompt_embeds.shape}")
                # print(f"cross_attention_kwargs: {self.cross_attention_kwargs}")
                # print(f"added_cond_kwargs: {added_cond_kwargs}")
                # print(f"t: {t}")

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                ## tree search here

                # compute the previous noisy sample x_t -> x_t-1
                # latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                is_guidance = i >= t_start and i < t_end

                ### here, different methods

                if search_method == "TreeG-SC":
                    latents = self.scheduler.step(
                        noise_pred,
                        t,
                        latents,
                        branch_size=branch_size if is_guidance else 1,
                        **extra_step_kwargs,
                        return_dict=False,
                    )[0]

                    ### evaluate and select
                    if is_guidance and branch_size > 1:
                        if i + 1 < len(self.scheduler.timesteps):
                            prev_t = self.scheduler.timesteps[i + 1]
                            latents, rewards = self.evaluate_then_select(
                                latents,
                                prev_t,
                                sample_size,
                                active_size,
                                prompt_embeds_for_tree_search,
                                generator,
                                output_type,
                            )

                        else:
                            latents, rewards = self.select_clean(
                                images=latents,
                                sample_size=sample_size,
                                top_size=active_size,
                                output_type=output_type,
                            )
                        
                        rewards_mean = rewards.mean().item()
                        progress_bar.set_postfix({'reward': f'{rewards_mean:.3f}'}, refresh=False)

                elif search_method == "TreeG-SD":
                    outputs = self.scheduler.step(
                        noise_pred,
                        t,
                        latents,
                        branch_size=1,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                    latents = outputs[0]
                    pred_x0 = outputs[1]
                    mean_pred = outputs[2]
                    sigma = outputs[3]
                    if is_guidance and branch_size > 1:
                        # pred_x0 = self.scheduler.get_original_sample(noise_pred, t, latent_model_input)
                        if i + 1 < len(self.scheduler.timesteps):
                            SD_kwargs=kwargs.get("SD_kwargs", {})
                            guided_outputs = self.xclean_branch_and_select(
                                t = t,
                                pred_xstart= pred_x0,
                                mean_pred=mean_pred,
                                num_samples=branch_size,
                                active_size=active_size,
                                output_type=output_type,
                                generator=generator,
                                SD_kwargs= SD_kwargs,
                            )

                                
                            guided_x0 = guided_outputs['guided_x0']
                            rewards = guided_outputs['rewards']
                            mean_pred = guided_outputs['mean_pred']


                            if SD_kwargs['dsg']:
                                epsilon = 1e-10
                                g_ascent = guided_x0 - guided_outputs['original_x0']
                                g_ascent_norm = torch.linalg.norm(g_ascent.view(g_ascent.size(0), -1), dim=1)
                                g_ascent_norm = g_ascent_norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)

                                _, c, h, w = mean_pred.size()
                                r = torch.sqrt(torch.tensor(c * h * w)) * sigma
                                d_star = r * g_ascent / (g_ascent_norm + epsilon)
                                d_sample = torch.randn_like(mean_pred) * sigma

                                guidance_rate = SD_kwargs.get('guidance_rate', 1.)
                                mix_direction = d_sample + guidance_rate * (d_star - d_sample)
                                mix_direction_norm = torch.linalg.norm(mix_direction.view(mix_direction.size(0), -1), dim=1)
                                mix_direction_norm = mix_direction_norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                                mix_step = mix_direction / (mix_direction_norm + epsilon) * r

                                sample = mean_pred + mix_step

                                latents = sample
                        
                        else:
                            latents, rewards = self.select_clean(
                                images=latents,
                                sample_size=sample_size,
                                top_size=active_size,
                                output_type=output_type,
                            )

                            rewards = rewards.mean().item()

                        ## set the progress bar postfix
                        if isinstance(rewards, list):
                            ## show all rewards for bar postfix
                            rewards_all = [f'{reward:.3f}' for reward in rewards]
                            progress_bar.set_postfix({'reward': rewards_all}, refresh=False)
                        else:
                            progress_bar.set_postfix({'reward': f'{rewards:.3f}'}, refresh=False)

                        



                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                # if XLA_AVAILABLE:
                #     xm.mark_step()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        # print(f"image shape: {image.shape}")

        ## last step: select the best image over the active size
        image_pt = self.image_processor.postprocess(image, output_type="pt", do_denormalize=[True]*image.shape[0])
        if active_size > 1:
            rewards = self.reward_fn(image_pt)
            rewards = torch.tensor(rewards, device=latents.device) 
            rewards = rewards.view(sample_size, active_size)   # [2, 3]
            best_idx = rewards.argmax(dim=1, keepdim=True)     

            C, H, W = image.shape[1:]                     

            image = image.view(sample_size, active_size, C, H, W)   # [2, 3, 3, 768, 768]

            best_idx_expanded = (
                best_idx
                .unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)          # [2, 1, 1, 1, 1]
                .expand(-1, -1, C, H, W)                            # [2, 1, 3, 768, 768]
            )
            image = image.gather(dim=1, index=best_idx_expanded)    # [2, 1, 3, 768, 768]
           
            image = image.squeeze(1) 


        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
        
        image_pil = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        image_pt = self.image_processor.postprocess(image, output_type="pt", do_denormalize=do_denormalize)


        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image_pil, has_nsfw_concept), image_pt

        return StableDiffusionPipelineOutput(images=image_pil, nsfw_content_detected=has_nsfw_concept), image_pt
    

    def set_reward_fn(self, reward_fn: Callable):
        """
        Set the reward function for the pipeline.

        Args:
            reward_fn (`Callable`):
                The reward function to be used for tree search sampling.
        """
        self.reward_fn = reward_fn


    def evaluate_then_select(self, latents, t, sample_size, active_size, prompt_embeds, generator, output_type):
        model_output = self.unet(
            latents,
            t,
            encoder_hidden_states=prompt_embeds,
            # return_dict=False,
        ).sample

        pred_x0 = self.scheduler.get_original_sample(model_output, t, latents)

        ### decode to image
        if output_type != "latent":
            images = self.vae.decode(pred_x0 / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
            images = self.image_processor.postprocess(images, output_type="pt", do_denormalize=[True]*images.shape[0])
            ### check input
            # print(f"max value: {images.max()}, min value: {images.min()}")
            rewards = self.reward_fn(images)
        else:
            rewards = self.reward_fn(pred_x0)
        # print(rewards)
        rewards = torch.tensor(rewards, device=latents.device)
        rewards = rewards.view(sample_size, -1)
        topk_rewards, topk_indices = torch.topk(rewards, active_size, dim=1, largest=True)

        latents = latents.view(sample_size, -1, *latents.shape[1:])
        latents = latents[torch.arange(sample_size).unsqueeze(1), topk_indices]
        latents = latents.view(-1, *latents.shape[2:])

        return latents, topk_rewards
    
    def select_clean(self, images, sample_size, top_size, output_type):
        if output_type != "latent":
            images_ = self.vae.decode(images / self.vae.config.scaling_factor, return_dict=False)[0]
            images = self.image_processor.postprocess(images, output_type="pt", do_denormalize=[True]*images.shape[0])
            rewards = self.reward_fn(images_)
        else:
            rewards = self.reward_fn(images)
        rewards = torch.tensor(rewards, device=images.device)
        rewards = rewards.view(sample_size, -1)
        topk_rewards, topk_indices = torch.topk(rewards, top_size, dim=1, largest=True)
        images = images.view(sample_size, -1, *images.shape[1:])
        images = images[torch.arange(sample_size).unsqueeze(1), topk_indices]
        images = images.view(-1, *images.shape[2:])

        return images, topk_rewards
    
    def xclean_branch_and_select(
            self,
            t,
            pred_xstart,
            mean_pred,
            num_samples, ## branch_size
            active_size,
            output_type,
            generator,
            SD_kwargs = None,
    ):
               
        batch = pred_xstart.shape[0] // active_size
        _, C, H, W = pred_xstart.shape

        n_iter = SD_kwargs.get("n_iter", 1)
        
        rewards_list = []
        
        pred_xstart_new = pred_xstart.detach().clone()

        for _ in range(n_iter):
            
            sample = pred_xstart_new.unsqueeze(1).expand(-1, num_samples, -1, -1, -1)
            noise = torch.randn_like(sample)

            alpha_bar_t = self.scheduler.alphas_cumprod[t]
            scale_t = torch.sqrt(1 - alpha_bar_t)
            pred_xstart_scale_t = scale_t / torch.sqrt(1+scale_t**2)
            # pred_xstart_scale_t = 1.0 # to modify
            var_scale = SD_kwargs.get("pred_xstart_scale", 1.)
            

            sample = sample + var_scale * pred_xstart_scale_t * noise
            sample_ = sample.contiguous().view(-1, C, H, W)   # (batch*active_size*num_samples) x C x H x W


            ### decode to image
            if output_type != "latent":
                ### cut to chunk
                CHUNK_SIZE = 12
                num_chunks = (sample_.shape[0] + CHUNK_SIZE - 1) // CHUNK_SIZE
                sample_chunks = sample_.chunk(num_chunks, dim=0)
                images = []
                for chunk in sample_chunks:
                    chunk_images = self.vae.decode(chunk / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
                    images.append(chunk_images)
                images = torch.cat(images, dim=0)
                # images = self.vae.decode(sample_ / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
                images = self.image_processor.postprocess(images, output_type="pt", do_denormalize=[True]*images.shape[0])
                ### check input
                # print(f"max value: {images.max()}, min value: {images.min()}")
                rewards = self.reward_fn(images)
            else:
                rewards = self.reward_fn(sample_)
            # print(rewards)
            rewards = torch.tensor(rewards, device=pred_xstart.device)
            rewards = rewards.view(batch, -1)
            top_rewards, top_ind = torch.topk(rewards, active_size, dim=1, largest=True)

           
            sample = sample.contiguous().view(-1, C, H, W).contiguous().view(batch, -1, C, H, W)
            pred_xstart_new = sample[torch.arange(batch).unsqueeze(-1), top_ind]
            pred_xstart_new = pred_xstart_new.contiguous().view(-1, C, H, W)

            ### get corresponding pred_xstart and mean_pred
            pred_xstart = pred_xstart.unsqueeze(1).expand(-1, num_samples, -1, -1, -1)
            pred_xstart = pred_xstart.contiguous().view(-1, C, H, W).contiguous().view(batch, -1, C, H, W)
            pred_xstart = pred_xstart[torch.arange(batch).unsqueeze(-1), top_ind]
            pred_xstart = pred_xstart.contiguous().view(-1, C, H, W)

            mean_pred = mean_pred.unsqueeze(1).expand(-1, num_samples, -1, -1, -1)
            mean_pred = mean_pred.contiguous().view(-1, C, H, W).contiguous().view(batch, -1, C, H, W)
            mean_pred = mean_pred[torch.arange(batch).unsqueeze(-1), top_ind]
            mean_pred = mean_pred.contiguous().view(-1, C, H, W)



            rewards_list.append(top_rewards.mean().item())

        return {'guided_x0': pred_xstart_new, 'original_x0':pred_xstart, 'mean_pred':mean_pred, 'rewards': rewards_list}






