from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from dataclasses import dataclass

from .transformer import (
    LayerNormFp32,
    LayerNorm,
    QuickGELU,
    MultimodalTransformer,
    MultimodalDecoder
)
from .model import CLIPTextCfg, CLIPVisionCfg, _build_vision_tower, _build_text_tower
from .transformer import ContinuousTransformer
from .image_tokenizer import build_image_tokenizer
try:
    from transformers import (
        BeamSearchScorer,
        LogitsProcessorList,
        TopPLogitsWarper,
        TopKLogitsWarper,
        RepetitionPenaltyLogitsProcessor,
        MinLengthLogitsProcessor,
        MaxLengthCriteria,
        StoppingCriteriaList
    )

    GENERATION_TYPES = {
        "top_k": TopKLogitsWarper,
        "top_p": TopPLogitsWarper,
        "beam_search": "beam_search"
    }
    _has_transformers = True
except ImportError as e:
    GENERATION_TYPES = {
        "top_k": None,
        "top_p": None,
        "beam_search": "beam_search"
    }
    _has_transformers = False


@dataclass
class MultimodalCfg(CLIPTextCfg):
    mlp_ratio: int = 4
    dim_head: int = 64
    heads: int = 8
    n_queries: int = 256
    attn_pooler_heads: int = 8
    context_length: int = 77
    context_length_image: int = 196
    context_length_text: int = 77
    vocab_size_image: int = 20000
    vocab_size_text: int = 49408
    tied_decoder: bool = False

@dataclass
class DecoderCfg(CLIPTextCfg):
    mlp_ratio: int = 4
    dim_head: int = 64
    heads: int = 8
    n_queries: int = 256
    attn_pooler_heads: int = 8
    context_length: int = 77
    vocab_size: int = 49408
    output_dim:int = 49408

@dataclass
class ImageTokenizerCfg():

    name: str = "image_patch" # or taming or icetk
    # only for taming
    config_path: str = "vqgan_imagenet_f16_16384.yaml"
    model_path: str = "vqgan_imagenet_f16_16384.ckpt"

    # compress_rate for icetk
    compress_rate: int = 16



def _build_decoder_tower(
        decoder_cfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
        discrete=True,
        input_dim=None,
):
    decoder_cfg = DecoderCfg(**decoder_cfg) if isinstance(decoder_cfg, dict) else decoder_cfg
    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = (
        LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    )

    decoder = MultimodalDecoder(
        context_length=decoder_cfg.context_length,
        width=decoder_cfg.width,
        heads=decoder_cfg.heads,
        layers=decoder_cfg.layers,
        ls_init_value=decoder_cfg.ls_init_value,
        output_dim=decoder_cfg.output_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
        vocab_size=decoder_cfg.vocab_size,
        discrete_input=discrete,
        input_dim=input_dim,

    )

    return decoder


class SymGen(nn.Module):
    def __init__(
        self,
        embed_dim,
        vision_cfg: CLIPTextCfg,
        text_cfg: CLIPTextCfg,
        image_decoder_cfg: DecoderCfg,
        text_decoder_cfg: DecoderCfg,
        image_tokenizer_cfg: ImageTokenizerCfg,
        quick_gelu: bool = False,
        init_logit_scale: float = np.log(1 / 0.07),
        init_logit_bias: Optional[float] = None,
        cast_dtype: Optional[torch.dtype] = None,
        pad_id: int = 0,
    ):
        super().__init__()
        self.image_tokenizer = build_image_tokenizer(**image_tokenizer_cfg)
        vision_cfg = CLIPTextCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg
        text_cfg = CLIPTextCfg(**text_cfg) if isinstance(text_cfg, dict) else text_cfg
        image_decoder_cfg = DecoderCfg(**image_decoder_cfg) if isinstance(image_decoder_cfg, dict) else image_decoder_cfg
        text_decoder_cfg = DecoderCfg(**text_decoder_cfg) if isinstance(text_decoder_cfg, dict) else text_decoder_cfg
        # causal text encoder
        self.text = _build_text_tower(
            embed_dim=embed_dim,
            text_cfg=text_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )
        # causal image encoder
        if self.image_tokenizer.discrete:
            self.visual = _build_text_tower(
                embed_dim=embed_dim,
                text_cfg=text_cfg,
                quick_gelu=quick_gelu,
                cast_dtype=cast_dtype,
            )
        else:
            self.visual = ContinuousTransformer(
                input_dim=self.image_tokenizer.dim,
                context_length=image_decoder_cfg.context_length,
                width=vision_cfg.width,
                heads=vision_cfg.heads,
                layers=vision_cfg.layers,
                pool_type="last",
                embed_cls=True,
                output_tokens=True,
                output_dim=embed_dim,
            )
        # causal text decoder
        self.text_decoder = _build_decoder_tower(
            decoder_cfg=text_decoder_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
            discrete=False,
            input_dim=embed_dim,
        )
        # causal image decoder
        self.image_decoder = _build_decoder_tower(
            decoder_cfg=image_decoder_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
            discrete=False,
            input_dim=embed_dim,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None
        self.pad_id = pad_id
        self.register_buffer("image_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
        self.use_contrastive = True
        self.use_image_decoder = True
        self.use_text_decoder = True
        self.use_unimodal_image = True
        self.use_unimodal_text = True
            
    def no_contrastive(self):
        self.text.ln_final.bias.requires_grad = False
        self.text.ln_final.weight.requires_grad = False
        self.text.text_projection.requires_grad = False

        self.visual.ln_final.bias.requires_grad = False
        self.visual.ln_final.weight.requires_grad = False
        self.visual.text_projection.requires_grad = False
        self.visual.cls_emb.requires_grad = False

        self.logit_scale.requires_grad = False
        self.use_contrastive = False
    
    def no_image_decoder(self):
        self.image_decoder.requires_grad = False
        self.use_image_decoder = False

    def no_text_decoder(self):
        self.text_decoder.requires_grad = False
        self.use_text_decoder = False

    def no_unimodal_text(self):
        self.use_unimodal_text = False

    def no_unimodal_image(self):
        self.use_unimodal_image = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)
        self.image_decoder.set_grad_checkpointing(enable)
        self.text_decoder.set_grad_checkpointing(enable)

    def _encode_image(self, images, normalize: bool = True, return_tokens=False):
        with torch.no_grad():
            images = (images * self.image_std + self.image_mean) if self.image_tokenizer.needs_0_1 else images
            image_tokens = self.image_tokenizer.tokenize(images)
        image_latent, tokens_embs = self.visual(image_tokens)
        image_latent = F.normalize(image_latent, dim=-1) if normalize else image_latent
        if return_tokens:
            return image_latent, tokens_embs, image_tokens
        else:
            return image_latent, tokens_embs

    def _encode_text(self, text, normalize: bool = True):
        text_latent, token_emb = self.text(text)
        text_latent = F.normalize(text_latent, dim=-1) if normalize else text_latent
        return text_latent, token_emb

    def encode_image(self, images, normalize: bool = True):
        image_latent, _ = self._encode_image(images, normalize=normalize)
        return image_latent

    def encode_text(self, text, normalize: bool = True):
        text_latent, _ = self._encode_text(text, normalize=normalize)
        return text_latent

    def forward(
            self,
            image,
            text: Optional[torch.Tensor] = None,
            image_tokens: Optional[torch.Tensor] = None,
            image_latent: Optional[torch.Tensor] = None,
            image_embs: Optional[torch.Tensor] = None,
    ):

        image_latent, image_embs, image_tokens = self._encode_image(image, return_tokens=True)
        text_latent, text_embs = self._encode_text(text)
    
        input_image = image_embs[:, 0:-1]
        labels_image = image_tokens[:, 1:]

        input_text = text_embs[:, 0:-1]
        labels_text = text[:, 1:]

        if self.use_text_decoder:
            logits_text = self.text_decoder(image_embs, input_text)
            if self.use_unimodal_text:
                logits_text_unimodal = self.text_decoder(None, input_text)
            else:
                logits_text_unimodal = None
        else:
            logits_text = None
            logits_text_unimodal = None

        if self.use_image_decoder:
            logits_image = self.image_decoder(text_embs, input_image)

            if self.use_unimodal_image:
                logits_image_unimodal = self.image_decoder(None, input_image)
            else:
                logits_image_unimodal = None
        else:
            logits_image = None
            logits_image_unimodal = None
        
        out_dict = {
            "image_features": image_latent,
            "text_features": text_latent,

            "logits_image": logits_image,
            "logits_text": logits_text,

            "logits_image_unimodal": logits_image_unimodal,
            "logits_text_unimodal": logits_text_unimodal,

            "labels_image": labels_image,
            "labels_text": labels_text,
            
            "logit_scale": self.logit_scale.exp()
        }
        if self.logit_bias is not None:
            out_dict["logit_bias"] = self.logit_bias
        return out_dict