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
            text_decoder_cfg: MultimodalCfg,
            image_decoder_cfg: MultimodalCfg,
            text_cfg: CLIPTextCfg,
            vision_cfg: CLIPVisionCfg,
            image_tokenizer_cfg: ImageTokenizerCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            cast_dtype: Optional[torch.dtype] = None,
            pad_id: int = 0,
    ):
        super().__init__()
        multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
        self.image_tokenizer = build_image_tokenizer(**image_tokenizer_cfg)
        text_cfg = CLIPTextCfg(**text_cfg) if isinstance(text_cfg, dict) else text_cfg
        vision_cfg = CLIPVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg

        self.text = _build_text_tower(
            embed_dim=embed_dim,
            text_cfg=text_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        self.visual = _build_vision_tower(
            embed_dim=embed_dim,
            vision_cfg=vision_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )
        self.tied_decoder = multimodal_cfg.tied_decoder
        self.text_decoder = _build_decoder_tower(
            multimodal_cfg.vocab_size_text,
            multimodal_cfg=multimodal_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
            context_length=multimodal_cfg.context_length_text,
        )
        self.image_decoder = _build_decoder_tower(
            multimodal_cfg.vocab_size_image,
            multimodal_cfg=multimodal_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
            context_length=multimodal_cfg.context_length_image,
            discrete=self.image_tokenizer.discrete,
            input_dim=None if self.image_tokenizer.discrete else self.image_tokenizer.dim,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None
        self.pad_id = pad_id

        self.context_length = multimodal_cfg.context_length

        self.register_buffer("image_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

        self.use_image_decoder = True
        self.use_text_decoder = True
        self.use_contrastive = True
            
    def no_contrastive(self):
        self.visual.proj.requires_grad = False
        self.text.ln_final.bias.requires_grad = False
        self.text.ln_final.weight.requires_grad = False
        self.text.text_projection.requires_grad = False
        self.logit_scale.requires_grad = False
        self.use_contrastive = False

    def no_image_decoder(self):
        self.image_decoder.requires_grad = False
        self.use_image_decoder = False

    def no_text_decoder(self):
        self.image_decoder.requires_grad = False
        self.use_image_decoder = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)
        self.text_decoder.set_grad_checkpointing(enable)

    def _encode_image(self, images, normalize: bool = True):
        image_latent, tokens_embs = self.visual(images)
        image_latent = F.normalize(image_latent, dim=-1) if normalize else image_latent
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
        if image_latent is None and image_embs is None:
            image_latent, image_embs = self._encode_image(image)

        if text is None:
            return {"image_features": image_latent, "image_embs": image_embs}

        if image_tokens is None and self.use_image_decoder:
            with torch.no_grad():
                image_0_1 = (image * self.image_std + self.image_mean) if self.image_tokenizer.needs_0_1 else image
                image_tokens = self.image_tokenizer.tokenize(image_0_1)


        if self.use_image_decoder or self.use_contrastive:
            text_latent, token_embs = self._encode_text(text)
        else:
            text_latent = None
            token_embs = None

        # TODO: add assertion to avoid bugs?
        #print(image_embs.shape, token_embs.shape, image_tokens.shape, text.shape)


        num_image_tokens = self.image_tokenizer.num_image_tokens if self.tied_decoder else 0

        input_text = text[:, 0:-1] + num_image_tokens
        labels_text = text[:, 1:] + num_image_tokens

        if self.use_image_decoder:
            input_image = image_tokens[:, 0:-1]
            labels_image = image_tokens[:, 1:]
        else:
            labels_image = None

        logits_text = self.text_decoder(image_embs, input_text)
        logits_text_unimodal = self.text_decoder(None, input_text)

        if self.use_image_decoder:
            logits_image = self.image_decoder(token_embs, input_image)
            logits_image_unimodal = self.image_decoder(None, input_image)
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

    def generate(
        self,
        image=None,
        text=None,
        seq_len=30,
        max_seq_len=77,
        temperature=1.,
        generation_type="top_p",
        top_p=0.1,  # keep tokens in the 1 - top_p quantile
        top_k=1,  # keeps the top_k most probable tokens
        pad_token_id=None,
        eos_token_id=None,
        sot_token_id=None,
        num_beams=6,
        num_beam_groups=3,
        min_seq_len=5,
        stopping_criteria=None,
        repetition_penalty=1.0,
        fixed_output_length=False # if True output.shape == (batch_size, seq_len)
    ):
        # taking many ideas and components from HuggingFace GenerationMixin
        # https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
        assert _has_transformers, "Please install transformers for generate functionality. `pip install transformers`."
        assert seq_len > min_seq_len, "seq_len must be larger than min_seq_len"

        with torch.no_grad():
            sot_token_id = 49406 if sot_token_id is None else sot_token_id
            eos_token_id = 49407 if eos_token_id is None else eos_token_id
            pad_token_id = self.pad_id if pad_token_id is None else pad_token_id
            logit_processor = LogitsProcessorList(
                [
                    MinLengthLogitsProcessor(min_seq_len, eos_token_id),
                    RepetitionPenaltyLogitsProcessor(repetition_penalty),
                ]
            )

            if stopping_criteria is None:
                stopping_criteria = [MaxLengthCriteria(max_length=seq_len)]

            stopping_criteria = StoppingCriteriaList(
                stopping_criteria
            )

            device = image.device if image else text.device

            if generation_type == "beam_search":
                output = self._generate_beamsearch(
                    image_inputs=image,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    sot_token_id=sot_token_id,
                    num_beams=num_beams,
                    num_beam_groups=num_beam_groups,
                    min_seq_len=min_seq_len,
                    stopping_criteria=stopping_criteria,
                    logit_processor=logit_processor,
                )
                if fixed_output_length and output.shape[1] < seq_len:
                    return torch.cat(
                        (output, torch.ones(output.shape[0], seq_len-output.shape[1], device=device, dtype=output.dtype) * self.pad_id),
                        dim=1
                    )
                return output

            elif generation_type == "top_p":
                logit_warper = GENERATION_TYPES[generation_type](top_p)
            elif generation_type == "top_k":
                logit_warper = GENERATION_TYPES[generation_type](top_k)
            else:
                raise ValueError(
                    f"generation_type has to be one of "
                    f"{'| ' + ' | '.join(list(GENERATION_TYPES.keys())) + ' |'}."
                )

                
            if image is not None:
                image_latent, image_embs = self._encode_image(image)
                N = len(image_embs)
                text_embs = None
            if text is not None:
                text_latent, text_embs = self._encode_text(text)
                N = len(text_embs)
                image_embs = None
            seq = None
            if seq is None:
                seq = torch.ones((N, 1), device=device, dtype=torch.long) * sot_token_id

            was_training = self.training
            num_dims = len(text.shape)

            if num_dims == 1:
                text = text[None, :]

            cur_len = text.shape[1]
            self.eval()
            out = seq

            while True:
                x = out[:, -max_seq_len:]
                cur_len = x.shape[1]
                #logits = self(image, x, image_latent=image_latent, image_embs=image_embs)["logits"][:, -1]
                #print(out.min(), out.max())
                if text_embs is not None:
                    logits = self.image_decoder(text_embs, out)[:, -1]
                else:
                    logits = self.text_decoder(image_embs, out)[:,-1]

                mask = (out[:, -1] == eos_token_id) | (out[:, -1] == pad_token_id)
                sample = torch.ones((out.shape[0], 1), device=device, dtype=torch.long) * pad_token_id

                if mask.all():
                    if not fixed_output_length:
                        break
                else:

                    logits = logits[~mask, :]
                    filtered_logits = logit_processor(x[~mask, :], logits)
                    filtered_logits = logit_warper(x[~mask, :], filtered_logits)
                    probs = F.softmax(filtered_logits / temperature, dim=-1)
                    #probs = F.softmax(logits / temperature, dim=-1)
                    if (cur_len + 1 == seq_len):
                        sample[~mask, :] = torch.ones((sum(~mask), 1), device=device, dtype=torch.long) * eos_token_id
                    else:
                        sample[~mask, :] = torch.multinomial(probs, 1)

                out = torch.cat((out, sample), dim=-1)
                cur_len += 1
                if stopping_criteria(out, None):
                    break

            if num_dims == 1:
                out = out.squeeze(0)

            self.train(was_training)
            return out

    def score(self, logits, texts):
        if logits.shape == texts.shape:
            return -nn.functional.mse_loss(logits, texts)
        else:
            # logits: (I, L, V)
            # texts: (I, L)
            I, L, V = logits.shape
            I2, L = texts.shape
            assert I == I2
            lp = logits.log_softmax(dim=-1)
            texts = texts.view(I, L, 1)
            lp = torch.gather(lp, 2, texts)
            lp[texts == 0] = 0
            ce = lp.sum(dim=(1,2))
            return ce

    def _generate_beamsearch(
            self,
            image_inputs,
            pad_token_id=None,
            eos_token_id=None,
            sot_token_id=None,
            num_beams=6,
            num_beam_groups=3,
            min_seq_len=5,
            stopping_criteria=None,
            logit_processor=None,
            logit_warper=None,
    ):
        device = image_inputs.device
        batch_size = image_inputs.shape[0]
        image_inputs = torch.repeat_interleave(image_inputs, num_beams, dim=0)
        image_latent, image_embs = self._encode_image(image_inputs)

        input_ids = torch.ones((batch_size * num_beams, 1), device=device, dtype=torch.long)
        input_ids = input_ids * sot_token_id
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=device,
            num_beam_groups=num_beam_groups,
        )
        # instantiate logits processors
        logits_processor = (
            LogitsProcessorList([MinLengthLogitsProcessor(min_seq_len, eos_token_id=eos_token_id)])
            if logit_processor is None
            else logit_processor
        )

        num_beams = beam_scorer.num_beams
        num_beam_groups = beam_scorer.num_beam_groups
        num_sub_beams = num_beams // num_beam_groups
        batch_size = len(beam_scorer._beam_hyps) // num_beam_groups
        batch_beam_size, cur_len = input_ids.shape
        beam_indices = None

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        beam_scores = torch.full((batch_size, num_beams), -1e9, dtype=torch.float, device=device)
        # initialise score of first beam of each group with 0 and the rest with 1e-9. This ensures that the beams in
        # the same group don't produce same tokens everytime.
        beam_scores[:, ::num_sub_beams] = 0
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while True:

            # predicted tokens in cur_len step
            current_tokens = torch.zeros(batch_size * num_beams, dtype=input_ids.dtype, device=device)

            # indices which will form the beams in the next time step
            reordering_indices = torch.zeros(batch_size * num_beams, dtype=torch.long, device=device)

            # do one decoder step on all beams of all sentences in batch
            model_inputs = prepare_inputs_for_generation(input_ids=input_ids, image_inputs=image_inputs)
            outputs = self(
                model_inputs['images'],
                model_inputs['text'],
                image_latent=image_latent,
                image_embs=image_embs
            )

            for beam_group_idx in range(num_beam_groups):
                group_start_idx = beam_group_idx * num_sub_beams
                group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
                group_size = group_end_idx - group_start_idx

                # indices of beams of current group among all sentences in batch
                batch_group_indices = []

                for batch_idx in range(batch_size):
                    batch_group_indices.extend(
                        [batch_idx * num_beams + idx for idx in range(group_start_idx, group_end_idx)]
                    )
                group_input_ids = input_ids[batch_group_indices]

                # select outputs of beams of currentg group only
                next_token_logits = outputs['logits'][batch_group_indices, -1, :]
                vocab_size = next_token_logits.shape[-1]

                next_token_scores_processed = logits_processor(
                    group_input_ids, next_token_logits, current_tokens=current_tokens, beam_group_idx=beam_group_idx
                )
                next_token_scores = next_token_scores_processed + beam_scores[batch_group_indices].unsqueeze(-1)
                next_token_scores = next_token_scores.expand_as(next_token_scores_processed)

                # reshape for beam search
                next_token_scores = next_token_scores.view(batch_size, group_size * vocab_size)

                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, 2 * group_size, dim=1, largest=True, sorted=True
                )

                next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
                next_tokens = next_tokens % vocab_size

                # stateless
                process_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
                beam_outputs = beam_scorer.process(
                    group_input_ids,
                    next_token_scores,
                    next_tokens,
                    next_indices,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    beam_indices=process_beam_indices,
                    group_index=beam_group_idx,
                )
                beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                input_ids[batch_group_indices] = group_input_ids[beam_idx]
                group_input_ids = torch.cat([group_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
                current_tokens[batch_group_indices] = group_input_ids[:, -1]

                # (beam_idx // group_size) -> batch_idx
                # (beam_idx % group_size) -> offset of idx inside the group
                reordering_indices[batch_group_indices] = (
                    num_beams * torch.div(beam_idx, group_size, rounding_mode="floor") + group_start_idx + (beam_idx % group_size)
                )

            input_ids = torch.cat([input_ids, current_tokens.unsqueeze(-1)], dim=-1)

            # increase cur_len
            cur_len = cur_len + 1
            if beam_scorer.is_done or stopping_criteria(input_ids, None):
                break

        final_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=final_beam_indices,
        )
        return sequence_outputs['sequences']


class SymGenDecoder(nn.Module):
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

    def _encode_image(self, images, normalize: bool = True):
        with torch.no_grad():
            images = (images * self.image_std + self.image_mean) if self.image_tokenizer.needs_0_1 else images
            image_tokens = self.image_tokenizer.tokenize(images)
        image_latent, tokens_embs = self.visual(image_tokens)
        image_latent = F.normalize(image_latent, dim=-1) if normalize else image_latent
        return image_latent, tokens_embs, image_tokens

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

        image_latent, image_embs, image_tokens = self._encode_image(image)
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