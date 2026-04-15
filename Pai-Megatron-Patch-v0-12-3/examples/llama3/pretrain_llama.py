# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import torch
from typing import Union
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.core.enums import ModelType
from megatron.training import pretrain
from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.spec_utils import import_module
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)

from megatron_patch.data import train_valid_test_datasets_provider
from megatron_patch.arguments import get_patch_args
from megatron_patch.tokenizer import build_tokenizer
import torch._dynamo
torch._dynamo.config.suppress_errors = True

def model_provider(pre_process=True, post_process=True) -> Union[GPTModel]:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel]: The returned model
    """
    args = get_args()
    build_tokenizer(args)
    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    config = core_transformer_config_from_args(args)
    if not args.use_legacy_models:
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm)

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor
        )
    else:
        assert(args.context_parallel_size == 1), "Context parallelism is only supported with Megatron Core!"
        from megatron_patch.model.llama3.gpt_model import GPTModel as GPTModelX
        model = GPTModelX(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process
        )

    return model

if __name__ == "__main__":
    from megatron_patch.template.helper import forward_step
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             extra_args_provider=get_patch_args)
