import torch
from transformers.generation.utils import GenerationMixin

_PATCH_APPLIED = False
_ORIGINAL_PREPARE_ATTENTION_MASK_FOR_GENERATION = (
    GenerationMixin._prepare_attention_mask_for_generation
)


def _patched_prepare_attention_mask_for_generation(
    self, inputs, pad_token_id=None, eos_token_id=None, *args, **kwargs
):
    if isinstance(inputs, torch.Tensor):
        device = inputs.device
        if pad_token_id is not None and not isinstance(pad_token_id, torch.Tensor):
            pad_token_id = torch.tensor([pad_token_id], device=device)
        if eos_token_id is not None and not isinstance(eos_token_id, torch.Tensor):
            eos_values = eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]
            eos_token_id = torch.tensor(eos_values, device=device)

    return _ORIGINAL_PREPARE_ATTENTION_MASK_FOR_GENERATION(
        self, inputs, pad_token_id, eos_token_id, *args, **kwargs
    )


def apply_generation_patches():
    global _PATCH_APPLIED

    if _PATCH_APPLIED:
        return

    GenerationMixin._prepare_attention_mask_for_generation = (
        _patched_prepare_attention_mask_for_generation
    )
    _PATCH_APPLIED = True


apply_generation_patches()

