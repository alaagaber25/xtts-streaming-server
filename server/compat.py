import importlib
import importlib.metadata as importlib_metadata
import inspect
import os

import torch
import transformers
from transformers.generation.logits_process import (
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    LogitsProcessorList,
    MinPLogitsWarper,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
)
from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel


_PATCHED = False


def _installed_package_version(package_name: str) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _torch_build_suffix() -> str | None:
    torch_version = getattr(torch, "__version__", "")
    if "+" not in torch_version:
        return None
    return torch_version.split("+", 1)[1]


def _torchaudio_repair_command() -> str:
    torch_version = getattr(torch, "__version__", "unknown")
    torch_release = torch_version.split("+", 1)[0]
    build_suffix = _torch_build_suffix()
    if build_suffix:
        index_url = f"https://download.pytorch.org/whl/{build_suffix}"
        return (
            "python -m pip install --force-reinstall --no-deps "
            f"--index-url {index_url} torchaudio=={torch_release}+{build_suffix}"
        )
    return f"python -m pip install --force-reinstall --no-deps torchaudio=={torch_release}"


def ensure_torchaudio_compatibility() -> None:
    torch_version = getattr(torch, "__version__", "unknown")
    torch_release = torch_version.split("+", 1)[0]
    torchaudio_version = _installed_package_version("torchaudio")

    if torchaudio_version is None:
        raise RuntimeError(
            "torchaudio is not installed. XTTS needs a torchaudio build that matches "
            f"torch=={torch_version}.\n"
            f"Repair with:\n  {_torchaudio_repair_command()}"
        )

    torchaudio_release = torchaudio_version.split("+", 1)[0]
    if torchaudio_release != torch_release:
        raise RuntimeError(
            "Installed torch and torchaudio versions do not match. XTTS imports torchaudio's "
            "native extension at startup, and mismatched wheels commonly fail with loader errors "
            f"such as WinError 127.\nInstalled torch=={torch_version}\n"
            f"Installed torchaudio=={torchaudio_version}\n"
            f"Repair with:\n  {_torchaudio_repair_command()}"
        )

    try:
        importlib.import_module("torchaudio")
    except Exception as exc:
        platform_hint = ""
        if os.name == "nt":
            platform_hint = "\nWindows virtualenv installs are especially sensitive here; Docker on Linux/WSL is more reliable."
        raise RuntimeError(
            "torchaudio is installed but failed to import its native extension. This usually means "
            "the torchaudio wheel was built for a different torch/CUDA runtime or a required DLL is missing.\n"
            f"Installed torch=={torch_version}\n"
            f"Installed torchaudio=={torchaudio_version}\n"
            f"Repair with:\n  {_torchaudio_repair_command()}{platform_hint}"
        ) from exc


def _patch_transformers_compat() -> None:
    if hasattr(transformers, "BeamSearchScorer"):
        return

    from transformers.generation.beam_constraints import (
        DisjunctiveConstraint,
        PhrasalConstraint,
    )
    from transformers.generation.beam_search import (
        BeamSearchScorer,
        ConstrainedBeamSearchScorer,
    )
    from transformers.generation.utils import GenerationMixin as ImportedGenerationMixin

    transformers.BeamSearchScorer = BeamSearchScorer
    transformers.ConstrainedBeamSearchScorer = ConstrainedBeamSearchScorer
    transformers.DisjunctiveConstraint = DisjunctiveConstraint
    transformers.PhrasalConstraint = PhrasalConstraint
    transformers.GenerationMixin = ImportedGenerationMixin


def _patch_torch_compat() -> None:
    try:
        torch.isin(elements=torch.tensor([0]), test_elements=0)
        return
    except TypeError:
        original_isin = torch.isin

        def _normalize_isin_operand(value):
            if torch.is_tensor(value):
                return value
            if isinstance(value, set):
                return torch.as_tensor(list(value))
            if isinstance(value, (list, tuple)):
                return torch.as_tensor(value)
            return value

        def _isin_compat(*args, **kwargs):
            remaining_args = args
            if "elements" in kwargs or "test_elements" in kwargs:
                elements = kwargs.pop("elements")
                test_elements = kwargs.pop("test_elements")
            else:
                elements, test_elements = remaining_args[:2]
                remaining_args = remaining_args[2:]

            elements = _normalize_isin_operand(elements)
            test_elements = _normalize_isin_operand(test_elements)
            if not torch.is_tensor(elements) and not torch.is_tensor(test_elements):
                return torch.tensor([elements == test_elements], dtype=torch.bool)

            return original_isin(elements, test_elements, *remaining_args, **kwargs)

        torch.isin = _isin_compat


def _patch_generation_mixin_compat() -> None:
    helper_names = [
        "_expand_inputs_for_generation",
        "_get_logits_processor",
        "_get_stopping_criteria",
        "_prepare_model_inputs",
        "_update_model_kwargs_for_generation",
        "_validate_model_kwargs",
    ]

    for helper_name in helper_names:
        if not hasattr(PreTrainedModel, helper_name) and hasattr(GenerationMixin, helper_name):
            setattr(PreTrainedModel, helper_name, getattr(GenerationMixin, helper_name))

    if not hasattr(PreTrainedModel, "_validate_model_class"):

        def _validate_model_class(self) -> None:
            return None

        PreTrainedModel._validate_model_class = _validate_model_class

    def _ensure_generation_config_tensors(model, generation_config, device=None):
        if generation_config is None:
            return

        target_device = device
        if target_device is None:
            try:
                target_device = model.device
            except (AttributeError, StopIteration):
                target_device = torch.device("cpu")

        def _tensor_or_none(token):
            if token is None:
                return None
            if torch.is_tensor(token):
                return token.to(target_device)
            return torch.tensor(token, device=target_device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(getattr(generation_config, "bos_token_id", None))
        eos_token_tensor = _tensor_or_none(getattr(generation_config, "eos_token_id", None))
        pad_token_tensor = _tensor_or_none(getattr(generation_config, "pad_token_id", None))
        decoder_start_token_tensor = _tensor_or_none(
            getattr(generation_config, "decoder_start_token_id", None)
        )

        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]

        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._decoder_start_token_tensor = decoder_start_token_tensor

    attention_mask_helper = getattr(PreTrainedModel, "_prepare_attention_mask_for_generation", None)
    needs_attention_mask_patch = attention_mask_helper is None
    if attention_mask_helper is not None:
        parameter_names = list(inspect.signature(attention_mask_helper).parameters)
        legacy_signature = len(parameter_names) >= 4 and parameter_names[2:4] == [
            "pad_token_id",
            "eos_token_id",
        ]
        needs_attention_mask_patch = not legacy_signature

    if needs_attention_mask_patch:

        def _prepare_attention_mask_for_generation(self, inputs_tensor, arg2=None, arg3=None):
            if hasattr(arg2, "_pad_token_tensor"):
                return GenerationMixin._prepare_attention_mask_for_generation(
                    self,
                    inputs_tensor,
                    arg2,
                    arg3,
                )

            pad_token_id = arg2
            eos_token_id = arg3
            if isinstance(pad_token_id, torch.Tensor) and pad_token_id.numel() == 1:
                pad_token_id = pad_token_id.item()
            if isinstance(eos_token_id, torch.Tensor):
                if eos_token_id.numel() == 1:
                    eos_token_id = eos_token_id.item()
                else:
                    eos_token_id = eos_token_id.flatten().tolist()

            is_input_ids = len(inputs_tensor.shape) == 2 and inputs_tensor.dtype in [torch.int, torch.long]
            if not is_input_ids:
                return torch.ones(inputs_tensor.shape[:2], dtype=torch.long, device=inputs_tensor.device)

            is_pad_token_in_inputs = pad_token_id is not None and inputs_tensor.eq(pad_token_id).any().item()
            if eos_token_id is None:
                is_pad_token_not_equal_to_eos_token_id = True
            elif isinstance(eos_token_id, (list, tuple, set)):
                is_pad_token_not_equal_to_eos_token_id = pad_token_id not in eos_token_id
            else:
                is_pad_token_not_equal_to_eos_token_id = pad_token_id != eos_token_id

            can_infer_attention_mask = is_pad_token_in_inputs and is_pad_token_not_equal_to_eos_token_id
            attention_mask_from_padding = inputs_tensor.ne(pad_token_id).long()

            if can_infer_attention_mask:
                return attention_mask_from_padding

            return torch.ones(inputs_tensor.shape[:2], dtype=torch.long, device=inputs_tensor.device)

        PreTrainedModel._prepare_attention_mask_for_generation = _prepare_attention_mask_for_generation

    if not hasattr(PreTrainedModel, "_get_logits_warper"):

        def _get_logits_warper(self, generation_config):
            warpers = LogitsProcessorList()
            min_tokens_to_keep = 2 if getattr(generation_config, "num_beams", 1) > 1 else 1

            temperature = getattr(generation_config, "temperature", None)
            if temperature is not None and temperature != 1.0:
                warpers.append(TemperatureLogitsWarper(temperature))

            top_k = getattr(generation_config, "top_k", None)
            if top_k is not None and top_k != 0:
                warpers.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=min_tokens_to_keep))

            top_p = getattr(generation_config, "top_p", None)
            if top_p is not None and top_p < 1.0:
                warpers.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=min_tokens_to_keep))

            typical_p = getattr(generation_config, "typical_p", None)
            if typical_p is not None and typical_p < 1.0:
                warpers.append(TypicalLogitsWarper(mass=typical_p, min_tokens_to_keep=min_tokens_to_keep))

            epsilon_cutoff = getattr(generation_config, "epsilon_cutoff", None)
            if epsilon_cutoff is not None and 0.0 < epsilon_cutoff < 1.0:
                warpers.append(
                    EpsilonLogitsWarper(epsilon=epsilon_cutoff, min_tokens_to_keep=min_tokens_to_keep)
                )

            eta_cutoff = getattr(generation_config, "eta_cutoff", None)
            if eta_cutoff is not None and 0.0 < eta_cutoff < 1.0:
                warpers.append(EtaLogitsWarper(epsilon=eta_cutoff, min_tokens_to_keep=min_tokens_to_keep))

            min_p = getattr(generation_config, "min_p", None)
            if min_p is not None and 0.0 < min_p:
                warpers.append(MinPLogitsWarper(min_p=min_p, min_tokens_to_keep=min_tokens_to_keep))

            return warpers

        PreTrainedModel._get_logits_warper = _get_logits_warper

    original_get_logits_warper = getattr(PreTrainedModel, "_get_logits_warper", None)
    if original_get_logits_warper is not None and not getattr(
        original_get_logits_warper,
        "_xtts_generation_config_compat",
        False,
    ):
        logits_warper_params = list(inspect.signature(original_get_logits_warper).parameters)
        logits_warper_accepts_device = len(logits_warper_params) >= 3 and logits_warper_params[2] == "device"

        def _get_logits_warper_compat(self, generation_config, device=None):
            resolved_device = device if device is not None else getattr(self, "device", torch.device("cpu"))
            _ensure_generation_config_tensors(self, generation_config, device=resolved_device)
            if logits_warper_accepts_device:
                return original_get_logits_warper(self, generation_config, resolved_device)
            return original_get_logits_warper(self, generation_config)

        _get_logits_warper_compat._xtts_generation_config_compat = True
        PreTrainedModel._get_logits_warper = _get_logits_warper_compat

    original_get_logits_processor = getattr(PreTrainedModel, "_get_logits_processor", None)
    if original_get_logits_processor is not None and not getattr(
        original_get_logits_processor,
        "_xtts_generation_config_compat",
        False,
    ):

        def _get_logits_processor_compat(
            self,
            generation_config,
            input_ids_seq_length,
            encoder_input_ids,
            prefix_allowed_tokens_fn,
            logits_processor,
            device=None,
            model_kwargs=None,
            negative_prompt_ids=None,
            negative_prompt_attention_mask=None,
        ):
            _ensure_generation_config_tensors(self, generation_config, device=device)
            return original_get_logits_processor(
                self,
                generation_config,
                input_ids_seq_length,
                encoder_input_ids,
                prefix_allowed_tokens_fn,
                logits_processor,
                device=device,
                model_kwargs=model_kwargs,
                negative_prompt_ids=negative_prompt_ids,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
            )

        _get_logits_processor_compat._xtts_generation_config_compat = True
        PreTrainedModel._get_logits_processor = _get_logits_processor_compat

    original_get_stopping_criteria = getattr(PreTrainedModel, "_get_stopping_criteria", None)
    if original_get_stopping_criteria is not None and not getattr(
        original_get_stopping_criteria,
        "_xtts_generation_config_compat",
        False,
    ):

        def _get_stopping_criteria_compat(
            self,
            generation_config,
            stopping_criteria=None,
            tokenizer=None,
            **kwargs,
        ):
            _ensure_generation_config_tensors(self, generation_config)
            return original_get_stopping_criteria(
                self,
                generation_config,
                stopping_criteria=stopping_criteria,
                tokenizer=tokenizer,
                **kwargs,
            )

        _get_stopping_criteria_compat._xtts_generation_config_compat = True
        PreTrainedModel._get_stopping_criteria = _get_stopping_criteria_compat


def apply_runtime_compatibility_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return

    _patch_transformers_compat()
    _patch_torch_compat()
    _patch_generation_mixin_compat()
    _PATCHED = True
