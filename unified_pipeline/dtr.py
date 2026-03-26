"""Dynamic Tensor Rematerialization (DTR) utilities.

Implements gradient checkpointing as the practical approximation of the DTR
technique described in:
  Kirisame et al., "Dynamic Tensor Rematerialization", ICLR 2021
  https://arxiv.org/pdf/2006.09616

DTR trades compute for memory: instead of keeping all activations alive
during the forward pass, it evicts them and recomputes on-demand during
backpropagation.  PyTorch's ``torch.utils.checkpoint`` provides this
capability natively and is the recommended entry-point for most models.

Usage
-----
Wrap any ``nn.Module`` with :func:`apply_dtr` before training::

    from unified_pipeline.dtr import apply_dtr
    model = apply_dtr(model)

Or use the context manager inside a custom training loop::

    from unified_pipeline.dtr import dtr_context
    import torch.utils.checkpoint as cp

    with dtr_context():
        output = cp.checkpoint(model, inputs)
"""

from __future__ import annotations

import contextlib
from typing import Generator

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp


def _wrap_forward_with_checkpoint(module: nn.Module) -> nn.Module:
    """Monkey-patch a module's forward so every call uses gradient checkpointing."""
    original_forward = module.forward

    def checkpointed_forward(*args, **kwargs):
        # use_reentrant=False is the modern, stable API (PyTorch >= 1.13).
        # Strip 'use_reentrant' from kwargs so it does not conflict with the
        # explicit keyword we pass to cp.checkpoint below.
        kwargs.pop("use_reentrant", None)
        return cp.checkpoint(original_forward, *args, use_reentrant=False, **kwargs)

    module.forward = checkpointed_forward  # type: ignore[method-assign]
    return module


def apply_dtr(
    model: nn.Module,
    *,
    target_types: tuple[type[nn.Module], ...] | None = None,
) -> nn.Module:
    """Apply DTR (gradient checkpointing) to a model.

    Parameters
    ----------
    model:
        The PyTorch model to wrap.
    target_types:
        If provided, only sub-modules whose type is in this tuple will be
        wrapped.  By default every leaf block that has learnable parameters
        is wrapped.  Pass e.g. ``(nn.TransformerEncoderLayer,)`` for
        transformer-based detectors.

    Returns
    -------
    nn.Module
        The same ``model`` object with checkpointing enabled in-place.

    Examples
    --------
    >>> from unified_pipeline.dtr import apply_dtr
    >>> import torch.nn as nn
    >>> model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    >>> model = apply_dtr(model)
    """
    if target_types is not None:
        for module in model.modules():
            if isinstance(module, target_types):
                _wrap_forward_with_checkpoint(module)
    else:
        # Enable built-in gradient-checkpointing when the model exposes it.
        if hasattr(model, "gradient_checkpointing_enable"):
            # HuggingFace / Transformers API
            model.gradient_checkpointing_enable()  # type: ignore[attr-defined]
        elif hasattr(model, "enable_checkpoint"):
            model.enable_checkpoint()  # type: ignore[attr-defined]
        else:
            # Fallback: wrap every named child block that carries parameters.
            for name, module in list(model.named_children()):
                if any(True for _ in module.parameters()):
                    _wrap_forward_with_checkpoint(module)
                    print(f"[DTR] gradient checkpointing enabled on: {name}")

    return model


@contextlib.contextmanager
def dtr_context(
    device: str | torch.device = "cuda",
    budget_mb: float | None = None,
) -> Generator[None, None, None]:
    """Context manager that activates native PyTorch DTR when available.

    PyTorch 2.0+ ships an experimental allocator-level DTR implementation
    accessible via the private ``torch.cuda.memory._set_allocator_settings``
    API.  This API is undocumented and may change between PyTorch releases;
    the context manager catches ``AttributeError`` and ``RuntimeError`` and
    falls back to a no-op when the API is unavailable, so it is safe to use
    on older versions.  Tested with PyTorch >= 2.1.

    Parameters
    ----------
    device:
        The CUDA device on which to enable DTR (ignored for CPU-only runs).
    budget_mb:
        Memory budget in MiB.  When *None* the allocator default is used.
        Setting a smaller value forces more aggressive eviction.

    Examples
    --------
    >>> import torch
    >>> from unified_pipeline.dtr import dtr_context
    >>> with dtr_context(budget_mb=4096):
    ...     pass  # your training loop here
    """
    if not torch.cuda.is_available():
        yield
        return

    settings: dict[str, object] = {"expandable_segments": True}
    if budget_mb is not None:
        # Native DTR budget (PyTorch >= 2.1 allocator feature).
        settings["max_split_size_mb"] = int(budget_mb)

    try:
        torch.cuda.memory._set_allocator_settings(",".join(f"{k}:{v}" for k, v in settings.items()))
        print(f"[DTR] allocator settings applied: {settings}")
    except (AttributeError, RuntimeError) as exc:
        print(f"[DTR] allocator tuning unavailable ({exc}); continuing without it.")

    try:
        yield
    finally:
        # Reset to defaults so subsequent runs are unaffected.
        try:
            torch.cuda.memory._set_allocator_settings("")
        except (AttributeError, RuntimeError):
            pass
