 
"""
utils.py — Shared PPO utilities

Includes:
    • masked_logits_softmax
    • normalize
    • entropy_from_logits
    • clip_value_function
    • to_device
"""

from __future__ import annotations
import torch
import torch.nn.functional as F


# ============================================================================
# Masked softmax for illegal actions
# ============================================================================

def masked_logits_softmax(logits: torch.Tensor,
                          legal_mask: torch.Tensor):
    """
    Apply a mask to logits before softmax.

    Args:
        logits: tensor of shape (N,) — raw logits for each action.
        legal_mask: bool tensor of shape (N,)
                    True  = legal action
                    False = illegal, must be excluded

    Returns:
        masked_logits: logits with illegal entries set to -1e9
        probs: softmax(masked_logits)
    """

    masked_logits = logits.masked_fill(~legal_mask, -1e9)

    # Standard softmax for categorical distribution
    probs = F.softmax(masked_logits, dim=0)

    # Ensure numerical stability: avoid all-zero probabilities
    if torch.isnan(probs).any() or probs.sum() == 0:
        # Assign uniform distribution over legal actions
        legal_count = legal_mask.sum().item()
        probs = torch.zeros_like(probs)
        probs[legal_mask] = 1.0 / legal_count

    return masked_logits, probs


# ============================================================================
# Advantage normalization
# ============================================================================

def normalize(x: torch.Tensor, eps: float = 1e-8):
    """
    Normalize tensor to zero mean, unit std.
    Used for stabilizing PPO training.
    """
    return (x - x.mean()) / (x.std() + eps)


# ============================================================================
# Entropy from logits
# ============================================================================

def entropy_from_logits(logits: torch.Tensor):
    """
    Compute categorical entropy directly from logits.
    Entropy = -sum(p * log(p))

    Args:
        logits: (N,)

    Returns:
        entropy: scalar tensor
    """
    log_probs = F.log_softmax(logits, dim=0)
    probs = log_probs.exp()
    return -(probs * log_probs).sum()


# ============================================================================
# Clipped value function (PPO variant)
# ============================================================================

def clip_value_function(values: torch.Tensor,
                        old_values: torch.Tensor,
                        clip_range: float = 0.2):
    """
    PPO value clipping variant:

        v_clipped = old_v + clip(v_new - old_v, -eps, eps)
        loss = max( (v_new - ret)^2 , (v_clipped - ret)^2 )

    This helper returns v_clipped.
    """
    diff = values - old_values
    diff_clipped = diff.clamp(-clip_range, clip_range)
    return old_values + diff_clipped


# ============================================================================
# Device helper
# ============================================================================

def to_device(batch, device):
    """
    Convenience function to move nested structures to a device.

    Supports:
        • torch.Tensors
        • lists of tensors
        • dict[str, tensors]

    Returns mirrored structure on the target device.
    """

    if torch.is_tensor(batch):
        return batch.to(device)

    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}

    if isinstance(batch, list):
        return [to_device(x, device) for x in batch]

    return batch  # fallback for unsupported types
