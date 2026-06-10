from __future__ import annotations

import torch


def rotation_to_vector(theta: torch.Tensor) -> torch.Tensor:
    """Convert angles to interleaved sin/cos representation.

    Args:
        theta: ``(B,)`` or ``(B, K)`` tensor of angles in radians.

    Returns:
        ``(B, 2)`` when input is ``(B,)`` or ``(B, 2*K)`` when input is
        ``(B, K)``, with pairs ``[sin_0, cos_0, sin_1, cos_1, ...]``.
    """
    if theta.dim() == 1:
        return torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)
    # (B, K) -> interleaved (B, 2*K)
    sin_vals = torch.sin(theta)  # (B, K)
    cos_vals = torch.cos(theta)  # (B, K)
    return torch.stack([sin_vals, cos_vals], dim=-1).reshape(theta.shape[0], -1)


def rotation_mse(pred: torch.Tensor, target_theta: torch.Tensor) -> torch.Tensor:
    if target_theta.dim() == 1:
        target_theta = target_theta.unsqueeze(-1)
        
    K = target_theta.shape[-1]
    pred = pred.reshape(-1, K, 2)
    
    # Calculate angle from predicted sin/cos
    pred_angle = torch.atan2(pred[..., 0], pred[..., 1])
    
    # Calculate true angle from target
    true_angle = torch.atan2(torch.sin(target_theta), torch.cos(target_theta))
    
    # Standard Circular MSE
    mse_standard = 1 - torch.cos(pred_angle - true_angle)
    
    # 180-degree flipped Circular MSE (for symmetric objects)
    mse_flipped = 1 - torch.cos(pred_angle - (true_angle + torch.pi))
    
    # Take the minimum error per axis (ignores 180 degree symmetry flips)
    min_mse = torch.minimum(mse_standard, mse_flipped)
    
    return min_mse.mean()


def angular_error(pred: torch.Tensor, target_theta: torch.Tensor) -> torch.Tensor:
    if target_theta.dim() == 1:
        target_theta = target_theta.unsqueeze(-1)
        
    K = target_theta.shape[-1]
    pred = pred.reshape(-1, K, 2)
    
    pred_angle = torch.atan2(pred[..., 0], pred[..., 1])
    true_angle = torch.atan2(torch.sin(target_theta), torch.cos(target_theta))
    
    # Standard MAE
    mae_standard = torch.abs(torch.atan2(
        torch.sin(pred_angle - true_angle),
        torch.cos(pred_angle - true_angle)
    ))
    
    # 180-degree flipped MAE
    mae_flipped = torch.abs(torch.atan2(
        torch.sin(pred_angle - (true_angle + torch.pi)),
        torch.cos(pred_angle - (true_angle + torch.pi))
    ))
    
    # Take the minimum error per axis
    min_mae = torch.minimum(mae_standard, mae_flipped)
    
    return min_mae.mean()
