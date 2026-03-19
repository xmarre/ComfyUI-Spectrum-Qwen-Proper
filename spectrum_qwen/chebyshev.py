from __future__ import annotations

import torch



def _normalize_sigmas(sigmas: torch.Tensor) -> torch.Tensor:
    if sigmas.numel() == 1:
        return torch.zeros_like(sigmas, dtype=torch.float32)
    sigma_min = torch.min(sigmas)
    sigma_max = torch.max(sigmas)
    denom = sigma_max - sigma_min
    if torch.abs(denom) < 1e-12:
        return torch.zeros_like(sigmas, dtype=torch.float32)
    normed = (sigmas - sigma_min) / denom
    return normed.to(dtype=torch.float32).mul(2.0).sub(1.0)



def chebyshev_basis(x: torch.Tensor, degree: int) -> torch.Tensor:
    x = x.to(dtype=torch.float32)
    basis = [torch.ones_like(x), x]
    if degree == 0:
        return basis[0].unsqueeze(-1)
    for _ in range(2, degree + 1):
        basis.append(2.0 * x * basis[-1] - basis[-2])
    return torch.stack(basis[: degree + 1], dim=-1)



def forecast_feature(
    history_sigmas: list[float],
    history_features: list[torch.Tensor],
    target_sigma: float,
    degree: int,
    ridge_lambda: float,
) -> torch.Tensor:
    if len(history_sigmas) != len(history_features):
        raise ValueError("history_sigmas and history_features length mismatch")
    if len(history_sigmas) < degree + 1:
        raise ValueError("not enough history points for requested Chebyshev degree")

    device = history_features[0].device
    feature_dtype = history_features[0].dtype
    feature_shape = history_features[0].shape

    sigma_tensor = torch.tensor(history_sigmas + [float(target_sigma)], device=device, dtype=torch.float32)
    x_all = _normalize_sigmas(sigma_tensor)
    x_hist = x_all[:-1]
    x_target = x_all[-1:]

    phi_hist = chebyshev_basis(x_hist, degree)
    phi_target = chebyshev_basis(x_target, degree)[0]

    y = torch.stack([feature.reshape(-1) for feature in history_features], dim=0).to(torch.float32)

    gram = phi_hist.transpose(0, 1) @ phi_hist
    if ridge_lambda > 0:
        gram = gram + torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype) * float(ridge_lambda)
    rhs = phi_hist.transpose(0, 1) @ y
    coeffs = torch.linalg.solve(gram, rhs)
    pred = phi_target @ coeffs
    return pred.reshape(feature_shape).to(dtype=feature_dtype)
