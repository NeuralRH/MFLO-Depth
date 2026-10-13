"""Flying Point Localization Module from MFLO-Depth."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def normalize_response(response, eps=1e-8):
    response = np.asarray(response, dtype=np.float32)
    finite = np.isfinite(response)
    out = np.zeros_like(response)
    if finite.any():
        low = response[finite].min()
        high = response[finite].max()
        out[finite] = (response[finite] - low) / (high - low + eps)
    return out


def depth_to_points(depth, K):
    height, width = depth.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    pixels = np.stack((xx, yy, np.ones_like(xx)), axis=-1)
    rays = pixels @ np.linalg.inv(K[:3, :3]).T
    return rays * depth[..., None]


def depth_gradient_response(depth):
    gx = cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return gx * gx + gy * gy


def normal_variation_response(points, eps=1e-8):
    right_points = np.concatenate((points[:, 1:], points[:, -1:]), axis=1)
    down_points = np.concatenate((points[1:], points[-1:]), axis=0)
    right = right_points - points
    down = down_points - points
    normals = np.cross(right, down)
    normals /= np.maximum(np.linalg.norm(normals, axis=2, keepdims=True), eps)
    normal_right = np.concatenate((normals[:, 1:], normals[:, -1:]), axis=1)
    normal_down = np.concatenate((normals[1:], normals[-1:]), axis=0)
    dot_right = np.sum(normals * normal_right, axis=2)
    dot_down = np.sum(normals * normal_down, axis=2)
    return np.maximum(1.0 - dot_right, 1.0 - dot_down).astype(np.float32)


def plane_residual_response(points, window_size=5, eps=1e-8):
    """Distance of each center point to the local least-squares plane."""
    height, width, _ = points.shape
    pad = window_size // 2
    padded = np.pad(points, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    windows = sliding_window_view(padded, (window_size, window_size), axis=(0, 1))
    windows = windows.transpose(0, 1, 3, 4, 2).reshape(height, width, -1, 3)
    centered = windows - windows.mean(axis=2, keepdims=True)
    covariance = np.einsum("...ni,...nj->...ij", centered, centered, optimize=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[..., 0]
    residual = np.abs(np.sum(normals * (points - windows.mean(axis=2)), axis=-1))
    return (residual / np.maximum(np.linalg.norm(normals, axis=-1), eps)).astype(np.float32)


def flying_point_map(depth, K, alpha=0.4, beta=0.2, gamma=0.4,
                     mad_scale=3.0, plane_window=5):
    """Return the MFLO-Depth flying-point mask and diagnostic cue maps."""
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError("depth must be a 2D array")
    valid_depth = np.isfinite(depth) & (depth > 0)
    safe_depth = np.where(valid_depth, depth, 0.0)
    points = depth_to_points(safe_depth, np.asarray(K, dtype=np.float32))

    gd = normalize_response(depth_gradient_response(safe_depth))
    gn = normalize_response(normal_variation_response(points))
    rp = normalize_response(plane_residual_response(points, plane_window))
    response = alpha * gd + beta * gn + gamma * rp

    valid_response = response[valid_depth & np.isfinite(response)]
    if valid_response.size:
        median = np.median(valid_response)
        mad = np.median(np.abs(valid_response - median))
        threshold = median + mad_scale * 1.4826 * mad
    else:
        threshold = np.inf
    mask = (response > threshold) & valid_depth
    return mask, {"Gd": gd, "Gn": gn, "Rp": rp, "C": response,
                  "threshold": float(threshold)}


def _load_depth(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path).squeeze().astype(np.float32)
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        for key in ("depth", "data", "arr_0"):
            if key in data:
                return data[key].squeeze().astype(np.float32)
        raise ValueError("NPZ file must contain a depth, data, or arr_0 array")
    raise ValueError("FPLM input must be .npy or .npz depth")


def main():
    parser = argparse.ArgumentParser(description="Compute MFLO-Depth FPLM flying-point masks")
    parser.add_argument("depth", help="predicted depth .npy or .npz")
    parser.add_argument("--intrinsics", nargs=4, type=float, required=True,
                        metavar=("FX", "FY", "CX", "CY"),
                        help="camera intrinsics at the depth map resolution")
    parser.add_argument("--output", default="fplm_output")
    parser.add_argument("--mad-scale", type=float, default=3.0,
                        help="k in median + k*1.4826*MAD")
    args = parser.parse_args()

    

    depth = _load_depth(args.depth)
    fx, fy, cx, cy = args.intrinsics
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    mask, cues = flying_point_map(depth, K, mad_scale=args.mad_scale)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "flying_points.npy", mask)
    for name in ("Gd", "Gn", "Rp", "C"):
        cv2.imwrite(str(output / f"{name}.png"),
                    np.clip(cues[name] * 255, 0, 255).astype(np.uint8))
    print(f"Removed candidates: {int(mask.sum())}/{mask.size}; threshold={cues['threshold']:.6g}")
    print(f"Saved FPLM maps under {output.resolve()}")


if __name__ == "__main__":
    main()
