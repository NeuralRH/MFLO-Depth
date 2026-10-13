"""Run Lite-Mono depth inference, remove flying points with FPLM, and export PLY."""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import networks
from FPLM import flying_point_map
from layers import disp_to_depth


def load_model(weights_dir, model_name, device):
    encoder_state = torch.load(os.path.join(weights_dir, "encoder.pth"), map_location="cpu")
    decoder_state = torch.load(os.path.join(weights_dir, "depth.pth"), map_location="cpu")
    height, width = int(encoder_state["height"]), int(encoder_state["width"])
    encoder = networks.LiteMono(model=model_name, height=height, width=width)
    encoder.load_state_dict({k: v for k, v in encoder_state.items() if k in encoder.state_dict()})
    decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
    decoder.load_state_dict({k: v for k, v in decoder_state.items() if k in decoder.state_dict()})
    return encoder.to(device).eval(), decoder.to(device).eval(), width, height


def write_ply(path, points, colors):
    valid_points = points.reshape(-1, 3)
    valid_colors = colors.reshape(-1, 3)
    with open(path, "w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(valid_points)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        rows = np.concatenate((valid_points, valid_colors.astype(np.float32)), axis=1)
        np.savetxt(stream, rows, fmt="%.6f %.6f %.6f %d %d %d")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_path", required=True, help="image file or directory")
    parser.add_argument("--weights", required=True, help="folder with encoder.pth and depth.pth")
    parser.add_argument("--output", default="point_cloud_output")
    parser.add_argument("--model", default="lite-mono-8m",
                        choices=["lite-mono", "lite-mono-small", "lite-mono-tiny", "lite-mono-8m"])
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=100.0)
    parser.add_argument("--kitti_intrinsics", nargs=4, type=float,
                        default=[721.5377, 721.5377, 609.5593, 172.8540],
                        metavar=("FX", "FY", "CX", "CY"),
                        help="camera intrinsics at the original calibration size")
    parser.add_argument("--calibration_size", nargs=2, type=int, default=[1242, 375],
                        metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--mad_scale", type=float, default=3.0,
                        help="FPLM threshold multiplier k in median + k*1.4826*MAD")
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    encoder, decoder, width, height = load_model(args.weights, args.model, device)
    paths = ([args.image_path] if os.path.isfile(args.image_path) else
             sorted(path for ext in ("jpg", "jpeg", "png", "bmp", "tif", "tiff")
                    for path in glob.glob(os.path.join(args.image_path, "**", f"*.{ext}"), recursive=True)))
    if not paths:
        raise FileNotFoundError(f"No input images found at {args.image_path}")
    os.makedirs(args.output, exist_ok=True)

    original_width, original_height = args.calibration_size
    fx, fy, cx, cy = args.kitti_intrinsics
    sx, sy = width / original_width, height / original_height
    K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]], dtype=np.float32)

    for image_path in paths:
        image = Image.open(image_path).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        rgb = np.asarray(image)
        tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
        with torch.no_grad():
            disp = decoder(encoder(tensor))[("disp", 0)]
            _, depth = disp_to_depth(disp, args.min_depth, args.max_depth)
        depth = depth.squeeze().cpu().numpy().astype(np.float32)

        flying, _ = flying_point_map(depth, K, mad_scale=args.mad_scale)
        points = np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=-1).astype(np.float32)
        pixels = np.concatenate((points, np.ones((height, width, 1), dtype=np.float32)), axis=-1)
        xyz = (pixels @ np.linalg.inv(K).T) * depth[..., None]
        keep = np.isfinite(depth) & (depth > 0) & ~flying
        out_path = os.path.join(args.output, os.path.splitext(os.path.basename(image_path))[0] + ".ply")
        write_ply(out_path, xyz[keep], rgb[keep])
        print(f"Saved {out_path} ({int(keep.sum())} points; removed {int(flying.sum())})")


if __name__ == "__main__":
    main()
