# MFLO-Depth

This project implements **MFLO-Depth: Self-Supervised Monocular Depth Estimation via Multi-Frame Loop Optimization**, using the Lite-Mono encoder and the Monodepth2 self-supervised training pipeline as its base.

## Method implemented

- Predict depth for each temporal source and the target image.
- Reproject target depth into each source, sample the source depth, then transform it back into the target view.
- Apply the Multi-view Geometric Verification (MGV) masks for in-bounds projections, positive depths, photometric validity, and an adaptive inverse-depth cycle threshold.
- Train for 60 epochs. The first 20 epochs use photometric reprojection and edge-aware disparity smoothing. After epoch 20, the inverse-depth cycle loss is introduced with a saturating exponential weight capped at 0.15 while the consistency percentile relaxes over time.
- Apply the Flying Point Localization Module (FPLM) only when exporting a point cloud. It fuses normalized depth-gradient, local-normal variation, and 5×5 plane-residual maps with weights 0.4, 0.2, and 0.4, then removes responses above a median/MAD threshold.

 Defaults are exposed as command-line options; the implementation uses percentiles 50→80, exponential rates 0.15, and MAD multiplier 3.0. 

## Environment

Install dependencies with `pip install -r requirements.txt`. Training the reported setup requires KITTI raw image sequences, an NVIDIA GPU, and the KITTI Eigen/Zhou split lists. Put KITTI under `kitti_data/` or pass its path with `--data_path`.

The paper reports 640×192 input, 60 epochs, batch size 12, AdamW weight decay 1e-2, starting learning rate 1e-4, and two RTX 3090 GPUs. Its 1024×320 result uses a separate higher-resolution run. A run on different hardware or training data will not reproduce the paper's reported benchmark figures automatically.

## Train on KITTI
python train.py --data_path path/to/kitti_data --model lite-mono-8m --height 192 --width 640 --num_epochs 60 --batch_size 12 --mypretrain 8m/lite-mono-8m-pretrain.pth


To train without the locally retained Lite-Mono ImageNet initialization, omit `--mypretrain`. Checkpoints and TensorBoard logs are written under `tmp/` by default.

To run the higher-resolution configuration, use `--height 320 --width 1024` and a batch size supported by the available GPU memory.

## Infer depth

python test_simple.py --image_path path/to/image_or_folder --load_weights_folder path/to/trained/models --model lite-mono-8m


## Export an FPLM-filtered point cloud


python point_cloud.py --image_path path/to/image_or_folder --weights path/to/trained/models --model lite-mono-8m --output point_cloud_output


The default camera calibration is KITTI camera 2 at 1242×375 and is scaled to the model input resolution. For other cameras, provide `--kitti_intrinsics FX FY CX CY --calibration_size WIDTH HEIGHT`. Point-cloud depth is the monocular model's predicted depth and therefore has the usual monocular scale ambiguity.

To generate the FPLM candidate mask for an existing depth array:
python FPLM.py depth.npy --intrinsics FX FY CX CY --output fplm_output


## Evaluate on KITTI
python evaluate_depth.py --data_path path/to/kitti_data --load_weights_folder path/to/trained/models --model lite-mono-8m




<img width="1036" height="267" alt="GITHUB" src="https://github.com/user-attachments/assets/3740cb4a-e1a7-4552-aa32-2f8e5dc1f808" />

## Acknowledgements

We sincerely thank the authors of the following excellent open-source projects, which inspired and supported this work:

- [Monodepth2](https://github.com/nianticlabs/monodepth2)
- [Lite-Mono](https://github.com/noahzn/Lite-Mono)
- [pixel-perfect-depth](https://github.com/gangweix/pixel-perfect-depth) 

Their outstanding work has greatly benefited our research



