# OASIS 2D Brain Segmentation with Improved U-Net (s4908819)

## Problem & Algorithm
This project solves 2D semantic segmentation on the OASIS brain dataset using an Improved U-Net/CAN variant. 
Target metric: **per-label Dice ≥ 0.9 on the test set** (Easy difficulty).

## How it works
We train a 2D encoder–decoder with skip connections and dilated/SE blocks (Improved U-Net). 
Loss = BCE + Dice; metrics include per-class and mean Dice. 
We plot losses/metrics during training and save best weights.

## Data (Rangpur paths)
- OASIS preprocessed: `/home/groups/comp3710/OASIS`  
(Do **not** commit any data or model files.)

## Environment & Reproducibility
- Python 3.10+, PyTorch 2.x, torchvision, numpy, matplotlib, nibabel (if using Nifti).
- Install: `pip install -r requirements.txt` (see versions in the file)
- Reproduce:
  ```bash
  # 1) Train
  python train.py --data_root /home/groups/comp3710/OASIS --epochs 20 --batch 16 --out runs/oasis_unet
  # 2) Predict (example visualization)
  python predict.py --weights runs/oasis_unet/best.pt --img example.png --out demo_pred.png
We fix random seeds where reasonable; minor nondeterminism from CuDNN may remain.

Preprocessing & Splits (justify briefly)
2D slices normalized per-image (z-score); masks kept as {0..K} integer labels and converted to one-hot for loss.

Split by subject (train/val/test) to avoid leakage; default 70/15/15.

Augmentations: random flips/crops (kept light to preserve anatomy).

Results (to fill after training)
Training curves (loss, mean Dice)

Test per-label Dice table (should be ≥0.9 each for Easy)

Example overlay images

References
Project spec & marking rubric (COMP3710 PatternAnalysis-2025).

Nifti reading examples for 2D/3D (Appendix B of the report).

