🧠 README.md — HipMRI 2D Prostate Segmentation (s4908819)
🧩 Project Overview

This project implements 2D prostate segmentation on the HipMRI dataset using an improved Residual U-Net (ResUNet) architecture.
The goal is to achieve a minimum Dice coefficient of 0.75 on the test prostate label.
Our final model achieved Dice = 0.9206 and IoU = 0.8588, surpassing the required threshold.

⚙️ 1. Method and Model Description
🧠 Network Architecture

Base model: Improved Residual U-Net (ResUNet), extending the classic U-Net by introducing:

Residual skip connections between encoder–decoder pairs.

Batch normalization after each convolution.

Dice + Cross-Entropy combined loss.

Input: 2D MRI slices (1 channel, 256×128).

Output: Binary mask (2 channels: background + prostate).

🔬 Key Features
Feature	Description
Residual blocks	Improve gradient flow and stability
Combined loss	Cross-Entropy + Dice for class balance
Nearest-neighbor mask resize	Preserve discrete label integrity
Binary prostate mapping	Use label = 2 as prostate class
📊 2. Dataset and Pre-Processing
Dataset

Source: HipMRI 2D pre-processed slices (.nii.gz).

Splits:

keras_png_slices_train/

keras_png_slices_validate/

keras_png_slices_test/

Each slice pair: case_XXX_week_Y_slice_Z.nii.gz and corresponding seg_XXX_week_Y_slice_Z.nii.gz.

Pre-processing
Step	Description
Normalization	Scale each image to [0, 1] individually
Label selection	Use label = 2 (prostate) → binary mask {0, 1}
Resize	Image = bilinear / Mask = nearest (256 × 128)
Split check	No overlap among train/val/test (all verified)
🧮 3. Training Setup
Hyperparameter	Value
Epochs	50
Batch size	8
Base filters	32
Learning rate	0.001
Optimizer	AdamW
Loss	BCE + Dice
Device	NVIDIA T4 (Google Colab GPU)
📈 4. Results
Quantitative Metrics
Split	Dice (↑)	IoU (↑)	Comments
Train	0.93	0.86	Consistent training convergence
Val	0.91	0.85	No overfitting observed
Test	0.9206	0.8588	✅ Meets requirement (≥ 0.75)
Qualitative Visualization

Below are examples of MRI slices with Ground Truth (GT) and Predicted masks:

MRI Image	GT (Prostate label 2)	Prediction

	
	

…	…	…

The overlay regions clearly correspond to the prostate zone in the pelvic area, confirming anatomical accuracy.

🧰 5. Reproducibility Instructions
Installation
pip install torch torchvision nibabel matplotlib

Training
python train_hipmri.py \
  --data_root /content/data/hipmri_slices/keras_slices_data \
  --num_classes 2 \
  --binary 1 \
  --epochs 50 \
  --batch 8 \
  --workers 2 \
  --base 32 \
  --resize 256 \
  --out runs/hipmri_resunet_binary

Evaluation
python eval_hipmri.py \
  --data_root /content/data/hipmri_slices/keras_slices_data \
  --weights runs/hipmri_resunet_binary/best.pt \
  --num_classes 2 --binary 1


Expected output:

== Test Dice (no-bg) ==
class_1: 0.9206
== Test IoU (no-bg, hard pred) ==
class_1: 0.8588
>> PROSTATE Dice (test) = 0.9206 (OK ≥ 0.75)

🧾 6. Implementation Notes

Implemented in PyTorch 2.x with Nibabel for NIfTI I/O.

All mask resizing verified via nearest neighbor interpolation to avoid label mixing.

Dataset inspection and debugging scripts (dataset_hipmri.py, train_hipmri.py) included for reproducibility.

Binary prostate mapping explicitly applied (plabel = 2).

🧩 7. Discussion

Strengths: High accuracy, stable convergence, anatomically aligned masks.

Limitations: Small number of test subjects; potential variability across MRI scanners.

Future work:

Explore 3D ResUNet for volumetric consistency.

Add data augmentation (elastic deformations).

Investigate semi-supervised learning for unlabeled slices.

📚 8. References

Ronneberger, O. et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.

Zhang Z. et al. (2018). Road Extraction by Deep Residual U-Net. IEEE Geoscience and Remote Sensing Letters.

HipMRI Dataset (2023). https://osf.io/xju2n/

PyTorch Documentation. https://pytorch.org/docs/

✅ 9. Submission Checklist

 All code in recognition/unet_hipmri_s4908819/

 No data or model files committed

 Train + Eval reproducible end-to-end

 Dice ≥ 0.75 (achieved 0.92)

 Pull Request submitted to topic-recognition

 README exported as PDF and submitted to Turnitin

🧑‍💻 Author

Yuqiao Geng (s4908819)
The University of Queensland — COMP3710 Pattern Analysis 2025
Supervisor: Shakes Chandra