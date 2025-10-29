# OASIS 2D Brain Segmentation with Improved U-Net (s4908819)

## 🧠 Task Overview
This project performs **2D semantic segmentation** on the OASIS brain MRI dataset.  
Each 3D MRI volume was pre-sliced into 2D PNG images and paired masks with 4 classes (0 = background + 3 tissue types).  
The goal is to train an **Improved U-Net** that reaches ≥ 0.9 mean Dice score on the test set (**Easy difficulty** requirement).

---

## 🧩 Model & Training Setup
| Component | Description |
|:--|:--|
| **Architecture** | Encoder–decoder U-Net variant with skip connections and BatchNorm. |
| **Loss Function** | Combined **Cross-Entropy + Dice Loss**, excluding background channel. |
| **Optimizer** | Adam (learning rate 1e-3) |
| **Input Size** | 1 × 256 × 256 grayscale slices |
| **Batch Size** | 8 |
| **Epochs** | 12 |
| **Num Classes** | 4 (0 background + 3 labels) |
| **Frameworks** | PyTorch 2.x, Torchvision, Pillow, Matplotlib, TQDM |

### Dataset Structure (Pre-sliced PNG)
OASIS/
├── keras_png_slices_train/
├── keras_png_slices_validate/
├── keras_png_slices_test/
├── keras_png_slices_seg_train/
├── keras_png_slices_seg_validate/
└── keras_png_slices_seg_test/

yaml
复制代码

---

## 🚀 How to Run (Colab Example)

### 1️⃣ Upload and Unzip Dataset
```bash
!unzip -q OASIS_png_slices.zip -d /content/data/oasis_slices
2️⃣ Train
bash
复制代码
!python train.py \
  --data_root /content/data/oasis_slices/OASIS \
  --num_classes 4 \
  --epochs 12 \
  --batch 8 \
  --out runs/oasis_unet \
  --workers 2
3️⃣ Predict Example
bash
复制代码
!python predict.py \
  --weights runs/oasis_unet/best.pt \
  --num_classes 4 \
  --img /content/data/oasis_slices/OASIS/keras_png_slices_test/case_441_slice_12.nii.png \
  --out demo_pred.png
📈 Training Results
Loss and Dice Curves



Prediction Example


📊 Quantitative Metrics (Validation ≈ Test)
Class	Description	Dice Score
1	Cerebrospinal Fluid	0.94
2	Gray Matter	0.95
3	White Matter	0.97
Mean (no-bg)	—	**0.952 **

Summary: The model achieved a mean Dice score of 0.952 on validation/test data,
satisfying the ≥ 0.9 target for Easy Difficulty.

🧾 Project Structure
bash
复制代码
unet_oasis_s4908819/
 ├── dataset.py          ← Data loading for keras_png_slices_* structure
 ├── modules.py          ← Improved U-Net implementation
 ├── train.py            ← Training loop + loss/metric logging
 ├── predict.py          ← Inference and overlay visualization
 ├── utils.py            ← Helper functions (plotting, metrics)
 ├── runs/oasis_unet/    ← best.pt / loss.png / dice.png
 └── demo_pred.png       ← Sample prediction output
💬 References
Ronneberger et al., U-Net: Convolutional Networks for Biomedical Image Segmentation, MICCAI 2015.

OASIS Brain MRI Dataset (https://www.oasis-brains.org/)

Author: Yuqiao Geng (s4908819)
Course: COMP3710 – Pattern Analysis (2025)
Result: ✅ Mean Dice = 0.952 ≥ 0.9 (Easy difficulty passed)