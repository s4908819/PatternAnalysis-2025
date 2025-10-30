# 🧠 OASIS 2D Brain MRI Segmentation with Improved U-Net (s4908819)

## 🧩 Task Overview
This project performs **2D semantic segmentation** on the **OASIS brain MRI dataset**.  
Each 3D MRI volume is preprocessed into 2D PNG slices paired with corresponding label maps (4 classes: 0 = background, 1–3 = tissue types).  
The model adopts an **Improved U-Net** architecture to achieve accurate segmentation of brain tissues (CSF, gray matter, white matter).

**Goal:**  
Train an Improved U-Net model that achieves a **mean Dice score ≥ 0.9** on the test set — satisfying the *Easy* difficulty requirement of the COMP3710 report.

**Task Details**
- **Dataset:** OASIS Brain MRI Dataset  
- **Classes:** Background + 3 brain tissues  
- **Difficulty:** Easy (max 10/20 Implementation marks)  
- **Framework:** PyTorch (2D U-Net)  
- **Final Performance:** Mean Dice = **0.952 ≥ 0.9**

---

## ⚙️ Model Principle
The project employs a **U-Net encoder–decoder architecture** for 2D semantic segmentation of OASIS brain MRI slices.  
The model extracts hierarchical features through downsampling (encoder) and restores spatial details via upsampling (decoder).  
**Skip connections** concatenate shallow and deep features, enabling precise pixel-level predictions.

The structure is fully symmetric:
- **Encoder:** progressively downsample to a bottleneck layer.  
- **Decoder:** progressively upsample, concatenating with corresponding encoder features.  
- Final **1×1 convolution** outputs per-pixel logits for each class.

---

## 🧩 Implemented Improvements
| Improvement | Description |
|:--|:--|
| **Batch Normalization** | Added after each convolution (`Conv → BN → ReLU ×2`) to stabilize training and speed convergence. |
| **Combined CE + Dice Loss** | Joint **Cross-Entropy + (1 − Dice)** loss. Dice is computed on softmax probabilities (excluding background). |
| **Consistent one-hot labels** | Dataset converts integer labels to one-hot (C×H×W) tensors aligned with loss/metric input. |
| **Safe interpolation** | Logits are resized using bilinear interpolation only when necessary to match label sizes — preventing mask interpolation artifacts. |
| **Light paired augmentation** | Random horizontal flips applied simultaneously to image–mask pairs to improve generalization. |
| **Optimized training details** | Used **AdamW** optimizer with weight decay and enabled `cudnn.benchmark` for stable GPU acceleration. |

*Note:* No Dropout or LeakyReLU was used; standard ReLU and skip concatenations were maintained for reproducibility.

---

## 🧠 Workflow Overview
1. **Input Stage**  
   - Single-channel grayscale MRI slices  
   - Z-score normalization per slice  
   - Default resolution: **256×256**

2. **Encoder (Downsampling)**  
   - Multiple `DoubleConv + MaxPool` blocks  
   - Channels increase from base to 16×base at the bottleneck  
   - Each DoubleConv: `Conv → BN → ReLU → Conv → BN → ReLU`

3. **Decoder (Upsampling)**  
   - Transposed convolutions (`ConvTranspose2d`)  
   - Concatenate encoder features (skip connections)  
   - Fused with DoubleConv for refinement  

4. **Output Stage**  
   - Final 1×1 convolution outputs **4 channels (background + 3 tissues)**  
   - Training: **Cross-Entropy** on logits + **Dice** on softmax probabilities (excluding background)  
   - Inference: **argmax** of per-pixel class probabilities

---

## 📈 Results & Visualization
- Training logs record **Loss** and **Mean Dice** curves (excluding background)  
- Inference results overlay predictions on original MRI slices with color-coded masks (semi-transparent)

---

## 🧰 Environment & Dependencies
All experiments were conducted in **Google Colab (GPU)**.

| Environment | Version / Info |
|:--|:--|
| Python | 3.12.12 |
| PyTorch | 2.8.0 + cu126 |
| CUDA | 12.6 |
| GPU | NVIDIA A100-SXM4-80GB |
| Dependencies | `torch`, `torchvision`, `numpy`, `pillow`, `matplotlib`, `tqdm`, `nibabel` |

**Install locally:**
```bash
pip install torch torchvision nibabel numpy pillow matplotlib tqdm
````

---

## 🚀 Training Command

```bash
!python train.py \
  --data_root /content/data/oasis_slices/OASIS \
  --num_classes 4 \
  --epochs 12 \
  --batch 8 \
  --out runs/oasis_unet \
  --workers 2
```

---

## 📂 Dataset Structure

Each 3D MRI is split into multiple 2D slices.

```
OASIS/
├── keras_png_slices_train/
├── keras_png_slices_validate/
├── keras_png_slices_test/
├── keras_png_slices_seg_train/
├── keras_png_slices_seg_validate/
└── keras_png_slices_seg_test/
```

| Split      | Images | Percentage |
| :--------- | -----: | ---------: |
| Train      |  9,664 |      85.3% |
| Validation |  1,120 |       9.9% |
| Test       |    544 |       4.8% |
| **Total**  | 11,328 |       100% |

* Dataset split is **patient-independent** (no leakage).
* Reproducible and consistent with official Rangpur/OASIS structure.

---

## ⚙️ Model & Training Configuration

### Model Configuration

| Component     | Description                                      |
| :------------ | :----------------------------------------------- |
| Architecture  | Symmetric U-Net (Encoder–Decoder) with BatchNorm |
| Encoder Depth | 4 layers (channels: 32 → 64 → 128 → 256)         |
| Activation    | ReLU                                             |
| Normalization | BatchNorm2d                                      |
| Dropout       | None                                             |
| Output        | 1×1 Conv (4 classes)                             |
| Upsampling    | ConvTranspose2d + skip concatenation             |
| Loss          | Cross-Entropy + Dice (excluding background)      |

### Training Parameters

| Parameter     | Value                     |
| :------------ | :------------------------ |
| Input Size    | 1×256×256                 |
| Classes       | 4                         |
| Batch Size    | 8                         |
| Epochs        | 12                        |
| Learning Rate | 1e-3                      |
| Optimizer     | AdamW (weight_decay=1e-5) |
| Scheduler     | None                      |
| Metric        | Mean Dice (no background) |
| Random Seed   | 4908819                   |

---

## 📊 Training & Evaluation

* Model converged after **~5 epochs** (~15 minutes total training).
* Outputs generated in `runs/oasis_unet/`:

  * `loss.png` — Training/validation loss curve
  * `dice.png` — Validation mean Dice curve
  * `best.pt` — Best-performing checkpoint

**Final Result**

| Metric                    |     Value |
| :------------------------ | --------: |
| Mean Dice (no background) | **0.952** |

✅ Meets **Easy difficulty** target (≥ 0.9).

---

## 📘 Summary

* Improved U-Net achieved **Mean Dice = 0.952** on OASIS dataset.
* Stable training, smooth curves, no overfitting.
* Meets *“Segment the OASIS dataset with Improved U-Net, Dice ≥ 0.9”* requirement.
* Fully reproducible, visually interpretable segmentation output.

---

## 💾 Project Structure

```
recognition/unet_oasis_s4908819/
 ├── dataset.py          ← Loads & preprocesses OASIS PNG images & masks
 ├── modules.py          ← Improved U-Net model (Encoder–Decoder + BN)
 ├── train.py            ← Training loop with CE + Dice loss & logging
 ├── predict.py          ← Inference & visualization (overlay masks)
 ├── utils.py            ← Metrics (Dice, IoU) and plotting functions
 ├── runs/oasis_unet/    ← Output folder (best.pt, loss.png, dice.png)
 ├── demo_pred.png       ← Example output visualization
 └── README.md           ← This documentation
```

---

## 📚 References

1. **Ronneberger, O., Fischer, P., & Brox, T. (2015).**
   *U-Net: Convolutional Networks for Biomedical Image Segmentation.*
   *MICCAI 2015, LNCS, Springer.* DOI: [10.1007/978-3-319-24574-4_28]

2. **OASIS Brain MRI Dataset.**
   *Open Access Series of Imaging Studies (OASIS).*
   [https://www.oasis-brains.org/](https://www.oasis-brains.org/)

3. **COMP3710 – Pattern Analysis (v1.64 Final Report, UQ 2025).**
   Defines grading and difficulty criteria:
   “Segment the 2D OASIS brain data set with an Improved U-Net, Dice ≥ 0.9.”

4. **PyTorch Official Documentation.**
   [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

---

## 👤 Author Information

* **Author:** Yuqiao Geng (s4908819)
* **Course:** COMP3710 – Pattern Analysis (Semester 2, 2025, The University of Queensland)
* **Project:** OASIS 2D Brain Segmentation with Improved U-Net
* **Result:** ✅ Mean Dice = 0.952 ≥ 0.9 (*Easy difficulty achieved*)

