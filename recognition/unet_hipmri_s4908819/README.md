# 🧠 Project Overview

This project implements **2D semantic segmentation** on the **HipMRI prostate cancer dataset**.  
Each 2D slice is accompanied by a corresponding segmentation label map; in this implementation, we map the **prostate (label = 2)** to a **binary foreground classification (prostate=1 / others=0)**, focusing on evaluating segmentation quality within the prostate region.  

The model adopts an **enhanced Residual U-Net (ResUNet)** architecture, designed to achieve stable and precise segmentation while maintaining inference speed.

**Objective:**  
Train an enhanced 2D ResUNet to achieve a **Dice coefficient ≥ 0.75** for the ‘prostate label’ on the test set, meeting the *Normal difficulty* requirement for HipMRI 2D (Improved U-Net/CAN) as specified in the course report.  
This project ultimately achieved **Dice = 0.9206** and **IoU = 0.8588**, significantly exceeding the threshold requirement (≥0.75).

---

## 🧩 Task Background

| Item | Description |
|:--|:--|
| **Data Source** | HipMRI Study (course-provided 2D pre-processed slice data with corresponding labels; divided into training/validation/test sets) |
| **Evaluation Label** | Prostate (original label value 2), uniformly mapped to binary classification in this implementation to focus on foreground quality |
| **Task Difficulty** | Normal (meeting the specification ‘2D Improved UNet/CAN on HipMRI, prostate Dice ≥ 0.75’) |
| **Implementation Framework** | PyTorch + Nibabel (NIfTI I/O) |
| **Final Model Performance** | Dice = 0.9206, IoU = 0.8588 (test set, prostate without background channel) |

---

# ⚙️ Model Principle

This project employs an **enhanced 2D Residual U-Net (UNetRes)** for semantic segmentation of prostate MRI.  
The network retains the classic Encoder–Decoder symmetric architecture with skip connections, introducing residual blocks and Batch Normalisation (BN) at each stage to enhance gradient flow and training stability, achieving superior convergence and accuracy at shallower depths.

**Corresponding code:**  
`UNetRes`, `ResBlock`, `Down`, `Up`, `OutConv` in `modules_resunet.py`;  
training and evaluation scripts: `train_hipmri.py` and `eval_hipmri.py`.

---

## 🧠 Model Architecture Overview

### Encoder
- Input: single-channel greyscale image `(1×H×W)`  
- Residual block for base width (default 32)
- Three successive downsampling stages:
  - `Down`: MaxPool2d(2) + Residual block  
  - Channels: base → 2×base → 4×base → 8×base  
- Skip connections fuse encoded and upsampled features

### Bottleneck
- Residual block expands to `16×base`

### Decoder
- Three symmetric upsampling stages:
  - `Up`: ConvTranspose2d → concat skip → Residual block  
  - Channels: 16b→8b, 8b→4b, 4b→2b  
- Output layer: `OutConv(1×1)` maps to n_classes (2 for binary)

### Output
- **Training/Evaluation:** Softmax on logits, Dice (exclude-bg)  
- **Inference:** Sigmoid + argmax for class labelling

---

### 🔍 Key Code Points
- **Upsampling Alignment:** Bilinear interpolation applied only to logits (`upsample_to_target / upsample_to_hw`)
- **Skip Alignment:** F.pad used when spatial dimensions mismatch in `Up.forward`

---

## 🧩 Theoretical Formulation

Let the input slice be \( x \in \mathbb{R}^{1 \times H \times W} \),  
and the network output logits \( z \in \mathbb{R}^{K \times H \times W} \).

**Softmax probability for class \( c \):**

\[
p_c = \text{softmax}(z)_c, \quad c = 0, \dots, K-1
\]

---

### Combined Loss Function

The total loss combines **Cross-Entropy (CE)** and **Dice Loss** (excluding background):

\[
L_{\text{total}} = \text{CE}(z, \arg\max(t)) + \left( 1 - \text{Dice}_{\text{mean}}(p_c, t_c) \right),
\quad c \in \mathcal{C} = \{1, \dots, K-1\}
\]

---

### Class-wise Dice Coefficient

For each class \( c \), the Dice score is defined as:

\[
\text{Dice}_c = \frac{2 \langle p_c, t_c \rangle + \varepsilon}
{\| p_c \|_1 + \| t_c \|_1 + \varepsilon}
\]

where \( \varepsilon \) is a small constant to ensure numerical stability.

# 🗂️ Data and Label Processing (Aligned with Dataset Implementation)

**Code:** `dataset_hipmri.py`

- **Normalization:** z-score per image  
- **Resize:** bilinear (image) / nearest (mask)  
- **Label Encoding:**
  - Binary → two-hot [bg, fg]  
  - Multi-class → one-hot (K×H×W)
- **File Formats:** PNG/JPG/TIF/NPY/NIfTI  
- **Pairing:** strip prefixes (`case_/seg_`) to align  
- **Augmentation:** random horizontal flip (optional)

---

# 🧮 Training & Evaluation Details

- **Loss:** CE + (1 - mean Dice_no-bg)  
- **Metrics:**
  - Dice (no-bg)
  - IoU (hard, no-bg)
  - Prostate Dice = class_1
- **Visualisation:** `--save_vis` saves IMG|GT|PRED

---

# 🧩 Textual Structure Diagram

```

Input (1×H×W, z-score)
↓
ResBlock(base)
↓
Down ×3 → Bottleneck(16×base)
↓
Up ×3 with Skip Connections
↓
OutConv(1×1 → n_classes)
↓
Softmax/Sigmoid → Mask

```

---

# ✅ Key Consistency Notes
- Architecture fully matches `UNetRes` structure  
- Loss = CE + (1 − mean Dice_no-bg)  
- Two-hot [bg, fg] output for prostate segmentation  
- Only logits interpolated (not masks)  
- Inference: sigmoid + argmax = softmax equivalence  

---

# 🧩 Improvements Implemented

| # | Item | Description |
|:--:|:--|:--|
| 1️⃣ | Residual Skip Connections | Adds `y=F(x)+x` in each ResBlock |
| 2️⃣ | Batch Normalisation | Conv→BN→ReLU sequence stabilises training |
| 3️⃣ | Composite Loss | `CE + (1 - Dice)` improves overlap learning |
| 4️⃣ | Binary Label Mapping | Only label=2 kept as foreground |
| 5️⃣ | Nearest-Neighbour Resize | Preserves mask edges |
| 6️⃣ | Validation Monitoring | Auto-save `best.pt` + loss/dice curves |
| 7️⃣ | Optimiser Enhancement | AdamW with weight_decay=1e-5 |
| 8️⃣ | Normalisation Pipeline | z-score + strict pairing logic |
| 9️⃣ | Inference Normalisation | Sigmoid + argmax = softmax ordering |
| 🔟 | Logging & Visualisation | loss.png, dice.png auto-saved |


---




# 🧠 Working Pipeline

**Stages:**  
Data Reading → Preprocessing → Model Training → Validation → Inference  

All executed on Google Colab GPU (A100).

---

## 1️⃣ Input Stage
- **Data:** single-channel `.nii/.nii.gz`
- **Loader:** nibabel.load()  
- **Dirs:**  
  - `/keras_png_slices_train/`, `/keras_png_slices_validate/`, `/keras_png_slices_test/`
  - `/keras_png_slices_seg_*`
- **Normalization:** z-score  
- **Mapping:** label 2→1, else 0  
- **Resize:** bilinear (img), nearest (mask)  
- **Example Log:**
```

[hipmri dataset] sample0 img (1,256,256) | mask (2,256,256)

```

---

## 2️⃣ Encoder Stage
- **Structure:** DoubleConv (3×3×2 + BN + ReLU)
- **Residual Path:** out = F(x) + x  
- **Downsampling:** MaxPool2d(2)
- **Channels:** 32→64→128→256  
✅ Extracts hierarchical features.

---

## 3️⃣ Decoder Stage
- **UpSampling:** ConvTranspose2d + concat + DoubleConv  
- **Diagram:**
```

Encoder_i ─────┐
concat ↑
UpConv ← Decoder_{i+1}

````

---

## ⚙️ 4️⃣ Output Stage
- Final Conv2d(1×1) → 2 channels  
- Softmax / Sigmoid + argmax  
- Dice / IoU computed excl. background

---

## 🧮 5️⃣ Training & Validation
- **Epochs:** 50  
- **Batch:** 8  
- **LR:** 1e-3  
- **Loss:** CE + Dice  
- **Optimiser:** AdamW  
- **Logs:** loss.png, dice.png  
- **Result:** Val Dice = 0.9242

---

## 🔍 6️⃣ Inference & Evaluation
```bash
!python eval_hipmri.py \
--data_root /content/data/hipmri_slices/keras_slices_data \
--weights runs/hipmri_resunet_binary_v2/best.pt \
--num_classes 2 --binary 1 --prostate_label 2 \
--batch 8 --workers 2 --base 32 --resize 256 \
--device cuda --save_vis 12 --vis_out /content/runs/hipmri_eval_vis
````

**Results:**

```
== Test Dice (no-bg) ==
class_1: 0.9206
== Test IoU (no-bg, hard pred) ==
class_1: 0.8588
```

---
## 🧩 Dependencies & Environment

| Package      | Version         |
|---------------|-----------------|
| **Python**    | 3.12.12         |
| **PyTorch**   | 2.8.0+cu126     |
| **TorchVision** | 0.23.0+cu126  |
| **TorchAudio** | 2.8.0+cu126    |
| **Nibabel**   | 5.3.2           |
| **NumPy**     | 2.0.2           |
| **Matplotlib** | 3.10.0         |

---

**Hardware:** Google Colab GPU — NVIDIA A100-SXM4-80GB  
**CUDA/cuDNN:** CUDA 12.6, cuDNN 91002  
**Reproducibility:** All random seeds fixed (`torch.manual_seed(49088191)`)


# 📊 Results & Visualisation

| Dataset         | Dice   | IoU    | Notes          |
| :-------------- | :----- | :----- | :------------- |
| Train           | ≈0.93  | ≈0.86  | Stable         |
| Validation      | 0.9242 | ≈0.86  | No overfitting |
| Test (Prostate) | 0.9206 | 0.8588 | ✅ Pass ≥ 0.75  |

## 📊 Training Curves

* **`loss.png`** — Loss decreases smoothly  
  ![Loss Curve](https://raw.githubusercontent.com/s4908819/PatternAnalysis-2025/topic-recognition/recognition/unet_hipmri_s4908819/run/loss.png)

* **`dice.png`** — Validation Dice peaks around 0.92  
  ![Dice Curve](https://raw.githubusercontent.com/s4908819/PatternAnalysis-2025/topic-recognition/recognition/unet_hipmri_s4908819/run/dice.png)

* `best.pt` saved at epoch 50

---

## 📈 Results Summary

| **Metric** | **Value** | **Description** |
|:------------|:----------:|:----------------|
| **Validation Dice** | 0.9242 | Reached peak at epoch 50 |
| **Test Dice (Prostate)** | 0.9206 | ✅ Exceeds the minimum requirement (≥ 0.75) |
| **Test IoU (Prostate)** | 0.8588 | Accurate boundary overlap |
| **Convergence** | Stable | Loss and Dice curves show consistent improvement |
| **Generalisation** | Strong | Validation–test gap < 0.02 |

---

The model demonstrated **stable convergence** throughout training with **no signs of overfitting**.  
Final performance on the test set **significantly surpasses the “Normal Difficulty” threshold (Dice ≥ 0.75)** specified in the COMP3710 Pattern Analysis report.

---

# 🧩 Discussion & Future Work

## ✅ Strengths

* Residual design enhances gradient flow
* CE + Dice handles imbalance
* Stable Dice/IoU > 0.85
* Fast inference (<0.1s)

## ⚠️ Limitations

* Blurred slice boundaries
* Binary only (no multi-organ)
* No data augmentation
* 2D context only

---

## 🔬 Future Directions

1️⃣ **3D Extension:** Convert Conv2d→Conv3d for volumetric segmentation
2️⃣ **Data Augmentation:** random flip, rotation, elastic deformation
3️⃣ **Lightweight UNet:** depthwise separable conv (MobileNet encoder)
4️⃣ **Multi-organ Segmentation:** bladder, rectum via `--num_classes 4`
5️⃣ **Semi-supervised Learning:** consistency regularisation & fine-tuning

---

# 📘 Summary

* Achieved **Dice=0.9206**, **IoU=0.8588**, **Val Dice=0.9242**
* Residual + BN + AdamW → stable convergence
* Fully reproducible pipeline with visualisation & metrics
* Foundation for 3D, lightweight, and clinical extensions

---

# 📚 References

1. **Ronneberger, O.** et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*.
   MICCAI 2015, LNCS 9351. [DOI:10.1007/978-3-319-24574-4_28]
2. **Zhang, Z.**, Liu, Q., Wang, Y. (2018). *Road Extraction by Deep Residual U-Net*.
   IEEE GRSL, 15(5), 749–753. [DOI:10.1109/LGRS.2018.2802944]
3. **COMP3710 Pattern Analysis Report** (v1.64 Final, 2025). UQ ITEE.
4. **PyTorch Documentation** (2024). [https://pytorch.org/docs/](https://pytorch.org/docs/)
5. **Nibabel Library** (2024). [https://nipy.org/nibabel/](https://nipy.org/nibabel/)

---



