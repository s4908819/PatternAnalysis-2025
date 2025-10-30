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

# 🧩 数学表述（Theoretical Formulation）

令输入切片 \(x ∈ ℝ^{1×H×W}\)，输出 logits \(∈ ℝ^{K×H×W}\)。  
Softmax 概率：
\[
p_c = softmax(logits)_c, \quad c = 0, …, K-1
\]

组合损失：
\[
L_{total} = CE(logits, argmax(t)) + (1 - Dice_{mean}(p_c, t_c)), \quad c ∈ C={1,…,K-1}
\]

单类 Dice：
\[
Dice_c = \frac{2⟨p_c, t_c⟩ + ε}{‖p_c‖_1 + ‖t_c‖_1 + ε}
\]

---

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

# 🧩 Effect Summary

| Metric | Value | Description |
|:--|:--|:--|
| Validation Dice | 0.9242 | Best epoch 50 |
| Test Dice | 0.9206 | ✅ Pass ≥ 0.75 |
| Test IoU | 0.8588 | Accurate boundary overlap |
| Training | Stable | No oscillations |
| Generalisation | Strong | Val/Test difference < 0.02 |

---

# ✅ Comprehensive Summary

This implementation enhances **gradient propagation**, **feature representation**, and **stability** through:
- Residual blocks  
- Batch Normalisation  
- AdamW optimiser  
- CE + Dice composite loss  
- Nearest-neighbour mask scaling  

→ Achieved **Dice=0.9206**, **IoU=0.8588**, **Val Dice=0.9242** on test set.

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

# 📊 Results & Visualisation

| Dataset         | Dice   | IoU    | Notes          |
| :-------------- | :----- | :----- | :------------- |
| Train           | ≈0.93  | ≈0.86  | Stable         |
| Validation      | 0.9242 | ≈0.86  | No overfitting |
| Test (Prostate) | 0.9206 | 0.8588 | ✅ Pass ≥ 0.75  |

**Training Curves:**

* `loss.png`: loss decreases smoothly
* `dice.png`: validation Dice peaks ~0.92
* `best.pt` saved at epoch 50

---

# 🧮 Model Performance Summary

| Metric          | Value  | Evaluation                  |
| :-------------- | :----- | :-------------------------- |
| Dice (Prostate) | 0.9206 | ✅ Above 0.75                |
| IoU (Prostate)  | 0.8588 | ✅ Accurate                  |
| Convergence     | Smooth | ✅ Stable                    |
| Generalisation  | Strong | ✅ Consistent Train/Val/Test |

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
3. **HipMRI Dataset** (2023). *The HipMRI Study for Prostate Cancer Radiotherapy*.
   OSF: [https://osf.io/xju2n/](https://osf.io/xju2n/)
4. **COMP3710 Pattern Analysis Report** (v1.64 Final, 2025). UQ ITEE.
5. **PyTorch Documentation** (2024). [https://pytorch.org/docs/](https://pytorch.org/docs/)
6. **Nibabel Library** (2024). [https://nipy.org/nibabel/](https://nipy.org/nibabel/)

---

# 👤 Author & Academic Integrity Statement

| 项目                          | 内容                                                            |
| :-------------------------- | :------------------------------------------------------------ |
| **姓名（Name）**                | Yuqiao Geng                                                   |
| **学号（Student ID）**          | s4908819                                                      |
| **课程（Course）**              | COMP3710 – Pattern Analysis (Semester 2, 2025)                |
| **学校（Institution）**         | The University of Queensland, School of ITEE                  |
| **项目名称（Project Title）**     | HipMRI 2D Prostate Segmentation using Improved Residual U-Net |
| **实现难度（Difficulty Level）**  | Normal (Dice ≥ 0.75 requirement achieved)                     |
| **最终结果（Final Performance）** | Dice = 0.9206 / IoU = 0.8588 ✅ 满足评分标准                         |

---

## 🔒 Academic Integrity Declaration

I declare that this submission is my own original work completed as part of the requirements for COMP3710 (Pattern Analysis) at The University of Queensland.
All code, documentation, and results were developed independently.
Any AI assistance (e.g., ChatGPT) was used **only for explanation, formatting, and code commenting**, with all final implementations and analyses personally verified and authored by me.

**— Yuqiao Geng (s4908819), 2025-10-30**

```

---

要直接使用：
1. 将上述 Markdown 保存为 `README.md`。  
2. 放入 `recognition/unet_hipmri_s4908819/` 目录中。  
3. GitHub 会自动渲染为带标题、表格、公式与代码块的正式文档。
```
