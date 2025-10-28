import os
import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm

def nifti_to_slices(img_path, mask_path, out_dir):
    os.makedirs(f"{out_dir}/images", exist_ok=True)
    os.makedirs(f"{out_dir}/masks", exist_ok=True)
    name = os.path.splitext(os.path.basename(img_path))[0]
    img = nib.load(img_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()

    for i in range(img.shape[2]):
        img_slice = img[:, :, i]
        mask_slice = mask[:, :, i]
        if mask_slice.sum() == 0:
            continue  # 跳过无内容切片
        img_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
        Image.fromarray((img_norm * 255).astype(np.uint8)).save(f"{out_dir}/images/{name}_{i:03d}.png")
        Image.fromarray(mask_slice.astype(np.uint8)).save(f"{out_dir}/masks/{name}_{i:03d}.png")

def main():
    root = "/home/groups/comp3710/OASIS"
    out_dir = "/home/$USER/oasis_slices"
    img_dir = os.path.join(root, "images")
    mask_dir = os.path.join(root, "masks")

    for f in tqdm(sorted(os.listdir(img_dir))):
        m = f.replace(".nii.gz", "_mask.nii.gz")
        nifti_to_slices(os.path.join(img_dir, f), os.path.join(mask_dir, m), out_dir)

if __name__ == "__main__":
    main()
