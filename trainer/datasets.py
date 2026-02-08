import glob
import random
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch





class ImageDataset(Dataset):
    def __init__(self, root,noise_level,count = None,transforms_1=None,transforms_2=None, unaligned=False):
        self.transform1 = transforms.Compose(transforms_1)
        self.transform2 = transforms.Compose(transforms_2)
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))
        self.unaligned = unaligned
        self.noise_level =noise_level
        
    def __getitem__(self, index):
        if self.noise_level == 0:
            # if noise =0, A and B make same transform
            seed = np.random.randint(2147483647) # make a seed with numpy generator 
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            item_A = self.transform2(np.load(self.files_A[index % len(self.files_A)]).astype(np.float32))

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            item_B = self.transform2(np.load(self.files_B[index % len(self.files_B)]).astype(np.float32))
        else:
            # if noise !=0, A and B make different transform
            item_A = self.transform1(np.load(self.files_A[index % len(self.files_A)]).astype(np.float32))
            item_B = self.transform1(np.load(self.files_B[index % len(self.files_B)]).astype(np.float32))
            
            
            
        return {'A': item_A, 'B': item_B}
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class ValDataset(Dataset):
    def __init__(self, root,count = None,transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))
        
    def __getitem__(self, index):
        item_A = self.transform(np.load(self.files_A[index % len(self.files_A)]).astype(np.float32))
        if self.unaligned:
            item_B = self.transform(np.load(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(np.load(self.files_B[index % len(self.files_B)]).astype(np.float32))
        return {'A': item_A, 'B': item_B}
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    
def _scan_packed_cases(root):
    """
    root/
      P001/CBCT_A/X.npy, Y.npy
      P001/CBCT_B/X.npy, Y.npy
      ...
    Returns list of dicts: {"x": x_path, "y": y_path, "z": Z}
    """
    cases = []
    patient_dirs = sorted([p for p in glob.glob(os.path.join(root, "P*")) if os.path.isdir(p)])
    for pdir in patient_dirs:
        for tag in ["CBCT_A", "CBCT_B", "CBCT_C"]:
            cdir = os.path.join(pdir, tag)
            x_path = os.path.join(cdir, "X.npy")
            y_path = os.path.join(cdir, "Y.npy")
            if os.path.isfile(x_path) and os.path.isfile(y_path):
                X = np.load(x_path, mmap_mode="r")  # (C,Z,H,W)
                Y = np.load(y_path, mmap_mode="r")  # (1,Z,H,W)
                if X.ndim != 4 or Y.ndim != 4:
                    raise RuntimeError(f"Bad X/Y dims in {cdir}: X{X.shape}, Y{Y.shape}")
                if X.shape[1:] != Y.shape[1:]:
                    raise RuntimeError(f"X/Y mismatch in {cdir}: X{X.shape}, Y{Y.shape}")
                cases.append({"x": x_path, "y": y_path, "z": int(X.shape[1])})
    return cases

def body_aware_random_crop(body_mask, crop_size):
    """
    img: (H, W) numpy array
    body_mask: (H, W) bool or 0/1
    crop_size: int

    Returns cropped img (crop_size, crop_size)
    """
    H, W = body_mask.shape

    ys, xs = np.where(body_mask > 0)
    if len(ys) == 0:
        # fallback: random crop
        y0 = np.random.randint(0, H - crop_size + 1)
        x0 = np.random.randint(0, W - crop_size + 1)
        return int(y0), int(x0)

    # Body bounding box
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    # Choose center inside body bbox
    cy = np.random.randint(y_min, y_max + 1)
    cx = np.random.randint(x_min, x_max + 1)

    # Convert center → crop corner
    y0 = np.clip(cy - crop_size // 2, 0, H - crop_size)
    x0 = np.clip(cx - crop_size // 2, 0, W - crop_size)

    return int(y0), int(x0)

def apply_box(img, y0, x0, crop_size):
    return img[y0:y0+crop_size, x0:x0+crop_size]

class PackedNPYSliceDataset(Dataset):
    """
    Baseline dataset for your packed volumes:
      X.npy: (C,Z,H,W)  -> use ONLY channel 0 for baseline
      Y.npy: (1,Z,H,W)

    Returns dict {'A': tensor, 'B': tensor} where A/B are (1,H,W).
    """

    def __init__(self, root, noise_level, count=None, transforms_1=None, transforms_2=None, unaligned=False,
                 x_channel_index=0, training = True, crop_size=256):
        self.transform1 = transforms.Compose(transforms_1)
        self.transform2 = transforms.Compose(transforms_2)
        self.unaligned = unaligned
        self.noise_level = noise_level
        self.x_channel_index = int(x_channel_index)
        self.training = bool(training)
        self.crop_size = int(crop_size)


        self.cases = _scan_packed_cases(root)
        if len(self.cases) == 0:
            raise RuntimeError(f"No packed cases found under: {root}")

        # Pre-build slice index: (case_id, z)
        self.slice_map = []
        for ci, c in enumerate(self.cases):
            for z in range(c["z"]):
                self.slice_map.append((ci, z))

    def __len__(self):
        return len(self.slice_map)

    def __getitem__(self, index):
        ci, z = self.slice_map[index % len(self.slice_map)]
        c = self.cases[ci]

        X = np.load(c["x"], mmap_mode="r")  # (C,Z,H,W)
        Y = np.load(c["y"], mmap_mode="r")  # (1,Z,H,W)
        img_A = X[self.x_channel_index, z].astype(np.float32)  # (H,W)
        img_B = Y[0, z].astype(np.float32)                     # (H,W)
        
        body_mask = (img_B != -1)
        
        if self.training:
            y0, x0 = body_aware_random_crop(body_mask, self.crop_size)
            img_A = apply_box(img_A, y0, x0, self.crop_size)
            img_B = apply_box(img_B, y0, x0, self.crop_size)

        else:
            ys, xs = np.where(body_mask > 0)
            if len(ys) > 0:
                cy = int(ys.mean())
                cx = int(xs.mean())
                H, W = img_B.shape
                y0 = np.clip(cy - self.crop_size // 2, 0, H - self.crop_size)
                x0 = np.clip(cx -self.crop_size // 2, 0, W - self.crop_size)
                img_A = img_A[y0:y0+self.crop_size, x0:x0+self.crop_size]
                img_B = img_B[y0:y0+self.crop_size, x0:x0+self.crop_size]
            else:
                # fallback: center crop
                img_A = img_A[:self.crop_size, :self.crop_size]
                img_B = img_B[:self.crop_size, :self.crop_size]
        # Baseline: use X[channel0] only
      

        if self.noise_level == 0:
            # Keep the SAME transform for A and B (same seed), as in your original ImageDataset
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            item_A = self.transform2(img_A)

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            item_B = self.transform2(img_B)
        else:
            # Different transforms
            item_A = self.transform1(img_A)
            item_B = self.transform1(img_B)

        return {"A": item_A, "B": item_B}


class PackedNPYValSliceDataset(Dataset):
    """
    Validation dataset aligned by (case, z). Same logic as ValDataset, but reads packed X/Y.
    """

    def __init__(self, root, count=None, transforms_=None, unaligned=False, x_channel_index=0,crop_size = 256):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.x_channel_index = int(x_channel_index)
        self.crop_size = int(crop_size)

        self.cases = _scan_packed_cases(root)
        if len(self.cases) == 0:
            raise RuntimeError(f"No packed cases found under: {root}")

        self.slice_map = []
        for ci, c in enumerate(self.cases):
            for z in range(c["z"]):
                self.slice_map.append((ci, z))

    def __len__(self):
        return len(self.slice_map)

    def __getitem__(self, index):
        ci, z = self.slice_map[index % len(self.slice_map)]
        c = self.cases[ci]

        X = np.load(c["x"], mmap_mode="r")
        Y = np.load(c["y"], mmap_mode="r")

        img_A = X[self.x_channel_index, z].astype(np.float32)
        img_B = Y[0, z].astype(np.float32)
        # Deterministic body-aware crop for fair validation
        body_mask = (img_B != -1)
        ys, xs = np.where(body_mask > 0)
        H, W = img_B.shape

        if len(ys) > 0 and H >= self.crop_size and W >= self.crop_size:
            cy = int(ys.mean())
            cx = int(xs.mean())
            y0 = np.clip(cy - self.crop_size // 2, 0, H - self.crop_size)
            x0 = np.clip(cx - self.crop_size // 2, 0, W - self.crop_size)
            img_A = img_A[y0:y0+self.crop_size, x0:x0+self.crop_size]
            img_B = img_B[y0:y0+self.crop_size, x0:x0+self.crop_size]
        else:
            # fallback
            img_A = img_A[:self.crop_size, :self.crop_size]
            img_B = img_B[:self.crop_size, :self.crop_size]

        item_A = self.transform(img_A)

        if self.unaligned:
            # if you ever enable unaligned, pick random slice from random case
            cj = random.randint(0, len(self.cases) - 1)
            cz = random.randint(0, self.cases[cj]["z"] - 1)
            Y2 = np.load(self.cases[cj]["y"], mmap_mode="r")
            img_B2 = Y2[0, cz].astype(np.float32)
            item_B = self.transform(img_B2)
        else:
            item_B = self.transform(img_B)

        return {"A": item_A, "B": item_B}
