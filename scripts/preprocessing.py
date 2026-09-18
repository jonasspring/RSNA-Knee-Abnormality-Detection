# Preprocessing functions
import os
import glob
import json
from pathlib import Path

import numpy as np 
import pandas as pd
import pydicom
import cv2

def get_orientation_axes(dcm):
    """Return row and col and normal vector of the image plane"""
    iop = np.array(dcm.ImageOrientationPatient, dtype=float)
    row_cosine = iop[:3]
    col_cosine = iop[3:]
    normal = np.cross(row_cosine, col_cosine)  
    return row_cosine, col_cosine, normal

def sort_slices_geometrically(slices):
    row_cosine, col_cosine, normal = get_orientation_axes(slices[0])

    # Normale should align with patients left side, to stay consistent
    if normal[0] < 0:
        normal = -normal

    def projected_position(dcm):
        pos = np.array(dcm.ImagePositionPatient, dtype=float)
        return np.dot(pos, normal)

    slices.sort(key=projected_position)
    return slices

def normalize_percentile(volume, low=1, high=99):
    lo, hi = np.percentile(volume, [low, high])
    volume = np.clip(volume, lo, hi)
    return (volume - lo) / (hi - lo + 1e-6)

def dominant_lr_axis(row_cosine, col_cosine, normal):
    """
    Determines which axis corresponds most with the patients left/right (x) Axis
    Returns: ("columns" | "rows" | "slices", x_component)
    """
    candidates = {
        "columns": row_cosine[0],
        "rows": col_cosine[0], 
        "slices": normal[0], 
    }
    axis = max(candidates, key=lambda k: abs(candidates[k]))
    return axis, candidates[axis]

def get_laterality(dcm):
    for tag in ("Laterality", "ImageLaterality"):
        val = getattr(dcm, tag, None)
        if val in ("R", "L"):
            return val
    return None

def normalize_laterality(volume, reference_dcm, laterality, canonical="L"):
    """
    volume: np.array of shape (Z, H, W) for one series
    reference_dcm: one dicom slice, to obtain orientation
    laterality: "L" or "R" of the study
    canonical: target side for normalization
    """
    if laterality is None or laterality == canonical:
        return volume

    row_cosine, col_cosine, normal = get_orientation_axes(reference_dcm)
    axis, _ = dominant_lr_axis(row_cosine, col_cosine, normal)

    if axis == "columns":
        return volume[:, :, ::-1].copy()
    elif axis == "rows":
        return volume[:, ::-1, :].copy()
    else:  # "slices"
        return volume[::-1, :, :].copy()


def build_volume(study_id, series_id, series_dir_root, num_slices=24, img_size=256):
    if pd.isna(series_id) or series_id is None:
        raise Exception("Faulty series id")
    
    series_dir = os.path.join(series_dir_root, study_id, series_id)
    if not os.path.exists(series_dir):
        raise Exception("No folder for series")

    dicom_files = [os.path.join(series_dir, f) for f in os.listdir(series_dir) if f.endswith('.dcm')]
    
    # read dicom and sort by instance number
    slices = []
    for f in dicom_files:
        try:
            dcm = pydicom.dcmread(f)
    
            if not dcm.pixel_array.astype(np.float32).size > 0:
                print('No pixel Data available')
                continue
            slices.append(dcm)
        except:
            print('Faulty dicom file')
            continue
        
    #slices.sort(key=lambda x: int(getattr(x, 'InstanceNumber', 0)))
    slices = sort_slices_geometrically(slices)

    # Limit slices by set amount
    indices = np.linspace(0, len(slices) - 1, num_slices).astype(int)
    slices = [slices[i] for i in indices]
    
    # 2. extract pixel arrays
    volume = []
    vol_max = -np.inf
    vol_min = np.inf
    for dcm in slices:
        img = dcm.pixel_array.astype(np.float32)

        vol_max = max(vol_max, img.max())
        vol_min = min(vol_min, img.min())
        
        # 2D Resize
        img = cv2.resize(img, (img_size, img_size))
        volume.append(img)

    volume = np.array(volume) # Shape: (num_slices, H, W)

    # normalization, to allow saving in uint8
    volume = (volume - vol_min) / (vol_max - vol_min + 1e-6) * 255.0

    # Lateral normalization
    volume_norm = normalize_laterality(volume, dcm, get_laterality(dcm))
    
    return volume

def preprocess_and_cache(study_id, study_dict, series_dir_root, cache_output_dir, cache_dataset_dir=None, consider_cache_dataset=False, num_slices=24, img_size=256):

    errors = None
    for volume_type in ['Sagittal', 'Coronal', 'Axial']:
        out_path = cache_output_dir / f"{study_id}" / f"{study_dict[study_id][volume_type]}.npy"
        if consider_cache_dataset:
            dataset_path = cache_dataset_dir / f"{study_id}" / f"{study_dict[study_id][volume_type]}.npy"
            dataset_cond = dataset_path.exists()
        else:
            dataset_cond = False

        if out_path.exists() or dataset_cond:
            continue  # already cached, skip

        try:
            volume = build_volume(study_id, study_dict[study_id][volume_type], series_dir_root, num_slices, img_size)
    
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, volume.astype(np.uint8)) 
        except Exception as e: 
            print(e)
            
            if not errors:
                errors = []
            errors.append(e)
            
            continue
    return errors