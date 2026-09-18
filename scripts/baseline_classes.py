import os
import glob
import json
from pathlib import Path

import numpy as np 
import pandas as pd
import pydicom
import cv2

import albumentations as A

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import timm

from preprocessing import normalize_percentile

from sklearn.metrics import roc_auc_score

class RSNAKneeDataset(Dataset):
    def __init__(self, df_labels, study_dict, data_dir, target_names, num_slices=24, img_size=256, transform=None,
                use_cached_data=True, has_labels=True):
        """
        df_labels: DataFrame with StudyInstanceUID und den 12 Labels (0,1)
        study_dict: Mapping of the studies to 3 series instances
        """
        self.df_labels = df_labels
        self.study_dict = study_dict
        self.data_dir = data_dir
        self.num_slices = num_slices
        self.img_size = img_size
        self.transform = transform
        self.use_cached_data = use_cached_data
        self.has_labels = has_labels
        
        self.targets = target_names

    def __len__(self):
        return len(self.df_labels)

    def _load_cached_data(self, study_id, series_id):
        if pd.isna(series_id) or series_id is None:
            # Return empty volume
            return np.zeros((self.num_slices, self.img_size, self.img_size), dtype=np.float32)

        volume_file = os.path.join(self.data_dir, study_id, series_id + '.npy')
        #if not os.path.exists(volume_file):
        #    return np.zeros((self.num_slices, self.img_size, self.img_size), dtype=np.float32)
        volume = np.load(volume_file).astype(np.float32)

        # Remaining preprocessing
        volume = normalize_percentile(volume)

        return volume

        
    def _load_dicom_volume(self, study_id, series_id):
        if pd.isna(series_id) or series_id is None:
            # Return empty volume
            return np.zeros((self.num_slices, self.img_size, self.img_size), dtype=np.float32)
        
        series_dir = os.path.join(self.data_dir, study_id, series_id)
        if not os.path.exists(series_dir):
            return np.zeros((self.num_slices, self.img_size, self.img_size), dtype=np.float32)

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
            
        slices.sort(key=lambda x: int(getattr(x, 'InstanceNumber', 0)))

        # Limit slices by set amount
        indices = np.linspace(0, len(slices) - 1, self.num_slices).astype(int)
        slices = [slices[i] for i in indices]
        
        # 2. extract pixel arrays
        volume = []
        vol_max = -np.inf
        vol_min = np.inf
        for dcm in slices:
            img = dcm.pixel_array.astype(np.float32)

            ## Min-Max Normalization
            #if img.max() > 0:
            #    img = (img - img.min()) / (img.max() - img.min())

            vol_max = max(vol_max, img.max())
            vol_min = min(vol_min, img.min())
                
            # 2D Resize
            img = cv2.resize(img, (self.img_size, self.img_size))
            volume.append(img)
            
        volume = np.array(volume) # Shape: (num_slices, H, W)

        # Normalization
        volume = normalize_percentile(volume)
        
        return volume

    def _transform_volume(self, volumes):
        if self.transform is not None:
            for p in range(volumes.shape[0]):
                replay = A.ReplayCompose(self.transform.transforms)(image=volumes[p, 0])
                for z in range(volumes.shape[1]):
                    volumes[p, z] = A.ReplayCompose.replay(replay['replay'], image=volumes[p, z])['image'].squeeze()
        return volumes

    def __getitem__(self, idx):
        row = self.df_labels.iloc[idx]
        study_id = row['StudyInstanceUID']

        volume_list = []
        for volume_type in ['Sagittal', 'Coronal', 'Axial']:
            if self.use_cached_data:
                vol_cur = self._load_cached_data(study_id, self.study_dict[study_id][volume_type])
            else:
                vol_cur = self._load_dicom_volume(study_id, self.study_dict[study_id][volume_type])

            volume_list.append(vol_cur.copy())
            
        
        # Shape after stack: (3, num_slices, img_size, img_size)
        volumes = np.stack(volume_list, axis=0)

        if self.transform is not None:
            volumes = self._transform_volume(volumes)
        
        # --- Label Handling ---
        if self.has_labels:
            labels_binary = row[self.targets].values.astype(np.float32)
        else:
            labels_binary = np.full(len(self.targets), -1, dtype=np.float32) # dummy for testdata
        
        # convert to tensors
        x = torch.tensor(volumes, dtype=torch.float32)
        y = torch.tensor(labels_binary, dtype=torch.float32)
        
        return x, y, study_id


class AttentionPooling(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()
        self.attention_V = nn.Linear(feature_dim, hidden_dim)
        self.attention_U = nn.Linear(feature_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask=None):
        # x: (B, Z, D) - D = Feature-Dimension pro Slice (z.B. 512 bei ResNet34)
        A_V = torch.tanh(self.attention_V(x))          # (B, Z, hidden_dim)
        A_U = torch.sigmoid(self.attention_U(x))        # (B, Z, hidden_dim)
        scores = self.attention_w(A_V * A_U).squeeze(-1)  # (B, Z)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        weights = torch.softmax(scores, dim=1)          # (B, Z), summiert zu 1 pro Sample
        pooled = torch.sum(weights.unsqueeze(-1) * x, dim=1)  # (B, D)
        return pooled, weights

class RSNAKnee_ResNet34(nn.Module):
    def __init__(self, backbone_name='resnet34', num_classes=12, pooling_type='attention'):
        """
        pooling_type: string of either "attention" or "max" for the type of ppoling in the classification head
        """
        super().__init__()

        self.pooling_type = pooling_type 

        # Init backbone without classifier head (num_classes=0)
        self.backbone = timm.create_model(backbone_name, pretrained=True, in_chans=1, num_classes=0)
        feature_dim = self.backbone.num_features 
        if self.pooling_type == "attention":
            self.attention_pool = AttentionPooling(feature_dim)
        
        # Classification Head
        # we'll concatenate the series (sagittal, coronal, axial) --> 3 * feature_dim
        self.head = nn.Linear(feature_dim * 3, num_classes)

    def forward(self, x):
        # Input Shape from Dataloader: (Batch, Planes, Slices, H, W)
        B, P, Z, H, W = x.shape
        
        # ---- Flattening for the 2D CNN (merge batch, planes and slices into one dim)
        x = x.view(B * P * Z, 1, H, W) 
        # New Shape: (288, 1, 256, 256)
        
        # --- Feature Extraktion ---
        features = self.backbone(x) 
        # New Shape (288, 512)
        
        # --- (Reshape) ---
        features = features.view(B, P, Z, -1) 
        # New Shape (4, 3, 24, 512)
        
        # --- Z-Axis pooling ---
        if self.pooling_type == "attention":
            pooled_per_plane = []
            for p in range(P):
                pooled, attn_weights = self.attention_pool(features[:, p, :, :])  # (B, D), (B, Z)
                pooled_per_plane.append(pooled)

            features = torch.cat(pooled_per_plane, dim=1)  # (B, P*D)
        else:
            # max over all slices
            features = features.max(dim=2)[0]
            # New Shape (4, 3, 512)
        
            # --- View Fusion ---
            # concatenate patches to one long vector
            features = features.view(B, -1) 
            # New Shape (4, 1536)  <-- (3 * 512 = 1536)
        
        # --- Classification ---
        logits = self.head(features)
        # New Shape: (4, 12)
        
        return logits
    


class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        """
        logits: Tensor of shape (Batch, 12) - raw network outputs
        targets: Tensor of shape (Batch, 12) - labels (0, 1 or -1)
        """
        # mask for valid target (not -1)
        mask = (targets != -1).float()
        
        # Create dummy targets
        valid_targets = targets.clone()
        valid_targets[valid_targets == -1] = 0.0
        
        # unreduced loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, valid_targets, reduction='none')
        
        # apply mask
        masked_loss = bce_loss * mask
        
        final_loss = masked_loss.sum() / (mask.sum() + 1e-8)
        
        return final_loss


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, accumulation_steps=2):
    model.train()
    running_loss = 0.0
    
    optimizer.zero_grad() # Wichtig: Einmal vor der Schleife Nullen
     
    for batch_idx, (images, targets, _) in enumerate(dataloader):
        # 1. Daten auf die GPU schieben
        images = images.to(device)
        targets = targets.to(device)
        
        # 2. Autocast für Mixed Precision (Halbiert den VRAM)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(images)
            loss = criterion(logits, targets)
            
            # Loss durch accumulation_steps teilen, da wir die Gradienten summieren
            loss = loss / accumulation_steps
            
        # 3. Scaler übernimmt den Backward-Pass
        scaler.scale(loss).backward()
        
        # 4. Optimizer Step nur ausführen, wenn wir genug Gradienten gesammelt haben
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        # Tracking (multipliziert mit accumulation_steps für korrekte Anzeige)
        running_loss += loss.item() * accumulation_steps

    # Final update for remaining steps
    if len(dataloader) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
    return running_loss / len(dataloader)


def val_epoch(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad(): # Keine Gradienten berechnen = spart VRAM!
        for images, targets, _ in dataloader:
            images = images.to(device)
            
            # Forward Pass (AMP optional, aber meistens wird Val in FP32 gemacht)
            logits = model(images)
            
            # LOGITS ZU WAHRSCHEINLICHKEITEN MACHEN
            preds = torch.sigmoid(logits) 
            
            # Sammeln und auf die CPU schieben für sklearn
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            
    # Listen zu großen Tensoren zusammenbauen
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    # --- Maskierte Macro AUC Berechnung ---
    auc_scores = []
    auc_scores_all = []
    
    # Iteriere über die 12 Klassen
    for i in range(12):
        class_preds = all_preds[:, i]
        class_targets = all_targets[:, i]
        
        # Maske: Wo ist das Target NICHT -1?
        valid_mask = (class_targets != -1)
        
        # Filtere die Arrays
        valid_preds = class_preds[valid_mask]
        valid_targets = class_targets[valid_mask]
        
        # AUC kann nur berechnet werden, wenn es in den echten Labels mind. eine 0 und eine 1 gibt
        if len(np.unique(valid_targets)) == 2:
            score = roc_auc_score(valid_targets, valid_preds)
            auc_scores.append(score)

        # AUC for all classes
        class_targets[class_targets == -1] = 0
        score_all = roc_auc_score(class_targets, class_preds)
        auc_scores_all.append(score_all)
            
    # Kaggle-Metrik: Durchschnitt aller berechneten AUCs
    macro_auc = np.mean(auc_scores)
    macro_auc_all = np.mean(auc_scores_all)
    
    return macro_auc, macro_auc_all, auc_scores_all, all_preds, all_targets