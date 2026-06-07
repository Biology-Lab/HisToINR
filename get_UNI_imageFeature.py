import numpy as np
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt
import os

# Select GPU device if available
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Path to the local UNI model
model_dir = "/home/guodingfei/MyProject/UNI_model"
weight_path = f"{model_dir}/pytorch_model.bin"

# Build the UNI backbone architecture
# The configuration must match the original model config
model = timm.create_model(
    "vit_large_patch16_224",
    pretrained=False,
    num_classes=0,
    global_pool="token",
    dynamic_img_size=True,
    init_values=1.0
)

# Load pretrained UNI weights
state_dict = torch.load(weight_path, map_location="cpu")

if list(state_dict.keys())[0].startswith("module."):
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)

model.eval()
model.to(device)

from torchvision import transforms

# Image preprocessing pipeline
# Resize image patches to 224×224 and normalize using transformer
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class roi_dataset(Dataset):
    """
    Dataset containing cropped image patches centered on spatial spots.
    """

    def __init__(self, img):
        super().__init__()

        self.transform = transform
        self.images_lst = img

    def __len__(self):
        """
        Return the total number of image patches.
        """
        return len(self.images_lst)

    def __getitem__(self, idx):
        """
        Load and preprocess a single image patch.
        """
        image = self.images_lst[idx].astype('uint8')
        image = self.transform(image)
        return image


def crop_image(img, x, y, crop_size=None):
    """
    Crop a local image patch centered at a given spatial spot.

    Parameters
    ----------
    img : ndarray
        Whole-slide image.
    x : int
        X coordinate of the spot center.
    y : int
        Y coordinate of the spot center.
    crop_size : list, optional
        Size of the cropped patch [width, height].

    Returns
    -------
    cropped_img : ndarray
        Cropped image patch centered on the spot.
    """

    if crop_size is None:
        crop_size = [195, 195]

    h, w = img.shape[:2]

    # Compute the top-left corner of the crop region
    left = max(0, x - crop_size[0] // 2)
    top = max(0, y - crop_size[1] // 2)

    # Compute the bottom-right corner while preventing overflow
    right = min(w, left + crop_size[0])
    bottom = min(h, top + crop_size[1])

    cropped_img = img[top:bottom, left:right]

    return cropped_img


def UNI_features(img_path, spatial):
    """
    Extract UNI image features for all spatial transcriptomics spots.

    Parameters
    ----------
    img_path : str
        Path to the histology image.
    spatial : ndarray or DataFrame
        Spatial coordinates of spots, where each row is (x, y).

    Returns
    -------
    feature_embs : ndarray
        UNI feature embeddings for all spots.
        Shape: [n_spots, feature_dim]
    """

    model.eval()
    model.to(device)

    # Load histology image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img)

    sub_images = []

    spot_num = len(spatial)
    loc = spatial

    # Convert DataFrame to numpy array if necessary
    if isinstance(loc, pd.DataFrame):
        loc = loc.values

    # Crop image patches centered on each spot
    for i in range(spot_num):
        x = loc[i, 0]
        y = loc[i, 1]

        sub_image = crop_image(img, x, y)
        sub_images.append(sub_image)

    # Construct dataset and dataloader
    test_datat = roi_dataset(sub_images)

    database_loader = torch.utils.data.DataLoader(
        test_datat,
        batch_size=512,
        shuffle=False
    )

    feature_embs = []

    # Extract UNI embeddings in batches
    with torch.inference_mode():
        for batch in database_loader:

            batch = batch.to(device)

            # Forward pass through UNI encoder
            feature_emb = model(batch)

            feature_embs.append(feature_emb.cpu())

        # Concatenate all spot embeddings
        feature_embs = np.concatenate(feature_embs, axis=0)

    return feature_embs