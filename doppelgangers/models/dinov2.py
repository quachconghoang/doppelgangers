import os
from pathlib import Path
import time
import math
import itertools
from typing import Any
from tqdm import tqdm

import numpy as np
import cv2 as cv
from PIL import Image
from typing import Tuple

import torch
import kornia as K
from transformers import AutoImageProcessor, AutoModel
from torch import Tensor as T
import torch.nn.functional as F
import torchvision.transforms as transforms

from sklearn.decomposition import PCA

def load_torch_image(file_name, device=torch.device("cpu")):
    """Loads an image and adds batch dimension"""
    img = K.io.load_image(file_name, K.io.ImageLoadType.RGB32, device=device)[None, ...]
    return img

def embed_images(
        paths: list[Path],
        model_name: str,
        device: torch.device = torch.device("cpu"),
) -> T:
    """Computes image embeddings.
    Returns a tensor of shape [len(filenames), output_dim]
    """
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)

    embeddings = []

    for i, path in tqdm(enumerate(paths), desc="Global descriptors"):
        image = load_torch_image(path)

        with torch.inference_mode():
            inputs = processor(images=image, return_tensors="pt", do_rescale=False).to(device)
            outputs = model(**inputs)  # last_hidden_state and pooled

            # Max pooling over all the hidden states but the first (starting token)
            # To obtain a tensor of shape [1, output_dim]
            # We normalize so that distances are computed in a better fashion later
            embedding = F.normalize(outputs.last_hidden_state[:, 1:].max(dim=1)[0], dim=-1, p=2)

        embeddings.append(embedding.detach().cpu())
    return torch.cat(embeddings, dim=0)

class Dinov2FeatureExtractor:
    def __init__(self, repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14", smaller_edge_size=-1,
                 half_precision=False, device="cuda"):
        self.repo_name = repo_name
        self.model_name = model_name
        self.smaller_edge_size = smaller_edge_size
        self.half_precision = half_precision
        self.device = device

        if self.half_precision:
            self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).half().to(self.device)
        else:
            self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).to(self.device)

        self.model.eval()

        if smaller_edge_size == -1:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # imagenet defaults
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size=smaller_edge_size, interpolation=transforms.InterpolationMode.BICUBIC,
                                  antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # imagenet defaults
            ])

    def set_smaller_edge_size(self, smaller_edge_size):
        if smaller_edge_size == -1:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # imagenet defaults
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size=smaller_edge_size, interpolation=transforms.InterpolationMode.BICUBIC,
                                  antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # imagenet defaults
            ])

    # https://github.com/facebookresearch/dinov2/blob/255861375864acdd830f99fdae3d9db65623dafe/notebooks/features.ipynb
    def prepare_image(self, rgb_image_numpy):
        image = Image.fromarray(rgb_image_numpy)
        image_tensor = self.transform(image)
        resize_scale = image.width / image_tensor.shape[2]

        # Crop image to dimensions that are a multiple of the patch size
        height, width = image_tensor.shape[1:]  # C x H x W
        cropped_width, cropped_height = width - width % self.model.patch_size, height - height % self.model.patch_size  # crop a bit from right and bottom parts
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]

        grid_size = (cropped_height // self.model.patch_size, cropped_width // self.model.patch_size)
        return image_tensor, grid_size, resize_scale

    def extract_features(self, rgb_image_numpy, device = "cpu"):
        image_tensor, grid_size, resize_scale = self.prepare_image(rgb_image_numpy)
        with torch.inference_mode():
            if self.half_precision:
                image_batch = image_tensor.unsqueeze(0).half().to(self.device)
            else:
                image_batch = image_tensor.unsqueeze(0).to(self.device)

            tokens = self.model.get_intermediate_layers(image_batch)[0].squeeze()
            # if device == "cpu":
            #     tokens = tokens.cpu().numpy()
        return tokens.cpu().numpy(), grid_size

    def get_embedding_visualization(self, tokens, grid_size, resized_mask=None):
        pca = PCA(n_components=3)
        if resized_mask is not None:
            tokens = tokens[resized_mask]
        reduced_tokens = pca.fit_transform(tokens.astype(np.float32))
        if resized_mask is not None:
            tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
            tmp_tokens[resized_mask] = reduced_tokens
            reduced_tokens = tmp_tokens
        reduced_tokens = reduced_tokens.reshape((*grid_size, -1))
        normalized_tokens = (reduced_tokens - np.min(reduced_tokens)) / (
                    np.max(reduced_tokens) - np.min(reduced_tokens))
        return normalized_tokens

    # def idx_to_source_position(self, idx, grid_size, resize_scale):
    #     row = (idx // grid_size[1]) * self.model.patch_size * resize_scale + self.model.patch_size / 2
    #     col = (idx % grid_size[1]) * self.model.patch_size * resize_scale + self.model.patch_size / 2
    #     return row, col