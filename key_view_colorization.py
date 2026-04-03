import torch
import os
import cv2
import clip
import numpy as np
import argparse
from PIL import Image
from torchvision import transforms
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)


def load_images_from_folder(folder_path):
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if f.endswith((".png", ".jpg", ".JPG"))]
    return image_files


def extract_clip_features(image_paths):
    features = []
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = model.encode_image(image)
            feature = feature / feature.norm(dim=-1, keepdim=True)
        features.append(feature.cpu())

    return torch.cat(features, dim=0)


def compute_max_entropy_keyview(clip_features):
    clip_features = clip_features / torch.norm(clip_features, dim=1, keepdim=True)
    S = torch.matmul(clip_features, clip_features.T)
    P = torch.softmax(S, dim=1)
    entropy = -torch.sum(P * torch.log(P + 1e-8), dim=1)
    keyview_index = torch.argmax(entropy).item()
    return keyview_index


def compute_max_coverage_keyview(clip_features):
    clip_features = clip_features / torch.norm(clip_features, dim=1, keepdim=True)
    S = torch.matmul(clip_features, clip_features.T).float()
    S = S - torch.eye(S.shape[0], device=S.device) * 10
    min_similarity = torch.min(S, dim=1)[0]
    keyview_index = torch.argmax(min_similarity).item()
    return keyview_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, required=True, help='Path to the image folder')
    args = parser.parse_args()

    folder_path = args.folder_path
    image_paths = load_images_from_folder(folder_path)
    print(f"Find {len(image_paths)} images")

    clip_features = extract_clip_features(image_paths).float()

    keyview_entropy = compute_max_entropy_keyview(clip_features)

    print(f"Key view index: {keyview_entropy}, Image file: {image_paths[keyview_entropy]}")

    img_colorization = pipeline(Tasks.image_colorization, model='damo/cv_ddcolor_image-colorization')
    result = img_colorization(image_paths[keyview_entropy])
    cv2.imwrite('example/key_view_colorized.png', result[OutputKeys.OUTPUT_IMG])