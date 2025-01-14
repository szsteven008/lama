import argparse
import os
import yaml
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf

from saicinpainting.training.trainers import load_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser("lama inference demo.", add_help=True)
    parser.add_argument(
        "--image", "-i", type=str, required=True, help="input image"
    )
    parser.add_argument(
        "--mask", "-mask", type=str, required=True, help="input mask"
    )

    args = parser.parse_args()

    # cfg
    config_file = "checkpoints/big-lama/config.yaml"  # change the path of the model config file
    checkpoint_path = "checkpoints/big-lama/models/best.ckpt"  # change the path of the model
    image_file = args.image
    mask_file = args.mask

    with open(config_file, 'r') as f:
        config = OmegaConf.create(yaml.safe_load(f))
    config.training_model.predict_only = True
    config.visualizer.kind = 'noop'

    model = load_checkpoint(config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    model.to('cpu')

    image = cv2.imread(image_file, cv2.IMREAD_COLOR)
    image = image.astype('float32') / 255.0
    image = np.transpose(image, (2, 0, 1))

    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = mask.astype('float32') / 255.0
    _, mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
    mask = mask[None, ...]

    batch = {}
    batch['image'] = torch.tensor(image).unsqueeze(0).to('cpu')
    batch['mask'] = torch.tensor(mask).unsqueeze(0).to('cpu')

    outputs = model(batch)
    cur_res = (outputs["inpainted"][0] * 255.0).clamp(0, 255).type(torch.uint8)
    cur_res = cur_res.permute(1, 2, 0).detach().cpu().numpy()

    cv2.imshow('image', cur_res)
    cv2.waitKey()
    cv2.destroyAllWindows()
