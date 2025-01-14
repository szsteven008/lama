import argparse
import os
import yaml
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
import onnx
import onnxruntime as ort

from saicinpainting.training.trainers import load_checkpoint

class LamaModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) :
        batch = {}
        batch['image'] = image
        batch['mask'] = mask
        outputs = model(batch)
        inpainted = outputs['inpainted'] * 255.0

#        masked_img = image * (1 - mask)
#        masked_img = torch.cat([masked_img, mask], dim=1)
#        predicted_image = self.model.generator(masked_img)
#        inpainted = (mask * predicted_image + (1 - mask) * image) * 255.0
        return inpainted.clamp(0, 255).type(torch.uint8)        

def export_onnx(model, output):
    onnx_file = output + "/" + "lama.onnx"
    torch.onnx.export(
        model,                  # model to export
        (
            torch.rand(1, 3, 512, 512).type(torch.float32), 
            torch.rand(1, 1, 512, 512).type(torch.float32), 
        ),        # inputs of the model,
        onnx_file,        # filename of the ONNX model
        input_names=[ "image", "mask" ],  # Rename inputs for the ONNX model
        output_names=[ "output" ],  # Rename inputs for the ONNX model
        opset_version = 17, 
        export_params = True, 
        do_constant_folding = True,
        dynamic_axes = {
            "image": { 0: "batch_size" },
            "mask": { 0: "batch_size" },
            "output": { 0: "batch_size", 2: 'height', 3: 'width' },
        })    
    print("export lama.onnx ok!")

    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)
    print("check lama.onnx ok!")

def inference_onnx(output):
    onnx_file = output + "/" + "lama.onnx"
    session = ort.InferenceSession(onnx_file)

    image = cv2.imread("images/1.png", cv2.IMREAD_COLOR)
    image = image.astype('float32') / 255.0
    image = np.transpose(image, (2, 0, 1))

    mask = cv2.imread("images/1_mask001.png", cv2.IMREAD_GRAYSCALE)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = mask.astype('float32') / 255.0
    _, mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
    mask = mask[None, ...]

    outputs = session.run(None, {
        "image": image[None, ...], 
        "mask": mask[None, ...],
    })

    cur_res = np.transpose(outputs[0].squeeze(0), (1, 2, 0))

    cv2.imshow('image', cur_res)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Export LAMA Model to ONNX", add_help=True)
    parser.add_argument("--inference", "-t", help="test lama.onnx model", action="store_true")
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    output_dir = args.output_dir
    
    # make dir
    os.makedirs(output_dir, exist_ok=True)
    
    with open(config_file, 'r') as f:
        config = OmegaConf.create(yaml.safe_load(f))
    config.training_model.predict_only = True
    config.visualizer.kind = 'noop'
    
    # Enable JIT version of FourierUnit, required for export
    config.generator.resnet_conv_kwargs.use_jit = True
    # Fix the configuration by setting the weight to zero
    config.losses.resnet_pl.weight = 0

    model = load_checkpoint(config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    model.to('cpu')

    lama = LamaModel(model)

    if args.inference:
        inference_onnx(output_dir)
    else:
        export_onnx(lama, output_dir)
