import numpy as np
import torch
import matplotlib.pyplot as plt
from configs import open_configs
from dataset import FullImageDataset
from trainer import Trainer
from model import *
import argparse
import PIL
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str)
    parser.add_argument('-i', '--image_path', type=str)
    parser.add_argument('-g', '--ground_truth_path', type=str)
    parser.add_argument('-j', '--jit', action='store_true')

    args = parser.parse_args()
    model_path = args.model_path
    image_path = args.image_path
    ground_truth_path = args.ground_truth_path
    jit = args.jit

    data_configs, training_configs, model_configs = open_configs()
    if jit:
        model = torch.jit.load(model_path)
    else:
        model = Desnow(model_configs=model_configs, data_configs=data_configs).to(training_configs["device"])
        model.load_state_dict(torch.load(model_path))
    model.eval()
    
    if ground_truth_path=="" or ground_truth_path is None:
        image_data = FullImageDataset(image_path, image_path)
    else:
        image_data = FullImageDataset(image_path, ground_truth_path)

    images = []
    image = PIL.Image.open(image_path)
    images.append(np.array(image))
        
    with torch.no_grad():
        out = model(image_data[0][0].unsqueeze(0)).to('cpu')
        out = (out + 1) * 127.5
        out = torch.clamp(out, 0, 255).to(torch.uint8).numpy()
        out = out.squeeze()

    images.append(out)

    if ground_truth_path!="" or ground_truth_path is not None:
        gt = PIL.Image.open(ground_truth_path)
        images.append(np.array(gt))

    fig, axs = plt.subplots(1, len(images))
    for i, image in enumerate(images):
        match i:
            case 0:
                axs[i].set_title("Input")
            case 1:
                axs[i].set_title("Output")
            case 2:
                axs[i].set_title("Ground Truth")
        axs[i].imshow(image)
        axs[i].axis('off')
    plt.savefig(os.path.join(training_configs['fig_dir'], f'{image_path}_vis_eval.png'))
    plt.show()
                

    




        
    

    
