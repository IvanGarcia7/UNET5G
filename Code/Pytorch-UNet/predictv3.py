import argparse
import logging
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from utils.data_loading import CustomDataset  # Update this import based on your dataset definition
from unet import UNet # Update this import based on your model definition
from torch.nn import DataParallel

def predict_regression(net, input_data, device):
    net.eval()
    input_data = torch.from_numpy(input_data).unsqueeze(0).float()

    #input_data = input_data.unsqueeze(0)
    input_data = input_data.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(input_data).cpu()
        #regression_values = output.view(-1).numpy()
        #regression_values = output.view(100, 100).numpy()  # Ajusta esta línea según tus necesidades
        regression_values = output


    return regression_values

def get_args():
    parser = argparse.ArgumentParser(description='Predict values from input matrices using regression model')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input matrices', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output files')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')

    return parser.parse_args()

def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.npy'

    return args.output or list(map(_generate_name, args.input))

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = get_output_filenames(args)

    net = torch.nn.DataParallel(UNet(n_channels=8, n_classes=1, bilinear=args.bilinear))  # Update with your model definition

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    net.load_state_dict(state_dict)


    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting values for matrix {filename} ...')
        input_matrix = np.load(filename)
        print(input_matrix.shape)

        regression_values = predict_regression(net=net,
                                               input_data=input_matrix,
                                               device=device)

        if not args.no_save:
            out_filename = out_files[i]
            np.save(out_filename, regression_values)
            logging.info(f'Regression values saved to {out_filename}')
