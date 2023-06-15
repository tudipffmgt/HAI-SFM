import torch
import os


def check_output_dir(filepath):

    # Check if the directory exists
    if not os.path.exists(filepath):
        # Create the directory
        os.makedirs(filepath)

def check_device(parameters):
    # Check if GPU is enabled
    if parameters['gpu']:
        if torch.cuda.is_available():
            # Set the device for PyTorch
            device = torch.device(parameters['gpu_device'])
            torch.cuda.set_device(device)
            print('Using GPU device ' + str(parameters['gpu_device']))
        else:
            print('GPU is not available. Switching to CPU.')
            device = torch.device('cpu')
    else:
        # Use CPU
        device = torch.device('cpu')
        print('Using CPU. Attention: Feature matching might be slow on CPU!')