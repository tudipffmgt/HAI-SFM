import torch


def check_device(parameters):
    # Check if GPU is enabled
    if parameters['gpu']:
        if torch.cuda.is_available():
            # Set the device for PyTorch
            device = torch.device(parameters['gpu_device'])
            torch.cuda.set_device(device)
            print('Using GPU device ' + parameters['gpu_device'])
        else:
            print('GPU is not available. Switching to CPU.')
            device = torch.device('cpu')
    else:
        # Use CPU
        device = torch.device('cpu')
        print('Using CPU. Attention: Feature matching might be slow on CPU!')