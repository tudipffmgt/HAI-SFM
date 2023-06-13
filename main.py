import os
import argparse
import sys
import torch
from pathlib import Path

from image_processing import downsample_images, rotate_images, split_images
from generate_image_pairs import get_image_pairs, get_image_tracks
from feature_matching import sg_feature_matching, merge_npz_files ,disk_feature_matching


def retrieve_image_orientation(input_dir, num_flightstrips):

    iteration = 0
    max_iterations = num_flightstrips
    image_list = []

    num_files = len((os.listdir(input_dir)))

    output_dir_downsampled = 'downsampled'
    output_dir_superglue = 'superglue-results'

    # Create the output directory if it doesn't exist.
    if not os.path.exists(output_dir_downsampled):
        os.makedirs(output_dir_downsampled)

    if not os.path.exists(output_dir_superglue):
        os.makedirs(output_dir_superglue)

    while True:

        iteration += 1
        if iteration > max_iterations:
            print('Maximum number of iterations reached. '
                  'Could not find the correct orientation of all images using SuperGlue.')
            break
        # Part 1: Downsampling and retrieve original extension
        downsample_factor, ext = downsample_images(input_dir, output_dir_downsampled, image_list)

        # Part 2: Get image pairs
        image_pairs = get_image_pairs(output_dir_downsampled, output_dir_downsampled)

        # Part 3: SuperGlue matching
        # Define the path to the SuperGlue repository.
        superglue_path = 'SuperGluePretrainedNetwork/match_pairs.py'

        # Perform SuperGlue feature matching on all image pairs.
        setting = 'outdoor'
        sg_feature_matching(output_dir_downsampled, superglue_path, image_pairs, setting, output_dir_superglue)

        # Part 4: Find feature tracks and check the length
        image_tracks = get_image_tracks(input_dir, output_dir_superglue, downsample_factor)
        if len(image_tracks) == 1 and len(list(image_tracks.values())[0]) == num_files:
            # print('Found the correct image orientation for all images. Proceeding with tile-based SuperGlue.')
            print('Found the correct image orientation for all images. Proceeding with DISK.')
            break

        # Part 5: Rotate images of the smallest separate track
        modified_images = rotate_images(input_dir, image_tracks, ext)
        if modified_images == image_list:
            print('Rotating the same list of images around 180° - '
                  'The correct rotation could not be found for all images.')
            break
        else:
            image_list = modified_images


def tile_based_approach(input_dir, image_list=[]):

    output_dir_downsampled = 'downsampled'
    output_dir_superglue = 'superglue-results'
    output_dir_split = 'split'

    superglue_path = 'SuperGluePretrainedNetwork/match_pairs.py'

    # Create the output directory if it doesn't exist.
    if not os.path.exists(output_dir_downsampled):
        os.makedirs(output_dir_downsampled)

    if not os.path.exists(output_dir_superglue):
        os.makedirs(output_dir_superglue)

    if not os.path.exists(output_dir_downsampled):
        os.makedirs(output_dir_downsampled)

    downsample_factor, ext = downsample_images(input_dir, output_dir_downsampled, image_list)
    image_tracks = get_image_tracks(input_dir, output_dir_superglue, downsample_factor)

    image_pairs = split_images(input_dir, output_dir_split)

    setting = 'outdoor'
    sg_feature_matching(output_dir_split, superglue_path, image_pairs, setting, output_dir_superglue)


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


def main(parameters):

    check_device(parameters)

    input_dir = parameters['image_dir']
    if not input_dir.exists():
        print('Error: The specified image directory does not exist.')
        sys.exit(1)

    if parameters['config'] == 'default':
        if parameters['rotation'] == 'not-rotated':
            print('Running with default configuration using a combination of SuperGlue and DISK. '
                  'Calculate the correct rotation for the historical aerial images.')
            num_flightstrips = parameters['flightstrips']
            retrieve_image_orientation(input_dir, num_flightstrips)

            #TODO Run DISK feature matching
            disk_feature_matching(input_dir, )

        elif parameters['rotation'] == 'rotated':
            print('Running with default configuration using a combination of SuperGlue and DISK. '
                  'The images are already rotated for the use of learned methods.')

            # TODO Run DISK feature matching
            # disk_feature_matching()

    elif parameters['config'] == 't-ba':
        if parameters['rotation'] == 'not-rotated':
            print('Running the tile-based approach using exclusively SuperGlue.'
                  'Calculate the correct rotation for the historical aerial images.')

            num_flightstrips = parameters['flightstrips']
            retrieve_image_orientation(input_dir, num_flightstrips)

            tile_based_approach(input_dir)

        elif parameters['rotation'] == 'rotated':
            print('Running the tile-based approach using exclusively SuperGlue.'
                  'The images are already rotated for the use of learned methods.')

            tile_based_approach(input_dir)

    elif parameters['config'] == 'disk':
        if parameters['rotation'] == 'not-rotated':
            print('Running the approach using exclusively DISK.'
                  'Calculate the correct rotation for the historical aerial images.')

            # TODO Use DISK features to find correct rotation

        elif parameters['rotation'] == 'rotated':
            print('Running the approach using exclusively DISK.'
                  'The images are already rotated for the use of learned methods.')

            # TODO Run DISK feature matching
            # disk_feature_matching()

    elif parameters['config'] == 'tests':
        print('testing')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', action='store_true', help='Enable GPU usage')
    parser.add_argument('--gpu_device', type=int, default=0, help='Specify the GPU device index')
    parser.add_argument('--image_dir', type=Path, required=True, help="Path to original images in .jpg, .png, or .tif.")
    parser.add_argument('--superglue_path', type=Path, required=True, help="Path to SuperGlue")
    parser.add_argument('--disk_path', type=Path, required=True, help="Path to DISK")
    parser.add_argument('--config', type=str, choices=['default', 't-ba', 'disk', 'tests'], default='default')
    parser.add_argument('--rotation', type=str, choices=['rotated', 'not-rotated'], default='not-rotated',
                        help='Specify if the images are already rotated to be usable for learned matchers.')
    parser.add_argument('--flightstrips', type=int, default=10, help='Number of flightstrips if known. '
                                                                     'If not known the parameter is set to 10.')

    args = parser.parse_args()
    args = vars(args)

    main(args)


