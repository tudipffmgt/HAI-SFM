import argparse
import sys
from pathlib import Path

from utils import check_device
from feature_matching import retrieve_image_orientation, disk_feature_matching, tile_based_approach, h5_to_colmap


def main(parameters):

    # Check CPU/GPU use
    check_device(parameters)

    # Retrieve paths
    superglue_path = parameters['superglue_path']
    disk_path = parameters['disk_path']

    input_dir = parameters['image_dir']
    if not input_dir.exists():
        print('Error: The specified image directory does not exist.')
        sys.exit(1)

    # Check if images are ordered and apply overlap
    if parameters['ordered']:
        images_in_order = True
        along_track_overlap = parameters['along-track-overlap']

    # Check the tested rotation angle
    if parameters['rotation'] == 'not-_rotated':
        rotation_angle = 180
    if parameters['rotation'] == 'not-rotated-90':
        rotation_angle = 90

    # Default configuration
    if parameters['config'] == 'default':
        if parameters['rotation'] == 'not-rotated':
            print('Running with default configuration using a combination of SuperGlue and DISK. '
                  'Calculate the correct rotation for the historical aerial images.')
            num_flightstrips = parameters['flightstrips']
            retrieve_image_orientation(input_dir, superglue_path, num_flightstrips, rotation_angle)

            disk_feature_matching(input_dir, disk_path)
            if parameters['colmap']:
                h5_to_colmap(input_dir, disk_path)

        elif parameters['rotation'] == 'rotated':
            print('Running with default configuration using a combination of SuperGlue and DISK. '
                  'The images are already rotated for the use of learned methods.')

            disk_feature_matching(input_dir, disk_path)
            if parameters['colmap']:
                h5_to_colmap(input_dir, disk_path)

    # Tile-based approach
    elif parameters['config'] == 't-ba':
        if parameters['rotation'] == 'not-rotated':
            print('Running the tile-based approach using exclusively SuperGlue.'
                  'Calculate the correct rotation for the historical aerial images.')

            num_flightstrips = parameters['flightstrips']
            retrieve_image_orientation(input_dir, superglue_path, num_flightstrips, rotation_angle)

            tile_based_approach(input_dir, superglue_path)
            if parameters['colmap']:
                h5_to_colmap(input_dir, disk_path)

        elif parameters['rotation'] == 'rotated':
            print('Running the tile-based approach using exclusively SuperGlue.'
                  'The images are already rotated for the use of learned methods.')

            tile_based_approach(input_dir, superglue_path)
            if parameters['colmap']:
                h5_to_colmap(input_dir, disk_path)

    # DISK approach
    elif parameters['config'] == 'disk':
        if parameters['rotation'] == 'not-rotated':
            print('Running the approach using exclusively DISK.'
                  'Calculate the correct rotation for the historical aerial images.')

            num_flightstrips = parameters['flightstrips']
            retrieve_image_orientation(input_dir, disk_path, num_flightstrips, rotation_angle)
            # TODO Use DISK features to find correct rotation!!

        elif parameters['rotation'] == 'rotated':
            print('Running the approach using exclusively DISK.'
                  'The images are already rotated for the use of learned methods.')

            disk_feature_matching(input_dir, disk_path)
            if parameters['colmap']:
                h5_to_colmap(input_dir, disk_path)

    # TODO remove
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
    parser.add_argument('--ordered', action='store_true', help='The images in the data directory are already ordered '
                                                               'according to the flight route')
    parser.add_argument('--along-track-overlap', type=float, default=0.6, help='If the images are ordered the '
                                                                               'along-track overlap can be specified '
                                                                               'to reduce matching time')
    parser.add_argument('--rotation', type=str, choices=['rotated', 'not-rotated', 'not-rotated-90'], default='not-rotated',
                        help='Specify if the images are already rotated to be usable for learned matchers.')
    parser.add_argument('--flightstrips', type=int, default=10, help='Number of flightstrips if known. '
                                                                     'If not known the parameter is set to 10.')
    parser.add_argument('--colmap', action='store_true', help='Generate a COLMAP database from h5 files.')

    args = parser.parse_args()
    args = vars(args)

    main(args)


