import os
import subprocess
import numpy as np
import re
import h5py

from image_processing import downsample_images, split_images, rotate_images
from generate_image_pairs import get_image_pairs, get_image_tracks


def sg_feature_matching(input_dir, superglue_path, image_pairs, setting, output_dir):
    # List all the downsampled image files in the output directory.
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.jpg')]

    if len(image_files) > 0:
        print(f'Found {len(image_files)} downsampled image files in {input_dir}.')
    else:
        print(f'No downsampled image files found in {input_dir}.')
        return

    # Clear the npz. files
    # TODO Could keep pre-processed feature files
    for file in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, file))

    # Run SuperGlue feature matching on each pair of consecutive images.
    cmd = f'python {superglue_path} --input_dir {input_dir} --input_pairs {image_pairs} --superglue {setting} ' \
          f'--output_dir {output_dir} --max_keypoints {1024} --resize {-1}'

    print(f'Running SuperGlue on {input_dir}')
    subprocess.run(cmd.split())

    print('SuperGlue feature matching completed.')


def retrieve_image_orientation(input_dir, superglue_path, num_flightstrips):

    iteration = 0
    max_iterations = num_flightstrips
    image_list = []

    num_files = len((os.listdir(input_dir)))

    output_dir_downsampled = 'output/downsampled'
    output_dir_superglue = 'output/superglue-results'

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
        superglue_path = os.path.join(superglue_path, 'match_pairs.py')

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


def disk_feature_matching(input_dir, disk_path):

    # List all the image files in the data directory.
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                   if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif')]

    #
    _, ext = os.path.splitext(image_files[0])

    disk_detection = os.path.join(disk_path, 'detect.py')
    disk_matching = os.path.join(disk_path, 'match.py')
    output_dir = 'output/disk-results'

    cmd_detect = f'python {disk_detection} {output_dir} {input_dir} ' \
                 f'--height 160 --width 160 --n 50 --image-extension {ext}'

    print(f'Running DISK feature detection on {input_dir}')
    subprocess.run(cmd_detect.split())

    cmd_match = f'python {disk_matching} {output_dir} --rt 0.95'

    print(f'Running DISK feature matching.')
    subprocess.run(cmd_match.split())


def tile_based_approach(input_dir, superglue_path, image_list=None):

    if image_list is None:
        image_list = []

    output_dir_downsampled = 'output/downsampled'
    output_dir_superglue = 'output/superglue-results'
    output_dir_split = 'output/split'

    superglue_path = os.path.join(superglue_path, 'match_pairs.py')

    # Create the output directory if it doesn't exist.
    if not os.path.exists(output_dir_downsampled):
        os.makedirs(output_dir_downsampled)

    if not os.path.exists(output_dir_superglue):
        os.makedirs(output_dir_superglue)

    if not os.path.exists(output_dir_downsampled):
        os.makedirs(output_dir_downsampled)

    downsample_factor, ext = downsample_images(input_dir, output_dir_downsampled, image_list)
    _ = get_image_tracks(input_dir, output_dir_superglue, downsample_factor)

    image_pairs = split_images(input_dir, output_dir_split)

    setting = 'outdoor'
    sg_feature_matching(output_dir_split, superglue_path, image_pairs, setting, output_dir_superglue)


def merge_npz_files(input_dir):
    # List all npz files in the data directory.
    npz_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.npz')]

    # Check if files were found
    if len(npz_files) > 0:
        print(f'Found {len(npz_files)} npz files in {input_dir}.')
    else:
        print(f'No npz files found in {input_dir}.')

    print(f'Reading npz files.')

    kp_file_path = 'output/h5/keypoints.h5'
    matches_file_path = 'output/h5/matches.h5'
    # Check if keypoint/matches file exist and delete
    if os.path.isfile(kp_file_path):
        os.remove(kp_file_path)
    if os.path.isfile(matches_file_path):
        os.remove(matches_file_path)

    # Dictionary to store keypoints with unique identifier as img1 or img2
    keypoints_dict = {}

    with h5py.File(kp_file_path, 'a') as keypoint_file, h5py.File(matches_file_path, 'a') as matches_file:
        for npz_file in npz_files:

            npz = np.load(npz_file)
            matches = npz['matches']
            matching_indices = np.where(matches > -1)[0]

            if matching_indices.size > 0:
                # Extract the original file names by using regular expressions
                filename = os.path.split(npz_file)[1]
                # Extract the original image names using regular expressions
                img1_match = re.search(r'(.+?)_', filename)
                img1 = img1_match.group(1)
                img2_match = re.search(r'x_(.*?)_y', filename)
                img2 = img2_match.group(1)

                # Extract the x-shift and y-shift values for each tile-image
                shift_values = re.findall(r'_y(\d+)y~x(\d+)x', filename)
                img1_yshift, img1_xshift = map(int, shift_values[0])
                img2_yshift, img2_xshift = map(int, shift_values[1])

                # Process keypoints and matches
                keypoints0 = npz['keypoints0'][matching_indices]
                keypoints0 = keypoints0 + np.array([img1_xshift, img1_yshift])

                keypoints1_indices = matches[matching_indices]
                keypoints1 = npz['keypoints1'][keypoints1_indices]
                keypoints1 = keypoints1 + np.array([img2_xshift, img2_yshift])

                if img1 not in keypoints_dict:
                    keypoints_dict[img1] = keypoints0
                    img1_indices = range(len(keypoints0))
                else:
                    prev_len = len(keypoints_dict[img1])
                    keypoints_dict[img1] = np.concatenate([keypoints_dict[img1], keypoints0], axis=0)
                    img1_indices = range(prev_len, prev_len + len(keypoints0))

                if img2 not in keypoints_dict:
                    keypoints_dict[img2] = keypoints1
                    img2_indices = range(len(keypoints1))
                else:
                    prev_len = len(keypoints_dict[img2])
                    keypoints_dict[img2] = np.concatenate([keypoints_dict[img2], keypoints1], axis=0)
                    img2_indices = range(prev_len, prev_len + len(keypoints1))

                # Write to file keypoints.h5
                if img1 in keypoint_file:
                    keypoint_file[img1].resize((keypoint_file[img1].shape[0] + len(img1_indices)), axis=0)
                    keypoint_file[img1][-len(img1_indices):] = keypoints_dict[img1][img1_indices]
                else:
                    keypoint_file.create_dataset(img1, data=keypoints_dict[img1][img1_indices], chunks=True,
                                                 maxshape=(None, 2))

                if img2 in keypoint_file:
                    keypoint_file[img2].resize((keypoint_file[img2].shape[0] + len(img2_indices)), axis=0)
                    keypoint_file[img2][-len(img2_indices):] = keypoints_dict[img2][img2_indices]
                else:
                    keypoint_file.create_dataset(img2, data=keypoints_dict[img2][img2_indices], chunks=True,
                                                 maxshape=(None, 2))
                # Write to file matches.h5
                group_name = f'/{img1}'
                if group_name in matches_file:
                    group = matches_file[group_name]
                else:
                    group = matches_file.create_group(group_name)

                dataset_name = f'{img2}'
                if dataset_name in group:
                    dataset = group[dataset_name]
                    dataset.resize((2, dataset.shape[1] + len(img1_indices)))
                    dataset[:, -len(img1_indices):] = np.vstack([img1_indices, img2_indices])
                else:
                    dataset = group.create_dataset(dataset_name, data=np.vstack([img1_indices, img2_indices]),
                                                   chunks=True, maxshape=(2, None))


def h5_to_colmap(input_dir, disk_path):

    disk_to_colmap = os.path.join(disk_path, 'colmap/h5_to_db.py')

    output_dir = 'output/colmap/database.db'

    # List all the image files in the data directory.
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                   if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif')]

    #
    _, ext = os.path.splitext(image_files[0])

    cmd_colmap = f'python {disk_to_colmap} --database-path {output_dir} output/h5 {input_dir} --single-camera ' \
                 f'--image-extension {ext}'

    print('Converting the h5 files to COLMAP database format.')
    subprocess.run(cmd_colmap.split())
