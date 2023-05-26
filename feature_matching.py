import os
import subprocess
import numpy as np
import re
import h5py


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
    # out_file = os.path.join(output_dir, f'{os.path.basename(img1)[:-4]}_{os.path.basename(img2)[:-4]}.h5')
    cmd = f'python {superglue_path} --input_dir {input_dir} --input_pairs {image_pairs} --superglue {setting} ' \
          f'--output_dir {output_dir} --max_keypoints {1024} --resize {-1}'

    print(f'Running SuperGlue on {input_dir}')
    subprocess.run(cmd.split())

    print('SuperGlue feature matching completed.')


# def disk_feature_matching(input_dir, disk_path, output_dir):
#
#     disk_feature_detection = os.path.join(disk_path, 'detect.py')
#
#     cmd = f'python {disk_feature_detection} {output_dir} {input_dir}'
#
#     print(f'Running DISK on {input_dir}')
#     subprocess.run(cmd.split())
#
#     print('DISK feature matching completed.')

def merge_npz_files(input_dir):
    # List all npz files in the data directory.
    npz_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.npz')]

    # Check if files were found
    if len(npz_files) > 0:
        print(f'Found {len(npz_files)} npz files in {input_dir}.')
    else:
        print(f'No npz files found in {input_dir}.')

    print(f'Reading npz files.')

    kp_file_path = 'h5/keypoints.h5'
    matches_file_path = 'h5/matches.h5'
    # Check if keypoint/matches file exist and delete
    if os.path.isfile(kp_file_path):
        os.remove(kp_file_path)
    if os.path.isfile(matches_file_path):
        os.remove(matches_file_path)

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

                # Extract only matched keypoints and apply the shift
                keypoints0 = npz['keypoints0'][matching_indices]
                keypoints0 = keypoints0 + np.array([img1_xshift, img1_yshift])

                keypoints1_indices = matches[matching_indices]
                keypoints1 = npz['keypoints1'][keypoints1_indices]
                keypoints1 = keypoints1 + np.array([img2_xshift, img2_yshift])

                # Write to keypoints.h5
                if img1 in keypoint_file:
                    keypoint_file[img1].resize((keypoint_file[img1].shape[0] + keypoints0.shape[0]), axis=0)
                    keypoint_file[img1][-keypoints0.shape[0]:] = keypoints0
                else:
                    keypoint_file.create_dataset(img1, data=keypoints0, chunks=True, maxshape=(None, 2))

                if img2 in keypoint_file:
                    keypoint_file[img2].resize((keypoint_file[img2].shape[0] + keypoints1.shape[0]), axis=0)
                    keypoint_file[img2][-keypoints1.shape[0]:] = keypoints1
                else:
                    keypoint_file.create_dataset(img2, data=keypoints1, chunks=True, maxshape=(None, 2))

                # Write to matches.h5
                # TODO fix order and size
                group_name = f'/{img1}'
                if group_name in matches_file:
                    group = matches_file[group_name]
                else:
                    group = matches_file.create_group(group_name)

                dataset_name = f'{img2}'
                if dataset_name in group:
                    dataset = group[dataset_name]
                    dataset.resize((dataset.shape[0] + keypoints0.shape[0]), axis=0)
                    dataset[-keypoints0.shape[0]:] = np.arange(keypoints0.shape[0])
                else:
                    dataset = group.create_dataset(dataset_name, data=np.arange(keypoints0.shape[0]), chunks=True,
                                                   maxshape=(None,))



