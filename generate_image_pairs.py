import os
import numpy as np


def get_image_pairs(input_dir, output_dir, ordered=True, along_track_overlap=0.6, cross_track_overlap=0.3, ):

    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.jpg')]

    # Create the matched_pairs file
    if os.path.exists(os.path.join(output_dir, 'matched_pairs.txt')):
        print(f'matched_pairs.txt already found in directory. Deleting old file...')
        os.remove(os.path.join(output_dir, 'matched_pairs.txt'))

        matched_pairs_file = os.path.join(output_dir, 'matched_pairs.txt')
    else:
        matched_pairs_file = os.path.join(output_dir, 'matched_pairs.txt')

    # Check if the files are ordered (user input)
    if ordered:
        print('ordered')
        # TODO implement the smart along_track and cross_track_overlap
    else:
        for i in range(len(image_files) - 1):
            image_path1 = image_files[i]
            for j in range(i + 1, len(image_files)):
                image_path2 = image_files[j]
                with open(matched_pairs_file, 'a') as f:
                    f.write(os.path.basename(image_path1) + ' ' + os.path.basename(image_path2) + '\n')

    return matched_pairs_file


def extract_valid_image_pairs(input_dir, confidence_threshold=0.5, matches_threshold=100):

    # List all npz files in the data directory.
    npz_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.npz')]

    # Check if files were found
    if len(npz_files) > 0:
        print(f'Found {len(npz_files)} image files in {input_dir}.')
    else:
        print(f'No npz files found in {input_dir}.')

    for npz_file in npz_files:

        print(f'Reading npz files.')
        npz = np.load(npz_file)
        # extract keypoints0, keypoints1, matches, and match_confidence
        keypoints0 = npz['keypoints0']
        keypoints1 = npz['keypoints1']
        matches = npz['matches']
        match_confidence = npz['match_confidence']

        # check the shape of keypoints0, keypoints1, matches, and match_confidence
        # print('keypoints0 shape:', keypoints0.shape)
        # print('keypoints1 shape:', keypoints1.shape)
        # print('matches shape:', matches.shape)
        # print('match_confidence shape:', match_confidence.shape)

        num_matches = np.sum(match_confidence > confidence_threshold)

        print(num_matches)
        # TODO print the image names to receive feature tracks!
