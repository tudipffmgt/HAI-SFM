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


def get_image_tracks(input_dir, confidence_threshold=0.5, matches_threshold=100):

    # List all npz files in the data directory.
    npz_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.npz')]

    # Check if files were found
    if len(npz_files) > 0:
        print(f'Found {len(npz_files)} image files in {input_dir}.')
    else:
        print(f'No npz files found in {input_dir}.')

    # create an empty dictionary to store matched images
    image_tracks = {}

    print(f'Reading npz files.')
    for npz_file in npz_files:

        npz = np.load(npz_file)

        # extract keypoints0, keypoints1, matches, and match_confidence
        # keypoints0 = npz['keypoints0']
        # keypoints1 = npz['keypoints1']
        # matches = npz['matches']
        match_confidence = npz['match_confidence']

        num_matches = np.sum(match_confidence > confidence_threshold)

        # print(num_matches)
        if num_matches > 100:

            basename = os.path.basename(npz_file)  # extract filename without directory path
            image_names = basename.split("_matches.npz")[0].split("_")
            image_name1 = image_names[0] + ".jpg"
            image_name2 = image_names[1] + ".jpg"

            # Check if either image name is already part of a track
            new_track = True
            for track_key in image_tracks.keys():
                if image_name1 in image_tracks[track_key] or image_name2 in image_tracks[track_key]:
                    image_tracks[track_key].add(image_name1)
                    image_tracks[track_key].add(image_name2)
                    new_track = False
                    break

            # If neither image name is part of a track, create a new track
            if new_track:
                track_key = f"{image_name1}_{image_name2}"
                image_tracks[track_key] = {image_name1, image_name2}

    # Print tracks
    for track in image_tracks.values():
        print(track)
        #print("Track:", ", ".join(sorted(list(track))))

    return image_tracks
    # TODO if tracks are empty use DISK!
