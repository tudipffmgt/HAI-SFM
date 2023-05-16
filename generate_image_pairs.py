import os
import numpy as np


def bounding_box(points, upsample_factor=1):
    x_coordinates, y_coordinates = zip(*points)

    return [(min(x_coordinates) * upsample_factor, min(y_coordinates) * upsample_factor),
            (max(x_coordinates) * upsample_factor, max(y_coordinates) * upsample_factor)]


def get_image_pairs(input_dir, output_dir, ordered=True, along_track_overlap=0.6, cross_track_overlap=0.3):

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
        print('Assuming that the images in the directory are' + '\033[32m' + ' ordered.' + '\033[0m')

        inv_overlap = 1 - along_track_overlap
        num_sequential_images = 0.96/inv_overlap  # one would expect 1/inv_overlap,
        # but we do not consider border regions (2% of image borders) for matching

        for i in range(len(image_files) - 1):
            image_path1 = image_files[i]
            for j in range(i + 1, len(image_files)):
                if j - i <= num_sequential_images:
                    image_path2 = image_files[j]
                    with open(matched_pairs_file, 'a') as f:
                        f.write(os.path.basename(image_path1) + ' ' + os.path.basename(image_path2) + '\n')
                else:
                    break

        # TODO cross_track_overlap
    else:
        print('Assuming that the images in the directory are' + '\033[33m' + ' not ordered. ' + '\033[0m' +
              'Write an exhaustive matching file.')
        for i in range(len(image_files) - 1):
            image_path1 = image_files[i]
            for j in range(i + 1, len(image_files)):
                image_path2 = image_files[j]
                with open(matched_pairs_file, 'a') as f:
                    f.write(os.path.basename(image_path1) + ' ' + os.path.basename(image_path2) + '\n')

    return matched_pairs_file

def get_image_pairs_tiles():
    # TODO Continue here!!
    input_file = os.path.join('data', 'matches_bounding_boxes.txt')

    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',')
                img1_path = parts[0].strip()
                img2_path = parts[1].strip()
                bbox1 = eval(parts[2].strip())
                bbox2 = eval(parts[3].strip())
                match = (img1_path, img2_path, bbox1, bbox2)
                print(match)
                #matches.append(match)



def get_image_tracks(input_dir_data, input_dir_superglue, downsample_factor, confidence_threshold=0.5, matches_threshold=40):

    # List all npz files in the data directory.
    npz_files = [os.path.join(input_dir_superglue, f) for f in os.listdir(input_dir_superglue) if f.endswith('.npz')]

    # Check if files were found
    if len(npz_files) > 0:
        print(f'Found {len(npz_files)} image files in {input_dir_superglue}.')
    else:
        print(f'No npz files found in {input_dir_superglue}.')

    # create an empty dictionary to store matched images
    image_tracks = {}

    # Write bounding boxes of feature matches to file
    file = open(os.path.join(input_dir_data, 'matches_bounding_boxes.txt'), 'w')

    print(f'Reading npz files.')
    for npz_file in npz_files:

        npz = np.load(npz_file)

        # extract match_confidence
        match_confidence = npz['match_confidence']
        high_confidence_indices = np.where(match_confidence > confidence_threshold)
        num_matches = np.sum(match_confidence > confidence_threshold)

        # print(num_matches)
        if num_matches > matches_threshold:

            # Extract the filenames
            basename = os.path.basename(npz_file)  # extract filename without directory path
            image_names = basename.split("_matches.npz")[0].split("_")

            # Extract keypoints, matches, and match confidence from the npz file
            keypoints0 = npz['keypoints0']
            keypoints1 = npz['keypoints1']
            matches = npz['matches']

            # Extract keypoints with high confidence
            high_confidence_keypoints0 = keypoints0[high_confidence_indices]
            high_confidence_keypoints1 = keypoints1[matches[high_confidence_indices]]

            # Calculate the bounding boxes for the keypoints
            b1 = bounding_box(high_confidence_keypoints0, downsample_factor)
            b2 = bounding_box(high_confidence_keypoints1, downsample_factor)

            # Print the extracted keypoints and matches
            print(f'{image_names[0]}, {image_names[1]}, {b1}, {b2}')
            file.write(f'{image_names[0]}, {image_names[1]}, {b1}, {b2}\n')

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

    file.close()

    # Print tracks
    for track in image_tracks.values():
        print(track)
        #print("Track:", ", ".join(sorted(list(track))))



    return image_tracks
    # TODO if tracks are empty use DISK!
