import os

from image_processing import downsample_images, rotate_images, split_images
from generate_image_pairs import get_image_pairs, get_image_tracks
from feature_matching import sg_feature_matching #, disk_feature_matching


def main():
    # Define the path to your data directory.
    # This is the intra-epoch workflow!

    # TODO while loop for the whole workflow - stop processing when image_tracks show only one track
    #  or no matches are found

    input_dir = 'data'
    num_files = len((os.listdir(input_dir)))
    output_dir_downsampled = 'downsampled'
    output_dir_split = 'split'
    output_dir_superglue = 'superglue-results'
    output_dir_disk = 'disk-results'
    image_list = []

    max_iterations = 10
    iteration = 0

    # Create directories if they do not exist
    # Create the output directory if it doesn't exist.
    if not os.path.exists(output_dir_downsampled):
        os.makedirs(output_dir_downsampled)

    if not os.path.exists(output_dir_superglue):
        os.makedirs(output_dir_superglue)

    if not os.path.exists(output_dir_disk):
        os.makedirs(output_dir_disk)

    if not os.path.exists(output_dir_split):
        os.makedirs(output_dir_split)

    # disk_path = 'disk'
    # disk_feature_matching(input_dir, disk_path, output_dir_disk)

    downsample_factor, ext = downsample_images(input_dir, output_dir_downsampled, image_list)
    image_tracks = get_image_tracks(input_dir, output_dir_superglue, downsample_factor)

    split_images(input_dir, output_dir_split)

    # while True:
    #
    #     iteration += 1
    #     if iteration > max_iterations:
    #         print('Maximum number of iterations reached. '
    #               'Could not find the correct orientation of all images using SuperGlue.')
    #         break
    #     # Part 1: Downsampling and retrieve original extension
    #     downsample_factor, ext = downsample_images(input_dir, output_dir_downsampled, image_list)
    #
    #     # Part 2: Get image pairs
    #     image_pairs = get_image_pairs(output_dir_downsampled, output_dir_downsampled)
    #
    #     # Part 3: SuperGlue matching
    #     # Define the path to the SuperGlue repository.
    #     superglue_path = 'SuperGluePretrainedNetwork/match_pairs.py'
    #
    #     # Perform SuperGlue feature matching on all image pairs.
    #     setting = 'outdoor'
    #     sg_feature_matching(output_dir_downsampled, superglue_path, image_pairs, setting, output_dir_superglue)
    #
    #     # Part 4: Find feature tracks and check the length
    #     image_tracks = get_image_tracks(output_dir_superglue)
    #     if len(image_tracks) == 1 and len(list(image_tracks.values())[0]) == num_files:
    #         print('Found the correct image orientation for all images. Proceeding with tile-based SuperGlue.')
    #         print('Found the correct image orientation for all images. Proceeding with DISK.')
    #         break
    #
    #     # Part 5: Rotate images of the smallest separate track
    #     modified_images = rotate_images(input_dir, image_tracks, ext)
    #     if modified_images == image_list:
    #         print('Rotating the same list of images around 180° - The correct rotation could not be found.')
    #         break
    #     else:
    #         image_list = modified_images


if __name__ == '__main__':
    main()


