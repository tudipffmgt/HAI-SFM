import os
import subprocess

from image_processing import downsample_images, rotate_images
from generate_image_pairs import get_image_pairs, get_image_tracks


def sg_feature_matching(input_dir, superglue_path, image_pairs, setting, output_dir):
    # List all the downsampled image files in the output directory.
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.jpg')]

    if len(image_files) > 0:
        print(f'Found {len(image_files)} downsampled image files in {input_dir}.')
    else:
        print(f'No downsampled image files found in {input_dir}.')
        return

    # Run SuperGlue feature matching on each pair of consecutive images.
    # out_file = os.path.join(output_dir, f'{os.path.basename(img1)[:-4]}_{os.path.basename(img2)[:-4]}.h5')
    cmd = f'python {superglue_path} --input_dir {input_dir} --input_pairs {image_pairs} --superglue {setting} ' \
          f'--output_dir {output_dir} --max_keypoints {1024} --resize {-1}'

    print(f'Running SuperGlue on {input_dir}')
    subprocess.run(cmd.split())

    print('SuperGlue feature matching completed.')


def main():
    # Define the path to your data directory.
    # This is the intra-epoch workflow!

    input_dir = 'data'
    output_dir_downsampled = 'downsampled'
    output_dir_superglue = 'superglue'

    # Create directories if they do not exist
    if not os.path.exists(output_dir_superglue):
        os.makedirs(output_dir_superglue)

    # Part 1: Downsampling
    ext = downsample_images(input_dir, output_dir_downsampled)

    # Part 2: Get image pairs
    image_pairs = get_image_pairs(output_dir_downsampled, output_dir_downsampled, False)

    # Part 3: SuperGlue matching
    # Define the path to the SuperGlue repository.
    superglue_path = 'SuperGluePretrainedNetwork/match_pairs.py'

    # Perform SuperGlue feature matching on all image pairs.
    setting = 'outdoor'
    sg_feature_matching(output_dir_downsampled, superglue_path, image_pairs, setting, output_dir_superglue)

    # Part 4: Find feature tracks
    image_tracks = get_image_tracks(output_dir_superglue)

    # Part 5: Rotate images of the smallest separate track
    # rotate_images(input_dir, image_tracks, ext)


main()


