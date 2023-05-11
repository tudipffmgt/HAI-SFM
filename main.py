import os
import subprocess

from image_processing import downsample_image


def sg_feature_matching(input_dir, superglue_path, setting, output_dir):
    # List all the downsampled image files in the output directory.
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.jpg')]

    if len(image_files) > 0:
        print(f'Found {len(image_files)} downsampled image files in {input_dir}.')
    else:
        print(f'No downsampled image files found in {input_dir}.')
        return

    # Run SuperGlue feature matching on each pair of consecutive images.
    # out_file = os.path.join(output_dir, f'{os.path.basename(img1)[:-4]}_{os.path.basename(img2)[:-4]}.h5')
    cmd = f'python {superglue_path} --input {input_dir} --superglue {setting} --output_dir {output_dir} --resize {-1} --no_display'
    print(f'Running SuperGlue on {input_dir}')
    subprocess.run(cmd.split())

    print('SuperGlue feature matching completed.')


def main():
    # Define the path to your data directory.
    input_dir = 'data'
    output_dir ='downsampled'

    # Create the output directory if it doesn't exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all the image files in the data directory.
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                   if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif')]

    # Check if files were found
    if len(image_files) > 0:
        print(f'Found {len(image_files)} image files in {input_dir}.')
    else:
        print(f'No image files found in {input_dir}.')

    # Process each image file.
    for image_path in image_files:
        # Downsample the image.
        print(f'Downsampling {image_path}...')
        downsampled_image = downsample_image(image_path, output_dir)

    # Define the path to the SuperGlue repository.
    superglue_path = 'SuperGluePretrainedNetwork/demo_superglue.py'
    output_dir_superglue = 'superglue'

    if not os.path.exists(output_dir_superglue):
        os.makedirs(output_dir_superglue)
    # Perform SuperGlue feature matching on all image pairs.
    setting = 'outdoor'
    sg_feature_matching(output_dir, superglue_path, setting, output_dir_superglue)


main()


