import os
import subprocess


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

