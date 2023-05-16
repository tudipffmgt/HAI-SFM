import os
import cv2
import sys
import math


def downsample_images(input_dir, output_dir, image_list):

    if not image_list:
        print('No image list was initialized. Processing all images...')

        # List all the image files in the data directory.
        image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                       if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif')]

        #
        _, ext = os.path.splitext(image_files[0])

        # Check if files were found
        if len(image_files) > 0:
            print(f'Found {len(image_files)} image files in directory {input_dir}.')

            # Rename files if necessary
            for image_path in image_files:
                if '_' in image_path:
                    new_image_path = image_path.replace('_', '-')
                    os.rename(image_path, new_image_path)
                    print('\033[91m' + 'Attention: ' + '\033[0m' + 'Renamed ' + image_path + ' to '
                          + new_image_path + ' because of ambiguous underscores')

                    # Update the image file list
                    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                                   if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif')]

                else:
                    pass  # print(f'No underscores found in {image_path}, no renaming necessary.')

        else:
            print(f'No image files found in {input_dir}.')
            sys.exit(1)

        # Process each image file.
        for image_path in image_files:
            # Downsample the image.
            print(f'Downsampling {image_path}...')

            # Load the input image.
            img = cv2.imread(image_path)

            # Determine the longer edge of the image.
            height, width = img.shape[:2]
            if height >= width:
                longer_edge = height
            else:
                longer_edge = width

            # Downsample the image if necessary.
            if longer_edge > 1600:
                downsample_factor = longer_edge / 1600
                new_height, new_width = int(height / downsample_factor), int(width / downsample_factor)
                img = cv2.resize(img, (new_width, new_height))

            # Construct the output file path.
            output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '.jpg')

            # Save the downscaled image to the output directory.
            cv2.imwrite(output_path, img)

        return downsample_factor, ext

    else:

        _, ext = os.path.splitext(image_list[0])

        # Process each image file.
        for image_path in image_list:
            # Downsample the image.
            print(f'Downsampling {image_path}...')

            # Load the input image.
            img = cv2.imread(image_path)

            # Determine the longer edge of the image.
            height, width = img.shape[:2]
            if height >= width:
                longer_edge = height
            else:
                longer_edge = width

            # Downsample the image if necessary.
            if longer_edge > 1600:
                downsample_factor = longer_edge / 1600
                new_height, new_width = int(height / downsample_factor), int(width / downsample_factor)
                img = cv2.resize(img, (new_width, new_height))

            # Construct the output file path.
            output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '.jpg')

            # Save the downscaled image to the output directory.
            cv2.imwrite(output_path, img)

    return ext


def rotate_images(input_dir, image_tracks, ext):

    # Find the smallest track
    smallest_track = None
    min_length = float('inf')

    for track in image_tracks.values():
        track_length = len(track)
        if track_length < min_length:
            min_length = track_length
            smallest_track = track

    print("Smallest track:", ", ".join(sorted(list(smallest_track))))

    modified_images = []
    for image_filename in smallest_track:
        # Construct the full path to the image file
        img_without_ext = os.path.splitext(image_filename)[0]

        img_path = os.path.join(input_dir, f'{img_without_ext}{ext}')

        print(f'Rotating image {img_path} by 180 degrees.')
        img = cv2.imread(img_path)
        img_rotated = cv2.rotate(img, cv2.ROTATE_180)

        cv2.imwrite(img_path, img_rotated)

        # Append the rotated image path to the list
        modified_images.append(img_path)

    return modified_images


def split_images(input_dir, output_dir, size_x=1600, size_y=1600):

    # List all the image files in the data directory.
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                   if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif')]

    # Process each image file.
    for image_path in image_files:
        # Downsample the image.
        print(f'Splitting {image_path}...')

        img = cv2.imread(image_path)

        num_tiles_x = math.ceil(img.shape[1] / size_x)
        num_tiles_y = math.ceil(img.shape[0] / size_y)

        image_name = os.path.splitext(os.path.basename(image_path))[0]

        for rows in range(num_tiles_y):
            y_start = rows * size_y
            y_end = min((rows + 1) * size_y, img.shape[0])

            for cols in range(num_tiles_x):
                x_start = cols * size_x
                x_end = min((cols + 1) * size_x, img.shape[1])

                current_tile = img[y_start:y_end, x_start:x_end]

                filepath = os.path.join(output_dir, f'{image_name}_ystart{y_start}_ystartxstart{x_start}_xstart.jpg')

                cv2.imwrite(filepath, current_tile)

