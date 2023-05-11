import os
import cv2


def downsample_images(input_dir, output_dir):

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

