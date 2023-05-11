import os
import cv2


def downsample_image(input_path, output_dir):

    # Load the input image.
    img = cv2.imread(input_path)

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
    output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(input_path))[0] + '.jpg')

    # Save the downscaled image to the output directory.
    cv2.imwrite(output_path, img)

    return img
