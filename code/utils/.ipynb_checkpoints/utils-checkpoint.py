import matplotlib.pyplot as plt
from PIL.Image import Image as PilImage
import textwrap
import cv2 as cv

def display_images(
    images, 
    columns=4, width=20, height=8, max_images=15, 
    label_wrap_length=50, label_font_size=8):

    if not images:
        print("No images to display.")
        return 

    if len(images) > max_images:
        print(f"Showing {max_images} images of {len(images)}:")
        images=images[0:max_images]

    height = max(height, int(len(images)/columns) * height)
    plt.figure(figsize=(width, height))
    i=0
    for image in images :

        plt.subplot(len(images) / columns + 1, columns, i + 1)
        img = plt.imread(image)
        plt.imshow(img)
        i+=1

        if hasattr(image, 'filename'):
            title=image.filename