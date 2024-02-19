from PIL import Image
import numpy as np
import os

def count_pixel_occurrences(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)
    flattened_array = img_array.flatten()
    unique_values, _ = np.unique(flattened_array, return_counts=True)
    return len(unique_values)

def max_classes_in_directory(directory_path):
    max_classes = 0

    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Add more image file extensions if needed
            image_path = os.path.join(directory_path, filename)
            num_classes = count_pixel_occurrences(image_path)

            # Update max_classes if the current image has more classes
            max_classes = max(max_classes, num_classes)

    return max_classes

# Example usage:
directory_path = "D:\COMPUTER_DEPARTMENT\\3RD_YEAR\AML\AML_Project\Project_Repository\ENet\content\\trainannot"  # Replace with the path to your image directory
max_classes = max_classes_in_directory(directory_path)

print(f"The maximum number of classes in one image is: {max_classes}")
