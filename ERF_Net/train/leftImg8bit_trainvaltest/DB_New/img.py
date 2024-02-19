import cv2
import numpy as np
#this code is to display one image dimensions with the number of unique pixels like the number of different classes in that image

# Load the image
# image_path = "D:\COMPUTER_DEPARTMENT\\3RD_YEAR\AML\AML_Project\Project_Repository\ERF_Net\\train\leftImg8bit_trainvaltest\DB_New\\valannot_m/frankfurt_000000_001016_gtFine_color.png"  # Provide the path to your image file
image_path = "D:\COMPUTER_DEPARTMENT\\3RD_YEAR\AML\AML_Project\Project_Repository\ENet\eval\Validation_Dataset\RoadAnomaly\labels_masks\\6.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

unique_values, counts = np.unique(image, return_counts=True)

# Print the number of unique pixel values
num_unique_values = len(unique_values)
print("Number of unique pixel values:", num_unique_values)
print("The unique values are: ",str(unique_values))
# Check the dimensions of the image
height, width = image.shape
print("Image dimensions: {} x {}".format(width, height))

# Find the maximum pixel value
max_pixel_value = image.max()
print("Maximum pixel value:", max_pixel_value)
