from PIL import Image
import numpy as np

# Open an image file
image_path = "D:\COMPUTER_DEPARTMENT\\3RD_YEAR\AML\AML_Project\Project_Repository\ENet\eval\Validation_Dataset\RoadAnomaly\labels_masks\\17.png"  # Provide the path to your image file
image = Image.open(image_path)
# print(image.shape)
# Convert the image to a numpy array
image_array = np.array(image)

# Flatten the array to count unique colors
flattened_array = image_array.reshape(-1, image_array.shape[-1])

# Get unique colors and their counts
unique_values, counts = np.unique(flattened_array, axis=0, return_counts=True)

# Display the number of unique colors
print("Number of unique colors in the image:", len(unique_values))