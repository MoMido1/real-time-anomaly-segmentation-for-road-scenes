from PIL import Image
import numpy as np
import os
import cv2
#this code has 2 count functions the first one will read the image as a grey scale image
# and according to that the max number of classes in one image will be 18 and here we are using
#a cv2 library

# the second function of count uploads the image as it is and transform it to 
#grey scale image and save it as a new image and we won't need that

def count_pixel_occurrences(image_path,filename):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    unique_values, counts = np.unique(image, return_counts=True)
    
    return len(unique_values)

# def count_pixel_occurrences(image_path,filename):
#     img = Image.open(image_path)
#     img_array = np.array(img)
#     # print(img_array.shape)
#     flattened_array = img_array.reshape(-1, img_array.shape[-1])

# # Get unique colors and their counts
#     unique_values, counts = np.unique(flattened_array, axis=0, return_counts=True)
#     color_to_gray_mapping = {tuple(color): i for i, color in enumerate(unique_values)}

# # Replace each color in the image array with its corresponding grayscale value
#     gray_image_array = np.array([[color_to_gray_mapping[tuple(pixel)] for pixel in row] for row in img_array])

# # Convert the grayscale array back to an image
#     gray_image = Image.fromarray(gray_image_array.astype(np.uint8))

# # Save or display the grayscale image
#     # gray_image.show()
#     gray_image.save("D:\COMPUTER_DEPARTMENT\\3RD_YEAR\AML\AML_Project\Project_Repository\ERF_Net\\train\leftImg8bit_trainvaltest\DB_New\\valannot_m\\"+filename)
#     # flattened_array = img_array.flatten()
#     # unique_values, _ = np.unique(flattened_array, return_counts=True)
#     # print(unique_values)
#     return len(unique_values)

def max_classes_in_directory(directory_path):
    max_classes = 0

    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        print(filename)
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Add more image file extensions if needed
            image_path = os.path.join(directory_path, filename)
            num_classes = count_pixel_occurrences(image_path,filename)
            # print(num_classes)
            # Update max_classes if the current image has more classes
            max_classes = max(max_classes, num_classes)

    return max_classes

# Example usage:
directory_path = "D:\COMPUTER_DEPARTMENT\\3RD_YEAR\AML\AML_Project\Project_Repository\ERF_Net\\train\leftImg8bit_trainvaltest\DB_New\\trainannot_f"  # Replace with the path to your image directory
max_classes = max_classes_in_directory(directory_path)

print(f"The maximum number of classes in one image is: {max_classes}")
