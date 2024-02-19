<<<<<<< HEAD
from PIL import Image
import numpy as np
import os
import cv2

allColors =[]
count = 0
def count_pixel_occurrences(image_path):
    img = Image.open(image_path)
    # print(img.size)
    img_array = np.array(img)
    flattened_array = img_array.flatten()
    unique_values, _ = np.unique(flattened_array, return_counts=True)
    return len(unique_values)

def count_pixels(image_path):
    # img = Image.open(image_path)
    # img_array = np.array(img)
    # flattened_array = img_array.flatten()
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('Grayscale Image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # img = np.array(img)
    #       # Resizing using nearest neighbor method
    # # img = cv2.resize(img, (320, 1000), cv2.INTER_NEAREST)
    # unique_values, _ = np.unique(img, return_counts=True)

    image = cv2.imread(image_path)
# Convert the image from BGR to RGB (OpenCV reads images in BGR format by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert the image to a 1D array
    pixels = np.reshape(image_rgb, (-1, 3))
    # Find unique colors and their counts
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    # add_missing_values(allColors,unique_colors)
    # print(unique_colors)
    concatenated_strings=[]
    for l in unique_colors:
        concatenated_strings.append(''.join(str(num) for num in l))

#     add_missing_values(allColors,concatenated_strings)

    # unique_colors = [i for i in range(len(unique_colors))]
    # print(unique_colors)
    # for color, count in zip(unique_colors, counts):
    #     print(f"Color: {color}, Count: {count}")
    return concatenated_strings


def add_missing_values(main_list, secondary_list):
    # print(secondary_list)
    for value in secondary_list:
        if value not in main_list:
            main_list.append(value)
    return main_list

def max_classes_in_directory(directory_path):
    max_classes = 0
    classes_nums = []
    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Add more image file extensions if needed
            image_path = os.path.join(directory_path, filename)
            num_classes = count_pixel_occurrences(image_path)

            classes = count_pixels(image_path)
            classes_nums=add_missing_values(classes_nums,classes)
            classes_nums.sort()
            print(f"The different classes along all the images are: {classes_nums}")
            print(f"The length of different classes along all the images are: {len(classes_nums)}")

            # Update max_classes if the current image has more classes
            max_classes = max(max_classes, num_classes)

    return max_classes

# Example usage:
directory_path = "D:\COMPUTER_DEPARTMENT\\3RD_YEAR\AML\AML_Project\Project_Repository\ENet\content\\valannot"  # Replace with the path to your image directory
max_classes = max_classes_in_directory(directory_path)
# print(allColors)
# print(len(allColors))
print(f"The maximum number of classes in one image is: {max_classes}")
=======
from PIL import Image
import numpy as np
import os
import cv2

allColors =[]
count = 0
def count_pixel_occurrences(image_path):
    img = Image.open(image_path)
    # print(img.size)
    img_array = np.array(img)
    flattened_array = img_array.flatten()
    unique_values, _ = np.unique(flattened_array, return_counts=True)
    return len(unique_values)

def count_pixels(image_path):
    # img = Image.open(image_path)
    # img_array = np.array(img)
    # flattened_array = img_array.flatten()
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('Grayscale Image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # img = np.array(img)
    #       # Resizing using nearest neighbor method
    # # img = cv2.resize(img, (320, 1000), cv2.INTER_NEAREST)
    # unique_values, _ = np.unique(img, return_counts=True)

    image = cv2.imread(image_path)
# Convert the image from BGR to RGB (OpenCV reads images in BGR format by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert the image to a 1D array
    pixels = np.reshape(image_rgb, (-1, 3))
    # Find unique colors and their counts
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    # add_missing_values(allColors,unique_colors)
    # print(unique_colors)
    concatenated_strings=[]
    for l in unique_colors:
        concatenated_strings.append(''.join(str(num) for num in l))

#     add_missing_values(allColors,concatenated_strings)

    # unique_colors = [i for i in range(len(unique_colors))]
    # print(unique_colors)
    # for color, count in zip(unique_colors, counts):
    #     print(f"Color: {color}, Count: {count}")
    return concatenated_strings


def add_missing_values(main_list, secondary_list):
    # print(secondary_list)
    for value in secondary_list:
        if value not in main_list:
            main_list.append(value)
    return main_list

def max_classes_in_directory(directory_path):
    max_classes = 0
    classes_nums = []
    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Add more image file extensions if needed
            image_path = os.path.join(directory_path, filename)
            num_classes = count_pixel_occurrences(image_path)

            classes = count_pixels(image_path)
            classes_nums=add_missing_values(classes_nums,classes)
            classes_nums.sort()
            print(f"The different classes along all the images are: {classes_nums}")
            print(f"The length of different classes along all the images are: {len(classes_nums)}")

            # Update max_classes if the current image has more classes
            max_classes = max(max_classes, num_classes)

    return max_classes

# Example usage:
directory_path = "D:\COMPUTER_DEPARTMENT\\3RD_YEAR\AML\AML_Project\Project_Repository\ENet\content\\valannot"  # Replace with the path to your image directory
max_classes = max_classes_in_directory(directory_path)
# print(allColors)
# print(len(allColors))
print(f"The maximum number of classes in one image is: {max_classes}")
>>>>>>> 3f06b50c51741a130c22f30daab1ef2b6285ee8c
