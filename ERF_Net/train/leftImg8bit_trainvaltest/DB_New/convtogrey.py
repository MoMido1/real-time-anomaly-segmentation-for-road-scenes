from PIL import Image
import numpy as np
import os
import cv2
#this code has 2 count functions the first one will read the image as a grey scale image
# and according to that the max number of classes in one image will be 18 and here we are using
#a cv2 library

# the second function of count uploads the image as it is and transform it to 
#grey scale image and save it as a new image and we won't need that
UNIQUE_CLR = {}
UNIQUE_IND = 0

def count_pixel_occurrences(image_path,filename):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    unique_values, counts = np.unique(image, return_counts=True)
    
    return unique_values

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

def add_missing_values(main_list, secondary_list):
    # print(secondary_list)
    for value in secondary_list:
        if value not in main_list:
            main_list.append(value)
    return main_list

def transformg(image_path,output_path,filename):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = np.reshape(image_rgb, (-1, 3))
    global UNIQUE_CLR
    global UNIQUE_IND

    # Iterate over each pixel
    if UNIQUE_IND !=20:
        for pixel in pixels:
            # Convert the pixel tuple to a string to use as a dictionary key
            pixel_str = ','.join(map(str, pixel))

            
            # Check if the RGB value is already in the dictionary
            if pixel_str not in UNIQUE_CLR:
                # If not, assign a unique index to it
                UNIQUE_CLR[pixel_str] = UNIQUE_IND
                UNIQUE_IND += 1
                print(UNIQUE_CLR)

    # Map each pixel to its unique index
    transformed_image = np.array([UNIQUE_CLR[','.join(map(str, pixel))] for pixel in pixels])

    # Reshape the transformed image to match the original image shape
    transformed_image = np.reshape(transformed_image, image_rgb.shape[:2])
    if filename.endswith('color.png'):
        # Replace 'color.png' with 'labelIds.png'
        filename = filename.replace('color.png', 'labelIds.png')
        
    cv2.imwrite(output_path+"\\"+filename, transformed_image)
   
def max_classes_in_directory(directory_path,outpth):
    # max_classes = 0
    # classes_nums =[]
    # cls_nums =[]
    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        print(filename)
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Add more image file extensions if needed
            image_path = os.path.join(directory_path, filename)
            
            transformg(image_path,outpth,filename)
            
    

# Example usage:
directory_path = "D:\COMPUTER_DEPARTMENT\\3RD_YEAR\AML\AML_Project\Project_Repository\ERF_Net\\train\leftImg8bit_trainvaltest\DB_New\\valannot_f"  # Replace with the path to your image directory
outpth ="D:\COMPUTER_DEPARTMENT\\3RD_YEAR\AML\AML_Project\Project_Repository\ERF_Net\\train\leftImg8bit_trainvaltest\DB_New\\vval"

if __name__ == '__main__':
    
    max_classes_in_directory(directory_path,outpth)
    # print(UNIQUE_CLR)
    # directory_path = "D:\COMPUTER_DEPARTMENT\\3RD_YEAR\AML\AML_Project\Project_Repository\ERF_Net\\train\leftImg8bit_trainvaltest\DB_New\\valannot_f"  # Replace with the path to your image directory
    # outpth ="D:\COMPUTER_DEPARTMENT\\3RD_YEAR\AML\AML_Project\Project_Repository\ERF_Net\\train\leftImg8bit_trainvaltest\DB_New\\valannot_nw"
    # max_classes_in_directory(directory_path,outpth)
    print("Done ")

