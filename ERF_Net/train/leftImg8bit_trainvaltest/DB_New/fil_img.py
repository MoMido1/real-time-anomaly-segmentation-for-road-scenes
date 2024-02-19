import cv2
import numpy as np
import os

def filter_image(img_path, colors, output_path):
    # Read the image
    img = cv2.imread(img_path)

    # Define the 19 colors in RGB format
    target_colors = np.array(colors)

    # Convert image to RGB format
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Mask to identify pixels matching the target colors
    mask = np.zeros_like(img_rgb[:,:,0], dtype=bool)
    for color in target_colors:
        mask = np.logical_or(mask, np.all(img_rgb == color, axis=-1))

    # Create a binary mask with 1s for pixels matching the target colors
    mask = mask.astype(np.uint8) * 255

    # Apply the mask to the original image
    filtered_img = cv2.bitwise_and(img, img, mask=mask)

    target_color_np = np.array([81,0,81], dtype=np.uint8)
    new_color_np = np.array([128, 64, 128], dtype=np.uint8)
    mask = np.all(img == target_color_np, axis=-1)
    filtered_img[mask] = new_color_np
    # filtered_img = cv2.bitwise_and(img, img, mask=mask)
    

    # Save the filtered image
    cv2.imwrite(output_path, filtered_img)

# Example usage
colors = [
    [128, 64, 128],    # Road
    [244, 35, 232],    # sidewalk
    [70, 70, 70],    # Building
    [102,102,156],   #Wall
    [190, 153, 153],  #fence
    [153, 153 ,153],  #pole
    [250,170, 30],    #traffic light
    [220, 220, 0],   #traffic sign
    [107, 142, 35],   #vegetation
    [152, 251, 152],  #terrain
    [70,130, 180],  #sky
    [220,20,60],  #person
    [255,0,0],   #rider
    [0,0,142],   #car
    [0,0,70],  #truck
    [0,60,100],  #bus
    [230,150,140], #train
    [0,0,230], #motorcycle
    [119,11,32]  # bicycle

    # Add more colors as needed
]

# img_path = 'D:\COMPUTER_DEPARTMENT\\3RD_YEAR\AML\AML_Project\Project_Repository\ERF_Net\\train\leftImg8bit_trainvaltest\DB_New\\trainannot\\aachen_000000_000019_gtFine_color.png'
# output_path = 'D:\COMPUTER_DEPARTMENT\\3RD_YEAR\AML\AML_Project\Project_Repository\ERF_Net\\train\leftImg8bit_trainvaltest\DB_New\\trainannot_f\\aachen_000000_000019_gtFine_color.png'

input_dir = "D:\COMPUTER_DEPARTMENT\\3RD_YEAR\AML\AML_Project\Project_Repository\ERF_Net\\train\leftImg8bit_trainvaltest\DB_New\\valannot\\"

# Directory to save the filtered images
output_dir = "D:\COMPUTER_DEPARTMENT\\3RD_YEAR\AML\AML_Project\Project_Repository\ERF_Net\\train\leftImg8bit_trainvaltest\DB_New\\valannot_f\\"
os.makedirs(output_dir, exist_ok=True)
if __name__ == '__main__':
    for filename in os.listdir(input_dir):
        if filename.endswith((".jpg", ".png")):  # Add more extensions if needed
            img_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir,filename)
            filter_image(img_path,colors,output_path)
    # filter_image(img_path, colors, output_path)
