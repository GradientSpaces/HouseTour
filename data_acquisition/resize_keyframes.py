import os
from PIL import Image
import imageio
from tqdm import tqdm

# Define the paths
input_folder = '../keyframes/'  # Folder with original images
output_folder = '../keyframes_resized/'  # Folder where resized images will be saved

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define the new size (width, height)
new_size = (512, 288)  # Example size, change as needed

# Process each image in the input folder
for filename in tqdm(os.listdir(input_folder)):
    if filename.endswith(('.png')):
        # Construct the full path to the image file
        input_path = os.path.join(input_folder, filename)
        
        # Read the image
        image = imageio.imread(input_path)
        
        # Convert to PIL Image for resizing
        pil_image = Image.fromarray(image)
        
        # Resize the image
        resized_image = pil_image.resize(new_size)
        
        # Save the resized image to the output folder
        output_path = os.path.join(output_folder, filename)
        resized_image.save(output_path)

print("All images have been resized and saved.")
