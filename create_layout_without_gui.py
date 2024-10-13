import numpy as np
import cv2
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import json

#claculating thickness through depth map


def estimate_object_thickness(depth_map, mask):
    """
    Estimate the thickness of an object using the depth map and object mask.
    
    :param depth_map: 2D numpy array of the depth map
    :param mask: 2D numpy array of the object mask
    :return: Estimated thickness of the object
    """
    # Extract depth values for the object
    object_depth = depth_map[mask > 0]
    
    if len(object_depth) == 0:
        return 0  # Return 0 if no depth values are found for the object
    
    # Calculate the difference between the furthest and closest points
    max_depth = np.max(object_depth)
    min_depth = np.min(object_depth)
    thickness = max_depth - min_depth
    
    return thickness



def get_layout_from_sam(image_path, depth_map_path, zoom_factor=1.5):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load the depth map
    depth_map = np.array(Image.open(depth_map_path).convert("L"))

    # Load SAM
    sam = sam_model_registry["default"](checkpoint="/home/zcy/segment-anything/sam_vit_h_4b8939.pth")
    predictor = SamPredictor(sam)

    # Set the image for SAM
    predictor.set_image(image)

    layout = []
    
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    while True:
        print("\nEnter two points to define a bounding box (format: x1,y1 x2,y2)")
        print("Or press Enter to finish")
        
        user_input = input("> ")
        if user_input.strip() == "":
            break
        
        try:
            p1, p2 = user_input.split()
            x1, y1 = map(int, p1.split(','))
            x2, y2 = map(int, p2.split(','))
        except ValueError:
            print("Invalid input. Please use the format: x1,y1 x2,y2")
            continue

        # Calculate bounding box
        left, right = min(x1, x2), max(x1, x2)
        top, bottom = min(y1, y2), max(y1, y2)

        # Generate mask using SAM
        input_box = np.array([left, top, right, bottom])
        mask, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )

        # Extract the object dimensions
        rows = np.any(mask[0] > 0, axis=1)
        cols = np.any(mask[0] > 0, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        # Calculate the original position (center of the object in the original image)
        original_center_x = int((x1 + x2) // 2)
        original_center_y = int((y1 + y2) // 2)

        depth_value = float(np.mean(depth_map[mask[0] > 0]))

        # Estimate object thickness
        thickness = estimate_object_thickness(depth_map, mask[0])

        # Calculate size
        # X: width (left to right)
        # Y: depth (thickness)
        # Z: height (bottom to top)
        size = [int(xmax - xmin), int(thickness), int(ymax - ymin)]

        # Create the position vector
        # X: increases from left to right
        # Y: represents depth (objects further away have larger Y values)
        # Z: increases from top to bottom (as in image coordinates)
        original_position = [original_center_x, depth_value, original_center_y]

        # Get object label from user
        label = input("Enter label for this object: ")

        # Add to layout
        layout.append({
            "label": label,
            "size": size,
            "original_position": original_position,
        })

        print(f"Added {label} to layout")

        # Now proceed with centering and saving the object image
        object_image = np.where(mask[0, :, :, None], image, 255)
        object_image = object_image[ymin:ymax+1, xmin:xmax+1]

        # Calculate the new dimensions for the white background
        object_height, object_width = object_image.shape[:2]
        new_dim = int(max(object_height, object_width) * zoom_factor)

        # Create a white background image
        white_background = np.ones((new_dim, new_dim, 3), dtype=np.uint8) * 255

        # Calculate the position to place the object at the center
        y_offset = (new_dim - object_height) // 2
        x_offset = (new_dim - object_width) // 2

        # Place the object on the white background
        white_background[y_offset:y_offset+object_height, x_offset:x_offset+object_width] = object_image

        # Save the object on white background as an image
        image_filename = f"{label}.png"
        cv2.imwrite(image_filename, cv2.cvtColor(white_background, cv2.COLOR_RGB2BGR))
        print(f"Saved image as {image_filename}")

    return layout

# After getting the layout, save it to a JSON file
def save_layout_to_json(layout, filename="layout.json"):
    with open(filename, 'w') as f:
        json.dump(layout, f, indent=2)
    print(f"Layout saved to {filename}")

# Usage
image_path = "cat_dog.png"
depth_map_path = "dog_cat/depth_colored/dog_cat_colored.png"

layout = get_layout_from_sam(image_path, depth_map_path, zoom_factor=1.5)

save_layout_to_json(layout)
