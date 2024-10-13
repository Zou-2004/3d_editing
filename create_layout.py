import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import cv2

def get_layout_from_sam(image_path, depth_map_path):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load the depth map
    depth_map = np.array(Image.open(depth_map_path).convert("L"))

    # Load SAM
    sam = sam_model_registry["default"](checkpoint="/home/zcy/segment-anything/sam_vit_h_4b8939.pth")
    predictor = SamPredictor(sam)

    # Preprocess the image for SAM
    long_side = max(image.shape[0], image.shape[1])
    scale = 1024 / long_side
    new_height = int(image.shape[0] * scale)
    new_width = int(image.shape[1] * scale)
    image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Set the image for SAM
    predictor.set_image(image_resized)

    layout = []
    
    fig, ax = plt.subplots()
    ax.imshow(image)
    
    while True:
        print("Click twice on an object to define its bounding box. Close the window to finish.")
        points = plt.ginput(n=2, timeout=-1)
        
        if len(points) < 2:
            break

        # Calculate bounding box from points
        x1, y1 = points[0]
        x2, y2 = points[1]
        left, right = min(x1, x2), max(x1, x2)
        top, bottom = min(y1, y2), max(y1, y2)

        # Scale the bounding box to match the resized image
        input_box = np.array([left * scale, top * scale, right * scale, bottom * scale])

        # Generate mask using SAM
        mask, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )

        # Resize mask back to original image size
        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Calculate size and position
        size = [right - left, bottom - top, np.mean(depth_map[mask > 0])]
        position = [(left + right) / 2, (top + bottom) / 2, np.mean(depth_map[mask > 0]) / 2]

        # Get object label from user
        label = input("Enter label for this object: ")

        # Add to layout
        layout.append({
            "label": label,
            "size": size,
            "position": position
        })

        # Visualize the segmentation
        ax.clear()
        ax.imshow(image)
        ax.imshow(mask, alpha=0.5)
        ax.add_patch(plt.Rectangle((left, top), right-left, bottom-top, 
                                   fill=False, edgecolor='red', linewidth=2))
        ax.set_title(f"Segmentation for {label}")
        plt.draw()

    plt.close()
    return layout

# Usage
image_path = "cat_dog.png"
depth_map_path = "pet/depth_colored/pet_pred_colored.png"
layout = get_layout_from_sam(image_path, depth_map_path)

# Save layout to JSON
import json
with open("layout.json", "w") as f:
    json.dump(layout, f, indent=2)