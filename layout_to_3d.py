import json
import numpy as np
import open3d as o3d
from PIL import Image

def load_layout(layout_path):
    with open(layout_path, 'r') as f:
        return json.load(f)

def load_depth_map(depth_map_path):
    depth_image = Image.open(depth_map_path)
    depth_array = np.array(depth_image)
    return depth_array

def load_3d_model(model_path):
    return o3d.io.read_triangle_mesh(model_path)

def scale_model(model, target_size):
    current_size = model.get_max_bound() - model.get_min_bound()
    scale_factors = target_size / current_size
    model.scale(scale_factors, center=model.get_center())
    return model

def place_3d_models(layout_path, models_dict, depth_map_path):
    layout = load_layout(layout_path)
    depth_map = load_depth_map(depth_map_path)
    
    scene = o3d.geometry.TriangleMesh()
    
    for item in layout:
        label = item['label']
        size = np.array(item['size'])
        position = np.array(item['position'])
        
        if label not in models_dict:
            print(f"No model found for {label}")
            continue
        
        model_path = models_dict[label]
        model = load_3d_model(model_path)
        
        # Scale the model
        model = scale_model(model, size)
        
        # Move the model to its position
        model.translate(position)
        
        # Add the model to the scene
        scene += model
    
    # Add a coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
    scene += coordinate_frame
    
    return scene

def visualize_scene(scene):
    o3d.visualization.draw_geometries([scene])

if __name__ == "__main__":
    layout_path = 'layout.json'
    depth_map_path = 'dog_cat/depth_colored/dog_cat_colored.png'
    models_dict = {
        'dog': 'dumps/zxhezexin/openlrm-mix-base-1.1/meshes/dog.ply',
        'black_cat': 'dumps/zxhezexin/openlrm-mix-base-1.1/meshes/black_cat.ply',
        'white_cat': 'dumps/zxhezexin/openlrm-mix-base-1.1/meshes/white_cat.ply'
    }

    try:
        scene = place_3d_models(layout_path, models_dict, depth_map_path)
        visualize_scene(scene)
    except Exception as e:
        print(f"An error occurred: {e}")