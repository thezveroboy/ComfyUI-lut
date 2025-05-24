import numpy as np
import cv2
from sklearn.cluster import KMeans
import os
import torch
from PIL import Image
import folder_paths

class ImageToLUTNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "num_colors": ("INT", {"default": 17, "min": 1, "max": 256, "step": 1}),
                "lut_size": ("INT", {"default": 17, "min": 2, "max": 64, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lut_filepath",)
    FUNCTION = "generate_lut"
    CATEGORY = "image/LUT"

    def generate_lut(self, image, num_colors, lut_size):
        image = image[0].cpu().numpy() * 255.0
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.reshape((-1, 3))

        kmeans = KMeans(n_clusters=num_colors, random_state=0)
        kmeans.fit(image)
        colors = kmeans.cluster_centers_
        colors = colors / 255.0

        num_entries = lut_size ** 3
        lut_data = np.zeros((num_entries, 3))
        idx = 0
        for b in range(lut_size):
            for g in range(lut_size):
                for r in range(lut_size):
                    lut_data[idx] = colors[(r * lut_size**2 + g * lut_size + b) % len(colors)]
                    idx += 1

        # Сохраняем файл в папке outputs
        output_dir = folder_paths.get_output_directory()
        lut_filepath = os.path.join(output_dir, "generated_lut.cube")
        with open(lut_filepath, 'w') as f:
            f.write("TITLE \"Generated LUT from Image\"\n")
            f.write(f"LUT_3D_SIZE {lut_size}\n")
            for rgb in lut_data:
                f.write(f"{rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}\n")

        return (lut_filepath,)

NODE_CLASS_MAPPINGS = {"ImageToLUT": ImageToLUTNode}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageToLUT": "Image to LUT (Cube)"}
