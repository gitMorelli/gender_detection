import cv2
from PIL import Image
import numpy as np

def extract_patches(image,gw=5,n_cc=10):
    """Splits image into grid patches and checks for text presence."""
    image_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    h, w = image_gray.shape
    patch_w = w // gw
    patch_h = patch_w
    gh= h // patch_h
    
    valid_patches = []
    
    for i in range(gh):
        for j in range(gw):
            x1, y1 = j * patch_w, i * patch_h
            x2, y2 = x1 + patch_w, y1 + patch_h
            patch = image_gray[y1:y2, x1:x2]
            
            # Check for text using connected components
            _, binary = cv2.threshold(patch, 180, 255, cv2.THRESH_BINARY_INV)
            num_labels, _, _, _ = cv2.connectedComponentsWithStats(binary)
            
            if num_labels > n_cc:  # More than 1 means text is present
                #patch_rgb = image.crop((x1, y1, x2, y2))  # Extract RGB patch
                valid_patches.append((x1, y1, x2, y2,num_labels))
    
    return valid_patches

def process_row(row):
    image = Image.open(row["file_name"])  # Open the image
    #print(row["file_name"])
    patches = extract_patches(image,gw=5,n_cc=10)  # Extract patches
    
    # Create a new row for each patch
    new_rows = []
    for (x, y, x2, y2,n_cc) in patches:
        new_row = row.copy()
        new_row["x"], new_row["y"], new_row["x2"], new_row["y2"],new_row["n_cc"] = x, y, x2, y2, n_cc
        new_rows.append(new_row)
    
    return new_rows