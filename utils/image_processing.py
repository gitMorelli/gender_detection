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
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary)
            
            if num_labels > n_cc:  # More than 1 means text is present
                #patch_rgb = image.crop((x1, y1, x2, y2))  # Extract RGB patch
                #areas = stats[1:, cv2.CC_STAT_AREA]  # skip background (label 0)
                black_pixel_count = np.sum(binary == 255)
                # Optionally: compute percentage of black
                total_pixels = binary.size
                black_ratio = black_pixel_count / total_pixels
                valid_patches.append((x1, y1, x2, y2,num_labels,black_ratio))
            # maybe you should filter based on the number of pixels
            #not on the number of different components (see): https://chatgpt.com/share/682d9a39-622c-8010-b50a-da370dcf214c
    return valid_patches

def process_row(row,gw=5,n_cc=10):
    image = Image.open(row["file_name"])  # Open the image
    #print(row["file_name"])
    patches = extract_patches(image,gw=gw,n_cc=n_cc)  # Extract patches
    
    # Create a new row for each patch
    new_rows = []
    for (x, y, x2, y2,n_cc,b_ratio) in patches:
        new_row = row.copy()
        new_row["x"], new_row["y"], new_row["x2"], new_row["y2"],new_row["n_cc"],new_row["black_ratio"] = x, y, x2, y2, n_cc, b_ratio
        new_rows.append(new_row)
    
    return new_rows