import cv2
from PIL import Image
import numpy as np

def get_sentence_regions(img, n_selected_lines=3, n_slices=3, spacing=15, r_mask=0.1, r_min=0.5):
    height, width = img.shape
    n_slices= n_slices
    patches = [[] for _ in range(n_slices)]
    projections = [[] for _ in range(n_slices)]
    patch_width = width // n_slices
    lines_in_patch = [[] for _ in range(n_slices)]
    r_min=0.5
    x_limits = []

    for i in range(n_slices):
        x1 = i * patch_width
        x2 = (i + 1) * patch_width if i < 2 else width  # last patch takes the rest
        patch = img[:, x1:x2]
        patches[i] = patch
        x_limits.append((x1, x2))
        # Step 1: Binarize the image using adaptive threshold
        binary = cv2.adaptiveThreshold(patch, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 10)
        #_, binary = cv2.threshold(patch, 180, 255, cv2.THRESH_BINARY_INV)
        horizontal_projection = np.sum(binary, axis=1)
        projections[i] = horizontal_projection
        lines = []
        start = None
        '''non_zero_vals = horizontal_projection[horizontal_projection > 0]
        ten_smallest = np.sort(non_zero_vals)[:100]
        min_sum = np.max(ten_smallest)  # Minimum sum to consider a line'''
        min_sum=r_min * np.mean(horizontal_projection)

        for j, row_sum in enumerate(horizontal_projection):
            if row_sum > min_sum and start is None:
                start = j
            elif row_sum < min_sum and start is not None:
                lines.append((start, j))
                start = None

        if start is not None:
            lines.append((start, len(binary)))
        lines_in_patch[i] = lines
    
    n_selected_lines = n_selected_lines
    selected_lines = [[] for _ in range(n_slices)]
    spacing = spacing
    r_mask=r_mask
    for i in range(n_slices):
        lines = lines_in_patch[i]
        width = np.array([l[1]-l[0] for l in lines])
        max_width = np.max(width)
        mask= width>r_mask*max_width
        filtered_widths = width[mask]
        width[~mask] = 0
        median_width = np.median(filtered_widths)
        for j in range(n_selected_lines):
            closest_idx = np.argmin(np.abs(np.array(width) - median_width))
            width[closest_idx] = 0
            selected_lines[i].append((lines[closest_idx][0]-spacing, lines[closest_idx][1]+spacing))
    return selected_lines, x_limits

def extract_patches(image,gw=5,n_cc=10, prop=1):
    """Splits image into grid patches and checks for text presence."""
    image_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    h, w = image_gray.shape
    patch_w = w // gw
    patch_h = int(patch_w*prop)
    gh= int(h // patch_h)
    #print(gw,gh,h/patch_h)
    
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

def process_row(row,gw=5,n_cc=10,prop=1):
    image = Image.open(row["file_name"])  # Open the image
    #print(row["file_name"])
    patches = extract_patches(image,gw=gw,n_cc=n_cc, prop=prop)  # Extract patches
    
    # Create a new row for each patch
    new_rows = []
    for (x, y, x2, y2,n_cc,b_ratio) in patches:
        new_row = row.copy()
        new_row["x"], new_row["y"], new_row["x2"], new_row["y2"],new_row["n_cc"],new_row["black_ratio"] = x, y, x2, y2, n_cc, b_ratio
        new_rows.append(new_row)
    
    return new_rows

def process_row_sentences(row,n_selected_lines=3, n_slices=3, spacing=15, r_mask=0.1, r_min=0.5):
    #image = Image.open(row["file_name"])  # Open the image
    image = cv2.imread(row["file_name"], cv2.IMREAD_GRAYSCALE)
    #print(row["file_name"])
    lines_per_page,x_lims = get_sentence_regions(image, n_selected_lines=3, n_slices=3, spacing=15, r_mask=0.1, r_min=0.5)
    
    # Create a new row for each patch
    new_rows = []
    for i in range(len(lines_per_page)):
        for (y1, y2) in lines_per_page[i]:
            x1, x2 = x_lims[i]
            new_row = row.copy()
            new_row["x"], new_row["y"], new_row["x2"], new_row["y2"] = x1, y1, x2, y2
            new_rows.append(new_row)
    
    return new_rows