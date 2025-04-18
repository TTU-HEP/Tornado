import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def sample_points_from_contour(contour, num_points=10, right_half_only=True, image_width=None):
    contour_pts = contour.reshape(-1, 2)
    if right_half_only and image_width:
        contour_pts = np.array([pt for pt in contour_pts if pt[0] > image_width // 6])
    if len(contour_pts) < num_points:
        return contour_pts.tolist()
    idxs = np.linspace(0, len(contour_pts) - 1, num_points, dtype=int)
    return [tuple(contour_pts[i]) for i in idxs]

def closest_point(point, contour):
    contour_pts = contour.reshape(-1, 2)
    dists = np.linalg.norm(contour_pts - point, axis=1)
    idx = np.argmin(dists)
    return tuple(contour_pts[idx]), dists[idx]

def process_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Mask for Hexaboard (Dark Region)
    _, mask_hexaboard = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    contours_red, _ = cv2.findContours(mask_hexaboard, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_red = max(contours_red, key=cv2.contourArea)

    # --- Mask for Dark Green Background (Custom Green Range)
    lower_green = np.array([24, 20, 30])
    upper_green = np.array([80, 190, 56])  # Upper bound for dark green
    mask_dark_green = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lower_green, upper_green)

    kernel = np.ones((7, 7), np.uint8)
    mask_dark_green = cv2.morphologyEx(mask_dark_green, cv2.MORPH_OPEN, kernel)
    mask_dark_green = cv2.morphologyEx(mask_dark_green, cv2.MORPH_CLOSE, kernel)
    contours_green, _ = cv2.findContours(mask_dark_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_green = max(contours_green, key=cv2.contourArea)

    # Sample points from green contour
    green_pts = sample_points_from_contour(max_green, num_points=7, right_half_only=True, image_width=img.shape[1])

    # Line colors and legend setup
    colors = [(255, 0, 0), (0, 255, 0), (0, 255, 255),
              (255, 0, 255), (255, 165, 0), (0, 128, 255),
              (128, 0, 128)]
    legend_entries = []

    # Draw output
    output = img_rgb.copy()
    cv2.drawContours(output, [max_green], -1, (0, 255, 0), 1)
    cv2.drawContours(output, [max_red], -1, (255, 0, 0), 1)

    for i, pt in enumerate(green_pts):
        color = colors[i % len(colors)]
        red_pt, dist = closest_point(np.array(pt), max_red)
        cv2.line(output, pt, red_pt, color, 2)
        legend_entries.append((color, f"Line {i+1}: {dist*2/103:.1f}mm"))

    # Legend creation
    legend_height = 20 * len(legend_entries) + 10
    legend = np.ones((legend_height, 300, 3), dtype=np.uint8) * 255
    for i, (color, text) in enumerate(legend_entries):
        cv2.line(legend, (10, 20 * i + 15), (40, 20 * i + 15), color, 5)
        cv2.putText(legend, text, (50, 20 * i + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Combine with output
    legend = cv2.resize(legend, (output.shape[1], legend.shape[1]))
    combined = np.vstack((output, legend))

    return combined

def process_multiple_images(image_folder):
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
    num_images = len(image_paths)
    
    # Determine grid size
    cols = 3  # Set the number of columns
    rows = (num_images // cols) + (num_images % cols > 0)  # Calculate the number of rows needed
    
    # Create the subplot grid
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()  # Flatten axes array for easy indexing

    for i, image_path in enumerate(image_paths):
        combined_output = process_image(image_path)
        ax = axes[i]
        ax.imshow(combined_output)
        # Use the image file name (without path and extension) as the title
        image_name = os.path.basename(image_path).split('.')[0]
        ax.set_title(f"{image_name}")
        ax.axis('off')

    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

image_folder = "/home/abhinav/Backside_Wirebonded_m113/"  # Change this to your folder path containing images
process_multiple_images(image_folder)
