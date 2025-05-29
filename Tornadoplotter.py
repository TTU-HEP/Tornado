import os
import cv2
import numpy as np
import glob
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
#import tensorflow
import seaborn as sns
#import imutils
from math import dist
import pandas as pd
from matplotlib.patches import RegularPolygon

def get_module_paths(module_name):
    #base_dir = os.path.join('/home/akshriva/AIgantry/Tornado/Web', 'Modules')
    current_dir = os.getcwd()
    current_dir = str(current_dir)
    base_dir = os.path.join(current_dir, 'Modules')
    images_dir = os.path.join(base_dir, module_name, 'Results')
    #images_dir = os.path.join(base_dir, module_name,)   
    labels_dir = os.path.join(base_dir, module_name, 'Results','labels')
    #print(current_dir)
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    return images_dir, labels_dir

def process_labels(label_path, img_path):
    dict1 = {}
    print("IMG PATH",img_path)
    for name in os.listdir(img_path):
        if ".jpg" in name:
            try:
                image_path = os.path.join(img_path, name)
                mask = cv2.imread(image_path)
                (h, w) = mask.shape[:2]
                txtname = name.split('.jpg')[0] + ".txt"
                label_file_path = os.path.join(label_path, txtname)
                with open(label_file_path, 'r') as filename:
                    lines = filename.readlines()
                    (x, y) = ((lines[0].split(" "))[1], (lines[0].split(" "))[2])
                    point = (int(float(x) * w), int(float(y) * h))
                    dict1[name] = [point[0], point[1]]
            except :
                pass
    return dict1

#def detect_hough_circles(img_path):
 #   dict2 = {}  
  #  for name in os.listdir(img_path):
   #     if ".jpg" in name:
    #        image_path=os.path.join(img_path,name)
     #       mask = cv2.imread(os.path.join(img_path)
      #      gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
       #     blurred = cv2.medianBlur(gray, 5)
        #    circles = cv2.HoughCircles(
         #       blurred, 
          #      cv2.HOUGH_GRADIENT, dp=1, minDist=150, param1=100, param2=50, minRadius=9

def detect_hough_circles(img_path):
    dict2 = {}  
    for name in os.listdir(img_path):
        if ".jpg" in name:
            image_path = os.path.join(img_path, name)
            mask = cv2.imread(image_path)
            gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            blurred = cv2.medianBlur(gray, 5)
            circles = cv2.HoughCircles(
                blurred, 
                cv2.HOUGH_GRADIENT, dp=1, minDist=150, param1=100, param2=50, minRadius=90, maxRadius=110
            )
            if circles is not None:
                detected_circles = np.uint16(np.around(circles))  
                for (x, y, r) in detected_circles[0, :]:
                    dict2[name] = [[x, y], r]  
    return dict2

def synchronize_dicts(dict1, dict2):
    # Identify keys that are in dict1 but not in dict2
    keys_to_drop = [key for key in dict1 if key not in dict2]
    for key in keys_to_drop:
        del dict1[key]

    # Identify keys that are in dict2 but not in dict1
    keys_to_drop = [key for key in dict2 if key not in dict1]
    for key in keys_to_drop:
        del dict2[key]

    return dict1, dict2

def process_and_generate_df(img_path, label_path):
    dict1 = process_labels(label_path, img_path)  
    print("DICT ! ",dict1)
    dict2 = detect_hough_circles(img_path)
    dict1, dict2 = synchronize_dicts(dict1, dict2)

    # Extract data from dict1 and dict2 and compute distances, angles
    X2 = [values[0] for values in dict1.values()]  # Extract X values from dict1
    Y2 = [values[1] for values in dict1.values()]  # Extract Y values from dict1

    X1 = [values[0][0] for values in dict2.values()]  # Extract X values from dict2
    Y1 = [values[0][1] for values in dict2.values()]  # Extract Y values from dict2

    # Calculate delta_X, delta_Y, and r (distance)
    changeinx = []
    changeiny = []
    r = []

    for x1, y1, x2, y2 in zip(X1, Y1, X2, Y2):
        delta_x = x2 - x1
        delta_y = y2 - y1
        distance = math.sqrt(delta_x**2 + delta_y**2)
        changeinx.append(delta_x)
        changeiny.append(delta_y)
        r.append(distance)

    # Calculate angles (in degrees)
    angles = []
    for delta_x, delta_y in zip(changeinx, changeiny):
        angle_in_radians = math.atan2(delta_y, delta_x)
        angle_in_degrees = math.degrees(angle_in_radians)
        angles.append(angle_in_degrees)

    # Prepare the image names list (from dict2, assuming filenames are the same)
    Images = list(dict2.keys())

    # Step 5: Create and return DataFrame
    df1 = pd.DataFrame({
        "Images": Images,
        "X1": X1,
        "Y1": Y1,
        "X2": X2,
        "Y2": Y2,
        "Delta_X": changeinx,
        "Delta_Y": changeiny,
        "r": r,
        "Angles": angles
    })
    df1 = sort_dataframe_by_image_number(df1)
    return df1

def sort_dataframe_by_image_number(df1):
    df2 = df1.copy()
    print(df2)
    #df2['ImageNumber'] = df2['Images'].str.extract('(\d+)').astype(int)
    #print('DF2:::',df2)
    df2['ImageNumber'] = df2['Images'].astype(str).str.extract('(\d+)').astype(float).astype('Int64')
    print('DF2:::',df2)
    df2 = df2.sort_values(by='ImageNumber')
    print(df2)
    df3=df2.copy()
    df3=df3.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    #df2 = df2.drop(columns=['ImageNumber'])
    print("DF 2",df2)
    print(df3)
    return df2,df3

def process_files_and_append_to_df(images_path,df3):
    working_dir = os.path.abspath(os.path.join(images_path, '..'))
    print(working_dir)
    os.chdir(working_dir)
    dict3 = {}
    for name in os.listdir():
        if ".txt" in name:
            with open(name, 'r') as filename:
                lines = filename.readlines()
                (x,y)=((lines[0].split(",")[0]),(lines[0].split(",")[1]))
                point=(float(x),float(y))
                dict3[name]=[point[0],point[1]] 
    x_values = []
    y_values = []
    for values in dict3.values():
        x_values.append(values[0])
        y_values.append(values[1])

    df3 = pd.DataFrame({
        "Images": list(dict3.keys()),
        "X": x_values,
        "Y": y_values,
    })
    #df3['ImageNumber'] = df3['Images'].str.extract('(\d+)').astype(int)
    #df3['ImageNumber'] = df3['Images'].astype(str).str.extract('(\d+)').astype(float).astype('Int64')
    df3['ImageKey'] = df3['Images'].str.replace('.txt', '', regex=False)
    #df2['ImageKey'] = df2['Images'].str.replace('.jpg', '', regex=False)
    df3['ImageNumber'] = df3['Images'].astype(str).str.extract('(\d+)').astype(float).astype('Int64')
    df3 = df3[df3['ImageNumber'] != -1]
    print('____1st_________',df3)
    #df3_sorted = df3.sort_values(by='ImageNumber')
    df3_sorted = df3.sort_values(by='ImageNumber')
    print('____1st__sorted_______',df3_sorted)
    df3_sorted=df3_sorted[df3_sorted['ImageNumber'].isin(df3['ImageNumber'])]
    print('_____________',df3_sorted)
    df3_sorted = df3_sorted.reset_index(drop=True)
    print('_____________',df3_sorted)
    #df3_sorted = df3_sorted.drop(columns=['ImageNumber'])
    #print('__________ihjafiosjgiopajsepgjkaeos[gk[___',df3_sorted)
    return df3_sorted

# def finaldf(df2, df3_sorted):
#     #finaldf = df2.merge(df3_sorted[['X','Y']], left_on='ImageNumber', right_index=True, how='left')
#     finaldf = df2.merge(df3_sorted[['ImageNumber', 'X', 'Y']], on='ImageNumber', how='left')

#     print("FINAL DF",finaldf)
#     return finaldf


def finaldf(df2, df3_sorted):
    df2['ImageKey'] = df2['Images'].str.replace('.jpg', '', regex=False)
    df3_sorted['ImageKey'] = df3_sorted['Images'].str.replace('.txt', '', regex=False)
    
    finaldf = df2.merge(df3_sorted[['ImageKey', 'X', 'Y']], on='ImageKey', how='left')
    print("FINAL DF", finaldf)
    return finaldf


def create_arrow_plot(finaldf, df3, images_path,module_name):
    x = np.array(finaldf['X'])
    y = np.array(finaldf['Y'])
    angles = np.array(finaldf['Angles'])
    r = np.array(finaldf['r'])
    
    #fig, ax = plt.subplots()
    fig, ax = plt.subplots(figsize=(8,7)) #rectangle
    #fig.set_size_inches(6, 6)  # Make it a square
    plt.tight_layout()  # Adjust layout to remove excess space
    #plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # Fine-tune margins

    U = [r_value * math.cos(math.radians(-angle)) for angle, r_value in zip(angles, r)]
    V = [r_value * math.sin(math.radians(-angle)) for angle, r_value in zip(angles, r)]

    print('got here')
    for i in range(len(U)):
        ax.quiver(x[i], y[i], U[i], V[i], scale=110, color='b', width=0.008)

    for i, label in enumerate(df3['ImageNumber']):
        print(label)
        plt.annotate(label, (x[i], y[i]), color='red', weight='bold', ha='center')

    '''ADDing hexagon '''
    center_x, center_y = np.mean(x), np.mean(y)  # Center at the middle of the arrows
    #hex_size = 1.12*max(x.max() - x.min(), y.max() - y.min()) / 2 
    #hex_size = 1.12*max( - x.min(), y.max() - y.min()) / 2 
    hex_size = 100
    #hexagon = RegularPolygon((center_x, center_y), numVertices=6, radius=hex_size, 
    #                    edgecolor='black', facecolor='none', linewidth=2,orientation=np.radians(30))
    hexagon = RegularPolygon((50, 90), numVertices=6, radius=hex_size, 
                        edgecolor='black', facecolor='none', linewidth=2,orientation=np.radians(30))
    ax.add_patch(hexagon)
    ax.set_aspect('equal') 


    #ax.set_xlim(-60, 150) #Nominal Values
    #ax.set_ylim(-60, 240)
    #ax.set_xlim(0 - 5, x.max() + 5)  # Remove extra padding on x-axis
    ax.set_xlim(-50,150)
    ax.set_ylim(-10,230)
    #ax.set_ylim(y.min() - 5, y.max() + 5)  # Remove extra padding on y-axis
    ax.set_aspect('equal') 
    #ax.grid(True)
    ax.set_xticks(np.arange(-50, 170, 50))  # Major ticks every 50 units
    ax.set_yticks(np.arange(-10, 240, 10)) #-40, 240, 10

    reference_arrow = finaldf[finaldf['r'] == finaldf['r'].max()].index[0]
    reference_arrow1 = finaldf[finaldf['r'] == finaldf['r'].min()].index[0]
    average_r = finaldf['r'].mean()
    reference_arrow2 = (finaldf['r'] - average_r).abs().idxmin()
    
    arrow_length = math.sqrt(U[reference_arrow]**2 + V[reference_arrow]**2)
    arrow_length1 = math.sqrt(U[reference_arrow1]**2 + V[reference_arrow1]**2)
    arrow_length2 = math.sqrt(U[reference_arrow2]**2 + V[reference_arrow2]**2)
    
    ax.quiver(x[reference_arrow], y[reference_arrow], U[reference_arrow], V[reference_arrow], scale=110, color='g', width=0.008)
    ax.quiver(x[reference_arrow1], y[reference_arrow1], U[reference_arrow1], V[reference_arrow1], scale=110, color='black', width=0.008)
    ax.quiver(x[reference_arrow2], y[reference_arrow2], U[reference_arrow2], V[reference_arrow2], scale=110, color='y', width=0.008)

    
    bottom_left_x, bottom_left_y = -40,200  # Adjust these values as needed base: -40 and 180
    ax.quiver(bottom_left_x, bottom_left_y, arrow_length, 0, scale=110, color='g', width=0.008)
    bottom_left_x1, bottom_left_y1 = -40, 210 #nominal values -40,195
    bottom_left_x2, bottom_left_y2 = -40, 220 #nominal values -40,220
    ax.quiver(bottom_left_x, bottom_left_y, arrow_length, 0, scale=110, color='g', width=0.008)
    ax.quiver(bottom_left_x1, bottom_left_y1, arrow_length1, 0, scale=110, color='black', width=0.008)
    ax.quiver(bottom_left_x1, bottom_left_y2, arrow_length2, 0, scale=110, color='y', width=0.008)


    ax.annotate(f'Length(Longest): {(arrow_length) * 2 / 103:.2f}mm', 
            (bottom_left_x + arrow_length + 10, bottom_left_y), color='red', fontsize=8, weight='bold', ha='left', 
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    ax.annotate(f'Length(Smallest): {(arrow_length1) * 2 / 103:.2f}mm', 
                (bottom_left_x1 + arrow_length1 + 10, bottom_left_y1), color='red', fontsize=8, weight='bold', ha='left', 
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    ax.annotate(f'Length(Mean): {(arrow_length2) * 2 / 103:.2f}mm', 
                (bottom_left_x1 + arrow_length2 + 10, bottom_left_y2), color='red', fontsize=8, weight='bold', ha='left', 
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    ax.set_title(f'Tornado plot {module_name}  ')
    #ax.grid(True)
    plt.tight_layout()
    #plt.grid(True)
    # Save the final image
    save_path = os.path.join(images_path, f"Tornado_plot_{module_name}.jpg")
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    print('image path ',images_path)


 

#     save_path = os.path.join(images_path, f"Tornado_plot_{module_name}.jpg")
#     plt.savefig(save_path, dpi=100, bbox_inches='tight')
#     #plt.show()

# def create_arrow_plot(finaldf, df3, images_path, module_name):
#     x = np.array(finaldf['X'])
#     y = np.array(finaldf['Y'])
#     angles = np.array(finaldf['Angles'])
#     r = np.array(finaldf['r'])

#     fig, ax = plt.subplots(figsize=(8, 7))

#     # Overlay background image
#     img = mpimg.imread('/lustre/work/akshriva/APD/Tornado/module_map.jpg')  # Make sure it's in your working directory
#     x_min, x_max = x.min() - 10, x.max() + 10
#     y_min, y_max = y.min() - 10, y.max() + 10
#     ax.imshow(img, extent=[x_min, x_max, y_min, y_max], aspect='auto', zorder=0)

#     # Arrows
#     U = [r_value * math.cos(math.radians(-angle)) for angle, r_value in zip(angles, r)]
#     V = [r_value * math.sin(math.radians(-angle)) for angle, r_value in zip(angles, r)]
#     ax.quiver(x, y, U, V, scale=110, color='b', width=0.008, zorder=1)

#     # Labels
#     for i, label in enumerate(df3['ImageNumber']):
#         ax.annotate(label, (x[i], y[i]), color='red', weight='bold', ha='center', zorder=2)

#     # Hexagon overlay (optional)
#     hexagon = RegularPolygon((50, 90), numVertices=6, radius=90, edgecolor='black',
#                              facecolor='none', linewidth=2, orientation=np.radians(30), zorder=2)
#     ax.add_patch(hexagon)

#     # Reference Arrows (Longest, Shortest, Mean)
#     ref_idx_long = finaldf['r'].idxmax()
#     ref_idx_short = finaldf['r'].idxmin()
#     ref_idx_mean = (finaldf['r'] - finaldf['r'].mean()).abs().idxmin()

#     def draw_ref_arrow(idx, color, label_offset_y, text):
#         ax.quiver(x[idx], y[idx], U[idx], V[idx], scale=110, color=color, width=0.008, zorder=2)
#         arrow_len = math.sqrt(U[idx]**2 + V[idx]**2)
#         base_x, base_y = -40, label_offset_y
#         ax.quiver(base_x, base_y, arrow_len, 0, scale=110, color=color, width=0.008, zorder=2)
#         ax.annotate(f'{text}: {arrow_len * 2 / 103:.2f} mm',
#                     (base_x + arrow_len + 10, base_y), color='red', fontsize=8,
#                     weight='bold', ha='left',
#                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

#     draw_ref_arrow(ref_idx_long, 'g', 200, 'Longest')
#     draw_ref_arrow(ref_idx_short, 'black', 210, 'Shortest')
#     draw_ref_arrow(ref_idx_mean, 'y', 220, 'Mean')

#     # Final plot setup
#     ax.set_xlim(x_min, x_max)
#     ax.set_ylim(y_min, y_max)
#     ax.set_xticks(np.arange(x_min, x_max + 1, 50))
#     ax.set_yticks(np.arange(y_min, y_max + 1, 10))
#     ax.set_aspect('equal')
#     ax.set_title(f'Tornado Plot Overlay for {module_name}')
#     plt.tight_layout()

#     # Save the final image
#     save_path = os.path.join(images_path, f"Tornado_overlay_{module_name}.png")
#     plt.savefig(save_path, dpi=150, bbox_inches='tight')
#     print(f"Saved overlay plot to: {save_path}")

def main():
    module_name = input("Enter the module name (e.g., 'M35'): ")
    images_path, labels_path = get_module_paths("320MLF3TCTT0"+module_name)
    print(images_path)
    df1, df3 = process_and_generate_df(images_path, labels_path)
    df3_sorted = process_files_and_append_to_df(images_path,df3)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print(f'Sorted DF {df3_sorted }')
    final_df = finaldf(df1, df3_sorted)
    print('finished final df')
    #create_arrow_plot(final_df, df3, images_path,module_name)
    create_arrow_plot(final_df, df3, images_path,module_name)
    print('done')
if __name__ == "__main__":
    main()
