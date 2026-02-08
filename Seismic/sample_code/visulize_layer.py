#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
import segyio
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    pathname = '/home/wangyh/DATAFOLDER/samples/train-choas/crossed/AYL-00002.npz'
    data = np.load(pathname)
    contour_num = 12
    
    Details = data['formatS']
    MDetails = data['formatD']
    Trends = data['gtime'] # Dimensions: x * y * z Shape: 256*256*256
    
    trend_slice = Trends[100, :, :].T
    fig,ax = plt.subplots(1,1, figsize=(15,5))
    im = ax.imshow(trend_slice, cmap='jet')
    
    # Extract and visualize contours
    contours = ax.contour(trend_slice, contour_num, colors='white', linewidths=0.5)
    ax.clabel(contours, inline=True, fontsize=8)

    fig.colorbar(im, ax=ax)
    ax.set_title('Trend Component Slice at X=100 with Contours')
    plt.show()
    #%%
    # Histogram of the Trend Slice
    plt.figure(figsize=(10, 6))
    plt.hist(trend_slice.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title('Histogram of Trend Component Slice Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    #%%
    # Classification based on Contour Levels (Quantization)
    print("Classifying based on contour levels...")
    
    # Get the levels from the previously generated contours
    levels = contours.levels
    print(f"Contour levels used values: {levels}")
    
    # Classify the image based on these levels
    # np.digitize returns the index of the bin to which each value belongs.
    # Data is divided into bins: (-inf, level[0]), [level[0], level[1]), ..., [level[-1], inf)
    classified_slice = np.digitize(trend_slice, levels)
    
    num_classes = len(levels) + 1
    print(f"Number of classes (regions between/outside contours): {num_classes}")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    # Use a colormap that distinguishes the classes well
    if num_classes <= 20:
        cmap = plt.get_cmap('tab20', num_classes)
    else:
        cmap = plt.get_cmap('viridis', num_classes)
    
    im_class = ax.imshow(classified_slice, cmap=cmap)
    
    # Adjust colorbar to show integer class labels centered on the color blocks
    cbar = fig.colorbar(im_class, ax=ax, ticks=np.arange(num_classes))
    cbar.set_label('Class ID')
    
    ax.set_title(f'Contour-based Classification\n{len(levels)} Levels -> {num_classes} Classes')
    plt.show()

    #%%
    # 3D Isosurface Visualization for Multiple Levels (Calculated from 3D Volume)
    print("Generating 3D isosurfaces using levels calculated from 3D volume...")
    
    # Calculate levels directly from the 3D volume statistic
    # Using np.linspace to divide the range of values in Trends into contour_num levels
    # We generate contour_num + 2 points and exclude the min and max to get internal levels
    levels_3d = np.linspace(Trends.min(), Trends.max(), contour_num + 2)[1:-1]
    print(f"3D Contour levels used: {levels_3d}")

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use the same colormap logic
    num_classes = len(levels_3d) + 1
    if num_classes <= 20:
        cmap = plt.get_cmap('tab20', num_classes)
    else:
        cmap = plt.get_cmap('viridis', num_classes)

    # Generate isosurfaces for each level
    print(f"Extracting isosurfaces for {len(levels_3d)} levels...")
    for i, level in enumerate(levels_3d):
        try:
            # Using step_size=2 to improve performance. Reduce to 1 for higher quality.
            verts, faces, normals, values = measure.marching_cubes(Trends, level=level, step_size=2)
            
            # Create mesh
            # Alpha needs to be low to see nested surfaces
            mesh = Poly3DCollection(verts[faces], alpha=0.15)
            
            # Color: Map the level index to a color in the colormap
            # We skip 0 (background class) and start from 1 for the first contour
            color = cmap(i + 1) 
            mesh.set_facecolor(color)
            
            ax.add_collection3d(mesh)
        except Exception as e:
            # This can happen if a level does not intersect the volume in a way marching cubes handles
            print(f"Skipping level {level}: {e}")

    ax.set_xlim(0, Trends.shape[0])
    ax.set_ylim(0, Trends.shape[1])
    ax.set_zlim(0, Trends.shape[2])
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f'3D Multi-Isosurfaces ({len(levels_3d)} Levels from 3D Volume)')
    
    # Create a dummy ScalarMappable for colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=num_classes-1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, ticks=np.arange(num_classes))
    cbar.set_label('Class/Level Index')
    
    plt.show()

    #%%
    # Save 3D Label Volume
    print("Generating and saving 3D classification volume...")
    
    # Generate the label volume using the same levels
    classified_volume = np.digitize(Trends, levels_3d)
    #%%
    output_filename = 'classified_trends.npz'
    # Save both the labels and the levels used for classification
    np.savez(output_filename, labels=classified_volume, levels=levels_3d)
    print(f"Saved 3D labels to {output_filename}")
    print(f"Volume shape: {classified_volume.shape}")
    print(f"Unique classes: {np.unique(classified_volume)}")

    #%%
    