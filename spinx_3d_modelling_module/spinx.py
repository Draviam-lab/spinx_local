import os
import sys
import numpy as np
from numpy import linalg, sqrt
from skimage import io
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rc
from io import BytesIO
import skimage
from skimage.color import rgb2gray, gray2rgb, label2rgb
from skimage.measure import label, regionprops, find_contours
from skimage.segmentation import clear_border
from skimage.morphology import erosion
from skimage.morphology import dilation
from skimage.morphology import square
from skimage.morphology import black_tophat
from skimage.morphology import disk
from skimage.morphology import remove_small_objects
from skimage.draw import circle_perimeter
from scipy import ndimage as ndi
from scipy import optimize
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull
import scipy
from skimage import util, draw
from skimage.filters import threshold_local
from skimage import img_as_bool
from skimage.morphology import convex_hull_image
import imageio
import math
from math import atan2, asin, pi
import platform
import pandas as pd
import datetime
import time
from statistics import mean, stdev, median # Statistics
import termtables as tt # Print table
from scipy.spatial.transform import Rotation as R
import copy
from itertools import product, combinations
from natsort import natsorted


########################
# SPINX MAIN FUNCTIONS #
# #######################

class SpinX():
    def __init__(self):
        pass
    
    def multi_importer(self, image_paths, n_slices, n_frames):
        # ================= Setting parameters ================= #
        convert_format = 1
        pref_ext = '.png'

        # ================= IMPORT LIST OF FILES ================= #
        # Check if it is a list
        if isinstance(image_paths, list): 
            print("your object is a list !") 
        else: 
            image_paths = [image_paths]

        file_type_list = []
        #def load_image_list
        image_list = []
        for filepath in image_paths:
            # Obtain filename
            filename = os.path.basename(filepath)
            # Keep file extension after first dot (.ome.tiff)
            if filename.split(os.extsep, 1)[1].lower() == 'ome.tiff':
                image_list.append(filepath)
                file_type_list.append('ome-tiff')
            elif os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg', '.tif']:
                image_list.append(filepath)
                file_type_list.append('tif-png')
        # Sort list alphabetically (libary)
        image_list = natsorted(image_list)

        # Conver to a set (get unique values) to check if the file format is consistent
        if len(set(file_type_list)) > 1:
            sys.exit('Mixed file formats were imported.')
        # Get file format of loaded images
        file_type = list(set(file_type_list))[0]

        if file_type == 'ome-tiff':
            filename_ome_list = []
            # Use OME-TIFF loader
            OME = OME_TIFF()
            # Most efficient way to stack numpy in a loop is by append to list
            ome_list = []
            meta_list = []
            for i, file in enumerate(image_list):
                # Read OME-TIFF: T, Z, C, Y, X
                ome_array, metadata, xml_str = OME.read_ometiff(file)
                # Obtain filename
                filename_ome_list.append(os.path.basename(file))
                ome_list.append(ome_array)
                meta_list.append(metadata)
            # Convert list to 6D array: S, T, Z, C, Y, X
            array6d_ome = np.stack(ome_list, axis=0)
            array6d_ome.shape
            # Meta data
            pixel_x = meta_list[0].image(0).Pixels.get_PhysicalSizeX()
            pixel_y = meta_list[0].image(0).Pixels.get_PhysicalSizeY()
            pixel_z = meta_list[0].image(0).Pixels.get_PhysicalSizeZ()

            # Print meta data
            print( 'Pixel size X: ' + str(pixel_x) )
            print( 'Pixel size Y: ' + str(pixel_y) )
            print( 'Pixel size Z: ' + str(pixel_z) )

            # Convert to SpinX 6D array format by permutation
            # From: [S, T, Z, C, Y, X] to [Y, X, Z, T, S, C]
            perm_ome_spinx = (4, 5, 2, 1, 0, 3 ) # Order

            array6d_sx = np.transpose(array6d_ome, perm_ome_spinx)
            array6d_sx.shape

            # Add indices in basefile name to get a list of 
            filename_list = self.expand_filename(filename_ome_list)       
        else:
            array6d_sx, _, filename_img_list = self.convert_6d(image_paths, n_slices, n_frames)
            filename_list = filename_img_list

        # Reshape list to 3D nested structure
        n_cells = len(filename_list) // (n_slices * n_frames)
        # Convert list in nested list
        filename_list = np.array(filename_list).reshape(n_cells, n_frames, n_slices)
        # ================= IMPORT LIST OF FILES END ================= #
        return array6d_sx, n_cells, filename_list

    def create_dir_structure(self, DIR, exp_name):
        """
        Create SpinX folder structure
        """
        # Output folder
        main_folder = exp_name
        # Generate folder path
        main_folder = os.path.join(DIR, main_folder)
        # Check for folder
        if not os.path.exists( main_folder ):
            # Main Folder
            os.makedirs( main_folder )
            # Model
            os.makedirs( os.path.join(main_folder, 'model') )
            os.makedirs( os.path.join(main_folder, 'model', 'complete_model') )
            os.makedirs( os.path.join(main_folder, 'model', 'cortex') )
            os.makedirs( os.path.join(main_folder, 'model', 'cortex', 'alignment') )
            os.makedirs( os.path.join(main_folder, 'model', 'cortex', 'segmentation') )
            os.makedirs( os.path.join(main_folder, 'model', 'spindle') )
            os.makedirs( os.path.join(main_folder, 'model', 'spindle', 'axes') )
            os.makedirs( os.path.join(main_folder, 'model', 'spindle', 'axes_no') )
            os.makedirs( os.path.join(main_folder, 'model', 'spindle', 'correct') )
            os.makedirs( os.path.join(main_folder, 'model', 'spindle', 'tracking') )
            # Plots
            os.makedirs( os.path.join(main_folder, 'plots') )
            os.makedirs( os.path.join(main_folder, 'plots', 'pole_cortex') )
            # Video
            os.makedirs( os.path.join(main_folder, 'video') )
            os.makedirs( os.path.join(main_folder, 'video', 'model') )
            os.makedirs( os.path.join(main_folder, 'video', 'raw') )
            # Video frames
            os.makedirs( os.path.join(main_folder, 'video_frames') )
            os.makedirs( os.path.join(main_folder, 'video_frames', 'fused') )
            os.makedirs( os.path.join(main_folder, 'video_frames', 'overlay') )
            print('Output: ##### Create exp output folder. #####')
        else:
            print('Output: ##### Exp output folder exists. #####')
        
    def get_list(self, image_paths):
        '''
        Input: List of images
        Return: List of images, Filetype'''
        file_type_list = []
        img_list = []
        for filepath in image_paths:
            # Obtain filename
            filename = os.path.basename(filepath)
            # Keep file extension after first dot (.ome.tiff)
            if filename.split(os.extsep, 1)[1].lower() == 'ome.tiff':
                image_list.append(filepath)
                file_type_list.append('ome-tiff')
            elif os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                img_list.append(filepath)
                file_type_list.append('tif-png')
        # Sort list alphabetically (libary)
        img_list = natsorted(img_list)
        
        # Conver to a set (get unique values) to check if the file format is consistent
        if len(set(file_type_list)) > 1:
            sys.exit('Mixed file formats were imported.')
        # Get file format of loaded images
        file_type = list(set(file_type_list))[0]
        return img_list, file_type
    
    def get_list_dir(self, DIR):
        '''
        Input: Directory
        Return: Sorted list of images, Filetype
        '''
        file_type_list = []
        img_list = []
        for filepath in os.listdir(DIR):
            # Obtain filename
            filename = os.path.basename(filepath)
            # Keep file extension after first dot (.ome.tiff)
            if filename.split(os.extsep, 1)[1].lower() == 'ome.tiff':
                img_list.append(os.path.join(DIR, filename))
                file_type_list.append('ome-tiff')
            elif os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                img_list.append(os.path.join(DIR, filename))
                file_type_list.append('tif-png')
        # Sort list alphabetically (libary)
        img_list = natsorted(img_list)
        
        # Convert to a set (get unique values) to check if the file format is consistent
        if len(set(file_type_list)) > 1:
            sys.exit('Mixed file formats were imported.')
        # Get file format of loaded images
        file_type = list(set(file_type_list))[0]
        return img_list, file_type
    
    
    def expand_filename(self, basename_list):
        '''
        Take a list of base filenames and expand with cell id, time id and z-sliced id
        '''
        expand_list = []
        # Loop through cells
        for c_id in range(2):
            fname_c = basename_list[c_id] + '_c' + str(c_id)
            # Loop through time
            for t_id in range(5):
                fname_t = fname_c + '_t' + str(t_id)
                # Loop through z-slices
                for z_id in range(3):
                    fname_z = fname_t + '_z' + str(z_id)
                    expand_list.append(fname_z)
        return expand_list

    def load_image_list(self, img_list, index):
        """
        Input: A list with image path to all images; index.
        Output: Read i-th image from the list.
        """
        file_path = img_list[index]
        # Obtain filename
        filename = os.path.basename(file_path)
        # Get only filename without path or extension
        #filename_wo_ext = os.path.splitext(filename)[0]
        # Read image
        img = skimage.io.imread(file_path)
        if len(img.shape)==3:
            img = rgb2gray(img)
        elif len(img.shape)>3:
            print('Input image dimension is larger than 3')
        output = np.array(img)
        return output, filename
    
    def convert_5d(self, img_list, n_slices, n_time):
        """
        Input: A list with image path to all images, number of z-slices; Number of frames.
        Output: 5D array with (H x W x D x T x C) or (Y x X x Z x T x C).
        Z refers to -slices, T refers to time points, C refers to cell id.
        """
        temp_array = []
        name_list = []
        n_cells = len(img_list)//(n_slices*n_time)
        for i in range(len(img_list)):
            mask, name_mask = self.load_image_list(img_list, i)
            # Read the image dimensions from first image
            if i == 1:
                # Image info
                img_height = mask.shape[0] # Y
                img_width = mask.shape[1] # X
                temp_array.append(mask)
                name_list.append(name_mask)
            else:
                temp_array.append(mask)
                name_list.append(name_mask)
           
        # use "F" Fortran for correct order
        array5d = np.dstack(temp_array).reshape(img_height, img_width, n_slices, n_time, n_cells, order='F')
        # Convert list in nested list
        name_list = np.array(name_list).reshape(n_cells, n_time, n_slices)
        return array5d, n_cells, name_list

    def convert_6d(self, img_list, n_slices, n_time, n_channel=None):
        """
        Input: A list with image path to all images, number of z-slices; Number of frames, n_channel.
        Output: 6D array with (H x W x D x T x S x C) or (Y x X x Z x T x S x C).
        Z refers to -slices, T refers to time points, S refers to Series/Cell, C refers to Channel.
        """
        temp_array = []
        name_list = []
        n_cells = len(img_list)//(n_slices*n_time)
        for i in range(len(img_list)):
            mask, name_mask = self.load_image_list(img_list, i)
            # Read the image dimensions from first image
            if i == 1:
                # Image info
                img_height = mask.shape[0] # Y
                img_width = mask.shape[1] # X
                temp_array.append(mask)
                name_list.append(name_mask)
            else:
                temp_array.append(mask)
                name_list.append(name_mask)
        # Set Channel
        if n_channel:
            pass
        else:
            n_channel = 1
        # use "F" Fortran for correct order
        array6d = np.dstack(temp_array).reshape(img_height, img_width, n_slices, n_time, n_cells, n_channel, order='F')
        
        # Convert list in nested list
        # name_list = np.array(name_list).reshape(n_cells, n_channel, n_time, n_slices)
        return array6d, n_cells, name_list
    
    def boundary_6d(self, array6d, dist_slices, pixelsize_micron_xy):
        """
        Input: 6D array (H x W x D x T x S x C) of grayscale images; 1D array with z-slice distances.
        Output: List with a boundary coordinates (Y x X x Z x C); measurement table [diameter,minor_axis, area, y0, x0].
        Access list: list[c_id][tp][z]
        """
        contours6d = []
        info6d = []
        # Loop over cells
        for c_id in range(len(array6d[0,0,0,0,:,0])):
            #print("---------------| Cell id:" + str(c_id))
            # Loop over time points
            contours5d = []
            info5d = []
            for tp in range(len(array6d[0,0,0,:,0,0])):
                #print("Time point:" + str(tp))  
                contours4d = []
                info4d = []
                # Loop over z-slices
                for z in range(len(array6d[0,0,:,0,0,0])):
                    # Binarize the image to avoid discontinued splines
                    bw = self.binarize_img(array6d[:,:,z,tp,c_id,0])
                    # Convert to unsiged byte image format
                    u_bw = bw.astype(np.uint8)
                    u_bw*=255
                    # Find boundary coordinates
                    contours = find_contours(u_bw, 0.9, fully_connected='high')
                    # Convert to micron
                    contours_um = []
                    for c in contours:
                        contours_um.append(c / pixelsize_micron_xy)
                        
                    #contours = contours / pixelsize_micron_xy
                    # Create column with z-coordinate (starts from focal plane = 0) and merge it with the contours array (Y, X)
                    z_column = np.full((len(contours_um[0]), 1), dist_slices[z])
                    temp_contours3d = np.append(contours_um[0], z_column, axis=1)
                    contours4d.append(temp_contours3d)
                    # Find diameter, area, centroid
                    label_bw = label(u_bw, connectivity = u_bw.ndim)
                    props = regionprops(label_bw)
                    diameter = props[0].major_axis_length / pixelsize_micron_xy
                    minor_axis = props[0].minor_axis_length / pixelsize_micron_xy
                    area = props[0].area / pixelsize_micron_xy
                    x0 = props[0].centroid[0] / pixelsize_micron_xy
                    y0 = props[0].centroid[1] / pixelsize_micron_xy
                    info_array = [diameter,minor_axis, area, x0, y0]
                    info4d.append(info_array)
                # Append 4D to 5D
                contours5d.append(contours4d)
                info5d.append(info4d)
            # Append 5D to 6D
            contours6d.append(contours5d)
            info6d.append(info5d)
        return contours6d, info6d

    def binarize_img(self, img):
        """
        Simple binarize function.
        """
        # Binarize image
        img_gray = rgb2gray(img)
        # Binarize image
        bw = img_gray > 0
        return bw
    
    def pxl_micron(value):
        """
        Convert pixel distances to microns
        Input: Pixel value (global: pixelsize_micron_xy)
        Output: micron
        """
        global pixelsize_micron_xy
        micron = value/pixelsize_micron_xy
        return micron

    def micron2pxl(value):
        """
        Convert micron to pixel distances
        Input: Micron (global: pixelsize_micron_xy)
        Output: Pixel
        """
        global pixelsize_micron_xy
        pixel = value*pixelsize_micron_xy
        return pixel 

    def micron2pxl_axial(value):
        """
        Convert micron to pixel distances
        Input: Micron (global: pixelsize_micron_xy)
        Output: Pixel
        """
        global pixelsize_micron_z
        pixel = value*pixelsize_micron_z
        return pixel 

    def create_folder_structure(DIR, exp_name):
        """
        Create folder structure
        """
        # Output folder
        main_folder = exp_name
        # Generate folder path
        main_folder = os.path.join(DIR, main_folder)
        # Check for folder
        if not os.path.exists( main_folder ):
            # Main Folder
            os.makedirs( main_folder )
            # Model
            os.makedirs( os.path.join(main_folder, 'model') )
            os.makedirs( os.path.join(main_folder, 'model', 'complete_model') )
            os.makedirs( os.path.join(main_folder, 'model', 'cortex') )
            os.makedirs( os.path.join(main_folder, 'model', 'cortex', 'alignment') )
            os.makedirs( os.path.join(main_folder, 'model', 'cortex', 'segmentation') )
            os.makedirs( os.path.join(main_folder, 'model', 'spindle') )
            os.makedirs( os.path.join(main_folder, 'model', 'spindle', 'axes') )
            os.makedirs( os.path.join(main_folder, 'model', 'spindle', 'axes_no') )
            os.makedirs( os.path.join(main_folder, 'model', 'spindle', 'correct') )
            os.makedirs( os.path.join(main_folder, 'model', 'spindle', 'tracking') )
            # Plots
            os.makedirs( os.path.join(main_folder, 'plots') )
            os.makedirs( os.path.join(main_folder, 'plots', 'pole_cortex') )
            # Video
            os.makedirs( os.path.join(main_folder, 'video') )
            os.makedirs( os.path.join(main_folder, 'video', 'model') )
            os.makedirs( os.path.join(main_folder, 'video', 'raw') )
            # Video frames
            os.makedirs( os.path.join(main_folder, 'video_frames') )
            os.makedirs( os.path.join(main_folder, 'video_frames', 'fused') )
            os.makedirs( os.path.join(main_folder, 'video_frames', 'overlay') )
            print('Output: ##### Create exp output folder. #####')
        else:
            print('Output: ##### Exp output folder exists. #####')
            
    def generate_seq(self, start, stop, step=1):
        """
        A function to create a sequence similar to function in R.
        """        
        n = int(round((stop - start)/float(step)))
        if n > 1:
            return([start + step*i for i in range(n+1)])
        elif n == 1:
            return([start])
        else:
            return([])

    def find_diameter(self, array, mode='max'):
        """
        Input: An array of diameter values from all z-slices at one time point.
        Output: Index and value of the largest diameter.
        """
        if mode == 'max':
            max_val = np.max(array)
            get_idx = np.where(array == max_val)
            max_idx = get_idx[0][0]
        elif mode == 'median':
            # list of elements to calculate median 
            n_num = array.tolist()
            n = len(n_num) 
            n_num.sort() 

            if n % 2 == 0: 
                median1 = n_num[n//2] 
                median2 = n_num[n//2 - 1] 
                median = (median1 + median2)/2
                # Rename
                max_val = median
            else: 
                median = n_num[n//2]
                # Rename
                max_val = median
            get_idx = np.where(array == max_val)
            max_idx = get_idx[0][0]
            
        return max_idx, max_val

    def slice_estimate(self, x0, y0, factor, diameter, slice_select):
        """
        Input: Centroid coordinates (X,Y,Z) obtained from the best focal plane.
        Z-coordinate is estimated by (diameter of best focal plane)/2.
        Factor defines the distance from each pixel to the centroid. The factor is in pixel.
        Output: Numpy Array
        est_array = [ [y, x, z],
                      [y, x, z] ] 
        """
        if slice_select == "bot":
            z0 = (diameter//2)*(-1)
            est_array = np.array([
                    [x0, y0, z0], 
                    [x0+factor, y0+factor, z0+factor],
                    [x0-factor, y0-factor, z0+factor], 
                    [x0+factor, y0-factor, z0+factor],
                    [x0-factor, y0+factor, z0+factor],
                    ])
        elif slice_select == "top":
            z0 = (diameter//2)
            est_array = np.array([
                                [x0, y0, z0], 
                                [x0+factor, y0+factor, z0-factor],
                                [x0-factor, y0-factor, z0-factor], 
                                [x0+factor, y0-factor, z0-factor],
                                [x0-factor, y0+factor, z0-factor],
                                ])
        return est_array

    def merge_array(self, x, y, z, dim):
        """
        Input: X, Y, Z array with shape (N x N)
        Output: Numpy Array
        xyz = [ [y, x, z],
                [y, x, z] ]
        Access:
        Row-wise: xyz[1]
        Column-wise: xyz[:,1]
        """
        x_1d = x.reshape(x.size, dim)
        y_1d = y.reshape(y.size, dim)
        z_1d = z.reshape(z.size, dim)
        xyz = np.hstack([x_1d, y_1d, z_1d])
        return xyz

    def align_stack(self, array, centroid, centroid_reference):
        """
        Align contours (x,y) through z-stack with
        respect to a reference centroid (z-slice with the largest diameter).
        Input: Array[x,y,z] coordinates; corresponding centroid [cx, cy];
        Reference centroid [cx_ref, cy_ref]
        Output: Shifted contour array [x,y,z]; corresponding centroid [cx, cy]."""
        cx = centroid[0]
        cy = centroid[1]

        cx_ref = centroid_reference[0]
        cy_ref = centroid_reference[1]

        array_new = array.copy()
        dif_x = cx_ref - cx
        cx_new = cx + dif_x
        array_new[:,0] = array[:,0] + dif_x
        dif_y = cy_ref - cy
        cy_new = cy + dif_y
        array_new[:,1] = array[:,1] + dif_y
        
        return array_new, cx_new, cy_new

    def align_stack_plot(self, raw_img, contours, centroids, suffix, exp_name, output_dir):
        """
        Plot overlay to visualise alignment.
        Input: Raw image[x,y,z]; contours[x,y,z]; centroids[x,y,z];
        corrected contour[x,y,z]; corrected_contour[x,y,z]
        Output: Save plot.
        """
        
        # Define colors for different z-slices.
        # Use RGB if number of slices is 3 or less
        
        if len(contours) <= 3:
            colors = ['r', 'g', 'b']
        else:     
            from random import randint
            # Define random color
            colors = []
            for i in range(255):
                colors.append('#%06X' % randint(0, 0xFFFFFF))
        
        # Create max projection image
        max_proj = np.max(raw_img, axis=2)

        # Overlay

        plt.figure()
        plt.axis("off")
        fig = plt.imshow(max_proj, cmap='gray')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        
        for i in range(len(contours)):
            # Plot centroid
            plt.plot(centroids[i][4], centroids[i][3], color=colors[i], marker='o', markersize=5, alpha=0.4)
            # Plot contour
            plt.plot(contours[i][:,1], contours[i][:,0], color=colors[i], linestyle='-', linewidth=3, alpha=0.4)
        # Save plot
        name_export = 'cell_c_' + str(c_id) + '_tp_' + str(tp)
        #full_name_export = 'figs/' + exp_name + '/model/cortex/alignment/' + name_export + suffix + '.pdf'
        full_name_export = os.path.join(output_dir, exp_name, 'model' , 'cortex', 'alignment', name_export + suffix + '.pdf')
        plt.savefig(full_name_export, dpi=300, bbox_inches='tight', pad_inches = 0, transparent=True)
        plt.close('all')



############################
# SPINX OME-TIFF FUNCTIONS #
# ###########################

class OME_TIFF():
    """
    Adapted from APEER: apeer-ometiff-library (https://github.com/apeer-micro/apeer-ometiff-library)
    and CellProfiler: python-bioformats (https://github.com/CellProfiler/python-bioformats)
    """
    def __init__(self):
        pass
    
    def read_ometiff(self, input_path):
        """
        Read OME-TIFF
        """
        import omexmlClass
        import tifffile
        with tifffile.TiffFile(input_path) as tif:
            array = tif.asarray()
            omexml_string = tif.ome_metadata

        # Turn Ome XML String to an Bioformats object for parsing
        metadata = omexmlClass.OMEXML(omexml_string)

        # Parse pixel sizes
        pixels = metadata.image(0).Pixels
        size_c = pixels.SizeC
        size_t = pixels.SizeT
        size_z = pixels.SizeZ
        size_x = pixels.SizeX
        size_y = pixels.SizeY

        # Expand image array to 5D of order (T, Z, C, X, Y)
        if size_c == 1:
            array = np.expand_dims(array, axis=-3)
        if size_z == 1:
            array = np.expand_dims(array, axis=-4)
        if size_t == 1:
            array = np.expand_dims(array, axis=-5)

        # Makes sure to return the array in (T, Z, C, X, Y) order

        dim_format = pixels.DimensionOrder

        if dim_format == "XYCZT":
            pass
        elif dim_format == "XYZCT":
            array = np.moveaxis(array, 1, 2)
        elif dim_format == "XYCTZ":
            array = np.moveaxis(array, 0, 1)
        elif dim_format == "XYZTC":
            array = np.moveaxis(array, 0, 2)
        elif dim_format == "XYTZC":
            array = np.moveaxis(array, 0, 2)
            array = np.moveaxis(array, 0, 1)
        elif dim_format == "XYTCZ":
            array = np.moveaxis(array, 1, 2)
            array = np.moveaxis(array, 0, 1)
        else:
            print(array.shape)
            raise Exception("Unknow dimension format") 

        return array, metadata, omexml_string

    def update_omexml(self, omexml, Image_ID=None, Image_Name=None, Image_AcquisitionDate=None, 
                      DimensionOrder=None, dType=None, SizeT=None, SizeZ=None, SizeC=None, SizeX=None, SizeY=None,
                      PhysicalSizeX=None, PhysicalSizeY=None, PhysicalSizeZ=None,
                      ExposureTime=None,
                      Channel_ID=None, Channel_Name=None, Channel_SamplesPerPixel=None):
        """
        Update OME-XML with user input.
        """
        import omexmlClass
        metadata = omexmlClass.OMEXML(omexml)

        if Image_ID:
            metadata.image().set_ID(Image_ID)
        if Image_Name:
            metadata.image().set_Name(Image_Name)
        if Image_AcquisitionDate:
            metadata.image().Image.AcquisitionDate = Image_AcquisitionDate

        if DimensionOrder: # Dimension order
            metadata.image().Pixels.DimensionOrder = DimensionOrder
        if dType: # The pixel bit type, for instance PT_UINT8
            metadata.image().Pixels.PixelType = dType
        if SizeT: # The dimensions of the image in the T direction in pixels
            metadata.image().Pixels.set_SizeT(SizeT)
        if SizeZ: # The dimensions of the image in the Z direction in pixels
            metadata.image().Pixels.set_SizeZ(SizeZ)
        if SizeC: # The dimensions of the image in the C direction in pixels
            metadata.image().Pixels.set_SizeC(SizeC)
        if SizeX: # The dimensions of the image in the X direction in pixels
            metadata.image().Pixels.set_SizeX(SizeX)
        if SizeY: # The dimensions of the image in the Y direction in pixels
            metadata.image().Pixels.set_SizeY(SizeY)
        if PhysicalSizeX: # The length of a single pixel in Y direction
            metadata.image().Pixels.set_PhysicalSizeX(PhysicalSizeX)
        if PhysicalSizeY: # The length of a single pixel in Y direction
            metadata.image().Pixels.set_PhysicalSizeY(PhysicalSizeY)
        if PhysicalSizeZ: # The length of a single pixel in Z direction
            metadata.image().Pixels.set_PhysicalSizeZ(PhysicalSizeZ)

        if ExposureTime: # Duration of exposure time in seconds
            metadata.image().Plane.set_ExposureTime = ExposureTime
        
        if Channel_ID:
            metadata.image().Channel.ID = Channel_ID
        if Channel_Name:
            metadata.image().Channel.Name = Channel_Name
        if Channel_SamplesPerPixel:
            metadata.image().Channel.SamplesPerPixel = Channel_SamplesPerPixel
    
        metadata = metadata.to_xml().encode()

        return metadata


    def gen_omexml(self, array):
        """
        Generate OME-XML template
        """
        import omexmlClass
        
        #Dimension order is assumed to be TZCYX
        dim_order = "TZCYX"

        metadata = omexmlClass.OMEXML()
        shape = array.shape
        assert ( len(shape) == 5), "Expected array of 5 dimensions"

        metadata.image().set_Name("IMAGE")
        metadata.image().set_ID("0")

        pixels = metadata.image().Pixels
        pixels.ome_uuid = metadata.uuidStr
        pixels.set_ID("0")

        pixels.channel_count = shape[2]

        pixels.set_SizeT(shape[0])
        pixels.set_SizeZ(shape[1])
        pixels.set_SizeC(shape[2])
        pixels.set_SizeY(shape[3])
        pixels.set_SizeX(shape[4])

        pixels.set_DimensionOrder(dim_order[::-1])

        pixels.set_PixelType(omexmlClass.get_pixel_type(array.dtype))

        for i in range(pixels.SizeC):
            pixels.Channel(i).set_ID("Channel:0:" + str(i))
            pixels.Channel(i).set_Name("C:" + str(i))

        for i in range(pixels.SizeC):
            pixels.Channel(i).set_SamplesPerPixel(1)

        pixels.populate_TiffData()

        return metadata.to_xml().encode()



    def write_ometiff(self, output_path, array, mode='minisblack', omexml_str = None):
        """
        Write OME-TIFF.
        """
        import tifffile
        if omexml_str is None:
            omexml_str = self.gen_omexml(array)

        tifffile.imwrite(output_path, array,  photometric = mode, description=omexml_str, metadata = None)
        return omexml_str
    
    
    def write_omexml(self, path, omexml_str):
        """
        Export for each cell an XML file (prettified).
        """
        import xml.dom.minidom #Prettify XML
        omexml_parse = xml.dom.minidom.parseString(omexml_str)
        omexml_pretty = omexml_parse.toprettyxml()
        
        if path == 'print':
            print(omexml_pretty) 
        else:
            # Export XML
            f =  open(path, "wb")
            f.write(omexml_pretty.encode())
            f.close()
            print(omexml_pretty)


##################################
# SPINX RECONSTRUCTION FUNCTIONS #
# #################################

# Source: https://github.com/minillinim/ellipsoid
# 
# There are different ways to fit an ellipsoid (e.g. PCA). However, it does not guarantee that the ellipse obtained from the decomposition (eigen value) will be minimum bounding ellipse since points outside the ellipse is an indication of the variance.
# 
# Here, we use the Minimum Volume Enclosing Ellipsoid based on Khachiyan Algorithm.

class EllipsoidTool:
    """Some stuff for playing with ellipsoids"""
    def __init__(self): pass
    
    def getMinVolEllipse(self, P=None, tolerance=0.01):
        """ Find the minimum volume ellipsoid which holds all the points
        
        Based on work by Nima Moshtagh
        http://www.mathworks.com/matlabcentral/fileexchange/9542
        and also by looking at:
        http://cctbx.sourceforge.net/current/python/scitbx.math.minimum_covering_ellipsoid.html
        Which is based on the first reference anyway!
        
        Here, P is a numpy array of N dimensional points like this:
        P = [[x,y,z,...], <-- one point per line
             [x,y,z,...],
             [x,y,z,...]]
        
        Returns:
        (center, radii, rotation)
        
        """
        (N, d) = np.shape(P)
        d = float(d)
    
        # Q will be our working array
        Q = np.vstack([np.copy(P.T), np.ones(N)]) 
        QT = Q.T
        
        # initializations
        err = 1.0 + tolerance
        u = (1.0 / N) * np.ones(N)

        # Khachiyan Algorithm
        while err > tolerance:
            V = np.dot(Q, np.dot(np.diag(u), QT))
            M = np.diag(np.dot(QT , np.dot(linalg.inv(V), Q)))    # M the diagonal vector of an NxN matrix
            j = np.argmax(M)
            maximum = M[j]
            step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
            new_u = (1.0 - step_size) * u
            new_u[j] += step_size
            err = np.linalg.norm(new_u - u)
            u = new_u

        # center of the ellipse 
        center = np.dot(P.T, u)
    
        # the A matrix for the ellipse
        A = linalg.inv(
                       np.dot(P.T, np.dot(np.diag(u), P)) - 
                       np.array([[a * b for b in center] for a in center])
                       ) / d
                       
        # Get the values we'd like to return
        U, s, rotation = linalg.svd(A)
        radii = 1.0/np.sqrt(s)
        return center, radii, rotation
    

    def ellipsoid_fit(self, X=None):
        # https://github.com/aleksandrbazhin/ellipsoid_fit_python
        # Based on: http://www.mathworks.com/matlabcentral/fileexchange/24693-ellipsoid-fit
        # for arbitrary axes
        x = X[:, 0]
        y = X[:, 1]
        z = X[:, 2]
        D = np.array([x * x + y * y - 2 * z * z,
                     x * x + z * z - 2 * y * y,
                     2 * x * y,
                     2 * x * z,
                     2 * y * z,
                     2 * x,
                     2 * y,
                     2 * z,
                     1 - 0 * x])
        d2 = np.array(x * x + y * y + z * z).T # rhs for LLSQ
        u = np.linalg.solve(D.dot(D.T), D.dot(d2))
        a = np.array([u[0] + 1 * u[1] - 1])
        b = np.array([u[0] - 2 * u[1] - 1])
        c = np.array([u[1] - 2 * u[0] - 1])
        v = np.concatenate([a, b, c, u[2:]], axis=0).flatten()
        A = np.array([[v[0], v[3], v[4], v[6]],
                      [v[3], v[1], v[5], v[7]],
                      [v[4], v[5], v[2], v[8]],
                      [v[6], v[7], v[8], v[9]]])

        center = np.linalg.solve(- A[:3, :3], v[6:9])

        translation_matrix = np.eye(4)
        translation_matrix[3, :3] = center.T

        R = translation_matrix.dot(A).dot(translation_matrix.T)

        evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])
        evecs = evecs.T

        radii = np.sqrt(1. / np.abs(evals))
        radii *= np.sign(evals)
        return center, radii, evecs, v

    def getEllipsoidVolume(self, radii):
        """Calculate the volume of the blob"""
        return 4./3.*np.pi*radii[0]*radii[1]*radii[2]

    def plotEllipsoid(self, center, radii, rotation, ax=None, plotAxes=False, rStride=4, cStride=4, cageColor='k', cageAlpha=0.1):
        """Plot an ellipsoid"""
        SX = SpinX()
        make_ax = ax == None
        if make_ax:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
        u = np.linspace(0.0, 2.0 * np.pi, 100)
        v = np.linspace(0.0, np.pi, 100)
        
        # cartesian coordinates that correspond to the spherical angles:
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
        # rotate accordingly
        for i in range(len(x)):
            for j in range(len(x)):
                [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center
    
        if plotAxes:
            # make some purdy axes
            axes = np.array([[radii[0],0.0,0.0],
                             [0.0,radii[1],0.0],
                             [0.0,0.0,radii[2]]])
            # rotate accordingly
            for i in range(len(axes)):
                axes[i] = np.dot(axes[i], rotation)
    
            axis_color = ('r', 'g', 'b')
            xyz_cell_axis = []
            # plot axes
            for ax_c, p in enumerate(axes):
                X3 = np.linspace(-p[0], p[0], 100) + center[0]
                Y3 = np.linspace(-p[1], p[1], 100) + center[1]
                Z3 = np.linspace(-p[2], p[2], 100) + center[2]
                ax.plot(X3, Y3, Z3, color=axis_color[ax_c])
                # Extract axis coordinates of spheroid
                axis_coord = SX.merge_array(X3,Y3,Z3, 1)
                xyz_cell_axis.append(axis_coord)
        else:
            # make some purdy axes
            axes = np.array([[radii[0],0.0,0.0],
                             [0.0,radii[1],0.0],
                             [0.0,0.0,radii[2]]])
            # rotate accordingly
            for i in range(len(axes)):
                axes[i] = np.dot(axes[i], rotation)
            xyz_cell_axis = []
            for ax_c, p in enumerate(axes):
                X3 = np.linspace(-p[0], p[0], 100) + center[0]
                Y3 = np.linspace(-p[1], p[1], 100) + center[1]
                Z3 = np.linspace(-p[2], p[2], 100) + center[2]
                # Extract axis coordinates of spheroid
                axis_coord = SX.merge_array(X3,Y3,Z3, 1)
                xyz_cell_axis.append(axis_coord)
            #xyz_cell_axis = []
        # plot ellipsoid
        ax.plot_wireframe(x, y, z,  rstride=rStride, cstride=cStride, color=cageColor, alpha=cageAlpha, antialiased=True)

        if make_ax:
            #plt.show(block=False)
            #plt.pause(3)
            plt.close(fig)
            del fig
            
        # # Extract surface coordinates of the spheroid. Merge x, y, z coords column-wise
        xyz_cell = SX.merge_array(x, y, z, 1)
        return xyz_cell, xyz_cell_axis
    
    def data_regularize(self, data=None, type="spherical", divs=10):
        """Plot an ellipsoid"""
        limits = np.array([
            [min(data[:, 0]), max(data[:, 0])],
            [min(data[:, 1]), max(data[:, 1])],
            [min(data[:, 2]), max(data[:, 2])]])

        regularized = []

        if type == "cubic": # take mean from points in the cube

            X = np.linspace(*limits[0], num=divs)
            Y = np.linspace(*limits[1], num=divs)
            Z = np.linspace(*limits[2], num=divs)

            for i in range(divs-1):
                for j in range(divs-1):
                    for k in range(divs-1):
                        points_in_sector = []
                        for point in data:
                            if (point[0] >= X[i] and point[0] < X[i+1] and
                                    point[1] >= Y[j] and point[1] < Y[j+1] and
                                    point[2] >= Z[k] and point[2] < Z[k+1]):
                                points_in_sector.append(point)
                        if len(points_in_sector) > 0:
                            regularized.append(np.mean(np.array(points_in_sector), axis=0))

        elif type == "spherical": #take mean from points in the sector
            divs_u = divs 
            divs_v = divs * 2

            center = np.array([
                0.5 * (limits[0, 0] + limits[0, 1]),
                0.5 * (limits[1, 0] + limits[1, 1]),
                0.5 * (limits[2, 0] + limits[2, 1])])
            d_c = data - center

            #spherical coordinates around center
            r_s = np.sqrt(d_c[:, 0]**2. + d_c[:, 1]**2. + d_c[:, 2]**2.)
            d_s = np.array([
                r_s,
                np.arccos(d_c[:, 2] / r_s),
                np.arctan2(d_c[:, 1], d_c[:, 0])]).T

            u = np.linspace(0, np.pi, num=divs_u)
            v = np.linspace(-np.pi, np.pi, num=divs_v)

            for i in range(divs_u - 1):
                for j in range(divs_v - 1):
                    points_in_sector = []
                    for k, point in enumerate(d_s):
                        if (point[1] >= u[i] and point[1] < u[i + 1] and
                                point[2] >= v[j] and point[2] < v[j + 1]):
                            points_in_sector.append(data[k])

                    if len(points_in_sector) > 0:
                        regularized.append(np.mean(np.array(points_in_sector), axis=0))
    # Other strategy of finding mean values in sectors
    #                    p_sec = np.array(points_in_sector)
    #                    R = np.mean(p_sec[:,0])
    #                    U = (u[i] + u[i+1])*0.5
    #                    V = (v[j] + v[j+1])*0.5
    #                    x = R*math.sin(U)*math.cos(V)
    #                    y = R*math.sin(U)*math.sin(V)
    #                    z = R*math.cos(U)
    #                    regularized.append(center + np.array([x,y,z]))
        return np.array(regularized)
    
    def set_axes_equal(ax):
        '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Input
          ax: a matplotlib axis, e.g., as output from plt.gca().
        '''

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])



##################################
# SPINX MODELLING FUNCTIONS #
# #################################

class SpinX_Modelling():
    def __init__(self):
        pass
    
    def pole_cortex_distance3d(self, pole, intersect):
        """
        Calculate pole-to-cortex distances
        Input: Pole array: (x,y,z) coordinates of first pole[0], second pole[1]
        Output: 3D Euclidean distances for both poles
        """    
        # Euclidean distance in 3D for pole 1
        d_1 = np.sqrt(np.sum((pole[0] - intersect[0])**2))
        # Euclidean distance in 3D for pole 2
        d_2 = np.sqrt(np.sum((pole[1] - intersect[1])**2))
        d = [d_1, d_2]

        # Measure length of the axis
        l = np.sqrt(np.sum((pole[1] - pole[0])**2))
        return d, l

    # Calculate 3D distance
    def distance3d(self, point1, point2):
        """
        Calculate 3D distance
        Input: Array of (x,y,z) coordinates of point 1 and 2.
        Output: 3D Euclidean distance
        """
        # Point 1:
        x1 = point1[0]
        y1 = point1[1]
        z1 = point1[2]
        # Point 2:
        x2 = point2[0]
        y2 = point2[1]
        z2 = point2[2]
        # Euclidean distance in 3D
        d = ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)**(1/2) 
        return d

    # other implementation
    def extend3d_line(self, p1, p2, length):
        """
        Extend 3D line
        Input:  p1&2[x]
                    [y]
                    [z]
                length (in percent): 5 == 500% of its length
        Output: First [x,y,z] point: pDest[0]
                Second [x,y,z] point: pDest[1]
        """
        x1 = p1[0]
        y1 = p1[1]
        z1 = p1[2]
        
        x2 = p2[0]
        y2 = p2[1]
        z2 = p2[2]
        pDest_x = [ x1 + length*(x2 - x1), x1 - length*(x2 - x1) ]
        pDest_y = [ y1 + length*(y2 - y1), y1 - length*(y2 - y1) ]
        pDest_z = [ z1 + length*(z2 - z1), z1 - length*(z2 - z1) ]
        
        pDest = np.transpose([pDest_x, pDest_y, pDest_z])
        return pDest

    # Use only anchor to speed up
    def pair_distance3d(self, mem_surface5d, anchors):
        """
        Calculate pair-wise distances
        Input:  surface_3dcoords: Array[x, y, z] coordinates of cell surface;
                anchors[1, 2] points of the extended 3D line.
        Output: List of distances
        """
        merged_d = []
        for i in range(len(mem_surface5d)):
            temp_coord = mem_surface5d[i]
            x1 = temp_coord[0]
            y1 = temp_coord[1]
            z1 = temp_coord[2]

            # Spindle
            x2 = anchors[0]
            y2 = anchors[1]
            z2 = anchors[2]
            d = ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)**(1/2) 
            merged_d.append(d)
        return merged_d

    def hrt(self,surface_3dcoords, anchors):
        """
        Heuristic Ray-Tracing method:
        Input:  surface_3dcoords: Array[x, y, z] coordinates of cell surface;
                anchors[1, 2] points of the extended 3D line.
        Output: Intersect point 1: intercept[0] and Intercept point 2: intercept[1]
        """

        # Get list with all calculated distances
        merged_d1 = self.pair_distance3d(surface_3dcoords, anchors[:,0]) # First point
        merged_d2 = self.pair_distance3d(surface_3dcoords, anchors[:,1]) # Second point

        # Find smallest distance (value and its index)
        val1, idx1 = min((val, idx) for (idx, val) in enumerate(merged_d1)) # First point
        val2, idx2 = min((val, idx) for (idx, val) in enumerate(merged_d2)) # Second point

        # Obtain membrane surface (x,y,z) with the smallest distance  
        intersect_1 = surface_3dcoords[idx1] # First point
        intersect_2 = surface_3dcoords[idx2] # Second point
        intersect = [intersect_1, intersect_2]
        return intersect

    def line_ellipsoid_intersection(self, p3, c, r):
        """
        Analytical Ray-Tracing method:
        Input:  Extended line p3 | first (x,y,z) - coordinate p3[0] | second (x,y,z) - coordinate p3[1];
                c: Centroid of the ellipsoid
                r: a,b,c radii of the ellipsoid
        Output: Intersect point 1: intersect[0] and Intersect point 2: intersect[1]
        """    
        def square(f):
            return f*f

        from math import sqrt

        # l1[0],l1[1],l1[2]  P1 coordinates (point of line)
        # l2[0],l2[1],l2[2]  P2 coordinates (point of line)
        # c[0],c[1],c[2]     P3 coordinates (centroid of the ellipsoid)
        # r[0],r[1],r[2]     Conic radii (a, b, c)
        
        # Output: P1(x,y,z) and P2(x,y,z) intersection coordinates
        #
        # This function returns a pointer array which first index indicates
        # the number of intersection point, followed by coordinate pairs.
        # For spheroid:
        # http://paulbourke.net/geometry/circlesphere/#:~:text=A%20line%20that%20passes%20through,these%20are%20called%20antipodal%20points.&text=A%20plane%20can%20intersect%20a,%22cut%22%20is%20a%20circle.
        # or using sphere coordinates:
        # https://johannesbuchner.github.io/intersection/intersection_line_ellipsoid.html
        # and:
        # https://cs.oberlin.edu/~bob/cs357.08/VectorGeometry/VectorGeometry.pdf

        p1 = p2 = None

        # Split coordinates
        l1 = p3[0]
        l2 = p3[1]
        
        chi = square(l2[0]-l1[0]) / square(r[0]) + \
        square(l2[1]-l1[1]) / square(r[1]) + \
        square(l2[2]-l1[2]) / square(r[2])
        
        gamma = (2*(l2[0]-l1[0])*(l1[0]-c[0])) / square(r[0]) + \
        (2*(l2[1]-l1[1])*(l1[1]-c[1])) / square(r[1]) + \
        (2*(l2[2]-l1[2])*(l1[2]-c[2])) / square(r[2])

        zeta = square(l1[0] - c[0]) / square(r[0]) + \
        square(l1[1] - c[1]) / square(r[1]) + \
        square(l1[2] - c[2]) / square(r[2]) - 1


        i = gamma * gamma - 4.0 * chi * zeta

        if i < 0.0:
            pass  # no intersections
            p1 = []
            p2 = []
        elif i == 0.0:
            # one intersection
            t = -gamma / (2.0 * chi)
            p1 = (l1[0] + t * (l2[0] - l1[0]),
                  l1[1] + t * (l2[1] - l1[1]),
                  l1[2] + t * (l2[2] - l1[2]),
                  )
            p2 = []
        elif i > 0.0:
            # first intersection
            t = (-gamma + sqrt(i)) / (2.0 * chi)
            p1 = (l1[0] + t * (l2[0] - l1[0]),
                  l1[1] + t * (l2[1] - l1[1]),
                  l1[2] + t * (l2[2] - l1[2]),
                  )

            # second intersection
            t = (-gamma - sqrt(i)) / (2.0 * chi)
            p2 = (l1[0] + t * (l2[0] - l1[0]),
                  l1[1] + t * (l2[1] - l1[1]),
                  l1[2] + t * (l2[2] - l1[2]),
                  )
        intersect = [p1, p2]
        return intersect


    def ellipsoid_volume(self, radii):
        """
        Calculate volume of any kind of ellipsoid:
        Input:  radii: a,b,c radii of the ellipsoid
        Output: Volume v
        """      
        from math import pi
        a = radii[0]
        b = radii[1]
        c = radii[2]
        v = (4/3)*pi*a*b*c
        return v

    # Surface:
    # The surface area of an ellipsoid cannot be calculated precisly.

    def ellipsoid_surface_area(self, radii):
        """
        Calculate surface area:
        There are different formula to calculate the surface area depending on the relation of a, b, c.
        
        Input:  radii: a,b,c radii of the ellipsoid
        Output: Surface area s in unit^2
        """
        SX = SpinX()
        from math import sqrt, pi, atanh, asin

        # There are different formula to calculate the surface area depending on the relation of a, b, c.

        # Create a sequence based on the number of radii
        full_idx = SX.generate_seq(0,len(radii)-1,1)
        # Find min and max indices
        min_v, min_idx = min((min_v, min_idx) for (min_idx, min_v) in enumerate(radii))
        max_v, max_idx = max((max_v, max_idx) for (max_idx, max_v) in enumerate(radii))
        # Create a list with min and max indices
        temp_idx = [min_idx, max_idx]
        # Find missing index
        [u_idx] = np.setdiff1d(full_idx,temp_idx)
        
        # Let a be the largest and c the smallest radius
        a = radii[max_idx]
        b = radii[u_idx]
        c = radii[min_idx]
        # Type of ellipsoids:
        # a is the equatorial, c is the polar radius
        # el[0] a = b = c: Spheroid
        # el[1] a = b > c: Oblate ellipsoid (https://mathworld.wolfram.com/OblateSpheroid.html)
        # el[2] a = b < c: Prolate ellipsoid (https://mathworld.wolfram.com/ProlateSpheroid.html)
        # el[3] a > b > c: Scalene ellipsoid (not considered here because differences between a and b is small)

        if a == b == c:
            el = 0
        elif a == b > c:
            el = 1
        elif a == b < c:
            el = 2   
        elif a > b > c:
            el = 3
       
        # Surface:
        if el == 0:
            r = (a*b*c)**(1./3.)
            s = 4*pi*r**2
        elif el == 1:
            k = sqrt(1 - (c**2/a**2))
            s = 2*pi*a**2*(1 + ((1-k**2)/k)*atanh(k))
        elif el == 2:
            l = sqrt(1 - (a**2/b**2))
            s = 2*pi*a**2*(1 + (c/(a*l))*asin(l))
        elif el == 3:
            p = 1.6075
            s = 4*pi*( (a**p * b**p + a**p * c**p + b**p * c**p)/3)**(1/p)        
        return s

    def sphericity3d(self, volume, surface_area):
        """
        Measure the sphericity of an object:
        Input:  volume, surface area
        Output: sphericity value (1=sphere, 0.84=hemisphere, 0.806=cube, 0.671=tetrahedron)
        https://en.wikipedia.org/wiki/Sphericity
        """      
        from math import pi
        V = volume
        A = surface_area
        psi = (pi**(1/3)*(6*V)**(2/3))/A
        return psi

    # Checks if a matrix is a valid rotation matrix.
    def is_rotation_matrix(self, rotation_matrix):
        """
        Check rotation matrix for identity.
        Input:  Rotation matrix (3x3)
        Output: True if n=0
        """   
        #https://www.learnopencv.com/rotation-matrix-to-euler-angles/
        R = rotation_matrix
        R_t = np.transpose(R)
        shouldBeIdentity = np.dot(R_t, R)
        I = np.identity(3, dtype = R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6


    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    def rotation_matrix2euler_angles(self, rotation_matrix):
        """
        Compute Euler Angles from rotation matrix
        Input:  Rotation matrix (3x3)
        Output: Angles (phi, theta, psi)
        https://www.gregslabaugh.net/publications/euler.pdf
        and:
        https://gist.github.com/crmccreary/1593090
        To cross-check: https://www.andre-gaschler.com/rotationconverter/
        """
        R = rotation_matrix
        assert(self.is_rotation_matrix(R))
        # Check if singular values s correspond to the identity matrix.
        sv = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sv < 1e-6
        if not singular:
            phi = math.atan2(R[2,1] , R[2,2])
            theta = math.atan2(-R[2,0], sv)
            psi = math.atan2(R[1,0], R[0,0])
        else:
            phi = math.atan2(-R[1,2], R[1,1])
            theta = math.atan2(-R[2,0], sv)
            psi = 0
        
        return np.array([phi, theta, psi])

    def pole_displacement3d(self, poles):
        """
        Calculate pole displacement through time
        Input:  List of poles (array of coodinates)
                poles[tp][pole1/2]
        Output: Displacement disp[pole1/2]
        """
        # Identify the current time point which is the last position of the list
        idx_t1 = len(poles)-1
        # Get index from the previous time point (t-1)
        idx_t0 = idx_t1-1
        # Pole 1
        t0_1 = poles[idx_t0][0] # t-1
        t1_1 = poles[idx_t1][0] # t
        # Pole 2
        t0_2 = poles[idx_t0][1]
        t1_2 = poles[idx_t1][1]

        disp1 = np.sqrt(np.sum((t1_1 - t0_1)**2))
        disp2 = np.sqrt(np.sum((t1_2 - t0_2)**2))
        
        disp = [disp1, disp2]
        return disp

    def centroid_displacement3d(self, centroid):
        """
        Calculate centroid displacement through time
        Input:  List of centroids (array of coodinates)
                poles[tp]
        Output: Displacement disp
        """
        # Identify the current time point which is the last position of the list
        idx_t1 = len(centroid)-1
        # Get index from the previous time point (t-1)
        idx_t0 = idx_t1-1
        # Centroid 1
        t0_1 = centroid[idx_t0] # t-1
        t1_1 = centroid[idx_t1] # t
        disp = np.sqrt(np.sum((t1_1 - t0_1)**2))
        return disp

    def decomposition3d(self, poles):
        """
        Note: Decomposition based on image canvas (not relative to spindle axis)
        Calculate equatorial displacement (parallel to y-axis) through time
        Input:  List of poles (array of coodinates)
                poles[tp][pole1/2]
        Output: Decomposed movements: decompose[long, eq, z]
        """
        # Identify the current time point which is the last position of the list
        idx_t1 = len(poles)-1
        # Get index from the previous time point (t-1)
        idx_t0 = idx_t1-1
        
        # Pole 1
        t0_1 = poles[idx_t0][0] # t-1
        t1_1 = poles[idx_t1][0] # t
        # Pole 2
        t0_2 = poles[idx_t0][1] # t-1
        t1_2 = poles[idx_t1][1] # t

        # Pole 1
        # Keep movement on the x-axis only. Other values cancel to 0.
        long_disp1 = np.sqrt( (t1_1[0] - t0_1[0])**2 )
        # Keep movement on the y-axis only. Other values cancel to 0.
        eq_disp1 = np.sqrt( (t1_1[1] - t0_1[1])**2 )
        # Keep movement on the z-axis only. Other values cancel to 0.
        z_disp1 = np.sqrt( (t1_1[2] - t0_1[2])**2 )
        
        # Pole 2
        # Keep movement on the x-axis only. Other values cancel to 0.
        long_disp2 = np.sqrt( (t1_2[0] - t0_2[0])**2 + 0 + 0 )
        # Keep movement on the y-axis only. Other values cancel to 0.
        eq_disp2 = np.sqrt( 0 + (t1_2[1] - t0_2[1])**2 + 0 )
        # Keep movement on the z-axis only. Other values cancel to 0.
        z_disp2 = np.sqrt( 0 + 0 + (t1_2[2] - t0_2[2])**2 )    
        
        decompose1 = [long_disp1, eq_disp1, z_disp1]
        decompose2 = [long_disp2, eq_disp2, z_disp2]
        return decompose1, decompose2

    def rotation_matrix_from_vectors(self, vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix

    def scale_line(self, poles, plot=True):
        """
        Scale the length of a line A to a reference line.
        Input: Poles[A/reference][Point1/2][xyz]
        Output: Points of the scaled line: scaled[0/1]
        """
        orig_line = np.asarray([poles[0][0], poles[0][1]])
        orig_line_length = np.sqrt(np.sum((orig_line[0] - orig_line[1])**2))
        
        ref_line = np.asarray([poles[1][0], poles[1][1]])
        ref_line_length = np.sqrt(np.sum((ref_line[0] - ref_line[1])**2))
        
        line_ratio = ref_line_length / orig_line_length

        cen = self.midpoint(np.asarray(poles[0][0]), np.asarray(poles[0][1]))
        
        
        t= line_ratio/2

        # Expand
        new1 = [cen[0] - t * (poles[0][1][0] - poles[0][0][0]),
                cen[1] - t * (poles[0][1][1] - poles[0][0][1]),
                cen[2] - t * (poles[0][1][2] - poles[0][0][2])
               ]
        new2 = [cen[0] + t * (poles[0][1][0] - poles[0][0][0]),
                cen[1] + t * (poles[0][1][1] - poles[0][0][1]),
                cen[2] + t * (poles[0][1][2] - poles[0][0][2])
               ]
        if plot == True:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            if old_3d == 1:
                #scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
                #ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)
                ax.set_aspect("equal") # Works only with older matplotlib==3.0.2 (unsolved bug with 3.3.1)
            else:
                xs = np.array([0,512])
                ys = np.array([0,512])
                zs = np.array([0,512])
                ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))  # aspect ratio is 1:1:1
                
            ax.set_xlim([0,23])
            ax.set_ylim([0,23])
            ax.plot([poles[0][0][0],poles[0][1][0]], [poles[0][0][1],poles[0][1][1]], 'r')
            ax.plot([new1[0],new2[0]], [new1[1],new2[1]], 'b', alpha=0.2)
            ax.plot([poles[1][0][0],poles[1][1][0]], [poles[1][0][1],poles[1][1][1]], 'g', alpha=0.2) # reference pole
        scaled = []
        scaled.append(new1)
        scaled.append(new2)
        return scaled

    def rotate_line3d(self, poles, scale=True, plot=True):
        """
        Rotate a line A to a reference line using a rotation matrix.
        Input: Poles[A/reference][Point1/2][xyz]
        Output: Points of the scaled line: scaled[0/1]
        """    
        # A corresponds to poles at t1
        p1_A = poles[0][0]
        p2_A = poles[0][1]
        # B corresponds to poles at t1
        p1_B = poles[1][0]
        p2_B = poles[1][1]

        # Convert to array
        A = np.array([p1_A,
                      p2_A])

        B = np.array([p1_B,
                      p2_B])
        # Find mid-point between 2 3D-points
        mid_pointA = self.midpoint(A[0], A[1])
        mid_pointB = self.midpoint(B[0], B[1])

        #Calculate centering coordinates
        Ac=A-np.ones((len(A),1))*mid_pointA
        # Get vectors A and B
        vecA = A[1] - A[0]
        vecB = B[1] - B[0]
        # Calculate rotation matrix (works with normalized or non-normalized vectors)
        R = self.rotation_matrix_from_vectors(vecA, vecB)
        # Rotating centred coordinates
        Arc=np.transpose( R.dot(np.transpose(Ac)) )
        # Rotating un-centred coordinates
        #Aruc=np.matmul(R,np.transpose(A))
        # Translate back to origin (rotation at midpoint of line)
        Ar = Arc+np.ones((len(A),1))*mid_pointA

        # Rotated line and translated points
        rot_trans_A = np.array(Ar[0]) + ( np.array(mid_pointB) - np.array(mid_pointA) ) 
        rot_trans_B = np.array(Ar[1]) + ( np.array(mid_pointB) - np.array(mid_pointA) )
        if scale == True:
            temp_poles = []
            temp_poles.append(Ar)
            temp_poles.append(poles[1])
            Ar = self.scale_line(temp_poles)
            Ar = np.asarray(Ar)
        # Convert to list (original input format)
        rot_line = Ar.tolist()
        # Keep reference poles in list
        parallel_lines = []
        parallel_lines.append(rot_line) # Rotated poles at t0
        parallel_lines.append(poles[1]) # Poles at t1
        
        if plot == True:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            if old_3d == 1:
                #scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
                #ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)
                ax.set_aspect("equal") # Works only with older matplotlib==3.0.2 (unsolved bug with 3.3.1)
            else:
                xs = np.array([0,512])
                ys = np.array([0,512])
                zs = np.array([0,512])
                ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))  # aspect ratio is 1:1:1
                
            ax.set_xlim([0,33])
            ax.set_ylim([0,33])
            ax.plot([p1_A[0],p2_A[0]], [p1_A[1],p2_A[1]], 'r') # Show line A
            ax.plot(mid_pointA[0], mid_pointA[1], 'go') # Show mid point of line A
            # Reference
            ax.plot([p1_B[0],p2_B[0]], [p1_B[1],p2_B[1]], 'b') # Show line B

            # Rotated
            ax.plot([Ar[0][0],Ar[1][0]], [Ar[0][1],Ar[1][1]], 'r', linestyle='dashed') # Show rotated line at orgin of line A
            #
            ax.plot([rot_trans_A[0],rot_trans_B[0]], [rot_trans_A[1],rot_trans_B[1]], 'm', linestyle='dashed') # Show translated of the rotated line A

            # Show Pole 1 only
            ax.plot(p1_A[0], p1_A[1], 'ro') # Line A
            ax.plot(Ar[0][0], Ar[0][1], 'ro') # Line A rotated
            ax.plot(rot_trans_A[0], rot_trans_A[1], 'ro') # Line A rotated and translated
        
        return parallel_lines


    def perpend3d(self, pole_t0, pole_t1, spindle_ax_start, spindle_ax_end):
        """
        Input: Single pole (e.g. poles[0][1])
        Output: Longitudinal, equatorial, axial
        """
        C_ext = np.asarray(spindle_ax_start)
        C = np.asarray(pole_t0)
        B_ext = np.asarray(spindle_ax_end)
        A = np.asarray(pole_t1)
        CB_dist = np.sqrt(np.sum((C_ext - B_ext)**2))
        vec_d = (C_ext - B_ext)/CB_dist
        vec_v = A - B_ext
        t = vec_v.dot(vec_d)
        vec_p = B_ext + t * vec_d
        long = np.sqrt(np.sum((C - vec_p)**2))
        eq = np.sqrt(np.sum((A - vec_p)**2))
        axial = np.sqrt(np.sum((C[2] - A[2])**2))
        decompose = [long, eq, axial]
        return decompose, vec_p

    def perpend2d(self, pole_t0, pole_t1, spindle_ax_start, spindle_ax_end):
        """
        Note: Decompose 
        Input: Single pole (e.g. poles[0][1])
        Output: Longitudinal, equatorial, axial
        """
        C_ext = np.asarray(spindle_ax_start[0:2])
        C = np.asarray(pole_t0[0:2])
        B_ext = np.asarray(spindle_ax_end[0:2])
        A = np.asarray(pole_t1[0:2])
        CB_dist = np.sqrt(np.sum((C_ext - B_ext)**2))
        vec_d = (C_ext - B_ext)/CB_dist
        vec_v = A - B_ext
        t = vec_v.dot(vec_d)
        vec_p = B_ext + t * vec_d
        long = np.sqrt(np.sum((C - vec_p)**2))
        eq = np.sqrt(np.sum((A - vec_p)**2))
        axial = np.sqrt(np.sum((pole_t0[2] - pole_t1[2])**2))
        decompose = [long, eq, axial]
        return decompose, vec_p

    def decomposition3d_v2(self, poles, plot=True, vers='3d'):
        """
        Decompose movement along the longitudinal, equatorial and axial movement with respect to spindle axis
        Note: Rotate spindle axis first to make sure they are parallel
        Input:  List of poles (array of coodinates)
                poles[tp][pole1/2]
        Output: Decomposed movements: decompose[long, eq, z]
        """

        # Formula: https://math.stackexchange.com/questions/1905533/find-perpendicular-distance-from-point-to-line-in-3d
        #vec3 d = (C - B) / C.distance(B);
        #vec3 v = A - B;
        #double t = v.dot(d);
        #vec3 P = B + t * d;
        #return P.distance(A);
        cen = self.midpoint(np.asarray(poles[0][0]), np.asarray(poles[0][1]))
        
        # Extend spindle axis at t0 (at centroid)
        t = np.sqrt(np.sum((np.asarray(poles[0][0]) - np.asarray(poles[0][1]))**2))*3
        p1_t0_ext1 = [cen[0] - t * (poles[0][1][0] - poles[0][0][0]),
                      cen[1] - t * (poles[0][1][1] - poles[0][0][1]),
                      cen[2] - t * (poles[0][1][2] - poles[0][0][2])
                     ]
        p1_t0_ext2 = [cen[0] + t * (poles[0][1][0] - poles[0][0][0]),
                      cen[1] + t * (poles[0][1][1] - poles[0][0][1]),
                      cen[2] + t * (poles[0][1][2] - poles[0][0][2])
                     ]
        
        #Calculate decomposed distances
        if vers == '3d':
            decompose1, proj_point_1 = self.perpend3d(pole_t0=poles[0][0], pole_t1=poles[1][0], spindle_ax_start=p1_t0_ext1, spindle_ax_end=p1_t0_ext2) # Pole 1
            decompose2, proj_point_2 = self.perpend3d(pole_t0=poles[0][1], pole_t1=poles[1][1], spindle_ax_start=p1_t0_ext1, spindle_ax_end=p1_t0_ext2) # Pole 2
        else:
            decompose1, proj_point_1 = self.perpend2d(pole_t0=poles[0][0], pole_t1=poles[1][0], spindle_ax_start=p1_t0_ext1, spindle_ax_end=p1_t0_ext2) # Pole 1
            decompose2, proj_point_2 = self.perpend2d(pole_t0=poles[0][1], pole_t1=poles[1][1], spindle_ax_start=p1_t0_ext1, spindle_ax_end=p1_t0_ext2) # Pole 2
            
        decompose = [decompose1, decompose2]
        proj_point = [proj_point_1, proj_point_2] # Projected point which is the intersection on the extended spindle axis
        if plot == True:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            if old_3d == 1:
                #scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
                #ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)
                ax.set_aspect("equal") # Works only with older matplotlib==3.0.2 (unsolved bug with 3.3.1)
            else:
                xs = np.array([0,512])
                ys = np.array([0,512])
                zs = np.array([0,512])
                ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))  # aspect ratio is 1:1:1
            
            ax.set_xlim([0,23])
            ax.set_ylim([0,23])
            ax.plot([poles[0][0][0],poles[0][1][0]], [poles[0][0][1],poles[0][1][1]], 'r')
            ax.plot([poles[1][0][0],poles[1][1][0]], [poles[1][0][1],poles[1][1][1]], 'b')
            ax.plot([p1_t0_ext2[0],p1_t0_ext1[0]], [p1_t0_ext2[1],p1_t0_ext1[1]], 'g', linestyle='dashed', alpha=0.2)
            ax.plot([poles[1][0][0],proj_point[0][0]], [poles[1][0][1],proj_point[0][1]], 'm', linestyle='dashed', alpha=0.2)

            ax.plot(poles[0][1][0], poles[0][1][1], 'r*')
            ax.plot(poles[1][1][0], poles[1][1][1], 'b*')


            ax.plot(proj_point[0][0], proj_point[0][1], 'bo')
            ax.plot(poles[0][0][0], poles[0][0][1], 'bo')
            ax.plot(poles[1][0][0], poles[1][0][1], 'bo')

            fig = plt.figure(figsize=(15,15))
            ax = fig.add_subplot(111, projection='3d')
            if old_3d == 1:
                #scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
                #ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)
                ax.set_aspect("equal") # Works only with older matplotlib==3.0.2 (unsolved bug with 3.3.1)
            else:
                xs = np.array([0,512])
                ys = np.array([0,512])
                zs = np.array([0,512])
                ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))  # aspect ratio is 1:1:1
            ax.view_init(elev=50, azim=0)
            ax.set_xlim([0,23])
            ax.set_ylim([0,23])
            ax.set_zlim([0,23])
            ax.plot3D([poles[0][0][0],poles[0][1][0]], [poles[0][0][1],poles[0][1][1]],[poles[0][0][2],poles[0][1][2]], 'r')  # Line at t0
            ax.plot3D([poles[1][0][0],poles[1][1][0]], [poles[1][0][1],poles[1][1][1]],[poles[1][0][2]],'b') # Line at t1
            ax.plot3D([p1_t0_ext2[0],p1_t0_ext1[0]], [p1_t0_ext2[1],p1_t0_ext1[1]], [p1_t0_ext2[2],p1_t0_ext1[2]], 'g', linestyle='dashed', alpha=0.2) # Extended line at t0
            ax.plot3D([poles[1][0][0],proj_point[0][0]], [poles[1][0][1],proj_point[0][1]], [poles[1][0][2],proj_point[0][2]], 'm', linestyle='dashed', alpha=0.2) # orthogonal line
     
            #ax.plot3D([poles[0][1][0]], [poles[0][1][1]], [poles[0][1][2]], 'r*')
            #ax.plot3D([poles[1][1][0]], [poles[1][1][1]], [poles[1][1][2]], 'b*')


            ax.plot3D([proj_point[0][0]], [proj_point[0][1]], [proj_point[0][2]], 'bo') # Pole 1 of projected (orthogonal line)
            ax.plot3D([poles[0][0][0]], [poles[0][0][1]], [poles[0][0][2]], 'bo') # Pole 1 of line at t0
            ax.plot3D([poles[1][0][0]], [poles[1][0][1]], [poles[1][0][2]], 'bo') # Pole 1 of line at t1
        return decompose, proj_point
        
        
    def spindle_ratio(self, listArray, norm_t=0):
        """
        Calculate spindle ratio at any timepoint t with respect to the first timepoint
        Input:  List of values (array of spindle length)
                list[tp]
        Output: ratio
        """
        if len(listArray) == 1:
            ratio = 1
        else:
            # Identify the current time point which is the last position of the list
            idx_t1 = len(listArray)-1
            # Get reference timepoint
            idx_t0 = norm_t
            t0 = listArray[idx_t0] # reference t
            t1 = listArray[idx_t1] # t
            ratio = t0 / t1
        return ratio


    def length_ratio(self, l1, l0, tp, norm_t=0):
        """
        Calculate temporal changes of spindle length (normalized by spindle lengh at t0).
        Note: 3D angles are not vectors and simply substracting values at t1-t0 won't work.
        
        Input:  List of spindle length
                list[tp]
        Output: Ratio
        """
        if tp == 0:
            ratio = 1
        else:
            ratio = l1/l0[norm_t]
        return ratio
        

    def diff_time3d(self, listArray):
        """
        Calculate temporal changes
        Input:  List of values (array of any measurements)
                list[tp]
        Output: difference value
        """
        from math import pi
        if len(listArray) == 1:
            dif_val = 0
        else:
            # Identify the current time point which is the last position of the list
            idx_t1 = len(listArray)-1
            # Get index from the previous time point (t-1)
            idx_t0 = idx_t1-1

            t0 = listArray[idx_t0] # t-1
            t1 = listArray[idx_t1] # t
            if t0 < 0:
                t0 = t0 + 2*pi
            if t1 < 0:
                t1 = t1 + 2*pi
            dif_val = abs(t1 - t0)
            print('t1: ' + str(t1))
            print('t0: ' + str(t0))
            print('dif: ' + str(dif_val))
        return dif_val

    def diff_angle3d(self, r1, r0, tp):
        """
        Calculate temporal changes angle changes.
        Note: 3D angles are not vectors and simply substracting values at t1-t0 won't work.
        
        Input:  List of rotation matrices (at t1 and t0)
                list[tp]
        Output: Relative rotation
        """
        from scipy.spatial.transform import Rotation as R
        if tp == 0:
            # xyz-convention
            phi = 0 # x
            theta = 0 # y
            psi = 0# z
            angles = [phi, theta, psi]
        else:
            #https://math.stackexchange.com/questions/2113634/comparing-two-rotation-matrices
            # Calculate the relative rotation matrix
            r_rel = np.matmul(np.transpose(r0), r1)
            
        r = R.from_dcm(r_rel)
        angles = r.as_euler('xyz', degrees=False)
        return angles

    def midpoint(self, p1, p2):
        """
        Calculate midpoint
        """
        mid_point = (p1+p2)/2
        return mid_point

    def rel_angle3d(self, vec3d_t0, vec3d_t1):
        """
        Calculate temporal changes angle changes.
        Note: 3D angles are not vectors and simply substracting values at t1-t0 won't work.
        
        Input:  List of rotation matrices (at t1 and t0)
                list[tp]
        Output: Relative rotation
        """
        t0_h = vec3d_t0[0]
        t0_w = vec3d_t0[1]
        t0_l = vec3d_t0[2]

        t1_h = vec3d_t1[0]
        t1_w = vec3d_t1[1]
        t1_l = vec3d_t1[2]


        # Find mid-point between 2 3D-points
        t0_cent = self.midpoint(t0_h[0], t0_h[1])
        t1_cent = self.midpoint(t1_h[0], t1_h[1])


        # Translate object vector from t0 to t1
        trans_cent = np.array(t0_cent) + ( np.array(t1_cent) - np.array(t0_cent) )  # Origin
        trans_h = np.array(t0_h[0]) + ( np.array(t1_cent) - np.array(t0_cent) ) 
        trans_w = np.array(t0_w[0]) + ( np.array(t1_cent) - np.array(t0_cent) ) 
        trans_l = np.array(t0_l[0]) + ( np.array(t1_cent) - np.array(t0_cent) ) 

        # Amount of translation (origin)
        dist3d = np.linalg.norm(trans_cent - t0_cent)

        # Normalise vector (translated vector from t0 to t1)
        vec_t0_h = ( trans_h - trans_cent)  / np.linalg.norm(trans_h - trans_cent)
        vec_t0_w = ( trans_w - trans_cent)  / np.linalg.norm(trans_w - trans_cent)
        vec_t0_l = ( trans_l - trans_cent)  / np.linalg.norm(trans_l - trans_cent)
        rot_t0 = np.vstack([vec_t0_h, vec_t0_w, vec_t0_l])
        rot_t0
        # Normalise vector (t1)
        vec_t1_h = ( t1_h[0] - t1_cent)  / np.linalg.norm(t1_h[0] - t1_cent)
        vec_t1_w = ( t1_w[0] - t1_cent)  / np.linalg.norm(t1_w[0] - t1_cent)
        vec_t1_l = ( t1_l[0] - t1_cent)  / np.linalg.norm(t1_l[0] - t1_cent)
        rot_t1 = np.vstack([vec_t1_h, vec_t1_w, vec_t1_l])
        rot_t1
        # Relative rotation matrix (https://math.stackexchange.com/questions/2113634/comparing-two-rotation-matrices)
        r_rel = np.matmul(np.transpose(rot_t0), rot_t1)
        r = R.from_dcm(r_rel)
        # To test that the rotation matrix is correct, aplly it to rot_t0 with: rot_t0.dot(r_rel)
        angles = r.as_euler('XYZ', degrees=True) # Obtain Eulers Angle (Capital for Intrinsic rotation)
        rad = r.as_euler('XYZ', degrees=False) # Obtain Eulers Angle (Capital for Intrinsic rotation)
        quat =  r.as_quat()  # [x,y,z,w]


        #fig = plt.figure(figsize=(18, 18))
        #ax = fig.add_subplot(111, projection='3d')
        #ax.set(xlim=(0,512), ylim=(0,512), zlim=(-256, 256))
        ##ax.axis("equal")
        ## Increase fontsize
        #ax.set_xlabel('$\mathbf{x}$')
        #ax.set_ylabel('$\mathbf{y}$')
        #ax.set_zlabel('$\mathbf{z}$')

        ## Change view
        ##ax.view_init(elev=40, azim=20)

        ## t0
        #ax.plot([trans_cent[0], trans_h[0]], [trans_cent[1], trans_h[1]], [trans_cent[2], trans_h[2]], 'r')
        #ax.plot([trans_cent[0], trans_w[0]], [trans_cent[1], trans_w[1]], [trans_cent[2], trans_w[2]], 'g')
        #ax.plot([trans_cent[0], trans_l[0]], [trans_cent[1], trans_l[1]], [trans_cent[2], trans_l[2]], 'b')
        #ax.plot([t0_h[0][0], t0_h[1][0]], [t0_h[0][1], t0_h[1][1]], [t0_h[0][2], t0_h[1][2]], 'm')
        #ax.plot([t0_w[0][0], t0_w[1][0]], [t0_w[0][1], t0_w[1][1]], [t0_w[0][2], t0_w[1][2]], 'm')
        #ax.plot([t0_l[0][0], t0_l[1][0]], [t0_l[0][1], t0_l[1][1]], [t0_l[0][2], t0_l[1][2]], 'm')
        ## t1
        #ax.plot([t1_cent[0], t1_h[0][0]], [t1_cent[1], t1_h[0][1]], [t1_cent[2], t1_h[0][2]], 'r', alpha = 0.5)
        #ax.plot([t1_cent[0], t1_w[0][0]], [t1_cent[1], t1_w[0][1]], [t1_cent[2], t1_w[0][2]], 'g', alpha = 0.5)
        #ax.plot([t1_cent[0], t1_l[0][0]], [t1_cent[1], t1_l[0][1]], [t1_cent[2], t1_l[0][2]], 'b', alpha = 0.5)
        return angles, rad, quat, dist3d

    def close_cortex(self, pole_cortex_dist, thres, num_poles=2):
        """
        Find local minima to determine time points where the spindle is close to the cortex
        Input: List with pole-cortex distances for both poles
        Output: Indices for each cell
        """
        from scipy.signal import argrelextrema
        # Convert as numpy array
        pole_cortex_dist = np.asarray(pole_cortex_dist)
        merged_close_cortex =[]
        for i in range(num_poles):
            # Find local minima
            local_min = argrelextrema(pole_cortex_dist[:,i], np.less)
            local_min = local_min[0].tolist()
            # Sometimes the spindle is already close to the cortex at t0
            local_min.insert(0,0)
            close_cortex = []
            # Keep only local minima that have values below the threshold
            for _, idx in enumerate(local_min):
                if pole_cortex_dist[idx,i] <= thres:
                    close_cortex.append(idx)
            merged_close_cortex.append(close_cortex)
        return merged_close_cortex

    def compute_msd(self, poles):
        """
        Caluclate Mean Squared Displacement (MSD): Displacement with respect to t0
        Input: Pole coordinates
        Output: disp[0]/[1] (Displacement for pole 1 and 2)
        total_disp (which is the sum of pole 1 and 2)
        """
        # Identify the current time point which is the last position of the list
        idx_t1 = len(poles)-1
        # Get index from the previous time point (t-1)
        idx_t0 = 0

        # Pole 1
        t0_1 = poles[idx_t0][0] # t-1
        t1_1 = poles[idx_t1][0] # t
        # Pole 2
        t0_2 = poles[idx_t0][1] # t-1
        t1_2 = poles[idx_t1][1] # t

        disp1 = np.sqrt(np.sum((t1_1 - t0_1)**2))
        disp2 = np.sqrt(np.sum((t1_2 - t0_2)**2))

        disp = [disp1, disp2]
        total_disp = disp1 + disp2
        
        
        return disp, total_disp

    def data_filter(self, in_array, n_windows=7):
        """
        Apply single Savitzky-Golay
        Input: 1-D array; multiplicator of consecutive odd window size
        Output: Filtered 1-D array; lowest RMSE (> 0), optimal window size.
        """   
        SX = SpinX()
        import scipy.signal
        from sklearn.metrics import mean_squared_error
        from math import sqrt    
        # Set default parameter of SG
        orders = 2
        # Set '2' to create an array of odd numbers
        steps = 2
        # n_window: Defines the number of possible odd values used to determine the optimal window size
        window_seq = SX.generate_seq(3, n_windows*steps, steps)
        y = in_array
        rmse_all = []
        for i in range(len(window_seq)):
            window_size = window_seq[i]
            y_hat_temp = scipy.signal.savgol_filter(y, window_size, orders, deriv = 0)
            rmse_temp = sqrt(mean_squared_error(y, y_hat_temp))
            rmse_all.append(rmse_temp)
        # Find index for the lowest RMSE value
        # Sometimes rmse gives value ~ 0(3.065e-15). In this case, chose the second smallest rmse.
        rmse, idx = min((val, idx) for (idx, val) in enumerate(rmse_all) if val > 1e-6)
        opt_window = window_seq[idx]
        y_hat = scipy.signal.savgol_filter(y, opt_window, orders, deriv = 0)
        return y_hat, rmse, opt_window

    def data_filter_multi(self, merged_y, n_windows):
        """
        Take average values across all cell trajectories and apply Savitzky-Golay on the final 1-D array.
        Input: List of 1-D arrays (merged_y[0/1/2])
        Output: Averaged 1-D array; filtered 1-D array; lowest RMSE (> 0); Optimal window size
        """    
        # Input must be a list of arrays
        # Number of trajectories
        n = len(merged_y)
        # Number of values (inspect first array)
        t = len(merged_y[0])
        
        # Preallocate initial array
        x_bar_temp = np.array([0]*t)

        for i in range(n):
            # Sum up all arrays
            x_bar_temp = x_bar_temp + merged_y[i]

        # Calculate average for each timepoint t
        x_bar = x_bar_temp/n
        
        # Apply SG on averaged data time series with optimal parameter
        y_hat, rmse, opt_window_size = self.data_filter(x_bar, n_windows)
        return x_bar, y_hat, rmse, opt_window_size

    def pole_tracking3d_v3(self, current_poles, previous_poles, spindle_axis):
        """
        Track spindle poles and correct them by calculating pair-wise distances
        Input:  Current spindle pole array[x, y, z] coordinates (t); Current Pole ID (1 or 2)
                List of previous pole array[x, y, z] (t-1).
        Output: Corrected pole ID; 1/0 to count correction
        """
        from scipy.spatial.distance import cdist

        t_current = current_poles
        t_prev = previous_poles
        
        D = cdist(t_current, t_prev)
        
        
        min_v = D.min()
        min_ij = np.where(D==min_v)
        min_ij = np.array([i.item() for i in min_ij])

        # Reconstruct the distance matrix
        # (0,0) = current[0] - previous[0]
        # (0,1) = current[0] - previous[1]
        # (1,0) = current[1] - previous[0]
        # (1,1) = current[1] - previous[1]

        if ( (min_ij[0] == 0) and (min_ij[1] == 0) ): # (0,0)
            pole_1 = t_current[0]
            pole_2 = t_current[1]
            corrected = 0
        elif ( (min_ij[0] == 0) and (min_ij[1] == 1) ): # (0,1)
            pole_1 = t_current[1]
            pole_2 = t_current[0]
            corrected = 1
        elif ( (min_ij[0] == 1) and (min_ij[1] == 0) ): # (1,0)
            pole_1 = t_current[1]
            pole_2 = t_current[0]
            corrected = 1
        elif ( (min_ij[0] == 1) and (min_ij[1] == 1) ): #(1,1)
            pole_1 = t_current[0]
            pole_2 = t_current[1]
            corrected = 0

        return pole_1, pole_2, corrected

    def fuse_img(self, img_spindle, img_membrane, img_memb_contours, cell_id, timepoint, time_interval, pixel_per_micron, exp_name, output_dir):
        """
        Fusing original raw images from membrane and spindle
        Input:  image stack of spindle and membrane (X,Y,Z)
        Output: Fused (X,Y) image
        """      
        import numpy as np
        from PIL import Image, ImageFont, ImageDraw
        from skimage.filters import threshold_yen
        from skimage.exposure import rescale_intensity
        from skimage import color, data, restoration
        from scipy.signal import convolve2d
        from skimage import exposure
        # Define font and fontsize
        arial = ImageFont.truetype("fonts/arial.ttf", size=40)
        location = (10, 10)
        text_color = (255, 255, 255)
        left  = (10, 80) 
        right = (10+pixel_per_micron*5, 80)
        # Generate max projection images
        # Input img is a N-dimensional stack image at one time point.
        # Use axis = 2 to identify the third dimension.
        img0 = np.max(img_spindle, axis=2)
        img1 = np.max(img_membrane, axis=2)
        

        # Apply Wiener filter 
        #psf = np.ones((5, 5)) / 25
        #img0 = convolve2d(img0, psf, 'same')
        #img0 += 0.1 * img0.std() * np.random.standard_normal(img0.shape)
        #img0 = restoration.wiener(img0, psf, 1100)
        
        img0 = exposure.equalize_adapthist(img0, kernel_size=None, clip_limit=0.001, nbins=256)
        # Normalize back from 0-1 to 0-255
        img0 = img0*255
        img0 = img0.astype(np.uint8)

        img1 = exposure.equalize_adapthist(img1, kernel_size=None, clip_limit=0.001, nbins=256)
        # Normalize back from 0-1 to 0-255
        img1 = img1*255
        img1 = img1.astype(np.uint8)
        
        # Apply Wiener filter 
        #psf = np.ones((5, 5)) / 25
        #img0 = convolve2d(img0, psf, 'same')
        #img0 += 0.1 * img0.std() * np.random.standard_normal(img0.shape)
        #img0 = restoration.wiener(img0, psf, 1100)

        # Convert both images to RGB
        img0_blank = img0*0
        img0 = np.dstack((img0,img0_blank, img0)) # Keep only red channel (RxGxB)
        img1 = np.dstack((img1, img1, img1))

     
        alpha = 0.6
        fused = alpha * img0 + (1 - alpha) * img1
        fused = fused.astype(np.uint8)
        
        d_img = Image.fromarray(fused)
        d = ImageDraw.Draw(d_img)
        # Write text on image
        d.text(location, 'Cell: ' + str(cell_id) + ' T: ' + str(timepoint*time_interval) + ' mins', font=arial, fill=text_color)
        d.line([left, left, right, right], fill=text_color, width=8)
        # Convert back
        fused_text = np.array(d_img)
        
        # Do the same but for spindle image only
        d2_img = Image.fromarray(img0)
        d2 = ImageDraw.Draw(d2_img)
        # Write text on image
        d2.text(location, 'Cell: ' + str(cell_id) + ' T: ' + str(timepoint*time_interval) + ' mins', font=arial, fill=text_color)
        d2.line([left, left, right, right], fill=text_color, width=8)
        # Convert back
        fused2_text = np.array(d2_img)    
        
        dpi = 80
        height, width, _ = fused2_text.shape
        # What size does the figure need to be in inches to fit the image?
        figsize = width / float(dpi), height / float(dpi)
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.imshow(fused2_text)
        ax.plot(img_memb_contours[:,1], img_memb_contours[:,0], color='navy', linewidth=4)
        ax.set(xlim=[-0.5, width - 0.5], ylim=[height - 0.5, -0.5], aspect=1)
        #plt.savefig('figs/' + exp_name + '/video_frames/overlay/overlay_c_' + str(cell_id) + '_tp_' + str(timepoint) + '.png', dpi=dpi)
        full_name_export = os.path.join(output_dir, exp_name, 'video_frames', 'overlay', 'overlay_c_' + str(cell_id) + '_tp_' + str(timepoint) + '.png' )
        plt.savefig(full_name_export, dpi=dpi)
        plt.close()
        #plt.figure(figsize=(16,16))
        #plt.imshow(fused)
        return fused_text
    
  
  
    # Python3 code for generating points on a 3-D line  
    # using Bresenham's Algorithm  
    def bresenham_3d(self, x1, y1, z1, x2, y2, z2):
        """
        Input: Array of (x,y,z) coordinates of point 1 and 2.
        Output: List of coordinates of points between point 1 and 2.
        """        
        ListOfPoints = [] 
        ListOfPoints.append( [x1, y1, z1] ) 
        dx = abs(x2 - x1) 
        dy = abs(y2 - y1) 
        dz = abs(z2 - z1) 
        if (x2 > x1): 
            xs = 1
        else: 
            xs = -1
        if (y2 > y1): 
            ys = 1
        else: 
            ys = -1
        if (z2 > z1): 
            zs = 1
        else: 
            zs = -1

        # Driving axis is X-axis" 
        if (dx >= dy and dx >= dz):         
            p1 = 2 * dy - dx 
            p2 = 2 * dz - dx 
            while (x1 != x2): 
                x1 += xs 
                if (p1 >= 0): 
                    y1 += ys 
                    p1 -= 2 * dx 
                if (p2 >= 0): 
                    z1 += zs 
                    p2 -= 2 * dx 
                p1 += 2 * dy 
                p2 += 2 * dz 
                ListOfPoints.append( [x1, y1, z1] ) 

        # Driving axis is Y-axis" 
        elif (dy >= dx and dy >= dz):        
            p1 = 2 * dx - dy 
            p2 = 2 * dz - dy 
            while (y1 != y2): 
                y1 += ys 
                if (p1 >= 0): 
                    x1 += xs 
                    p1 -= 2 * dy 
                if (p2 >= 0): 
                    z1 += zs 
                    p2 -= 2 * dy 
                p1 += 2 * dx 
                p2 += 2 * dz 
                ListOfPoints.append( [x1, y1, z1] ) 

        # Driving axis is Z-axis" 
        else:         
            p1 = 2 * dy - dz 
            p2 = 2 * dx - dz 
            while (z1 != z2): 
                z1 += zs 
                if (p1 >= 0): 
                    y1 += ys 
                    p1 -= 2 * dz 
                if (p2 >= 0): 
                    x1 += xs 
                    p2 -= 2 * dz 
                p1 += 2 * dy 
                p2 += 2 * dx 
                ListOfPoints.append( [x1, y1, z1] )
        # Convert to array       
        ListOfPoints = np.asarray(ListOfPoints)
        return ListOfPoints 
    
    
    def plot_pole_cortex(self, data_raw_1, data_raw_2, time_raw_mins_1, time_raw_mins_2, cell_id, time_point, value_1, value_2, max_tp, time_int, exp_name, output_dir):
        # Rename
        ii = time_point
        c_id = cell_id
        x_axis_end = max_tp*time_int
        y_axis_end = 12.0
        line_thick = 3
        # Enlarge measurement
        data_raw_1 = np.append(data_raw_1, value_1)
        data_raw_2 = np.append(data_raw_2, value_2)
        # Enlarge time points
        time_raw_mins_1 = np.append(time_raw_mins_1, ii)
        time_raw_mins_2 = np.append(time_raw_mins_2, ii)    


        fig = plt.figure(figsize=(12,5))
        ax1=fig.add_subplot(111, label="1")
        ax2=fig.add_subplot(111, label="2", frame_on=False)

        # Define ticks for plot 1
        ax1.set_xlim([0, x_axis_end])
        ax1.set_ylim([0, y_axis_end])

        ax1.plot(time_raw_mins_1*time_int, data_raw_1, '-', color = 'darkorange', linestyle='--', label = 'Pole 1', linewidth=line_thick)
        ax1.set_xlabel('Time (mins)', fontsize=18, labelpad = 5)
        ax1.set_ylabel('Pole-Cortex 3D distance ($\mu$m)', fontsize=18)
        ax1.legend(fontsize=18, loc="upper left", frameon=False)
        ax1.xaxis.set_tick_params(width=5)
        ax1.yaxis.set_tick_params(width=5)
        # Change the fontsize of minor ticks label 
        ax1.tick_params(axis='both', which='major', labelsize=20)
        ax1.tick_params(axis='both', which='minor', labelsize=8)

        # Define ticks for plot 2
        ax2.set_xlim([0, x_axis_end])
        ax2.set_ylim([0, y_axis_end])

        ax2.plot(time_raw_mins_2*time_int, data_raw_2, '-', color = 'black', linestyle='--', label = 'Pole 2', linewidth=line_thick)
        ax2.set_xlabel('Time (mins)', fontsize=18, labelpad = 5)
        ax2.set_ylabel('Pole-Cortex 3D distance ($\mu$m)', fontsize=18)
        ax2.legend(fontsize=18, loc="upper right", frameon=False)
        ax2.xaxis.set_tick_params(width=5)
        ax2.yaxis.set_tick_params(width=5)
        # Change the fontsize of minor ticks label 
        ax2.tick_params(axis='both', which='major', labelsize=20)
        ax2.tick_params(axis='both', which='minor', labelsize=8)

        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(5.0)

        # Remove double 0 at plot origin
        # ax1.yaxis.get_major_ticks()[0].label1.set_visible(False) for newer matplotlib
        xticks = ax1.xaxis.get_major_ticks() 
        xticks[0].label1.set_visible(False)

        # Save plot
        name_export = 'pole_cortex_c_' + str(c_id) + '_tp_' + str(time_point)
        #full_name_export = 'figs/' + exp_name + '/plots/pole_cortex/' + name_export + '.pdf'
        full_name_export = os.path.join(output_dir, exp_name, 'plots' , 'pole_cortex', name_export + '.pdf')
        plt.savefig(full_name_export, dpi=300, transparent=True)
        plt.close('all')
        return data_raw_1, data_raw_2, time_raw_mins_1, time_raw_mins_2

    def rad2deg(self, rad):
        from math import pi
        rad_d = rad * 180.0 / pi # Convert to degrees
        if rad_d < 0:
            rad_d += 360.0 # Convert negative to positive angles
        return rad_d
    
    
    # Search grid
    def crop_center(self, IM_MAX, pole, win_height = 50, win_width = 50):
        #win_height = 50
        #win_width = 50

        # Pole 1
        #pole_1 = merged_axis5d_s[cell][timep][2][0]
        #pole_2 = merged_axis5d_s[cell][timep][2][-1]

        # Crop area
        left_top_x = int(pole[0]-(win_width/2))
        left_top_y = int(pole[1]-(win_height/2))

        right_top_x = left_top_x + win_width
        right_top_y = left_top_y

        left_bot_x = left_top_x
        left_bot_y = left_top_y + win_width

        right_bot_x = left_bot_x + win_width
        right_bot_y = left_bot_y

        crop = IM_MAX[left_top_x:left_top_x+win_height,left_top_y:left_top_y+win_width]
        return crop, left_top_x, left_top_y


    def find_new_pole(self, pole_1, pole_2, distance_away, mode):
        d = sqrt((pole_1[0] - pole_2[0])**2 + (pole_1[1] - pole_2[1])**2 + (pole_1[2] - pole_2[2])**2)
        # Ratio
        t = distance_away/d
        if mode == 1: 
            xt = ((1 - t) * pole_1[0] + t * pole_2[0])
            yt = ((1 - t) * pole_1[1] + t * pole_2[1])
            zt = ((1 - t) * pole_1[2] + t * pole_2[2])
        elif mode == 2:
            xt = ((1 - t) * pole_2[0] + t * pole_1[0])
            yt = ((1 - t) * pole_2[1] + t * pole_1[1])
            zt = ((1 - t) * pole_2[2] + t * pole_1[2])
        p3 = [xt, yt, zt]
        p3 = np.asarray(p3)
        return p3

    def correct_spindle_pole(self, IM_MAX, pole_1, pole_2):
        from scipy.signal import find_peaks
        from scipy.ndimage import gaussian_filter
        from numpy import sqrt
        list_p3 = []
        # Get total distance between 2 poles
        d_total = sqrt((pole_2[0] - pole_1[0])**2 + (pole_2[1] - pole_1[1])**2 + (pole_2[2] - pole_1[2])**2)

        p = np.vstack([pole_1, pole_2])
        #
        t_array = np.linspace(0,1,int(d_total))
        for _, t in enumerate(t_array):
            x3 = p[0,0] + t*(p[1,0]-p[0,0])
            y3 = p[0,1] + t*(p[1,1]-p[0,1])
            z3 = p[0,2] + t*(p[1,2]-p[0,2])
            p3 = [x3,y3,z3]
            list_p3.append(p3)
        # Convert to int
        list_p3 = np.asarray(list_p3)

        # Apply gaussian filter on max projection image
        IM_MAX_g = gaussian_filter(IM_MAX, sigma=1, mode='nearest')
        # Get intensity values along the spindle axis
        line_profile = IM_MAX_g[list_p3[:,0].astype(int), list_p3[:,1].astype(int)]
        # Create x-axis
        line_profile_x = np.linspace(0, len(line_profile), len(line_profile))
        # Plot
        plt.plot(line_profile_x, line_profile)

        # Find peaks
        peaks = find_peaks(line_profile, width=1)

        # Find local maxima (first and last)
        peak_left = int(peaks[0][0])
        peak_right = int(peaks[0][-1])
        # Get coordinates from the list
        pole_cor_1 = list_p3[peak_left]
        pole_cor_2 = list_p3[peak_right]
        plt.plot([peak_left, peak_right], line_profile[[peak_left, peak_right]], "x", color='r')
        plt.margins(x=0)
        plt.xlabel('Spindle length axis')
        plt.ylabel('Intensity')

        d1_3d = sqrt((pole_cor_1[0] - pole_1[0])**2 + (pole_cor_1[1] - pole_1[1])**2 + (pole_cor_1[2] - pole_1[2])**2)
        #print(d1_3d)
        d2_3d = sqrt((pole_cor_2[0] - pole_2[0])**2 + (pole_cor_2[1] - pole_2[1])**2 + (pole_cor_2[2] - pole_2[2])**2)
        #print(d2_3d)

        p3_a = self.find_new_pole(pole_1, pole_2, d1_3d, mode=1)
        p3_b = self.find_new_pole(pole_1, pole_2, d2_3d, mode=2)
        plt.close('all')

        d_new = sqrt((p3_b[0] - p3_a[0])**2 + (p3_b[1] - p3_a[1])**2 + (p3_b[2] - p3_a[2])**2)

        return p3_a, p3_b, d_total, d_new


########################
# SPINX PSF functions #
# #######################

class SpinX_PSF():
    def __init__(self):
        pass
    
    # Read TIF of imaged bead
    def convert_3d(self, image_list, n_slices):
        """
        Input: A list with image path to all images, number of z-slices; Number of frames.
        Output: 3D array with (W x H x D) or (Y x X x Z).
        """
        SX = SpinX()
        temp_array = []
        for i in range(len(image_list)):
            mask, name_mask = SX.load_image_list(image_list, i)
            # Read the image dimensions from first image
            if i == 1:
                # Image info
                img_height = mask.shape[0]
                img_width = mask.shape[1]
                temp_array.append(mask)
            else:
                temp_array.append(mask)       
        # use "F" Fortran for correct order
        array3d = np.dstack(temp_array).reshape(img_height, img_width, n_slices, order='F')
        return array3d

    def find_max_slice(self, array3d):
        # Search through z-plane for max intensity
        max_all = array3d.max()
        # Get index[x, y, z]
        idx = (array3d == max_all).nonzero()
        # Consider only z-slice index
        z_slice_max = idx[2]
        # Select mean if there are multiple results
        z_slice_max = mean(z_slice_max).astype(int)
        # Extract x-y plane with the highest intensity
        slice_xyz = array3d[:, :, z_slice_max]
        slice_xy = slice_xyz.squeeze()
        return slice_xy, z_slice_max

    def select_slice(self, array3d, idx):
        slice_xy = array3d[:, :, idx]
        slice_xy = slice_xy.squeeze()    
        return slice_xy

    def gaussian_mle(self, data):
        from math import sqrt, pi
        mu = data.mean(axis=0)                                                                                                                                                                         
        var = (data-mu).T @ (data-mu) / data.shape[0] #  this is slightly suboptimal, but instructive
        # Get height of gaussian: http://davidmlane.com/hyperstat/A25726.html
        h = 1/(sqrt(2*pi*var) )*e ** (-(data-mu)**2/ (2*var) ).max()
        return mu, var, h

    def psf_load(self, BEAD_DIR):
        SX = SpinX()
        # Load z-stack of bead
        bead_list = SX.get_list_dir(BEAD_DIR)
        bead3d = self.convert_3d(bead_list, len(bead_list))    
        return bead3d

    def psf_fit(self, PSF, thres):
        '''
        Input: 3D array, threshold
        Output: Fitted PSF
        '''
        from sklearn import mixture
        from astropy.modeling import models, fitting
        from astropy import modeling
        from math import sqrt
        import numpy as np
        # rename
        bead3d = PSF
        # Find best z-slice (brightest local maxima)
        best_z, z_idx = self.find_max_slice(bead3d)

        # Find centroid x,y
        yc_raw, xc_raw = np.where(best_z == best_z.max())
        yc_raw = mean(yc_raw).astype(int)
        xc_raw = mean(xc_raw).astype(int)
        #yc_raw = int(yc_raw)
        #xc_raw = int(xc_raw)
        # Crop bounding box around PSF through all z-slices
        bb = 15 # half of the length
        bead3d_crop = bead3d[yc_raw-bb:yc_raw+bb,xc_raw-bb:xc_raw+bb, :]
        plane_xy = self.select_slice(bead3d_crop, z_idx)

        # Pick up pixels that belongs to the PSF
        y, x = np.where((plane_xy[:,:]>thres))
        coords = np.vstack((y,x))

        # Before using a 2D Gaussian fit, use GMM to determine best parameter
        # Transpose
        coords_t = np.transpose(coords)
        gmm = mixture.GaussianMixture(n_components = 1, covariance_type = 'full',  
                              max_iter = 1000, random_state = 5)
        gmm.fit(coords_t)

        # print('converged or not: ', gmm.converged_)

        # Create grid of same size as best plane to put fit data in
        yp, xp = plane_xy.shape
        y_m, x_m, = np.mgrid[:yp, :xp]
        # -- XY
        # Get estimated parameters from GMM
        print('GMM --- cov: ' + str(gmm.covariances_[0]) )
        print(' x_mean: ' + str(gmm.means_[0][0]) + ' y_mean: ' + str(gmm.means_[0][0]))
        # Declare what function you want to fit to your data
        f_init = models.Gaussian2D(amplitude = 2**16, cov_matrix=gmm.covariances_[0], x_mean = gmm.means_[0][0], y_mean = gmm.means_[0][1], theta=None)

        # Declare what fitting function you want to use
        fit_f = fitting.LevMarLSQFitter()

        # Fit the model to best plane
        fit_model_xy = fit_f(f_init, x_m, y_m, plane_xy)


        # Plot the data with the best-fit model
        plt.figure(figsize=(16, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(plane_xy)
        plt.title("Data")
        plt.subplot(1, 3, 2)
        plt.imshow(fit_model_xy(x_m, y_m))
        plt.title("Model")
        plt.subplot(1, 3, 3)
        plt.imshow(plane_xy - fit_model_xy(x_m, y_m))
        plt.title("Residual")
        plt.show()
        print('xy_amplitude: ' + str(fit_model_xy.amplitude[0]))
        print('x_mean: ' + str(fit_model_xy.x_mean[0]) + ' x_stdev: ' + str(fit_model_xy.x_stddev[0]) + ' x_fwhm: ' + str(fit_model_xy.x_fwhm))
        print('y_mean: ' + str(fit_model_xy.y_mean[0]) + ' y_stdev: ' + str(fit_model_xy.y_stddev[0]) + ' y_fwhm: ' + str(fit_model_xy.y_fwhm))


        from mpl_toolkits.axes_grid1 import make_axes_locatable
        # Get max pixel idices
        yc, xc = np.where(plane_xy==plane_xy.max())
        yc = mean(yc).astype(int)
        xc = mean(xc).astype(int)
        #yc = int(yc)
        #xc = int(xc)

        # Get line profile vector for xy
        x_cross = plane_xy[yc,:-1]
        x_cross_fit = fit_model_xy(x_m, y_m)[yc,:-1]
        # Get line profile vector for xy
        y_cross = plane_xy[:-1,xc]
        y_cross_fit = fit_model_xy(x_m, y_m)[:-1,xc]

        # Define axis for both plots
        x_ax = np.linspace(0, len(x_cross), len(x_cross))
        y_ax = np.linspace(0, len(y_cross), len(y_cross))

        fig, axScatter = plt.subplots(figsize=(11, 11))

        # the scatter plot:
        axScatter.tick_params(axis = 'both', which = 'major', labelsize = 14)

        axScatter.imshow(plane_xy)
        axScatter.set_aspect(1)
        #axScatter.axis('off')

        # create new axes on the right and on the top of the current axes
        # The first argument of the new_vertical(new_horizontal) method is
        # the height (width) of the axes to be created in inches.
        divider = make_axes_locatable(axScatter)
        axHistx = divider.append_axes("top", 1.4, pad=0.3, sharex=axScatter)
        axHisty = divider.append_axes("right", 1.4, pad=0.3, sharey=axScatter)

        # make some labels invisible
        axHistx.xaxis.set_tick_params(labelbottom=False)
        axHisty.yaxis.set_tick_params(labelleft=False)


        axHistx.margins(0, 0)
        axHistx.scatter(x_ax, x_cross, color='k')
        axHistx.plot(x_ax, x_cross_fit, color='r')

        axHisty.margins(0, 0)
        axHisty.scatter(y_cross, y_ax, color='k')
        axHisty.plot(y_cross_fit, y_ax, color='r')

        # the xaxis of axHistx and yaxis of axHisty are shared with axScatter,
        # thus there is no need to manually adjust the xlim and ylim of these
        # axis.

        axHistx.set_ylabel('Intensity', fontsize=10)
        axHistx.tick_params(axis = 'both', which = 'major', labelsize = 14)
        axHistx.set_yticks([0, max(x_cross)/2, max(x_cross)])

        axHisty.set_xlabel('Intensity', fontsize=10)
        axHisty.tick_params(axis = 'both', which = 'major', labelsize = 14)
        axHisty.set_xticks([0, max(y_cross)/2, max(y_cross)])

        #plt.savefig('psf_fitting_xy.pdf', dpi=300, transparent=True, bbox_inches = 'tight', pad_inches = 0.1)  

        # -- Z

        # May need this in some cases: Estimate mean and standard deviation
        #meam = sum(x * y)
        #sigma = sum(y * (x - m)**2)
        # Alternative fit: https://stackoverflow.com/questions/14459340/gaussian-fit-with-scipy-optimize-curve-fit-in-python-with-wrong-results

        # Obtain intensity values from the centroid through all z-stacks
        yc, xc = np.where(plane_xy==plane_xy.max())
        yc = mean(yc).astype(int)
        xc = mean(xc).astype(int)
        z_array = bead3d_crop[yc, xc, :]

        fitter = modeling.fitting.LevMarLSQFitter()
        model = modeling.models.Gaussian1D()   # depending on the data you need to give some initial values
        z_vec = x = np.linspace(0, len(z_array), len(z_array))
        fit_model_z = fitter(model, z_vec, z_array)

        fig4 = plt.figure(figsize=(13,5))
        ax4 = fig4.add_subplot(111)
        ax4.scatter(z_vec, z_array, color='k')
        ax4.plot(z_vec, fit_model_z(x), color='r')
        plt.ylabel('Intensity', fontsize=14, labelpad=12)
        plt.xlabel('Z-slices', fontsize=14, labelpad=12)
        plt.tick_params(axis = 'both', which = 'major', labelsize = 18)

        #plt.savefig('psf_fitting_z.pdf', dpi=300, transparent=True, bbox_inches = 'tight', pad_inches = 0.1)

        print('z_amplitude: ' + str(fit_model_z.amplitude[0]))
        print('z_mean: ' + str(fit_model_z.mean[0]) + ' z_stdev: ' + str(fit_model_z.stddev[0]) + ' z_fwhm: ' + str(fit_model_z.fwhm))


        # With normalized bead
        z_array_norm = (z_array-min(z_array))/(max(z_array)-min(z_array))
        z_vec_norm = np.linspace(0, len(z_array_norm), len(z_array_norm))
        model_norm = modeling.models.Gaussian1D(amplitude=fit_model_z.amplitude[0], mean=fit_model_z.mean[0], stddev=fit_model_z.stddev[0])
        fit_model_z_norm = fitter(model_norm, z_vec_norm, z_array_norm)
        print('z_amplitude: ' + str(fit_model_z_norm.amplitude[0]))
        print('z_mean: ' + str(fit_model_z_norm.mean[0]) + ' z_stdev: ' + str(fit_model_z_norm.stddev[0]) + ' z_fwhm: ' + str(fit_model_z_norm.fwhm))

        fig5 = plt.figure(figsize=(13,5))
        ax5 = fig5.add_subplot(111)
        ax5.scatter(z_vec_norm, z_array_norm, color='k')
        ax5.plot(z_vec_norm, fit_model_z_norm(x), color='r')
        plt.ylabel('Normalized Intensity', fontsize=14, labelpad=12)
        plt.xlabel('Z-slices', fontsize=14, labelpad=12)
        plt.tick_params(axis = 'both', which = 'major', labelsize = 18)

        return fit_model_xy, fit_model_z, fit_model_z_norm


    # Normalise image input and reference image is important because bead images are usually taken in 16 bit (2^16)

    def norm_img(self, img):
        '''
        Normalise image by min and max intensity
        '''
        norm = (img - np.min(img)) / (np.max(img) - np.min(img))
        return norm
    
    def generate_psf(self, wl=0.605, NA=1.42, M=100, ns=1.34, ng0=1.522, ni0=1.33, ti0=150, tg0=170, r_lat=0.1, r_ax=0.2, pZ=2):
        # Generate PSF function
        # http://kmdouglass.github.io/posts/implementing-a-fast-gibson-lanni-psf-solver-in-python/
        import scipy.special
        from scipy.interpolate import interp1d
        from matplotlib import animation
        #========================================================================================================#
        # Image properties
        # Size of the PSF array, pixels
        size_x = 256
        size_y = 256
        size_z = 128

        # Precision control
        num_basis    = 100  # Number of rescaled Bessels that approximate the phase function
        num_samples  = 1000 # Number of pupil samples along radial direction
        oversampling = 2    # Defines the upsampling ratio on the image space grid for computations
        # Excitation/Emission
        # DAPI:       350/435nm
        # FITC:       490/525nm
        # TRITC:      555/605nm
        # Cy5:        645/705nm
        # Live Filter
        # mCherry:    572/632nm
        # DeltaVision Core
        # http://www.sussex.ac.uk/gdsc/intranet/pdfs/DeltaVision%20Core%20and%20personal%20DV%20Users%20Manual_D.pdf
        # DeltaVision Elite
        # http://www.gelifesciences.co.kr/wp-content/uploads/2016/11/DeltaVision_Elite_Filter_Set_Charts1.pdf
        # Microscope parameters
        NA          = NA    # Numerical aperture
        wavelength  = wl    # microns (Emitted)
        M           = M     # magnification
        ns          = ns    # specimen refractive index (RI)
        ng0         = ng0   # coverslip RI design value (use oil calculator app)
        ng          = ng0   # coverslip RI experimental value (use oil calculator app)
        ni0         = ni0   # immersion medium RI design value
        ni          = ni0   # immersion medium RI experimental value
        ti0         = ti0   # microns, working distance (immersion medium thickness) design value (http://facilities.igc.gulbenkian.pt/microscopy/microscopy-dv.php) / https://cdn.cytivalifesciences.com/dmm3bwsv3/AssetStream.aspx?mediaformatid=10061&destinationid=10016&assetid=27720
        tg0         = tg0   # microns, coverslip thickness design value
        tg          = tg0   # microns, coverslip thickness experimental value
        res_lateral = r_lat # microns (Chapter 6: https://cdn.southampton.ac.uk/assets/imported/transforms/content-block/UsefulDownloads_Download/F21A6D82AB864B598A07D487C756A92E/Delta%20Vision%20Elite%20User%20Manual.pdf)
        res_axial   = r_ax  # microns
        pZ          = pZ    # microns, particle distance from coverslip

        # Scaling factors for the Fourier-Bessel series expansion
        min_wavelength = wl # microns
        scaling_factor = NA * (3 * np.arange(1, num_basis + 1) - 2) * min_wavelength / wavelength

        #========================================================================================================#
        # Create the coordinate systems
        # Place the origin at the center of the final PSF array
        x0 = (size_x - 1) / 2
        y0 = (size_y - 1) / 2

        # Find the maximum possible radius coordinate of the PSF array by finding the distance
        # from the center of the array to a corner
        max_radius = round(sqrt((size_x - x0) * (size_x - x0) + (size_y - y0) * (size_y - y0))) + 1;

        # Radial coordinates, image space
        r = res_lateral * np.arange(0, oversampling * max_radius) / oversampling

        # Radial coordinates, pupil space
        a = min([NA, ns, ni, ni0, ng, ng0]) / NA
        rho = np.linspace(0, a, num_samples)

        # Stage displacements away from best focus
        z = res_axial * np.arange(-size_z / 2, size_z /2) + res_axial / 2

        # Define the wavefront aberration
        OPDs = pZ * np.sqrt(ns * ns - NA * NA * rho * rho) # OPD in the sample
        OPDi = (z.reshape(-1,1) + ti0) * np.sqrt(ni * ni - NA * NA * rho * rho) - ti0 * np.sqrt(ni0 * ni0 - NA * NA * rho * rho) # OPD in the immersion medium
        OPDg = tg * np.sqrt(ng * ng - NA * NA * rho * rho) - tg0 * np.sqrt(ng0 * ng0 - NA * NA * rho * rho) # OPD in the coverslip
        W    = 2 * np.pi / wavelength * (OPDs + OPDi + OPDg)

        # Sample the phase
        # Shape is (number of z samples by number of rho samples)
        phase = np.cos(W) + 1j * np.sin(W)

        # Define the basis of Bessel functions
        # Shape is (number of basis functions by number of rho samples)
        J = scipy.special.jv(0, scaling_factor.reshape(-1, 1) * rho)

        # Compute the approximation to the sampled pupil phase by finding the least squares
        # solution to the complex coefficients of the Fourier-Bessel expansion.
        # Shape of C is (number of basis functions by number of z samples).
        # Note the matrix transposes to get the dimensions correct.
        C, residuals, _, _ = np.linalg.lstsq(J.T, phase.T, rcond=None)

        # Compute the PSF

        b = 2 * np. pi * r.reshape(-1, 1) * NA / wavelength

        # Convenience functions for J0 and J1 Bessel functions
        J0 = lambda x: scipy.special.jv(0, x)
        J1 = lambda x: scipy.special.jv(1, x)

        # See equation 5 in Li, Xue, and Blu
        denom = scaling_factor * scaling_factor - b * b
        R = (scaling_factor * J1(scaling_factor * a) * J0(b * a) * a - b * J0(scaling_factor * a) * J1(b * a) * a)
        R /= denom

        # The transpose places the axial direction along the first dimension of the array, i.e. rows
        # This is only for convenience.
        PSF_rz = (np.abs(R.dot(C))**2).T

        # Normalize to the maximum value
        PSF_rz /= np.max(PSF_rz)

        # Create the fleshed-out xy grid of radial distances from the center
        xy      = np.mgrid[0:size_y, 0:size_x]
        r_pixel = np.sqrt((xy[1] - x0) * (xy[1] - x0) + (xy[0] - y0) * (xy[0] - y0)) * res_lateral

        PSF = np.zeros((size_y, size_x, size_z))

        for z_index in range(PSF.shape[2]):
            # Interpolate the radial PSF function
            PSF_interp = interp1d(r, PSF_rz[z_index, :])

            # Evaluate the PSF at each value of r_pixel
            PSF[:,:, z_index] = PSF_interp(r_pixel.ravel()).reshape(size_y, size_x)
        # Normalize PSF
        PSF_norm = self.norm_img(PSF)
        # Scale PSF to 16bit
        PSF_16bit = PSF_norm*(2**16)
        return PSF_norm, PSF_16bit, res_axial
