import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import tifffile as tiff
import os
import stardist.models as stmod
from csbdeep.utils import normalize



def generate_tissue_images_from_HE_image(large_image, segment_size=(3000,3000), overlap_pixels=50): 
    """
    Splits a large HE image into segments of size segment_size and creates two folders: tissue_images (to store segments containing tissue) 
    and not_tissue_images (to store segments without tissue). The segments overlap by overlap_pixels, necessary for creating single-cell 
    images. If a nucleus is located near the image edge and it's not possible to create a single-cell image with the required dimensions, 
    that nucleus won't be used for creating single-cell images. Due to image overlapping, no nuclei will be omitted during this process.
    Note: The large HE image needs to be divided into smaller segments to enable segmentation.
    Note: The method for determining the presence of tissue on a segment needs improvement!!! (this line of code: is_tissue = (255- segment_image_gray).mean() > 20)
    ------------------------------------------------------------------
    Arguments:
        - large_image - large HE image
        - segment_size - the size of the smaller segments
        - overlap_pixels - number of pixels by which the segments overlap; it is equal to half the dimension of the single-cell image

    Return: - (creates two folders: tissue_images and not_tissue_images)
    ------------------------------------------------------------------
    """ 
    height, width = large_image.shape[:2]
    segment_height = segment_size[0]
    segment_width = segment_size[1]
    
    out_f_tissue = 'tissue_images'
    if (not os.path.exists(out_f_tissue)):
        os.mkdir(out_f_tissue)
        
    out_f_not_tissue = 'not_tissue_images'
    if (not os.path.exists(out_f_not_tissue)):
        os.mkdir(out_f_not_tissue)
    
    for y in range(0, height, segment_height):
        for x in range(0, width, segment_width):
            segment_start_y = int(max(0,y-overlap_pixels))  
            segment_end_y = int(min(y + segment_height+overlap_pixels, height))
            segment_start_x = int(max(0, x-overlap_pixels))  
            segment_end_x = int(min(x + segment_width+overlap_pixels, width))
      
            segment = large_image[segment_start_y:segment_end_y, segment_start_x:segment_end_x] #cropping segments from original HE image
            
            segment_image_gray = cv2.cvtColor(segment, cv2.COLOR_BGR2GRAY)
            is_tissue = (255- segment_image_gray).mean() > 20              #better way to select images with tissue needs to be found!!!
            
            if is_tissue: 
                cv2.imwrite(f"{out_f_tissue}/segment_{y}_{x}.tif", segment)
            else:
                cv2.imwrite(f"{out_f_not_tissue}/segment_{y}_{x}.tif", segment)
                
    return 0
                
                
                
def segmentation(tissue_images_folder_path):
    """
    All segments containing tissue from the tissue_images folder are loaded, and segmentation is performed on them using the stardist model. 
    A new folder is created (masks) to store the segmentation masks (binary images with values 0 and 255).
    Note: This function can be used independently if there is a folder with tissue images.
    Note: The stardist model parameters that provide the best segmentation need to be found (put them as input to function)!!!
    ------------------------------------------------------------------
    Arguments:
        - tissue_images_folder_path - path to folder with tissue images

    Return: - (creates folder: masks)
    ------------------------------------------------------------------
    """ 
    for filename in os.listdir(tissue_images_folder_path):
        image_path = os.path.join(tissue_images_folder_path, filename)
        image = np.array(Image.open(image_path))
        
        if image.dtype != 'uint8':
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        model = stmod.StarDist2D.from_pretrained('2D_versatile_he') #find the best parameters for stardist, needs to be input to segmentation function!!!
        labels, _ = model.predict_instances(normalize(image))  #also the best normalization needs to be found!!!
        label_uint8 = cv2.normalize(labels, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        label_uint8[label_uint8>0]=255
        
        out_f_mask = 'masks'
        if (not os.path.exists(out_f_mask)):
            os.mkdir(out_f_mask)
            
        base_name = os.path.splitext(filename)[0]  #remove extension
        new_filename = base_name + "_mask"
        cv2.imwrite(f"{out_f_mask}/{new_filename}.png", label_uint8)
        
    return 0
        
        
        
def find_centers_of_contours(contours, thresh): 
    """
    The function finds the center of each nucleus. Specifically, it finds the center of each contour. Contours are obtained using the 
    cv2.findContours function on the binary mask image. Certain contours (nucleus) are filtered out because they have a small size and are 
    considered noise. The filtering threshold is determined by the parameter 'thresh'.
    ------------------------------------------------------------------
    Arguments:
        - contours - contours obtained from the segmentation mask using the cv2.findContours function
        - thresh - nuclei below a thresh size are filtered out

    Return: 
       - the centers of each nucleus (contours)
    ------------------------------------------------------------------
    """ 
    centers = []
    for contour in contours: 
        area = cv2.contourArea(contour)

        if area > thresh:
            moments = cv2.moments(contour)
            centar_x = int(moments['m10'] / moments['m00'])
            centar_y = int(moments['m01'] / moments['m00'])
            centers.append((centar_x, centar_y))
            
    return centers


def generate_single_cell_images(path_to_tissue_images_folder, path_to_mask_folder, overlap_pixels = 50, thresh = 40):
    """
    Creates single-cell images and segmentation masks for each of these images and stores them in two separate folders:
    single_cell_images and single_cell_masks.
    ------------------------------------------------------------------
    Arguments:
        - path_to_tissue_images_folder - path to the folder with tissue images (generated in the function 'generate_tissue_images_from_HE_image')
        - path_to_mask_folder -  path to the folder with segmentation masks (generated in the 'segmentation' function)
        - overlap_pixels - number of pixels by which the segments overlap; it is equal to half the dimension of the single-cell image
        - thresh - nuclei below a thresh size are filtered out

    Return: - (creates two folders: single_cell_images and single_cell_masks)
    ------------------------------------------------------------------
    """ 
    dimensions = (overlap_pixels*2, overlap_pixels*2) #dimension of single cell image
    for tissue_image_filename, mask_filenema in zip(np.sort(os.listdir(path_to_tissue_images_folder)), np.sort(os.listdir(path_to_mask_folder))):
        tissue_image_path = os.path.join(path_to_tissue_images_folder, tissue_image_filename)
        tissue_image = np.array(Image.open(tissue_image_path))
        mask_path = os.path.join(path_to_mask_folder, mask_filenema)
        mask = np.array(Image.open(mask_path))
    
        contours, _ = cv2.findContours(mask, 1,2)
        centers = find_centers_of_contours(contours, thresh) #find the best thresh, filter out contours that are too small!!!
    
        out_f_single_cell = 'single_cell_images'
        if (not os.path.exists(out_f_single_cell)):
            os.mkdir(out_f_single_cell)
            
        out_f_single_cell_mask = 'single_cell_masks'
        if (not os.path.exists(out_f_single_cell_mask)):
            os.mkdir(out_f_single_cell_mask)
            
        for i, centar in enumerate(centers):
            centar_x, centar_y = centar
            half_width = dimensions[1] // 2 
            half_height = dimensions[0] // 2 

            upper_corner_x = centar_x - half_width
            upper_corner_y = centar_y - half_height
            down_corner_x = centar_x + half_width
            down_corner_y = centar_y + half_height
        
            if upper_corner_x>=0 and upper_corner_y>=0 and down_corner_x<=tissue_image.shape[1] and down_corner_y<=tissue_image.shape[0]:
                single_cell = tissue_image[upper_corner_y:upper_corner_y+dimensions[0], upper_corner_x:upper_corner_x+dimensions[1]]
                single_cell_mask = mask[upper_corner_y:upper_corner_y+dimensions[0], upper_corner_x:upper_corner_x+dimensions[1]]
                
                base_tissue_image_filename = os.path.splitext(tissue_image_filename)[0]  #remove extension
                cv2.imwrite(f"{out_f_single_cell}/{base_tissue_image_filename}_single_cell_{i+1}.tif", single_cell)
        
                base_mask_filenema = os.path.splitext(mask_filenema)[0]  #remove extension
                cv2.imwrite(f"{out_f_single_cell_mask}/{base_mask_filenema}_single_cell_mask_{i+1}.png", single_cell_mask)
            
    return 0