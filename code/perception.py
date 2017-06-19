import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select
#find the rocks, filters for yellow
def color_thresh1(img, rgb_thresh=(100, 100, 10)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all thre threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] < rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select  

# Define a function to convert to rover-centric coordinates
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = np.absolute(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[0]).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to apply a rotation to pixel positions
def rotate_pix(xpix, ypix, yaw):
    yaw_rad = yaw * np.pi / 180
    x_rotated = xpix * np.cos(yaw_rad) - ypix * np.sin(yaw_rad)
    y_rotated = xpix * np.sin(yaw_rad) + ypix * np.cos(yaw_rad)
    # Convert yaw to radians
    # Apply a rotation
    xpix_rotated = x_rotated
    ypix_rotated = y_rotated 
    return xpix_rotated, ypix_rotated


# Define a function to perform a translation
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    scale = 10
    x_rotated=xpix_rot
    y_rotated=ypix_rot
    # Perform translation and convert to integer since pixel values can't be float
    x_world = np.int_(xpos + (x_rotated / scale))
    y_world = np.int_(ypos + (y_rotated / scale))
    # Apply a scaling and a translation
    xpix_translated = x_world
    ypix_translated = y_world
    return xpix_translated, ypix_translated

# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
   
    rover_yaw = Rover.yaw
    rover_xpos = Rover.pos[0]
    rover_ypos = Rover.pos[1]
    #  Define source and destination points for perspective transform
    dst_size = 5 
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - bottom_offset],
                  [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - bottom_offset],
                  [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset], 
                  [Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset],
                  ])   
    #  Apply perspective transform
    warped = perspect_transform(Rover.img, source, destination)  
    #  Apply color threshold to identify navigable terrain/obstacles/rock samples
    threshed = color_thresh(warped)
    threshed1=color_thresh1(warped)
    sample=np.sum(threshed1)
    # Update Rover.vision_image (this will be displayed on left side of screen)
      
    Rover.vision_image[:,:,2] =threshed # navigable terrain color-thresholded binary image
    Rover.vision_image[:,:,1]=threshed1
    # Convert map image pixel values to rover-centric coords

    xpix, ypix = rover_coords(threshed)
    dist, angles = to_polar_coords(xpix, ypix)
    mean_dir = np.mean(angles)    
        # Convert rover-centric pixel values to world coordinates
    scale = 100
        # Get navigable pixel positions in world coords
    x_world, y_world = pix_to_world(xpix, ypix, rover_xpos, 
                                    rover_ypos, rover_yaw, 
                                    Rover.worldmap.shape[0], scale)    
        #Update Rover worldmap (to be displayed on right side of screen)

    #creating high map fidelity by eliminating map images when the pitch and roll have been compromised
    if (Rover.pitch<1 or Rover.pitch>359) and (Rover.roll<1 or Rover.roll>359):
        Rover.worldmap[y_world,x_world, 2] += 1
    #sending direction info to the rover
    Rover.nav_dists = dist
    Rover.nav_angles = angles
    
    #image processing for rock samples
    xpix, ypix = rover_coords(threshed1)
    dist, angles = to_polar_coords(xpix, ypix)
    mean_dir = np.mean(angles)    
    scale = 100
       # plotting location of rocks
    x_world, y_world = pix_to_world(xpix, ypix, rover_xpos, 
                                   rover_ypos, rover_yaw,  
                                   Rover.worldmap.shape[0], scale)   
    
    #Send rock position information

    Rover.samples_pos=[dist,angles]
    #Returns distance of rock to rover
    if dist.any():
        Rover.distance=dist[0]    
    
    
    
    
    return Rover