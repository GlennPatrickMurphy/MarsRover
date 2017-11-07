# MarsRover
Udacity Nanodegree Project
# README

   This repository is for the first project of the *Udacity Robotics Nanodegree*. The projects goal was to navigate through a virtual Mars terrain with a rover by processing images recorded from the rover's onboard camera. The project utilizes perspective transforms, and the rotation and translation of images. The manipulation of images were to highlight directions, obstacles, and rocks for the rover to explore. A map of the navigable terrain was developed through these distorted images.The document **Rover_Project_Test_Notebook** goes into great depths of how the aforementioned tasks were accomplished using Python. The rover's image processing can be seen in **perception.py** in the repository.

   Once an image was processed, a decision by the robot had to be made for it to continue. These decisions were based off the information created by the **perception.py** code, and can be seen in **decision.py**. Both **perception.py** & **decision.py** were accessed by the code **drive_rover.py**, which initalizes the RoverState() class and connects to the Unity software. The final project was able to explore over 50% of the map and collect 2 rock samples on an average run.This README will further address how the following repository met the *Udacity Robotic Nanodegree*'s rubric by exploring the code of the **Rover_Project_Test_Notebook**, **perception.py** and **decision.py**.
   
## Rover_Project_Test_Notebook

  This notebook outlines the basis of the functions used in the **perception.py**.
  
 ### Rock Sample
 A Rock sample was identified by performing a perspective transform on the image. The image was threshed at a different RGB level (> 100,> 100,< 10). This highlighted the yellow in the image. The image was then pushed through simlar functions to get a direction to the rock sample.
 
```python
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
```
![Rock2](https://github.com/GlennPatrickMurphy/MarsRover/blob/master/code/photos/example_rock2.png)

![Rock1](https://github.com/GlennPatrickMurphy/MarsRover/blob/master/code/photos/warped_example1.png)

![Rock3](https://github.com/GlennPatrickMurphy/MarsRover/blob/master/code/photos/warped_threshed1.png)

### def process_image(img):

This part of the code called on the important transformation function to finally create information for our rover to move. The image had a perspective transform applied, then had its color thressed to highlight navigable terain. The navigable terrain was then changed into rover coordinates and had polar coordinates solved also (polar coordinates are used to define the rovers direction). Finally the image was then translated and scaled to begin tracing a map.

A sample video of one the rover's manual runs were analysed with this function.

```python
# This function will be used by moviepy to create an output video
def process_image(img):
    # Example of how to use the Databucket() object defined above
    # to print the current x, y and yaw values 
    rover_yaw = data.yaw[data.count]
    rover_xpos = data.xpos[data.count]
    rover_ypos = data.ypos[data.count]
    #print(data.xpos[data.count], data.ypos[data.count], data.yaw[data.count])

    # TODO: 
    # 1) Define source and destination points for perspective transform
    dst_size = 5 
    # Set a bottom offset to account for the fact that the bottom of the image 
    # is not the position of the rover but a bit in front of it
    # this is just a rough guess, feel free to change it!
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  ])
    # 2) Apply perspective transform
    warped = perspect_transform(grid_img, source, destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    threshed = color_thresh(warped)
    # 4) Convert thresholded image pixel values to rover-centric coords
    xpix, ypix = rover_coords(threshed)
    dist, angles = to_polar_coords(xpix, ypix)
    mean_dir = np.mean(angles)
    # 5) Convert rover-centric pixel values to world coords
    scale = 100
    # Get navigable pixel positions in world coords
    x_world, y_world = pix_to_world(xpix, ypix, rover_xpos, 
                                rover_ypos, rover_yaw, 
                                data.worldmap.shape[0], scale)
    # Add pixel positions to worldmap
    # 6) Update worldmap (to be displayed on right side of screen)
    data.worldmap[y_world,x_world, 0] += 255

    # 7) Make a mosaic image, below is some example code
        # First create a blank image (can be whatever shape you like)
    output_image = np.zeros((img.shape[0] + data.worldmap.shape[0], img.shape[1]*2, 3))
        # Next you can populate regions of the image with various output
        # Here I'm putting the original image in the upper left hand corner
    output_image[0:img.shape[0], 0:img.shape[1]] = img

        # Let's create more images to add to the mosaic, first a warped image
    warped = perspect_transform(img, source, destination)
        # Add the warped image in the upper right hand corner
    output_image[0:img.shape[0], img.shape[1]:] = warped

        # Overlay worldmap with ground truth map
    map_add = cv2.addWeighted(data.worldmap, 1, data.ground_truth, 0.1, 0)
        # Flip map overlay so y-axis points upward and add to output_image 
    output_image[img.shape[0]:, 0:data.worldmap.shape[1]] = np.flipud(map_add)


        # Then putting some text over the image
    cv2.putText(output_image,"Populate this image with your analyses to make a video!", (20, 20), 
                cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    data.count += 1 # Keep track of the index in the Databucket()
    
    return output_image
```   
```python  
import io
import base64
from IPython.display import HTML
video = io.open('C:/Users/glenn/OneDrive/Documents/GitHub/RoboND-Rover-Project/output/test_mapping.mp4', 'r+b').read()
encoded_video = base64.b64encode(video)
HTML(data='''<video alt="test" controls>
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded_video.decode('ascii')))
    
    
```
![vid](https://github.com/GlennPatrickMurphy/MarsRover/blob/master/code/test_mapping.mp4)

