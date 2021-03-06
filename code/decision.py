import numpy as np


# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):
        

    #to avoid getting stuck
    if Rover.vel<=0.2 and Rover.total_time>5:
        if Rover.time_stopped==0:
            Rover.time_stopped= Rover.total_time
            stuck=0
        # Implement conditionals to decide what to do given perception data
        # Here you're all set up with some basic functionality but you'll need to
    print Rover.samples_pos[0]
    if not Rover.samples_pos[0].any():
        # Check if we have vision data to make decisions with
        if Rover.nav_angles is not None:
            # Check for Rover.mode status
            # check if stuck 
            if Rover.vel <= 0.2 and (Rover.total_time-Rover.time_stopped)>5:
                Rover.throttle = 0
                # Set brake to stored brake value
                Rover.brake = Rover.brake_set
                Rover.steer = 0
                Rover.mode = 'stop'
                Rover.time_stopped=Rover.total_time
                Rover.stuck=1
                Rover.stuck_yaw=Rover.yaw
            
                
                
            if Rover.mode == 'forward': 
                
                # Check the extent of navigable terrain
                if len(Rover.nav_angles) >= Rover.stop_forward:  
                    # If mode is forward, navigable terrain looks good 
                        # and velocity is below max, then throttle 
                        if Rover.vel < Rover.max_vel:
                            # Set throttle value to throttle setting
                            Rover.throttle = Rover.throttle_set
                        else: # Else coast
                            Rover.throttle = 0
                        Rover.brake = 0
                        # Set steering to average angle clipped to the range +/- 15
                 
                        Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi)+10, -30, 30)
                    # If there's a lack of navigable terrain pixels then go to 'stop' mode
                elif len(Rover.nav_angles) < Rover.stop_forward:
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'

                # If we're already in "stop" mode then make different decisions
            elif Rover.mode == 'stop':
                # If we're in stop mode but still moving keep braking
                if Rover.vel > 0.2:
                    Rover.throttle = 0
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    # If we're not moving (vel < 0.2) then do something else
                elif Rover.vel <= 0.2:
                    # Now we're stopped and we have vision data to see if there's a path forward
                    if len(Rover.nav_angles) < Rover.go_forward or Rover.stuck==1 :
                        Rover.throttle = 0
                        # Release the brake to allow turning
                        Rover.brake = 0
                        # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                        Rover.steer = -15 # Could be more clever here about which way to turn
                        if np.absolute(Rover.stuck_yaw-Rover.yaw)>10:
                            Rover.stuck=0
                        # If we're stopped but see sufficient navigable terrain in front then go!
                    if len(Rover.nav_angles) >= Rover.go_forward and Rover.stuck<0.5:
                        # Set throttle back to stored value
                        Rover.throttle = Rover.throttle_set
                        # Release the brake
                        Rover.brake = 0
                        # Set steer to mean angle
                        Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi)+10, -30, 30)
                        Rover.mode = 'forward'
                        Rover.stuck=0
            # Just to make the rover do something 
            # even if no modifications have been made to the code
        else:
            Rover.throttle = Rover.throttle_set
            Rover.steer = 0
            Rover.brake = 0
    else:   
        
        #picking up the rocks
        Rover.steer=np.clip(np.mean(Rover.samples_pos[1]*180/np.pi),-15,15)
        Rover.mode=='forward'
        Rover.throttle=0

        if Rover.distance<14:               
            Rover.throttle = 0
            # Set brake to stored brake value
            Rover.brake = Rover.brake_set
            Rover.steer = 0
            Rover.mode = 'stop'
            Rover.near_sample=1
            Rover.samples_found=Rover.samples_found+1
            
        # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
        Rover.near_sample=0

    return Rover
