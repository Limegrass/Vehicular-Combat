'''
Path Finding using Q-Learning
Author: JN, AAL
'''
import math
import numpy
from random import uniform
#from simulation_interface import VehicleTrackSystem
import simulation_interface
from utils import SimulationError

CRASH_PUNISHMENT = -2000
TRACK_INNER_RADIUS_X = 500.0
TRACK_OUTER_RADIUS_X = 550.0
TRACK_INNER_RADIUS_Y = 300.0
TRACK_OUTER_RADIUS_Y = 350.0
RADIUS_OVER_INERTIA = 1.03212E-7

def distance_travelled(x, y, vx):
    theta = math.atan2(y/TRACK_OUTER_RADIUS_Y, x/TRACK_OUTER_RADIUS_X) 
    if(theta < 0):
        theta += math.pi/2
    elif (theta > (math.pi/2) and vx > 0):
        theta += math.pi
    return theta*TRACK_OUTER_RADIUS_X

    
#return distance to outer wall
def dist_from_outer_wall(x, y):
    theta = math.atan2(y/TRACK_OUTER_RADIUS_Y, x/TRACK_OUTER_RADIUS_X)     
    x_boundary = TRACK_OUTER_RADIUS_X * math.cos(theta)
    y_boundary = TRACK_OUTER_RADIUS_Y * math.sin(theta)
    return ((x-x_boundary)**2 + (y-y_boundary)**2)**.5


#return distance to inner wall
def dist_from_inner_wall(x, y):
    theta = math.atan2(y/TRACK_INNER_RADIUS_Y, x/TRACK_INNER_RADIUS_X)     
    x_boundary = TRACK_INNER_RADIUS_X * math.cos(theta)
    y_boundary = TRACK_INNER_RADIUS_Y * math.sin(theta)
    return ((x-x_boundary)**2 + (y-y_boundary)**2)**.5

# crude lap testing
def full_lap(system):
    return distance_travelled > 2*math.pi*TRACK_OUTER_RADIUS_X

def reward(system):
    
    #punish crashes severely
    if not system.is_on_track:
        return CRASH_PUNISHMENT
    
    reward = 0 
    #theta = math.atan2(y, x)
    
    #reward positive velocity
    reward += (system.cur_vx**2 + system.cur_vy**2)**.5
    #reward positive distance travelled
    reward += distance_travelled(system.vehicle_position_history[-1].x, system.vehicle_position_history[-1].y, system.cur_vx) 
    #reward distance from walls
    reward -= abs(dist_from_inner_wall(system.vehicle_position_history[-1].x, system.vehicle_position_history[-1].y) 
                  - dist_from_outer_wall(system.vehicle_position_history[-1].x, system.vehicle_position_history[-1].y))
    #negative torques are fine

    
    return reward

def qVal():
    #Steering Angle
    #Distance from Center/Top/Bot of Track
    #Velocity
    #(Theta(v))?
    #(Torque(Angular Momentum/Yaw, something))?
    #
    #Use a learning rate 1/t
    pass

def f0_constant(steering_angle, front_wheel_torque, rear_wheel_torque, lat_vel, long_vel):
    return 1

'''
How should I be normalizing these?
'''

def f1_steering_angle(steering_angle, front_wheel_torque, rear_wheel_torque, lat_vel, long_vel):
    return steering_angle

def f2_fwt(steering_angle, front_wheel_torque, rear_wheel_torque, lat_vel, long_vel):
    return front_wheel_torque

def f3_rwt(steering_angle, front_wheel_torque, rear_wheel_torque, lat_vel, long_vel):
    return rear_wheel_torque

def f4_Vx(steering_angle, front_wheel_torque, rear_wheel_torque, lat_vel, long_vel):
    return lat_vel[-1]

def f4_Vy(steering_angle, front_wheel_torque, rear_wheel_torque, lat_vel, long_vel):
    return long_vel[-1]

#def least_squares(weight, change)

'''
Larger than the distance to the inner/outer wall would lead to a wall assuming a timestep of 1
Ratio of vx or vy vs dist-wall-x and dist-wall-y
Not sure if it's a useful parameter since we already have vy&vx
'''
def f5_highVx(steering_angle, front_wheel_torque, rear_wheel_torque, lat_vel, long_vel):
    return true
#def f6_vy equiv

#theta as a function of velocity might be useful since going too fast limits turning
#def f7_thetaV

def oursim():
    
    DISCOUNT_FACTOR = 1
    
    front_wheel_torque = 500.0
    rear_wheel_torque = 500.0 
    steering_angle = 0.0
    #distance_travelled = 0
    
    time = 1
    learning_rate= 1
    weights= []
    #Initialize random weights
    #Define a function fn and put in array so we can call w_n f_n
      
    for i in range(10000):
        
        cur_model = VehicleTrackSystem()
        time = 1
        learning_rate = 1/time
        angle = -.5
        best_angle = 0
        best_q_val = 0
       
        while True:
            #Actually, we should do a +- range from current steering angle so we don't hard steer 
            #Also for a range of torques
            while angle < 0.51:
                #thing with angle
                system.tick_simulation(front_wheel_torque=front_wheel_torque,
                                               rear_wheel_torque=rear_wheel_torque,
                                               steering_angle=steering_angle)                
                
                angle += .05
                
            time+=1
        #Use history and create new weights
        #Use old weight values and new points to
        #Modify our new weight values
        #Reassign old weights to new weights for persistence
        
        #I just wanted to simulate to pause per step to see the effects
        #var = raw_input("Give me  the input here: ")
        
    
        #Update weights
        
        #system.plot_history()

def main():
    print dist_from_inner_wall(500*math.cos(0.5*math.pi),300*math.sin(0.5*math.pi))
    print math.atan2(-1,-1)
    #oursim()
                               

if __name__ == "__main__":
    main()
