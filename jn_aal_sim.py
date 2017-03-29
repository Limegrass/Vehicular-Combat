'''

'''
import math
import numpy
from random import uniform
#from simulation_interface import VehicleTrackSystem
import simulation_interface
from utils import SimulationError

TRACK_INNER_RADIUS_X = 500.0
TRACK_OUTER_RADIUS_X = 550.0
TRACK_INNER_RADIUS_Y = 300.0
TRACK_OUTER_RADIUS_Y = 350.0
FRONT_WHEEL_TORQUE = 500.0
REAR_WHEEL_TORQUE = 500.0 
STEERING_ANGLE = 0.0
DISTANCE_TRAVELLED = 0
LEARNING_RATE = 1
WEIGHTS = []
QVALS = []

def dist_from_outer_wall(x, y):
    theta = math.atan2(x, y)
    print theta
    x_boundary = TRACK_OUTER_RADIUS_X * math.cos(theta)
    y_boundary = TRACK_OUTER_RADIUS_Y * math.sin(theta)
    #return distance from outer wall
    return ((x-x_boundary)**2 + (y-y_boundary)**2)**.5
    
def dist_from_inner_wall(x, y):
    theta = math.atan2(x, y)
    print theta
    x_boundary = TRACK_INNER_RADIUS_X * math.cos(theta)
    y_boundary = TRACK_INNER_RADIUS_Y * math.sin(theta)
    #return distance from outer wall
    return ((x-x_boundary)**2 + (y-y_boundary)**2)**.5
    
def reward():
    #do something 
    return DISTANCE_TRAVELLED - abs(dist_from_inner_wall() - dist_from_outer_wall())

def qVal():
    #Steering Angle
    #Distance from Center/Top/Bot of Track
    #Velocity
    #(Theta(v))?
    #(Torque(Angular Momentum/Yaw, something))?
    #
    #Use a learning rate 1/t
    pass

def oursim():
    time = 1
    
    while True:
        
        cur_model = VehicleTrackSystem()
        LEARNING_RATE = 1/time
        angle = -.5
        best_angle = 0
        best_q_val = 0
       
        while True:
            while angle < 0.51:
                #thing with angle
                system.tick_simulation(front_wheel_torque=front_wheel_torque,
                                               rear_wheel_torque=rear_wheel_torque,
                                               steering_angle=steering_angle)                
                angle += .05
        system.plot_history()
        var = raw_input("Give me  the input here: ")
        
    
        #Update weights
        
        time+=1

def main():
    print dist_from_inner_wall(300,500)
    #oursim()

if __name__ == "__main__":
    main()
