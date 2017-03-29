"""
Program containing two sample vehicle simulation runs.

Author: RR
"""
import copy
import math
from random import uniform
from simulation_interface import VehicleTrackSystem
from utils import SimulationError

def sim1():
    """
    Sample simulation 1: drives in a straight line until we crash into a wall,
    and then plots vehicle trajectory.
    """
    system = VehicleTrackSystem()
        
    # for all-wheel drive, apply same torque to front and rear wheels
    front_wheel_torque = 500.0
    rear_wheel_torque = 500.0
    steering_angle = 0.0
    
    try:
        while True:
            system.tick_simulation(front_wheel_torque=front_wheel_torque,
                                   rear_wheel_torque=rear_wheel_torque,
                                   steering_angle=steering_angle)
    except SimulationError:
        system.plot_history()


def sim2():
    """
    Sample simulation 2: steers randomly until either crashing into a wall, or
    until 500 simulation "tics" have elapsed.
    """
    system = VehicleTrackSystem()

    front_wheel_torque = 500.0
    rear_wheel_torque = 500.0 
    steering_angle = 0.0
    
    try:
        for i in range(2):
            system.tick_simulation(front_wheel_torque=front_wheel_torque,
                                   rear_wheel_torque=rear_wheel_torque,
                                   steering_angle=steering_angle)
            # change steering angle (measured in radians) by a random amount
            steering_angle += uniform(-0.05, 0.05)
            print system.simulate_inputs(front_wheel_torque, rear_wheel_torque, steering_angle)
            print system.vehicle_position_history
    except SimulationError:
        pass
    
    system.plot_history()

FRONT_WHEEL_TORQUE = 500.0
REAR_WHEEL_TORQUE = 500.0 
STEERING_ANGLE = 0.0
DISTANCE_TRAVELLED = 0
LEARNING_RATE = 1
WEIGHTS = []
QVALS = []

def dist_from_inner_wall():
    pass
def dist_from_outer_wall():
    pass
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
def ourSim():
    angle = -.5
    best_angle = 0
    best_q_val = 0
    
    
    time = 1
    while True:
        cur_model = VehicleModel()
        LEARNING_RATE = 1/time
        while True:
            while angle < 0.5:
                #thing with angle
                angle += .05
        system.plot_history()
        var = raw_input("Give me  the input here: ")
    
        #Update weights
        
        time+=1
    
def main():
    #sim1()
    sim2()
    

if __name__ == "__main__":
    main()
