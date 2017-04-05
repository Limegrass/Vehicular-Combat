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
    steering_angle = 0

    print "VX,", "Angle"
    try:
        while True:
            front_wheel_torque = 00.0
            rear_wheel_torque = 00.0 

            #print system.cur_vx, ",", math.atan2((system.vehicle_position_history[-1].x/325)**2, (system.vehicle_position_history[-1].x/525)**2)-(math.pi)

            #steering_angle = math.atan2(system.vehicle_position_history[-1].y, (system.vehicle_position_history[-1].x)*.8160)- (math.pi/2)
#            steering_angle =  math.acos(system.vehicle_position_history[-1].x/525) - (math.pi/2)
            #steering_angle -= math.pi/32
            steering_angle = system.estimate_angle(system.vx)
            print steering_angle
            #print system.vx, system.vy,  steering_angle
            system.tick_simulation(front_wheel_torque=front_wheel_torque,
                                   rear_wheel_torque=rear_wheel_torque,
                                   steering_angle=steering_angle)
            #print system.cur_vx
            # change steering angle (measured in radians) by a random amount
            #steering_angle += uniform(-0.05, 0.05)
            #for j in range(100):
                #system.simulate_inputs(front_wheel_torque, rear_wheel_torque, steering_angle)

    except SimulationError:
        print "crash"
        system.plot_history()
    except WindowsError:
        print "slow"
        system.plot_history()
    
def main():
    #sim1()
    sim2()
    

if __name__ == "__main__":
    main()
