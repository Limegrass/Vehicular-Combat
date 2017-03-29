"""
Defines a type that encapsulates the representation and simulation of the
Vehicle-Track system.

As per Python convention, method names that begin with an underscore are
meant to be treated as though they were private --- these methods are
not a part of the public API and you should refrain from calling them.
All other methods are public.

Feel free to add more methods to this class definition if necessary ---
for example, you may want to add a method for computing how far your
current position is from the central line around the track.

Author: RR
"""
import numpy as np
import matplotlib.pyplot as plt
from model_interface import VehicleModel
from utils import SIM_RESOLUTION, Position, SimulationError


class VehicleTrackSystem:
    TRACK_INNER_RADIUS_X = 500.0
    TRACK_OUTER_RADIUS_X = 550.0
    TRACK_INNER_RADIUS_Y = 300.0
    TRACK_OUTER_RADIUS_Y = 350.0
    
    def __init__(self):
        self._vehicle_state = VehicleModel()
        self.vehicle_position_history = [Position(x=0.0, y=325.0)]
        self.is_on_track = True
        
        
    '''
    Computes the new position for a few time steps given from the simualation
    '''
    def _compute_displacement(self, initial_position, velocity_over_time):
        new_positions = [initial_position] * len(velocity_over_time)
        new_positions[0] = (initial_position + 
                            (velocity_over_time[0] * SIM_RESOLUTION))
        for i in range(1, len(new_positions)):
            new_positions[i] = (new_positions[i - 1] +
                                (velocity_over_time[i] * SIM_RESOLUTION))
        return new_positions
    
    def _is_outer_wall_collision(self, x, y):
        return (((x / self.TRACK_OUTER_RADIUS_X) ** 2 +
                 (y / self.TRACK_OUTER_RADIUS_Y) ** 2) >= 1.0)
    
    def _is_inner_wall_collision(self, x, y):
        return (((x / self.TRACK_INNER_RADIUS_X) ** 2 +
                 (y / self.TRACK_INNER_RADIUS_Y) ** 2) <= 1.0)
    
    def is_collision(self, x, y):
        return (self._is_inner_wall_collision(x, y) or 
                self._is_outer_wall_collision(x, y))

    def tick_simulation(self,
                        front_wheel_torque,
                        rear_wheel_torque,
                        steering_angle):
        if not self.is_on_track:
            raise SimulationError('vehicle is already off the track!')
        
        # determine new vehicle velocity
        lat_velocity, long_velocity = self._vehicle_state.simulate_inputs(
            front_wheel_torque, rear_wheel_torque, steering_angle)
        
        # update vehicle position on track
        new_x = self._compute_displacement(self.vehicle_position_history[-1].x,
                                           long_velocity)
        new_y = self._compute_displacement(self.vehicle_position_history[-1].y,
                                           lat_velocity)

        # update history
        self.vehicle_position_history.extend([Position(x=x, y=y)
                                              for x, y in zip(new_x, new_y)])
        
        # bounds check
        if any(self.is_collision(x, y) for x, y in zip(new_x, new_y)):
            self.is_on_track = False
            raise SimulationError('vehicle has collided with a wall!')
        
    def plot_history(self):
        x_outer = np.linspace(start=-self.TRACK_OUTER_RADIUS_X,
                              stop=self.TRACK_OUTER_RADIUS_X,
                              num=100000)
        y_outer = np.sqrt((1 - (x_outer / self.TRACK_OUTER_RADIUS_X) ** 2) * 
                          (self.TRACK_OUTER_RADIUS_Y ** 2))

        x_inner = np.linspace(start=-self.TRACK_INNER_RADIUS_X,
                              stop=self.TRACK_INNER_RADIUS_X,
                              num=100000)
        y_inner = np.sqrt((1 - (x_inner / self.TRACK_INNER_RADIUS_X) ** 2) * 
                          (self.TRACK_INNER_RADIUS_Y ** 2))

        x_center = (x_outer + x_inner) / 2.0
        y_center = (y_outer + y_inner) / 2.0

        vehicle_x = [p.x for p in self.vehicle_position_history]
        vehicle_y = [p.y for p in self.vehicle_position_history]
        
        plt.plot(x_outer, y_outer, 'b-', x_outer, -y_outer, 'b-',
                 x_inner, y_inner, 'b-', x_inner, -y_inner, 'b-',
                 x_center, y_center, 'b--', x_center, -y_center, 'b--',
                 vehicle_x, vehicle_y, 'r-')
        plt.show()        
    
    '''
    Simulates series of time step to be able to calculate the Q values
    Using the simulate_inputs model from the Model class
    '''
    def simulate_inputs(self, front_wheel_torque, rear_wheel_torque, steering_angle):
        return self._vehicle_state.simulate_inputs(front_wheel_torque, rear_wheel_torque, steering_angle)   
    
    '''
    Maybe useful for feature testing a test input collision
    Incomplete. Must use outputs from simulate inputs to calcuate new positions
    First create new position function
    
    
    def model_outer_wall_collision(self, x, y):
        return (((x / self.TRACK_OUTER_RADIUS_X) ** 2 +
                 (y / self.TRACK_OUTER_RADIUS_Y) ** 2) >= 1.0)
    
    def model_inner_wall_collision(self, x, y):
        return (((x / self.TRACK_INNER_RADIUS_X) ** 2 +
                 (y / self.TRACK_INNER_RADIUS_Y) ** 2) <= 1.0)                     
    '''