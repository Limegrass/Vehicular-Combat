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
import math
import numpy as np
import matplotlib.pyplot as plt
from model_interface import VehicleModel
from utils import SIM_RESOLUTION, Position, SimulationError


class VehicleTrackSystem:
    TRACK_INNER_RADIUS_X = 500.0
    TRACK_OUTER_RADIUS_X = 550.0
    TRACK_INNER_RADIUS_Y = 300.0
    TRACK_OUTER_RADIUS_Y = 350.0
    TRACK_MIDDLE_RADIUS_X = 525.0
    TRACK_MIDDLE_RADIUS_Y = 325.0
    RADIUS_OVER_INERTIA = 1.03212E-7
    
    def __init__(self):
        self._vehicle_state = VehicleModel()
        self.vehicle_position_history = [Position(x=0.0, y=325.0)]
        self.is_on_track = True
        self.vx = 10.0
        self.vy = 0.0
        self.speed = (self.vx**2 + self.vy**2)**.5
        self.x = 0.0
        self.y = 325.0
        self.distance = 0
        
        
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
        
        # Save resulting velocity of input
        self.vy = lat_velocity[-1]
        self.vx = long_velocity[-1]
        self.speed = (self.vx**2 + self.vy**2)**.5
        
        # update vehicle position on track
        new_x = self._compute_displacement(self.vehicle_position_history[-1].x,
                                           long_velocity)
        new_y = self._compute_displacement(self.vehicle_position_history[-1].y,
                                           lat_velocity)

        # update history
        self.vehicle_position_history.extend([Position(x=x, y=y)
                                              for x, y in zip(new_x, new_y)])
        self.x = self.vehicle_position_history[-1].x
        self.y = self.vehicle_position_history[-1].y
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
    
    
    
    def predict_states(self,
                    front_wheel_torque,
                    rear_wheel_torque,
                    steering_angle):
        
        total_torque = (front_wheel_torque + rear_wheel_torque)
        #Not true theta since radii not inputted
        #theta = math.atan2(self.vehicle_position_history[-1].y, self.vehicle_position_history[-1].x)
        vx = (math.cos(steering_angle) * (self.RADIUS_OVER_INERTIA * total_torque)) + self.vx 
        vy = (math.sin(steering_angle) * (self.RADIUS_OVER_INERTIA * total_torque)) + self.vy
        x = self.vehicle_position_history[-1].x + self.vx + (total_torque * math.cos(steering_angle) * self.RADIUS_OVER_INERTIA / 2)
        y = self.vehicle_position_history[-1].y + self.vy + (total_torque * math.sin(steering_angle) * self.RADIUS_OVER_INERTIA / 2)
        
        #print "X: ", self.vehicle_position_history[-1].x, " Y: ", self.vehicle_position_history[-1].y, " PX: ", x, " PY: ", y, " TORQUE: ", front_wheel_torque
        #Is this legal syntax? How do I access this?
        return x, y, vx, vy        

    
    
    #Methods borrowed from Arthur Chen and Caleb Warren for Ellipse Calculation

    def ellipse_tan_dot(self, rx, ry, px, py, theta):
        '''Dot product of the equation of the line formed by the point
        with another point on the ellipse's boundary and the tangent of the ellipse
        at that point on the boundary.
        '''
        return ((rx ** 2 - ry ** 2) * cos(theta) * sin(theta) -
                px * rx * sin(theta) + py * ry * cos(theta))
    
    
    def ellipse_tan_dot_derivative(self, rx, ry, px, py, theta):
        '''The derivative of ellipe_tan_dot.
        '''
        return ((rx ** 2 - ry ** 2) * (cos(theta) ** 2 - sin(theta) ** 2) -
                px * rx * cos(theta) - py * ry * sin(theta))
    
    
    def estimate_distance(self, x, y, rx=525, ry=325, error=1e-5):
        '''Given a point (x, y), and an ellipse with major - minor axis (rx, ry),
        will return the distance between the ellipse and the
        closest point on the ellipses boundary.
        '''
        theta = atan2(rx * y, ry * x)
        while fabs(self.ellipe_tan_dot(rx, ry, x, y, theta)) > error:
            theta -= self.ellipe_tan_dot(
                rx, ry, x, y, theta) / \
                self.ellipe_tan_dot_derivative(rx, ry, x, y, theta)
    
        px, py = rx * cos(theta), ry * sin(theta)
        #update current position
        self.CURRENT_X = px
        self.CURRENT_Y = py
        dis = ((x - px) ** 2 + (y - py) ** 2) ** .5 
        return dis       
    
    def on_which_side(self, x, y, rx=525, ry=325):
        '''Given a point (x, y), return true if it is inside the central
        line; false if it is outside the central line
        '''
        if (((x / rx) ** 2 +
                         (y / ry) ** 2) >= 1.0):
            return True
        else:
            return False
    
    def get_velocity(self):
        '''returns an array of velocity
        '''
        #lat_velocity, long_velocity = self._vehicle_state.simulate_inputs(
                    #front_wheel_torque, rear_wheel_torque, steering_angle)
        #v = [long_velocity[-1], lat_velocity[-1]]
        return self.velocity
    
    def curvature(self, x, y):
        return
        
    def estimate_angle(self, v, rx=525, ry=325):
        x = self.x
        y = self.y
        
        d = self.derivative(x, y)
        v_e = [1, d]
        return self.angle_between(v, v_e)
    
    def derivative(self, x, y):
        return (-1) * (169 * x) / (411 * y)
    
    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  
        http://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
        """
        return vector / np.linalg.norm(vector)
    
    def angle_between(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::
                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))    