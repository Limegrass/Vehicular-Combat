
import math
from simulation_interface import VehicleTrackSystem
class QFunc:
    CRASH_PUNISHMENT = -25000
    MAX_TORQUE = 5.0
    TORQUE_INCREMENT = 1.0
    MAX_DELTA_STEERING_ANGLE = math.pi/1.0
    STEERING_ANGLE_INCREMENT = math.pi/32.0

    LEARNING_RATE = 0.2 #change this
    DISCOUNT_FACTOR = 0.5 #change this
    NUM_FEATURES = 6 #This must be changed to equal to the amount features in the __init__
    
    def __init__(self, weights):
        '''PLAY WITH FEATURES'''
        #features = [self._f0_constant, self._f1_steering_angle, self._f2_fwt, self._f3_rwt, self._f7_distance, self._f8_centerness]
        #features = [self._f1_steering_angle, self._f4_vx, self._f5_vy, self._f6_high_v_tangential, self._f7_distance, self._f8_centerness]
        #features = [self._f0_constant, self._f1_steering_angle, self._f2_fwt, self._f3_rwt, self._f4_vx, self._f5_vy, 
         #           self._f7_distance, self._f8_centerness, self._f9_pvx, self._f10_pvy, self._f11_pdistance, self._f12_pcenterness,
          #          self._f14_delta_theta, self._f15_low_velocity]
        #features = [_f1_steering_angle]
        self.features = [self._f0_constant, self._f1_steering_angle, self._f2_fwt, self._f3_rwt, self._f7_distance, self._f8_centerness]
        self.feature_evaluations = []
        self.q_used = []
        self.rewards = []
        self.q_best = []
        self.weights = weights

    def qVal(self, system, steering_angle, fwt, rwt, dtheta):
        q = 0
        px, py, pvx, pvy = system.predict_states(fwt, rwt, steering_angle)
        f_vals = []
        #Steering Angle
        for i in range(len(self.features)):
            f = self.features[i](steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta)
            f_vals.append(f)
            q += self.weights[i]*f
        return q, f_vals

    def find_max_a():
        possible_actions = []

        max = 0
        #find max a if you try every action at current state
        for i in possible_actions:
            #compute features
            #call qval()
            #update max if necessary
            pass
        
    def update_weights(self):

        new_weights= []
        for i in range(len(self.weights)):
            new_weights.append(self.weights[i])
        for i in range(len(self.weights)):
            for j in range(len(self.q_used)-1):
                new_weights[i] = self.weights[i] + self.LEARNING_RATE*(self.rewards[j] + self.DISCOUNT_FACTOR*self.q_best[j+1] - self.q_used[j])*self.feature_evaluations[j][i]
        return new_weights        
    
    '''=======================================FEATURES=========================================================='''
    
    def _f0_constant(self, steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta):
        return 1.0
    
    def _f1_steering_angle(self, steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta):
        #Find the angle it should face for forward movement
        #theta = math.atan2(system.y/VehicleTrackSystem.TRACK_MIDDLE_RADIUS_Y, system.x/VehicleTrackSystem.TRACK_MIDDLE_RADIUS_X) 
        
        theta = math.atan2(system.vehicle_position_history[-1].y, (system.vehicle_position_history[-1].x)*.81601)
        theta -= math.pi/2
        if theta < -math.pi:
            theta += (math.pi*2)
        if steering_angle > math.pi or steering_angle < -math.pi:
            print steering_angle
    
        
        clockwise = abs((steering_angle-theta)/math.pi)
        if clockwise > 1:
            return (2*math.pi - abs(steering_angle) - abs(theta))/math.pi
    
            
        return clockwise
    
    def _f2_fwt(self, steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta):
        return abs((fwt+self.MAX_TORQUE)/(self.MAX_TORQUE*2))
    
    def _f3_rwt(self, steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta):
        return abs((rwt+self.MAX_TORQUE)/(self.MAX_TORQUE*2))
    
    def _f4_vx(self, steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta):
        theta = math.atan2(system.y/VehicleTrackSystem.TRACK_MIDDLE_RADIUS_Y, system.x/VehicleTrackSystem.TRACK_MIDDLE_RADIUS_X) 
        theta -= math.pi/2
        return abs(((system.vx/system.speed - math.cos(theta))) / math.cos(theta))
    
    def _f5_vy(self, steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta):
        theta = math.atan2(system.y/TRACK_MIDDLE_RADIUS_Y, system.x/TRACK_MIDDLE_RADIUS_X) 
        theta -= math.pi/2
        if theta == 0:
            return abs(((system.vy/system.speed - math.sin(theta))))
        return abs(((system.vy/system.speed - math.sin(theta))) / math.sin(theta))
    
    def _f6_high_v_tangential(self, steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta): 
        theta = math.atan2(system.y/TRACK_MIDDLE_RADIUS_Y, system.x/TRACK_MIDDLE_RADIUS_X) 
        theta -= math.pi/2
        #If the tangential velocity to a wall relative to the direction the car should face is >50
        return abs(system.vx*math.cos(theta) - system.vy * math.sin(theta))/50
    
    def _f7_distance(self, steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta): 
        return abs(self.distance_travelled(system.x, system.y, system.vx) / (2*math.pi*VehicleTrackSystem.TRACK_OUTER_RADIUS_X))
        #return distance_travelled(system.x, system.y, system.vx)
    
    
    def _f8_centerness(self, steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta):
        return abs(self.follow_centerness(system.x, system.y) / 50)
    
    def _f9_pvx(self, steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta):
        theta = math.atan2(py/VehicleTrackSystem.TRACK_MIDDLE_RADIUS_Y, px/VehicleTrackSystem.TRACK_MIDDLE_RADIUS_X) 
        theta -= math.pi/2
        return abs(((pvx/(pvx**2 + pvy**2)**.5 - math.cos(theta))) / math.cos(theta))
    
    def _f10_pvy(self, steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta):
        theta = math.atan2(py/VehicleTrackSystem.TRACK_MIDDLE_RADIUS_Y, px/VehicleTrackSystem.TRACK_MIDDLE_RADIUS_X) 
        theta -= math.pi/2
        if theta == 0:
            return abs(((pvy/(pvx**2 + pvy**2)**.5 - math.sin(theta))))
        return abs(((pvy/(pvx**2 + pvy**2)**.5 - math.sin(theta))) / math.sin(theta))
    
    def _f11_pdistance(self, steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta): 
        return abs(distance_travelled(px, py, pvx) / (2*math.pi*VehicleTrackSystem.TRACK_OUTER_RADIUS_X))
        #return abs(distance_travelled(px, py, pvx))
    
    
    def _f12_pcenterness(self, steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta): 
        return abs(follow_centerness(px, py)/50)
    
    def _f13_crash(self, steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta): 
        return 1 if system.is_on_track else 0
    def _f14_delta_theta(steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta): 
        return abs(dtheta/self.MAX_DELTA_STEERING_ANGLE)
    
    def _f15_low_velocity(self, steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta): 
        speed = system.speed
        return (.3/speed) if speed > .1 else 1.0
    
    def _f16_pdist_outer_wall(self, steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta): 
        return dist_from_outer_wall(px, py)/50
                                    
    
    def f17_pdist_inner_wall(self, steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta): 
        return dist_from_inner_wall(px, py)/50
    
    
    #def f9_circle_velocity(steering_angle, fwt, rwt, x, y, vx, vy):
     #   return 0
    #theta as a function of velocity might be useful since going too fast limits turning
    #def f10_thetaV
    
    '''=======================END FEATURES========================'''    
    
    '''-------------------Functions that should work------------------'''
    #One idea to simplify all the code could be to swap x and y s.t we don't have to bother with 4 quadrants and instead deal with 2 hemispheres
    #This is if we began at 0 and also travel ccw
    def distance_travelled(self, x, y, vx):
        theta = math.atan2(y/VehicleTrackSystem.TRACK_MIDDLE_RADIUS_Y, x/VehicleTrackSystem.TRACK_MIDDLE_RADIUS_X) 
        if(theta < 0):
            theta = math.pi/2 - theta
        elif (theta > (math.pi/2) and vx > 0):
            theta = math.pi + theta
        else:
            theta = math.pi/2 - theta
        return theta*VehicleTrackSystem.TRACK_OUTER_RADIUS_X
    
        
    #return distance to outer wall
    def dist_from_outer_wall(self, x, y):
        theta = math.atan2(y/VehicleTrackSystem.TRACK_OUTER_RADIUS_Y, x/VehicleTrackSystem.TRACK_OUTER_RADIUS_X)     
        x_boundary = VehicleTrackSystem.TRACK_OUTER_RADIUS_X * math.cos(theta)
        y_boundary = VehicleTrackSystem.TRACK_OUTER_RADIUS_Y * math.sin(theta)
        return ((x-x_boundary)**2 + (y-y_boundary)**2)**.5
    
    
    #return distance to inner wall
    def dist_from_inner_wall(self, x, y):
        theta = math.atan2(y/VehicleTrackSystem.TRACK_INNER_RADIUS_Y, x/VehicleTrackSystem.TRACK_INNER_RADIUS_X)     
        x_boundary = VehicleTrackSystem.TRACK_INNER_RADIUS_X * math.cos(theta)
        y_boundary = VehicleTrackSystem.TRACK_INNER_RADIUS_Y * math.sin(theta)
        return ((x-x_boundary)**2 + (y-y_boundary)**2)**.5        
    
    def follow_centerness(self, x, y):
        return abs(self.dist_from_inner_wall(x, y) 
                      - self.dist_from_outer_wall(x, y)) 
    
    def reward(self, system):
        #punish crashes severely
        if not system.is_on_track:
            return self.CRASH_PUNISHMENT
        r = 0 
        #theta = math.atan2(y, x)
        #reward positive velocity
        r += self.circle_velocity(system) 
        #reward positive distance travelled
        r += self.distance_travelled(system.x, system.y, system.vx) 
        #reward distance from walls. Best in the center with a value of 0
        r -= self.follow_centerness(system.x, system.y)*2
        r -= (.1/(system.vx**2 + system.vy**2))
        #negative torques are fine
        return r
    
    
    #Not quite radial velocity, so can't name it as such
    def circle_velocity(self, system):
        velocity = 0
        #Quadrant 1
        if(system.x >= 0 and system.y >= 0):
            velocity -= system.vy
            velocity += system.vx
        #Quadrant 4
        elif(system.x >= 0 and system.y <= 0):
            velocity -= system.vy
            velocity -= system.vx
        #Quadrant 3
        elif(system.x <= 0 and system.y <= 0):
            velocity += system.vy
            velocity -= system.vx
        #Quadrant 2
        elif(system.x <= 0 and system.y >= 0):
            velocity += system.vy
            velocity += system.vx
        
        return velocity