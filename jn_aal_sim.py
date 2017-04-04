'''
Path Finding using Q-Learning
Author: JN, AAL
'''
import random
import math
import numpy
from random import uniform
from simulation_interface import VehicleTrackSystem
#import simulation_interface
from utils import SimulationError

CRASH_PUNISHMENT = -25000
TRACK_INNER_RADIUS_X = 500.0
TRACK_OUTER_RADIUS_X = 550.0
TRACK_INNER_RADIUS_Y = 300.0
TRACK_OUTER_RADIUS_Y = 350.0
TRACK_MIDDLE_RADIUS_X = (TRACK_INNER_RADIUS_X+TRACK_OUTER_RADIUS_X)/2
TRACK_MIDDLE_RADIUS_Y = (TRACK_INNER_RADIUS_Y+TRACK_OUTER_RADIUS_Y)/2
RADIUS_OVER_INERTIA = 1.03212E-7
MAX_TORQUE = 5.0
TORQUE_INCREMENT = 1.0
MAX_DELTA_STEERING_ANGLE = math.pi/4.0
STEERING_ANGLE_INCREMENT = math.pi/32.0
SIMULATION_MAX_TIME = 200
DISCOUNT_FACTOR = .5
LEARNING_RATE = .2
NUM_TORQUE_INCREMENTS = MAX_TORQUE/TORQUE_INCREMENT
NUM_ANGLE_INCREMENTS = (MAX_DELTA_STEERING_ANGLE/STEERING_ANGLE_INCREMENT)

    
               
                                
'''-------------------Functions that should work------------------'''
#One idea to simplify all the code could be to swap x and y s.t we don't have to bother with 4 quadrants and instead deal with 2 hemispheres
#This is if we began at 0 and also travel ccw
def distance_travelled(x, y, vx):
    theta = math.atan2(y/TRACK_MIDDLE_RADIUS_Y, x/TRACK_MIDDLE_RADIUS_X) 
    if(theta < 0):
        theta = math.pi/2 - theta
    elif (theta > (math.pi/2) and vx > 0):
        theta = math.pi + theta
    else:
        theta = math.pi/2 - theta
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
    return distance_travelled(system.vehicle_position_history[-1].x, system.vehicle_position_history[-1].y, system.cur_vx) > 2*math.pi*TRACK_INNER_RADIUS_X

def follow_centerness(x, y):
    return abs(dist_from_inner_wall(x, y) 
                  - dist_from_outer_wall(x, y)) 

def reward(system):
    #punish crashes severely
    if not system.is_on_track:
        return CRASH_PUNISHMENT
    reward = 0 
    #theta = math.atan2(y, x)
    #reward positive velocity
    reward += circle_velocity(system) 
    #reward positive distance travelled
    reward += distance_travelled(system.vehicle_position_history[-1].x, system.vehicle_position_history[-1].y, system.cur_vx) 
    #reward distance from walls. Best in the center with a value of 0
    reward -= follow_centerness(system.vehicle_position_history[-1].x, system.vehicle_position_history[-1].y)*2
    reward -= (.1/(system.cur_vx**2 + system.cur_vy**2))
    #negative torques are fine
    return reward


#Not quite radial velocity, so can't name it as such
def circle_velocity(system):
    velocity = 0
    #Quadrant 1
    if(system.vehicle_position_history[-1].x >= 0 and system.vehicle_position_history[-1].y >= 0):
        velocity -= system.cur_vy
        velocity += system.cur_vx
    #Quadrant 4
    elif(system.vehicle_position_history[-1].x >= 0 and system.vehicle_position_history[-1].y <= 0):
        velocity -= system.cur_vy
        velocity -= system.cur_vx
    #Quadrant 3
    elif(system.vehicle_position_history[-1].x <= 0 and system.vehicle_position_history[-1].y <= 0):
        velocity += system.cur_vy
        velocity -= system.cur_vx
    #Quadrant 2
    elif(system.vehicle_position_history[-1].x <= 0 and system.vehicle_position_history[-1].y >= 0):
        velocity += system.cur_vy
        velocity += system.cur_vx
    
    return velocity
     

def qVal(weights, features, steering_angle, fwt, rwt, system, dtheta):
    q = 0
    px, py, pvx, pvy = system.predict_states(fwt, rwt, steering_angle)
    #Steering Angle
    for i in range(len(features)):
        q += weights[i]*features[i](steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta)
    #(Theta(v))?
    #(Torque(Angular Momentum/Yaw, something))?
    return q

def feature_evaluations(features, steering_angle, fwt, rwt, system, dtheta):
    evals = []
    px, py, pvx, pvy = system.predict_states(fwt, rwt, steering_angle)

    for i in range(len(features)):
        evals.append(features[i](steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta))
    return evals

'''=======================================FEATURES=========================================================='''

def f0_constant(steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta):
    return 1.0

def f1_steering_angle(steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta):
    #Find the angle it should face for forward movement
    theta = math.atan2(system.vehicle_position_history[-1].y/TRACK_MIDDLE_RADIUS_Y, system.vehicle_position_history[-1].x/TRACK_MIDDLE_RADIUS_X) 
    theta -= math.pi/2
    
    
    if theta < -math.pi:
        theta += math.pi*2
    if steering_angle > math.pi or steering_angle < -math.pi:
        print steering_angle
    
        
    
    clockwise = abs((steering_angle-theta)/math.pi)
    if clockwise > 1:
        return (2*math.pi - abs(steering_angle) - abs(theta))/math.pi

        
    return clockwise

def f2_fwt(steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta):
    return abs((fwt+MAX_TORQUE)/(MAX_TORQUE*2))

def f3_rwt(steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta):
    return abs((rwt+MAX_TORQUE)/(MAX_TORQUE*2))

def f4_vx(steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta):
    theta = math.atan2(system.vehicle_position_history[-1].y/TRACK_MIDDLE_RADIUS_Y, system.vehicle_position_history[-1].x/TRACK_MIDDLE_RADIUS_X) 
    theta -= math.pi/2
    #Return the difference between the expected fractional component of the current angle
    # and the current steering angle, divided by the expected
    # Possibly add a factor of 550/350 ?
    return abs(((system.cur_vx/(system.cur_vx**2 + system.cur_vy**2)**.5 - math.cos(theta))) / math.cos(theta))

def f5_vy(steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta):
    theta = math.atan2(system.vehicle_position_history[-1].y/TRACK_MIDDLE_RADIUS_Y, system.vehicle_position_history[-1].x/TRACK_MIDDLE_RADIUS_X) 
    theta -= math.pi/2
    if theta == 0:
        return abs(((system.cur_vy/(system.cur_vx**2 + system.cur_vy**2)**.5 - math.sin(theta))))
    return abs(((system.cur_vy/(system.cur_vx**2 + system.cur_vy**2)**.5 - math.sin(theta))) / math.sin(theta))

def f6_high_v_tangential(steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta): 
    theta = math.atan2(system.vehicle_position_history[-1].y/TRACK_MIDDLE_RADIUS_Y, system.vehicle_position_history[-1].x/TRACK_MIDDLE_RADIUS_X) 
    theta -= math.pi/2
    #If the tangential velocity to a wall relative to the direction the car should face is >50
    return abs(system.cur_vx*math.cos(theta) - system.cur_vy * math.sin(theta))/50

def f7_distance(steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta): 
    return abs(distance_travelled(system.vehicle_position_history[-1].x, system.vehicle_position_history[-1].y, system.cur_vx) / (2*math.pi*TRACK_OUTER_RADIUS_X))
    #return distance_travelled(system.vehicle_position_history[-1].x, system.vehicle_position_history[-1].y, system.cur_vx)


def f8_centerness(steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta):
    return abs(follow_centerness(system.vehicle_position_history[-1].x, system.vehicle_position_history[-1].y) / 50)

def f9_pvx(steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta):
    theta = math.atan2(py/TRACK_MIDDLE_RADIUS_Y, px/TRACK_MIDDLE_RADIUS_X) 
    theta -= math.pi/2
    #Return the difference between the expected fractional component of the current angle
    # and the current steering angle, divided by the expected
    # Possibly add a factor of 550/350 ?
    return abs(((pvx/(pvx**2 + pvy**2)**.5 - math.cos(theta))) / math.cos(theta))

def f10_pvy(steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta):
    theta = math.atan2(py/TRACK_MIDDLE_RADIUS_Y, px/TRACK_MIDDLE_RADIUS_X) 
    theta -= math.pi/2
    if theta == 0:
        return abs(((pvy/(pvx**2 + pvy**2)**.5 - math.sin(theta))))
    return abs(((pvy/(pvx**2 + pvy**2)**.5 - math.sin(theta))) / math.sin(theta))

def f11_pdistance(steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta): 
    return abs(distance_travelled(px, py, pvx) / (2*math.pi*TRACK_OUTER_RADIUS_X))
    #return abs(distance_travelled(px, py, pvx))


def f12_pcenterness(steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta): 
    return abs(follow_centerness(px, py)/50)

def f13_crash(steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta): 
    return 1 if system.is_on_track else 0
def f14_delta_theta(steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta): 
    return abs(dtheta/MAX_DELTA_STEERING_ANGLE)

def f15_low_velocity(steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta): 
    speed = (system.cur_vx**2 + system.cur_vy**2)**.5
    return (.3/speed) if speed > .1 else 1.0

def f16_pdist_outer_wall(steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta): 
    return dist_from_outer_wall(px, py)/50
                                

def f17_pdist_inner_wall(steering_angle, fwt, rwt, system, px, py, pvx, pvy, dtheta): 
    return dist_from_inner_wall(px, py)/50


#def f9_circle_velocity(steering_angle, fwt, rwt, x, y, vx, vy):
 #   return 0
#theta as a function of velocity might be useful since going too fast limits turning
#def f10_thetaV

'''=======================END FEATURES========================'''

    
'''++++++++++++++++++++++++++++++++++++Stuff that probably doesn't work perfectly++++++++++++++++++++++++++++++++'''
def oursim():
    front_wheel_torque = 0.0
    rear_wheel_torque = 0.0 
    #distance_travelled = 0
    weights= []
    #features = [f0_constant, f1_steering_angle, f6_high_v_tangential, f7_distance, f8_centerness,  f11_pdistance, f12_pcenterness]
    #features = [f1_steering_angle, f4_vx, f5_vy, f6_high_v_tangential, f7_distance, f8_centerness]
    features = [f0_constant, f1_steering_angle, f2_fwt, f3_rwt, f4_vx, f5_vy, 
                f7_distance, f8_centerness, f9_pvx, f10_pvy, f11_pdistance, f12_pcenterness,
                f14_delta_theta, f15_low_velocity]
    #features = [f1_steering_angle]
    for i in range(len(features)):
        weights.append(uniform(-100.0, 100.0))
    #Initialize random weights
    #Define a function fn and put in array so we can call w_n f_n
    
    
    best_reward = 0
    for episode in range(SIMULATION_MAX_TIME):
        system = VehicleTrackSystem()
        try:
           #i represents time/episode
            epsilon = 1.0/(episode+1.0)
            q_values = []
            rewards = []
            feature_evals = []
            last_angle = 0.0
            last_torque = 0.0
            best_qs = []
            steering_angle = 0.0
           
            while True:
                #Actually, we should do a +- range from current steering angle so we don't hard steer 
                #Also for a range of torques
                '''
                speed = (system.cur_vx**2 + system.cur_vy**2)**.5
                if speed < .1:
                    print speed
                    '''
                best_q_val = CRASH_PUNISHMENT
                best_torque_multiplier = 0
                best_angle_multplier = 0 
                    
                        

                for torque_multiplier in range(int(-NUM_TORQUE_INCREMENTS), int(NUM_TORQUE_INCREMENTS+1)):
                    test_torque = torque_multiplier*TORQUE_INCREMENT
                    for angle_multiplier in range(int(-NUM_ANGLE_INCREMENTS), int(NUM_ANGLE_INCREMENTS+1)):
                        
                        test_angle = steering_angle + (angle_multiplier*STEERING_ANGLE_INCREMENT)
                        if test_angle > math.pi:
                            test_angle -= math.pi*2.0
                        if test_angle < -math.pi:
                            test_angle += math.pi*2.0

                        test_q = qVal(weights, features, test_angle, test_torque, test_torque, system, (angle_multiplier*STEERING_ANGLE_INCREMENT))
                        #print test_q
                        #thing with angle
                        if test_q > best_q_val:
                            best_torque_multiplier = torque_multiplier
                            best_angle_multplier = angle_multiplier
                            best_q_val = test_q
                        #Test parameters
                        #Calculate predicted Q vals
                        #Save best angle and torque
                    
                best_qs.append(best_q_val)

                        
                if (system.cur_vx**2 + system.cur_vy**2)**.5 < 1.0:
                    best_torque_multiplier = 5
                #print best_angle_multplier
                #print best_torque_multiplier
                if uniform(0.0,1.0) < epsilon: 
                    
                  # print best_angle_multplier
                   # print best_torque_multiplier
                    best_torque_multiplier = random.randint(int(-NUM_TORQUE_INCREMENTS), int(NUM_TORQUE_INCREMENTS))
                    best_angle_multplier = random.randint(int(-NUM_ANGLE_INCREMENTS), int(NUM_ANGLE_INCREMENTS))
                    
                    
                steering_angle+=(best_angle_multplier*STEERING_ANGLE_INCREMENT)

                if steering_angle > math.pi:
                    steering_angle -= math.pi*2
                if steering_angle < -math.pi:
                    steering_angle += math.pi*2


                q_values.append(qVal(weights, features, steering_angle, TORQUE_INCREMENT*best_torque_multiplier, TORQUE_INCREMENT*best_torque_multiplier, system, best_angle_multplier*STEERING_ANGLE_INCREMENT))

                rewards.append(reward(system))
                feature_evals.append(feature_evaluations(features, steering_angle, 
                                     TORQUE_INCREMENT*best_torque_multiplier, TORQUE_INCREMENT*best_torque_multiplier, system, best_angle_multplier*STEERING_ANGLE_INCREMENT))                    
                
                rounded_angle = round(steering_angle, 2)
                rounded_torque = round(best_torque_multiplier*TORQUE_INCREMENT, 2)

                system.tick_simulation(front_wheel_torque=rounded_torque,
                                               rear_wheel_torque=rounded_torque,
                                               steering_angle=rounded_angle)                       
                '''
                system.tick_simulation(front_wheel_torque=(best_torque_multiplier*TORQUE_INCREMENT), 
                                               rear_wheel_torque=(best_torque_multiplier*TORQUE_INCREMENT), 
                                               steering_angle=rounded_angle)                       
                '''
                                               

            #Use history and create new weights
            #Use old weight values and new points to
            #Modify our new weight values
            #Reassign old weights to new weights for persistence

        except SimulationError :
            print rewards[-1]
            
            
            #rewards.append(CRASH_PUNISHMENT)
            
            '''
            rewards.append(reward(system))
            rewards.append(reward(system))
            best_qs.append(CRASH_PUNISHMENT)
            q_values.append(CRASH_PUNISHMENT)
            '''
            last_reward = reward(system)
            rewards.append(last_reward)
            best_qs.append(last_reward)
            q_values.append(last_reward)
            #print "Rewards: ", rewards
            #print "Features: ", feature_evals
            #print "Q: ", q_values
            #print "MAX Q: ", best_qs
            print "Simulation Crash ", episode, " weights: " , weights
            weights = update_weights(weights, q_values, best_qs, rewards, feature_evals)
            '''
            if episode==SIMULATION_MAX_TIME-1:
                system.plot_history()
            
            '''
                
            if rewards[-2] > best_reward:
                system.plot_history()
                best_reward = rewards[-2]

            elif(episode%20==0):
                system.plot_history()
 
            elif rewards[-2] > 500:
                system.plot_history()
        except WindowsError:
            print (system.cur_vx**2 + system.cur_vy**2)**.5
            
            print rewards[-1]
            #rewards.append(CRASH_PUNISHMENT)
            '''
            rewards.append(reward(system))
            best_qs.append(CRASH_PUNISHMENT)
            q_values.append(CRASH_PUNISHMENT)
            '''
            
            last_reward = reward(system)
            rewards.append(last_reward)
            best_qs.append(last_reward)
            q_values.append(last_reward)
            #print "Rewards: ", rewards
            #print "Features: ", feature_evals
            #print "Q: ", q_values
            #print "MAX Q: ", best_qs
            print "Simulation Low V", episode, " weights: " , weights
            weights = update_weights(weights, q_values, best_qs, rewards, feature_evals)
                 
            if rewards[-2] > best_reward:
                system.plot_history()
                best_reward = rewards[-2]
            elif(episode%20==0):
                system.plot_history()
            elif rewards[-2] > 500:
                system.plot_history()
            
 
                '''
            if(episode%1==0):
                system.plot_history()
                
                '''

def update_weights(weights, q_values, best_qs, rewards, feature_evals):

    new_weights= []
    for i in range(len(weights)):
        new_weights.append(weights[i])
    for i in range(len(weights)):
        for j in range(len(q_values)-1):
            new_weights[i] = weights[i] + LEARNING_RATE*(rewards[j] + DISCOUNT_FACTOR*best_qs[j+1] - q_values[j])*feature_evals[j][i]
            #new_weights[i] = weights[i] + LEARNING_RATE*(rewards[j] + DISCOUNT_FACTOR*q_values[j+1] - q_values[j])*feature_evals[j][i]

    '''
    weights_double = []
    for i in range(len(weights)):
        weights_double.append(weights[i])
    for i in range(len(weights)):
        for j in range(len(q_values)-1):
            weights_double[i] = weights[i] + LEARNING_RATE*(rewards[-j-1] + DISCOUNT_FACTOR*best_qs[-j-2] - q_values[-j-1])*feature_evals[-j-1][i]
    
    for i in range(len(weights)):
        new_weights[i] = round(new_weights[i], 6)
    '''
   
    return new_weights
                              
def main():
    #print dist_from_inner_wall(500*math.cos(0.5*math.pi),300*math.sin(0.5*math.pi))
    #print math.atan2(350,0)
    '''
    features = [f0_constant, f1_steering_angle, f2_fwt, f3_rwt, f4_vx, f5_vy, f6_high_v_tangential, f7_distance, f8_centerness]
    angle = -math.pi/4
    torque = 20
    tx = 371
    ty = 230
    tvx = .707
    tvy = -.707
    
    for i in range (len(features)):
        print features[i](angle, torque, torque, tx, ty, tvx, tvy)
    '''
    oursim()
 
if __name__ == "__main__":
    main()

