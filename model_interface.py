"""
Defines a type that encapsulates the representation and simulation of a Bicycle
vehicle model.

Author: RR
"""
import os
import numpy as np
from ctypes import CDLL, c_double, pointer
from utils import SIM_RESOLUTION, ROOT_PATH, iterable_to_ctypes_array, \
     ctype_array_to_list


class VehicleModel:
    # model constants -- do not modify!
    NUM_INPUTS = 3
    NUM_OUTPUTS = 3
    NUM_TELEMETRY_CHANNELS = 8
    
    # pointer types for input, output, telemetry - do not modify!
    input_type = c_double * NUM_INPUTS
    output_type = c_double * NUM_OUTPUTS
    telemetry_type = c_double * NUM_TELEMETRY_CHANNELS
    
    def __init__(self):
        dll_path = os.path.join(ROOT_PATH, 'VehicleInterface.dll')
        model_path = os.path.join(ROOT_PATH, 'Bicycle.FMU')
        temp_path = os.path.join(ROOT_PATH, 'Temp')
        config_path = os.path.join(ROOT_PATH, 'config.json')
        self._dll = CDLL(dll_path)
        self._vehicle_model = self._dll.SimInstantiate(model_path,
                                                       temp_path,
                                                       config_path)
        self._dll.SimInitialize(self._vehicle_model)
        self._current_sim_time = -SIM_RESOLUTION
        
    def _pack_inputs(self,
                     front_wheel_torque,
                     rear_wheel_torque,
                     steering_angle):
        start_time = self._current_sim_time + SIM_RESOLUTION
        end_time = start_time + (10 * SIM_RESOLUTION)
        time = np.linspace(start=start_time, stop=end_time, num=10, endpoint=False)
        front_wheel_torque = front_wheel_torque * np.ones(len(time))
        rear_wheel_torque = rear_wheel_torque * np.ones(len(time))
        steering_angle = steering_angle * np.ones(len(time))
        
        packed_input = np.array([front_wheel_torque,
                                 steering_angle,
                                 rear_wheel_torque]).T
        return time, packed_input
    
    def simulate_inputs(self, 
                        front_wheel_torque, 
                        rear_wheel_torque, 
                        steering_angle):
        """ Simulates the response of the vehicle to the given inputs.

        The simulation is advanced by SIM_RESOLUTION seconds from its current
        state; the new lateral and longitudinal velocities of the vehicle are
        returned.
        
        Parameters:
            front_wheel_torque: a float representing the torque on the front
                                wheel. Positive value indicate acceleration,
                                negative value indicates braking.
            rear_wheel_torque: a float representing the torque on the rear
                               wheel. Positive value indicates acceleration,
                               negative value indicates braking.
            steering_angle: float representing the angle of the steering wheel.
                            in radians. Can be positive or negative.
                            
        Returns:
            A tuple representing the new (lateral, longitudinal) velocities of
            the vehicle.
        """                 
        # prepare inputs
        time, packed_input = self._pack_inputs(front_wheel_torque,
                                               rear_wheel_torque,
                                               steering_angle) 
        # allocate space for outputs
        output = iterable_to_ctypes_array(np.zeros(self.NUM_OUTPUTS),
                                          self.output_type)
        telemetry = iterable_to_ctypes_array(np.zeros(self.NUM_TELEMETRY_CHANNELS),
                                             self.telemetry_type)
        output_ptr = pointer(output)
        telemetry_ptr = pointer(telemetry)
        all_outputs = list()
        all_telemetry = list()
        
        # run simulation
        for i in range(len(time)):
            input_vector = iterable_to_ctypes_array(packed_input[i],
                                                    self.input_type)
            input_ptr = pointer(input_vector)
            self._dll.SimSetInput(self._vehicle_model, input_ptr,
                                  self.NUM_INPUTS)
            self._dll.SimDoStep(self._vehicle_model, c_double(time[i]),
                                c_double(SIM_RESOLUTION))
            self._dll.SimGetOutput(self._vehicle_model, output_ptr, 
                                   self.NUM_OUTPUTS)
            self._dll.SimGetTelemetry(self._vehicle_model, telemetry_ptr,
                                      self.NUM_TELEMETRY_CHANNELS)
            all_outputs.append(ctype_array_to_list(output))
            all_telemetry.append(ctype_array_to_list(telemetry))
            
        # return velocity at tics throughout simulation -- for now, telemetry
        # (i.e., tire data) and yaw rate are not being returned
        all_outputs = np.array(all_outputs)
        longitudinal_velocity = list(all_outputs[:, 0])
        lateral_velocity = list(all_outputs[:, 1])
        self._current_sim_time = time[-1]
        
        return lateral_velocity, longitudinal_velocity
