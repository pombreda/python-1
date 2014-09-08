from exceptions import InvalidParamterException
import string
from math import pi

class LDMRSParams(object):
    """ parameter object converts between ROS and ldmrs parameter values and
        checks validity
    """
    deg2ticks = 32  # ticks per degree
    hz2tickfreq = 256 # tick freq has units 1/256 Hz

    # Device parameter limits
    #-------------------------------------
    # Scan frequencies
    # device uses 1/256 Hz values
    valid_tick_freqs = [3200,  6400,  12800]  # ticks per sec, corresponds to 12.5, 25, and 50 Hz
    valid_freqs = [12.5,  25,  50] #Hz

    # Start angle limits
    start_ticks_min = -1919
    start_angle_min = -59.96  # deg
    start_ticks_max = 1600
    start_angle_max = 50.0    # deg

    # End angle limits
    end_ticks_min = -1920
    end_angle_min = -60.0   # deg
    end_ticks_max = 1599
    end_angle_max = 49.96   # deg

    # Sync angle limits
    sync_ticks_min = -5760
    sync_angle_min = -180.0   # deg
    sync_ticks_max = 5759
    sync_angle_max = 179.96   # deg

    # set up parameter names. We assume these are in the node private namespace
    start_angle = 'start_angle'
    end_angle = 'end_angle'
    scan_frequency = 'scan_frequency'
    sync_angle_offset = 'sync_angle_offset'
    constant_angular_res = 'constant_angular_res'
    use_first_echo = 'use_first_echo'
    frame_id_prefix = 'frame_id_prefix'

    # mapping from LD-MRS parameter names to parameter index byte strings
    parameter_index = {"ip_address":['\x00\x10', "uint32"],
                           "tcp_port":['\x01\x10', "uint32"],  # use hex notation
                           "subnet_mask":['\x02\x10', "uint32"],   # use hex notation
                           "standard_gateway":['\x03\x10', "uint32"],
                           "data_output_flag":['\x12\x10', "uint16"],
                           "start_angle":['\x00\x11', "int16"],
                           "end_angle":['\x01\x11', "int16"],
                           "scan_frequency":['\x02\x11', "uint16"],
                           "sync_angle_offset":['\x03\x11', "int16"],  # actually int14
                           "constant_angular_res":['\x04\x11', "uint16"],
                           "angle_ticks_per_rotation":['\x05\x11', "uint16"]}

    ros_params = {}
    ldmrs_params = {}


    def update_config(self,  config,  level,  fix_errors=False):
        """ Update parameters from passed in config object
            @param config: dictionary of parameter value pairs
            @type config: dict {ros_param_string:value }
            @param level: runlevel (currently unused)
            @type level: int
            @param fix_errors: fix invalid input values to be within limits
            @type fix_errors: bool
            @return: (possibly updated) config
            @rtype: dict
        """
        # note save values we see an apply_changes True parameter
        config = self._check_values(config,  fix_errors)
        self.ros_params = config.copy()
        self._ros_to_ldmrs()
        return config


    def _ros_to_ldmrs(self):
        ''' Convert ROS parameter values to LDMRS appropriate values
        '''
        # copy over the keys and values for everything and convert to ldmrs 
        # values where necessary
        for k, v in self.ros_params.iteritems():
            self.ldmrs_params[k] = v

        param_name = self.start_angle
        self.ldmrs_params[param_name] = int(round(self.ros_params[param_name]*self.deg2ticks))

        param_name = self.end_angle
        self.ldmrs_params[param_name] = int(round(self.ros_params[param_name]*self.deg2ticks))

        param_name = self.sync_angle_offset
        self.ldmrs_params[param_name] = int(round(self.ros_params[param_name]*self.deg2ticks))

        param_name = self.constant_angular_res
        self.ldmrs_params[param_name] = int(self.ros_params[param_name])

        param_name = self.scan_frequency
        self.ldmrs_params[param_name] = self.valid_tick_freqs[self.ros_params[param_name]]


    def _check_values(self,  config,  fix_errors=False):
        ''' Check that the parameter values are valid,
            throw an exception if an invalid combination is detected.
            Clamping is applied to out of range values and
            invalid configurations are corrected
            Collate all parameter setting errors to facilitate faster debugging
            @param config: configuration dictionary to check for consistency
            @type config: dict {ros_param_string:value}
            @param fix_errors: True: adjust the parameters to stay within appropriate bounds,
                False: raise an exception on error
            @type fix_errors: bool
            @return: config (possibly updated)
            @rtype: dict
        '''

        # flag exceptions by adding messages to the list
        # no exception is raised if the list is empty
        exception_msg = []  # list of exception messages

        for param_name, value in config.iteritems():

            # handle value < minimum allowed
            min = None if not param_name in ldmrsConfig.min else ldmrsConfig.min[param_name]
            if min is not None and min != '' and value < min:
                msg = str(err_num) + ": Parameter '%s' value %s exceeds minimum allowed value of %s. "\
                        %(param_name, str(value), str(min))
                if fix_errors:
                    msg += "Using the minimum allowed value of %s"%str(min)
                    rospy.logwarn(msg)
                    config[param_name] = min
                else:
                    rospy.logerr(msg)
                    raise InvalidParameterException(msg)
                    exception_msg.append(msg)

            # handle value > maximum allowed
            max = None if not param_name in ldmrsConfig.max else ldmrsConfig.max[param_name]
            if max is not None and max != '' and value > max:
                msg = str(err_num) + ": Parameter '%s' value %s exceeds maximum allowed value of %s. "\
                        %(param_name, str(value), str(max))
                if fix_errors:
                    msg += "Using the maximum allowed value of %s"%str(max)
                    rospy.logwarn(msg)
                    config[param_name] = max
                else:
                    rospy.logerr(msg)
                    raise InvalidParameterException(msg)

        # check that the angular_res_type is appropriate, given the scan frequency
        scan_freq = config[self.scan_frequency]
        res_type = config[self.constant_angular_res]
        if  not res_type and (scan_freq != 0):
            rospy.logerr("Focused angular resolution mode is only available when scan frequency is 12.5Hz but " \
                        "current frequency is " + str(self.valid_freqs[scan_freq]))
            if fix_errors:
                rospy.logwarn("Changing angular resolution type to constant and keeping scan_frequency at %d"%self.valid_freqs[scan_freq])
                config[self.constant_angular_res] = True
            else:
                rospy.logerr("Either change the scan frequency to 12.5Hz (value 0) or set the constant_angular_res parameter")
                raise InvalidParameterException('Illegal scan_freqency / constant_angular_res parameter combination')

        # check that start angle > end_angle
        if config["start_angle"] <= config["end_angle"]:
            rospy.logerr("Start angle must be greater than end angle.")
            if fix_errors:
                rospy.logwarn("Using default values of start_angle=%s, end_angle=%s"
                              %(ldmrsConfig.defaults[self.start_angle], ldmrsConfig.defaults[self.end_angle]))
                config[self.start_angle] = ldmrsConfig.defaults[self.start_angle]
                config[self.end_angle] = ldmrsConfig.defaults[self.end_angle]
            else:
                raise InvalidParamterException("End angle parameter is greater than start angle.")

        return config

    def get_ldmrs_param(self, param_name):
        ''' Get the value of ldmrs parameter <param_name> (in LDMRS format)
            @param param_name: ldmrs parameter name to get the value of 
            @type param_name: string
            @return: ldmrs parameter value for parmeter param_name
            @rtype: type of parameter param_name
        '''
        return None if param_name not in self.ldmrs_params else self.ldmrs_params[param_name]

    def get_ros_param(self, param_name):
        ''' Get the value of ROS parameter <param_name>
            @param param_name: ROS parameter name to get the value of 
            @type param_name: string
            @return: ROS parameter value for parmeter param_name
            @rtype: type of parameter param_name
        '''
        return None if param_name not in self.ros_params else self.ros_params[param_name]








