'''In this file you need to implement remote procedure call (RPC) server

* There are different RPC libraries for python, such as xmlrpclib, json-rpc. You are free to choose.
* The following functions have to be implemented and exported:
 * get_angle
 * set_angle
 * get_posture
 * execute_keyframes
 * get_transform
 * set_transform
* You can test RPC server with ipython before implementing agent_client.py
'''

# add PYTHONPATH
import os
import sys
from SimpleXMLRPCServer import SimpleXMLRPCServer
from threading import Thread
from agent_client import NumpyMarshall
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'kinematics'))

from inverse_kinematics import InverseKinematicsAgent


class ServerAgent(InverseKinematicsAgent):
    '''ServerAgent provides RPC service
    '''
    # YOUR CODE HERE

    def __init__(self):
        super(ServerAgent, self).__init__()
        self.np_marshall = NumpyMarshall()
        self.rpc_server = SimpleXMLRPCServer(("localhost", 8000))
        self.rpc_server.register_function(self.get_angle)
        self.rpc_server.register_function(self.set_angle)
        self.rpc_server.register_function(self.get_posture)
        self.rpc_server.register_function(self.execute_keyframes)
        self.rpc_server.register_function(self.get_transform)
        self.rpc_server.register_function(self.set_transform)
        print("All functions registered")
        self.server_thread = Thread(target=self.serve)
        self.server_thread.start()

    def serve(self):
        print("Server started")
        self.rpc_server.serve_forever()
    
    def get_angle(self, joint_name):
        '''get sensor value of given joint'''
        # YOUR CODE HERE
        return self.perception.joint[joint_name]
    
    def set_angle(self, joint_name, angle):
        '''set target angle of joint for PID controller
        '''
        # YOUR CODE HERE
        self.target_joints[joint_name] = angle
        return True

    def get_posture(self):
        '''return current posture of robot'''
        # YOUR CODE HERE
        return self.posture

    def execute_keyframes(self, keyframes):
        '''excute keyframes, note this function is blocking call,
        e.g. return until keyframes are executed
        '''
        # YOUR CODE HERE
        self.reset_time()
        self.keyframes = keyframes
        self.keyframe_running = True
        while self.keyframe_running:
            pass

        print("Done running keyframe")
        return True

    def get_transform(self, name):
        '''get transform with given name
        '''
        # YOUR CODE HERE
        return self.np_marshall.marshall(self.transforms[name])

    def set_transform(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''
        # YOUR CODE HERE
        self.set_transforms(effector_name, self.np_marshall.unmarshall(transform))
        self.execute_keyframes(keyframes=self.keyframes)
        return True


if __name__ == '__main__':
    agent = ServerAgent()
    agent.run()

