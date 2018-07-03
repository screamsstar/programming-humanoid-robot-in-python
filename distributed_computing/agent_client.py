'''In this file you need to implement remote procedure call (RPC) client

* The agent_server.py has to be implemented first (at least one function is implemented and exported)
* Please implement functions in ClientAgent first, which should request remote call directly
* The PostHandler can be implement in the last step, it provides non-blocking functions, e.g. agent.post.execute_keyframes
 * Hints: [threading](https://docs.python.org/2/library/threading.html) may be needed for monitoring if the task is done
'''

import weakref
from xmlrpc.client import ServerProxy
from threading import Thread
from numpy import zeros
# Testing
from numpy import identity
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'joint_control'))
from keyframes import hello


class PostHandler(object):
    '''the post hander wraps function to be excuted in paralle
    '''
    def __init__(self, obj):
        self.proxy = weakref.proxy(obj)

    def execute_keyframes(self, keyframes):
        '''non-blocking call of ClientAgent.execute_keyframes'''
        # YOUR CODE HERE
        t = Thread(target=self.proxy.execute_keyframes, args=[keyframes])
        t.start()

    def set_transform(self, effector_name, transform):
        '''non-blocking call of ClientAgent.set_transform'''
        # YOUR CODE HERE
        t = Thread(target=self.proxy.set_transform, args=[effector_name, transform])
        t.start()


class ClientAgent(object):
    '''ClientAgent request RPC service from remote server
    '''
    # YOUR CODE HERE
    def __init__(self):
        self.post = PostHandler(self)
        self.np_marshall = NumpyMarshall()
        self.rpc_proxy = ServerProxy("http://localhost:8000/")
    
    def get_angle(self, joint_name):
        '''get sensor value of given joint'''
        # YOUR CODE HERE
        return self.rpc_proxy.get_angle(joint_name)
    
    def set_angle(self, joint_name, angle):
        '''set target angle of joint for PID controller
        '''
        # YOUR CODE HERE
        self.rpc_proxy.set_angle(joint_name, angle)
        return

    def get_posture(self):
        '''return current posture of robot'''
        # YOUR CODE HERE
        return self.rpc_proxy.get_posture()

    def execute_keyframes(self, keyframes):
        '''excute keyframes, note this function is blocking call,
        e.g. return until keyframes are executed
        '''
        # YOUR CODE HERE
        self.rpc_proxy.execute_keyframes(keyframes)
        return

    def get_transform(self, name):
        '''get transform with given name
        '''
        # YOUR CODE HERE
        return self.np_marshall.unmarshall(self.rpc_proxy.get_transform(name))

    def set_transform(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''
        # YOUR CODE HERE
        self.rpc_proxy.set_transform(effector_name, self.np_marshall.marshall(transform))
        return


class NumpyMarshall(object):

    def marshall(self, transform):
        value_list = []
        for i in range(4):
            for j in range(4):
                value_list.append(transform[i, j].item())

        return value_list

    def unmarshall(self, value_list):
        transform = zeros([4, 4])
        index = 0
        for i in range(4):
            for j in range(4):
                transform[i, j] = value_list[index]
                index += 1

        return transform


if __name__ == '__main__':
    agent = ClientAgent()
    # TEST CODE HERE
    # agent.execute_keyframes(keyframes=hello())
    print(agent.get_transform("RAnkleRoll"))
    T = identity(4)
    T[0, 0] = 0
    T[0, 2] = 1
    T[2, 0] = 0
    T[0, 2] = 1
    T[0, -1] = 200
    T[1, -1] = -60
    T[2, -1] = -20
    agent.set_transform("RLeg", T)
    print(agent.get_transform("RAnkleRoll"))
    print("Done")

