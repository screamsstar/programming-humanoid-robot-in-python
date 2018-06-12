'''In this exercise you need to implement forward kinematics for NAO robot

* Tasks:
    1. complete the kinematics chain definition (self.chains in class ForwardKinematicsAgent)
       The documentation from Aldebaran is here:
       http://doc.aldebaran.com/2-1/family/robots/bodyparts.html#effector-chain
    2. implement the calculation of local transformation for one joint in function
       ForwardKinematicsAgent.local_trans. The necessary documentation are:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    3. complete function ForwardKinematicsAgent.forward_kinematics, save the transforms of all body parts in torso
       coordinate into self.transforms of class ForwardKinematicsAgent

* Hints:
    the local_trans has to consider different joint axes and link parameters for different joints
'''

# add PYTHONPATH
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'joint_control'))

from autograd.numpy import cos, sin, matrix, identity, dot, sqrt

from angle_interpolation import AngleInterpolationAgent


class ForwardKinematicsAgent(AngleInterpolationAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(ForwardKinematicsAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.transforms = {n: identity(4) for n in self.joint_names}

        # chains defines the name of chain and joints of the chain
        self.chains = {'Head': ['HeadYaw', 'HeadPitch'],
                       'LArm': ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll'],
                       'RArm': ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll'],
                       'LLeg': ['LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll'],
                       'RLeg': ['RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll']
                       }
        self.yaw_rotators=['HeadYaw','LElbowRoll', 'RElbowRoll', 'LShoulderRoll', 'RShoulderRoll'] # this is so weird
        self.roll_rotators=['LHipRoll', 'LAnkleRoll', 'RHipRoll', 'RAnkleRoll', 'LElbowYaw', 'RElbowYaw']
        self.pitch_rotators = ['HeadPitch', 'LShoulderPitch', 'RShoulderPitch','LHipPitch', 'LKneePitch', 'LAnklePitch', 'RHipPitch', 'RKneePitch', 'RAnklePitch']

        # great task... this took way to long
        self.distances = {# Head
                        'HeadYaw': [0, 0, 126.5],
                        'HeadPitch': [0, 0, 0],

                        # Left arm
                        'LShoulderPitch': [0, 98, 100],
                        'LShoulderRoll': [0, 0, 0],
                        'LElbowYaw': [105, 15, 0],
                        'LElbowRoll': [0, 0, 0],

                        # Right arm
                        'RShoulderPitch': [0, -98, 100],
                        'RShoulderRoll': [0, 0, 0],
                        'RElbowYaw': [105, -15, 0],
                        'RElbowRoll': [0, 0, 0],

                        # Left leg
                        'LHipYawPitch': [0, 50, -85],
                        'LHipRoll': [0, 0, 0],
                        'LHipPitch': [0, 0, 0],
                        'LKneePitch': [0, 0, -100],
                        'LAnklePitch': [0, 0, -102.9],
                        'LAnkleRoll': [0, 0, 0],

                        # Right leg
                        'RHipYawPitch': [0, -50, -85],
                        'RHipRoll': [0, 0, 0],
                        'RHipPitch': [0, 0, 0],
                        'RKneePitch': [0, 0, -100],
                        'RAnklePitch': [0, 0, -102.9],
                        'RAnkleRoll': [0, 0, 0],
                        }
        self.forward_kinematics_debug = True
        self.forward_kinematics_debug_single_transforms = False
        self.inverse_debug = True

    def think(self, perception):
        self.forward_kinematics(perception.joint)
        return super(ForwardKinematicsAgent, self).think(perception)

    def local_trans(self, joint_name, joint_angle):
        '''calculate local transformation of one joint

        :param str joint_name: the name of joint
        :param float joint_angle: the angle of joint in radians
        :return: transformation
        :rtype: 4x4 matrix
        '''
        # determine right rotation_matrix values
        normal_value_for_2 = sqrt(0.5)

        if self.yaw_rotators.__contains__(joint_name):
            ux, uy, uz = 0, 0, 1
        elif self.roll_rotators.__contains__(joint_name):
            ux, uy, uz = 1, 0, 0
        elif self.pitch_rotators.__contains__(joint_name):
            ux, uy, uz = 0, 1, 0
        elif joint_name == 'LHipYawPitch':
            ux, uy, uz = 0, normal_value_for_2, -normal_value_for_2
        elif joint_name == 'RHipYawPitch':
            ux, uy, uz = 0, -normal_value_for_2, -normal_value_for_2
        else:
            print "weird jointname"
            ux, uy, uz = 0, 0, 0

        cost = cos(joint_angle)
        sint = sin(joint_angle)
        distance_lengths = self.distances[joint_name]
        # rotation matrix in 3d around any normalized axis
        T = matrix([[cost+ux**2*(1-cost), ux*uy*(1-cost)-uz*sint, ux*uz*(1-cost)+uy*sint, distance_lengths[0]],
                    [uy*ux*(1-cost)+uz*sint, cost+uy**2*(1-cost), uy*uz*(1-cost)-ux*sint, distance_lengths[1]],
                    [uz*ux*(1-cost)-uy*sint, uz*uy*(1-cost)+ux*sint, cost+uz**2*(1-cost), distance_lengths[2]],
                    [0, 0, 0, 1]])

        return T

    def forward_kinematics(self, joints):
        '''forward kinematics

        :param joints: {joint_name: joint_angle}
        '''
        for chain_name, chain_joints in self.chains.iteritems():
            T = identity(4)
            for joint in chain_joints:
                angle = joints[joint]
                Tl = self.local_trans(joint, angle)
                if self.forward_kinematics_debug_single_transforms:
                    print(joint)
                    print(Tl)
                T = dot(T, Tl)
                self.transforms[joint] = T
            if self.forward_kinematics_debug:
                print(chain_name)
                print(T)



    def forward_kinematics_for_one_chain(self, chain_name, angles):
        chain = self.chains[chain_name]
        T = identity(4)
        for joint in chain:
            angle = angles[joint]
            Tl = self.local_trans(joint, angle)
            T = dot(T, Tl)
            self.transforms[joint] = T


if __name__ == '__main__':
    agent = ForwardKinematicsAgent()
    agent.run()
