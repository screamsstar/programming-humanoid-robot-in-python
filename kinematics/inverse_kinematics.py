'''In this exercise you need to implement inverse kinematics for NAO's legs

* Tasks:
    1. solve inverse kinematics for NAO's legs by using analytical or numerical method.
       You may need documentation of NAO's leg:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    2. use the results of inverse kinematics to control NAO's legs (in InverseKinematicsAgent.set_transforms)
       and test your inverse kinematics implementation.
'''


from forward_kinematics import ForwardKinematicsAgent
from numpy.matlib import identity, zeros, matrix
from math import cos, sin, pi, atan2, sqrt
from numpy.linalg import norm, inv
from numpy import dot, arccos



class InverseKinematicsAgent(ForwardKinematicsAgent):
    def inverse_kinematics(self, effector_name, transform):
        '''solve the inverse kinematics

        :param str effector_name: name of end effector, e.g. LLeg, RLeg
        :param transform: 4x4 transform matrix
        :return: list of joint angles
        '''
        joint_angles = []
        # YOUR CODE HERE
        if effector_name=="Lleg":
            hipyawpitch="LHipYawPitch"
        else:
            hipyawpitch="RHipYawPitch"

        knee_pitch_length = abs(self.distances["LKneePitch"][2])
        ankle_pitch_length = abs(self.distances["LAnklePitch"][2])

        # correct simple distances
        T = zeros((4, 4))
        for i in range(3):
            T[i, 3] = self.distances[hipyawpitch][i]
        hip_to_endeffector_transform = transform + T

        hip_to_endeffector_vector = hip_to_endeffector_transform[0:3, 3]
        hip_to_ankle_length = norm(hip_to_endeffector_vector)

        Rx_pi4 = [[1, 0, 0], [0, cos(pi/4), -sin(pi/4)], [0, sin(pi/4), cos(pi/4)]]
        hip_to_endeffector_rotated_vector = dot(Rx_pi4, hip_to_endeffector_vector)

        # compute theta 4, 5 and 6 with these values
        theta6 = atan2(hip_to_endeffector_rotated_vector[1], hip_to_endeffector_rotated_vector[2])
        theta5 = arccos((norm(hip_to_endeffector_vector)**2 + ankle_pitch_length**2 - knee_pitch_length**2) / (2*ankle_pitch_length*norm(hip_to_endeffector_vector))) \
            + atan2(hip_to_endeffector_rotated_vector[0], norm(hip_to_endeffector_vector))
        theta4 = pi - arccos((ankle_pitch_length**2 + knee_pitch_length**2 - norm(hip_to_endeffector_vector)**2)/(2 * ankle_pitch_length * knee_pitch_length))

        # build transform for these (left and right are the same for those)
        theta4_to_6_transform = self.local_trans("LKneePitch", theta4).dot(self.local_trans("LAnklePitch", theta5)).dot(self.local_trans("LAnkleRoll", theta6)).dot(inv(hip_to_endeffector_transform))

        theta1 = atan2(sqrt(2)*theta4_to_6_transform[1,0], theta4_to_6_transform[1,2])

        normal_value_for_2 = sqrt(0.5)
        ux, uy, uz = 0, normal_value_for_2, normal_value_for_2
        c = cos(theta1)
        s = sin(theta1)
        rotation_matrix = matrix([[c + ux ** 2 * (1 - c), ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s],
                                  [uy * ux * (1 - c) + uz * s, c + uy ** 2 * (1 - c), uy * uz * (1 - c) - ux * s],
                                  [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz ** 2 * (1 - c)]])

        theta2 = atan2( - rotation_matrix[1, 2], rotation_matrix[1, 1])
        theta3 = atan2( - rotation_matrix[2, 0], - pi + rotation_matrix[0, 0])
        joint_angles = [theta1, theta2, theta3, theta4, theta5, theta6]
        return joint_angles

    def set_transforms(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''
        # YOUR CODE HERE
        self.keyframes = ([], [], [])  # the result joint angles have to fill in

        angles = self.inverse_kinematics(effector_name, transform)
        name = list()
        times = list()
        keys = list()

        for chain in self.chains:
            if chain == 'LLeg':
                for i, joint in enumerate(self.chains[chain]):
                    name.append(joint)
                    keys.append([[angles[i], [0., 0., 0.], [0., 0., 0.]]])
                    times.append([10.0])
            else:
                for joint in self.chains[chain]:
                    name.append(joint)
                    keys.append([[0, [0., 0., 0.], [0., 0., 0.]]])
                    times.append([1.0])

        self.keyframes = (name, times, keys)  # the result joint angles have to fill in


if __name__ == '__main__':
    agent = InverseKinematicsAgent()
    # test inverse kinematics
    T = identity(4)
    T[-1, 1] = 0.05
    T[-1, 2] = 0.26
    agent.set_transforms('LLeg', T)
    agent.run()
