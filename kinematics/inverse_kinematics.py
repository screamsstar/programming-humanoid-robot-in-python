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
from autograd.numpy.linalg import grad_norm
from autograd.numpy import dot, identity, matrix, sqrt, arctan2, arcsin
import autograd.numpy as np
from autograd import grad
from collections import deque



class InverseKinematicsAgent(ForwardKinematicsAgent):
    def inverse_kinematics(self, effector_name, transform):
        '''solve the inverse kinematics

        :param str effector_name: name of end effector, e.g. LLeg, RLeg
        :param transform: 4x4 transform matrix
        :return: list of joint angles
        '''

        target = self.from_transform(transform)
        thetas = []
        for i, joint in enumerate(self.chains[effector_name]):
            thetas.append(self.perception.joint[joint])

        last_error = deque(np.zeros(10), maxlen=10)

        print("Starting numerical calculation (takes some time...)")

        if self.inverse_debug:
            print("Target Vector:" + str(target) + "\n")
            print("Start angles:" + str(thetas) + "\n")
            print("Error\t\t\t|\t\tAngles")
            print("------------------------------------------------------------")

        while True:
            error = self.error_func(effector_name, thetas, target)

            if self.inverse_debug:
                print(str(error) + "\t\t" + str(thetas))

            if error+0.1 > last_error[0] > error-0.1:
                print("Best possible solution found with an error of: " + str(error))
                break

            changes_list = []
            changes_list.extend(thetas)
            for i, joint in enumerate(self.chains[effector_name]):
                func = lambda t: self.error_func_one_theta(effector_name, thetas, target, i, t)
                func_grad = grad(func)

                d = func_grad(thetas[i])
                changes_list[i] = d

            summed_d = 0
            for i in changes_list:
                summed_d += abs(i)

            for i, joint in enumerate(self.chains[effector_name]):
                thetas[i] -= (changes_list[i] / summed_d) * 1e-2

            last_error.append(error)

        result = self.forward_kinematics_2(effector_name, thetas)

        if self.inverse_debug:
            print("Done! Thetas:" + str(thetas))
            print("Resulting transform (theoretical):")
            print(result)

        return thetas

    def forward_kinematics_2(self, effector_name, thetas):
        T = identity(4)
        for i, joint in enumerate(self.chains[effector_name]):
            angle = thetas[i]
            Tl = self.local_trans(joint, angle)
            T = dot(T, Tl)
        return T

    def error_func_one_theta(self, effector_name, thetas, target, theta_index, value):
        theta_new = []
        theta_new.extend(thetas)
        theta_new[theta_index] = value
        return self.error_func(effector_name, theta_new, target)

    def error_func(self, effector_name, thetas, target):
        Ts = self.forward_kinematics_2(effector_name, thetas)
        Te = self.from_transform(Ts)
        e = target - Te
        return np.sum(e * e)

    def from_transform(self, transform):
        # euler angles
        # xangle = arctan2(transform[2, 1], transform[2, 2])
        # yangle = - arcsin(transform[2, 0])
        # zangle = arctan2(transform[1, 0], transform[0, 0])
        # TODO arctan or arcsin or arccos seem to not work with autograd!!

        # this is a simplified substitute for the angles
        # --> distance between the corresponding vector and the standard axis-vector and that squared
        #  (easier but not as good--> orientation is only kinda working)
        xangle = (transform[0, 0]-1)**2 + transform[0, 1]**2 + transform[0, 2]**2
        yangle = transform[1, 0]**2 + (transform[1, 1]-1)**2 + transform[1, 2]**2
        zangle = transform[2, 0]**2 + transform[2, 1]**2 + (transform[2, 2]-1)**2

        return np.array([transform[0, 3], transform[1, 3], transform[2, 3], xangle, yangle, zangle])


    def set_transforms(self, effector_name, transform):
        '''
        solve the inverse kinematics and control joints use the results
        '''

        angles = self.inverse_kinematics(effector_name, transform)

        names = []
        times = []
        keys = []
        for chain_name in self.chains.keys():
            if chain_name != effector_name:
                for joint_name in self.chains[chain_name]:
                    names.append(joint_name)
                    times.append([5.0, 8.0])
                    keys.append([[0, [1, 1.00000, 0.00000], [1, -1.00000, 0.00000]], [0, [3, 0.00000, 0.00000], [3, 0.00000, 0.00000]]])

        for i, joint_name in enumerate(self.chains[effector_name]):
            names.append(joint_name)
            times.append([5.0, 8.0])
            keys.append([[angles[i], [1, -1.00000, 0.00000], [1, -1.00000, 0.00000]], [angles[i], [3, 0.00000, 0.00000], [3, 0.00000, 0.00000]]])

        self.keyframes = (names, times, keys)


if __name__ == '__main__':
    agent = InverseKinematicsAgent()
    # test inverse kinematics
    T = identity(4)
    T[0, 0] = 0
    T[0, 2] = 1
    T[2, 0] = 0
    T[0, 2] = 1
    T[0, -1] = 200
    T[1, -1] = -60
    T[2, -1] = -20
    print("Target transform:" + str(T) + "\n")
    agent.set_transforms('RLeg', T)
    agent.run()
