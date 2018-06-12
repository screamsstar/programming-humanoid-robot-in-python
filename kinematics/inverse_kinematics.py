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



class InverseKinematicsAgent(ForwardKinematicsAgent):
    def inverse_kinematics(self, effector_name, transform):
        '''solve the inverse kinematics

        :param str effector_name: name of end effector, e.g. LLeg, RLeg
        :param transform: 4x4 transform matrix
        :return: list of joint angles
        '''

        target = self.from_transform(transform)
        print("Target:" + str(target))
        thetas = []
        for i, joint in enumerate(self.chains[effector_name]):
            thetas.append(self.perception.joint[joint])

        print(thetas)

        while True:
            error = self.error_func(effector_name, thetas, target)

            print(str(error) + "\t" + str(thetas))

            if error < 1e-2:
                break
            changed_list = []
            changed_list.extend(thetas)
            for i, joint in enumerate(self.chains[effector_name]):
                func = lambda t: self.error_func_one_theta(effector_name, thetas, target, i, t)
                func_grad = grad(func)

                d = func_grad(thetas[i]) * 1e-5 * 0.5
                changed_list[i] -= d

            thetas = changed_list

        print("Done: " + str(thetas))

        result = self.forward_kinematics_2(effector_name, thetas)

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
        xangle, yangle, zangle = 0, 0, 0
        return np.array([transform[0, 3], transform[1, 3], transform[2, 3], xangle, yangle, zangle])


    def set_transforms(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''
        # YOUR CODE HERE
        self.keyframes = ([], [], [])  # the result joint angles have to fill in

        angles = self.inverse_kinematics(effector_name, transform)

        names = list()
        times = list()
        keys = list()
        for chain_name in self.chains.keys():
            if (chain_name == effector_name):
                i = 0
                for joint_name in self.chains[effector_name]:
                    print(joint_name)
                    names.append(joint_name)
                    times.append([1.0, 2.0])
                    keys.append([[angles[i], [3, 0.00000, 0.00000], [3, 0.00000, 0.00000]], [angles[i], [3, 0.00000, 0.00000], [3, 0.00000, 0.00000]]])
                    i = i + 1
            else:
                for joint_name in self.chains[chain_name]:
                    names.append(joint_name)
                    times.append([1.0, 2.0])
                    keys.append([[0, [3, 0.00000, 0.00000], [3, 0.00000, 0.00000]], [0, [3, 0.00000, 0.00000], [3, 0.00000, 0.00000]]])

        self.keyframes = (names, times, keys)
        print(self.keyframes)



if __name__ == '__main__':
    agent = InverseKinematicsAgent()
    # test inverse kinematics
    T = identity(4)
    T[0, -1] = 20
    T[1, -1] = -50
    T[2, -1] = -200
    print(T)
    print(agent.perception.joint)
    agent.set_transforms('LLeg', T)
    agent.run()
