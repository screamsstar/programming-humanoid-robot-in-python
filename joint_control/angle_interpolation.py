'''In this exercise you need to implement an angle interploation function which makes NAO executes keyframe motion

* Tasks:
    1. complete the code in `AngleInterpolationAgent.angle_interpolation`,
       you are free to use splines interploation or Bezier interploation,
       but the keyframes provided are for Bezier curves, you can simply ignore some data for splines interploation,
       please refer data format below for details.
    2. try different keyframes from `keyframes` folder

* Keyframe data format:
    keyframe := (names, times, keys)
    names := [str, ...]  # list of joint names
    times := [[float, float, ...], [float, float, ...], ...]
    # times is a matrix of floats: Each line corresponding to a joint, and column element to a key.
    keys := [[float, [int, float, float], [int, float, float]], ...]
    # keys is a list of angles in radians or an array of arrays each containing [float angle, Handle1, Handle2],
    # where Handle is [int InterpolationType, float dTime, float dAngle] describing the handle offsets relative
    # to the angle and time of the point. The first Bezier param describes the handle that controls the curve
    # preceding the point, the second describes the curve following the point.
'''


from pid import PIDAgent
from keyframes import hello, leftBackToStand, leftBellyToStand, rightBackToStand, rightBellyToStand, wipe_forehead


class AngleInterpolationAgent(PIDAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(AngleInterpolationAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.keyframes = ([], [], [])
        self.__start_time = -1
        self.__start_joints = -1

    def think(self, perception):
        target_joints = self.angle_interpolation(self.keyframes, perception)
        self.target_joints.update(target_joints)
        return super(AngleInterpolationAgent, self).think(perception)

    def angle_interpolation(self, keyframes, perception):
        target_joints = {}
        # YOUR CODE HERE

        if self.__start_time < 0:
            self.__start_time = perception.time
            self.__start_joints = perception.joint

        names, times, keys = keyframes

        current_time = perception.time - self.__start_time

        for name_index, joint_name in enumerate(names):

            if joint_name in self.joint_names:

                for time_index in range(len(times[name_index]) - 1):

                    if current_time < times[name_index][0]:
                        p0 = self.__start_joints[joint_name]
                        p1 = self.__start_joints[joint_name]
                        p2 = keys[name_index][0][0] + keys[name_index][0][1][2]
                        p3 = keys[name_index][0][0]

                        i = current_time / times[name_index][0]

                        target_joints[joint_name] = self.cubic_bezier(p0, p1, p2, p3, i)

                    elif times[name_index][time_index] < current_time < times[name_index][time_index + 1]:
                        p0 = keys[name_index][time_index][0]
                        p1 = keys[name_index][time_index][0] + keys[name_index][time_index][2][2]
                        p2 = keys[name_index][time_index + 1][0] + keys[name_index][time_index + 1][1][2]
                        p3 = keys[name_index][time_index + 1][0]

                        t_curr_keyframe = current_time - times[name_index][time_index]
                        t_abs_keyframe = times[name_index][time_index + 1] - times[name_index][time_index]
                        i = t_curr_keyframe / t_abs_keyframe

                        target_joints[joint_name] = self.cubic_bezier(p0, p1, p2, p3, i)

                    elif times[name_index][len(times[name_index]) - 1] < current_time:
                        target_joints[joint_name] = keys[name_index][len(times[name_index]) - 1][0]

        return target_joints

    def cubic_bezier(self, p0, p1, p2, p3, i):
        return (1 - i)**3 * p0 + 3 * (1 - i)**2 * i * p1 + 3 * (1 - i) * i**2 * p2 + i**3 * p3

    def reset_time(self):
        self.__start_time = -1


if __name__ == '__main__':
    agent = AngleInterpolationAgent()
    agent.keyframes = leftBackToStand()  # CHANGE DIFFERENT KEYFRAMES
    agent.run()
