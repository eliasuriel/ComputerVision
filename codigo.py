#!/usr/bin/env python
import rospy
import pinocchio
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState

class PinocchioNode:
    def __init__(self):
        rospy.init_node('pinocchio_node')

        urdf_filename = '/home/humanoid/reemc_public_ws/src/whole_body_state_msgs/urdf/reemc_full_ft_hey5.urdf'
        self.free_flyer = pinocchio.JointModelFreeFlyer()

        self.model = pinocchio.buildModelFromUrdf(urdf_filename, self.free_flyer)
        print('Model name: ' + self.model.name)

        self.data = self.model.createData()
        self.q = pinocchio.neutral(self.model)
        print('q: %s' % self.q.T)

        pinocchio.forwardKinematics(self.model, self.data, self.q)

        for name, oMi in zip(self.model.names, self.data.oMi):
            print(("{:<24} : {:.3f} {:.3f} {:.3f}".format(name, *oMi.translation.T)))

        rospy.Subscriber("/floating_base_pose_simulated", Odometry, self.baseCallback)
        rospy.Subscriber("/joint_states", JointState, self.jointCallback)

        rospy.spin()

    def baseCallback(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        self.q[0] = position.x
        self.q[1] = position.y
        self.q[2] = position.z
        self.q[3] = orientation.x
        self.q[4] = orientation.y
        self.q[5] = orientation.z
        self.q[6] = orientation.w

    def jointCallback(self, msg):
        joint_values = msg.position
        num_joints = len(joint_values)
        self.q[7:7+num_joints] = joint_values

if __name__ == '__main__':
    pinocchio_node = PinocchioNode()
