#######################################
#   Work in progress: Do Not Review
#######################################


from cv_bridge import CvBridge
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
import message_filters as mf
import pdb

from sensor_msgs.msg import Image
from angel_msgs.msg import HandJointPosesUpdate


BRIDGE = CvBridge()
    

class SynchronizedBagParser(Node):

    def __init__(self):
        super().__init__(self.__class__.__name__)

        self._image_topic = self.declare_parameter("image_topic", "PVFrames").get_parameter_value().string_value
        self._hand_topic = self.declare_parameter("hand_pose_topic", "HandJointPoseData").get_parameter_value().string_value
        self._frames_per_det = self.declare_parameter("frames_per_det", 32.0).get_parameter_value().double_value

        log = self.get_logger()
        log.info(f"Image topic: {self._image_topic}")
        log.info(f"Hand topic: {self._hand_topic}")
        log.info(f"Frames per detection: {self._frames_per_det}")

        self.subscription_list = []
        # Image subscription
        self.subscription_list.append(mf.Subscriber(self, Image, self._image_topic))
        # Hand pose subscription
        self.subscription_list.append(mf.Subscriber(self, HandJointPosesUpdate, self._hand_topic))
        self.time_sync = mf.TimeSynchronizer(
            self.subscription_list,
            self._frames_per_det
        )
        self.time_sync.registerCallback(self.multimodal_listener_callback)

    def multimodal_listener_callback(self, image, hand_pose):
        log = self.get_logger()
        log.info("Got a synchronized data sample!")

        # Convert ROS img msg to CV2 image and add it to the frame stack
        rgb_image = BRIDGE.imgmsg_to_cv2(image, desired_encoding="rgb8")
        rgb_image_np = np.asarray(rgb_image)
        lhand, rhand = self.get_hand_pose_from_msg(hand_pose)

        self.save_in_h2o_format(rgb_image, lhand, rhand)

    def get_hand_pose_from_msg(self, msg):
        hand_joints = [{"joint": m.joint,
                        "position": [ m.pose.position.x,
                                      m.pose.position.y,
                                      m.pose.position.z]} 
                      for m in msg.joints]

        # Rejecting joints not in OpenPose hand skeleton format
        reject_joint_list = ['ThumbMetacarpalJoint', 
                            'IndexMetacarpal', 
                            'MiddleMetacarpal', 
                            'RingMetacarpal', 
                            'PinkyMetacarpal']
        joint_pos = []
        for j in hand_joints:
            if j["joint"] not in reject_joint_list:
                joint_pos.append(j["position"])
        joint_pos = np.array(joint_pos).flatten()

        if msg.hand == 'Right':
            rhand = joint_pos
            lhand = np.zeros_like(joint_pos)
        elif msg.hand == 'Left':
            lhand = joint_pos
            rhand = np.zeros_like(joint_pos)
        else:
            lhand = np.zeros_like(joint_pos)
            rhand = np.zeros_like(joint_pos)

        return lhand, rhand

    def save_in_h2o_format(self, img, lh_pose, rh_pose):
        pass


def main():
    rclpy.init()

    sync_bag_parser = SynchronizedBagParser()

    rclpy.spin(sync_bag_parser)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    sync_bag_parser.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
