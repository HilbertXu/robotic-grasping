import sys

from numpy.ma.core import count
sys.path.append("/home/hilbertxu/air_ws/robotic-grasping")

import cv2
import torch
import rospy
import numpy as np
import message_filters
from copy import deepcopy
from datetime import datetime
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image

from inference.post_process import post_process_output
from utils.visualisation.plot import plot_results


class GRConvNetROS:
    def __init__(self) -> None:

        self.count = 0

        self.bridge = CvBridge()
        self.net = torch.load("/home/hilbertxu/air_ws/NextageManipulation/src/Python/grasping/network/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98").cuda().eval()

        self.count = 0


        rospy.init_node("grconvnet_ros")

        rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/camera/depth/image_rect_raw', Image)
        ts = message_filters.ApproximateTimeSynchronizer((rgb_sub, depth_sub), 10, 1, allow_headerless=True)
        ts.registerCallback(self.imageCallback)

        rospy.spin()
    

    def preprocess(self, rgb_msg, depth_msg):
        # Convert RGB message to RGB image
        cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        (rows,cols,channels) = cv_image.shape
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        

        # Convert depth message to Depth image
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        shape = depth.shape
        rgb = cv2.resize(rgb, (shape[1], shape[0]))

        # Crop images
        # print(rgb.shape, depth.shape)
        center = (shape[0]/2, shape[1]/2)
        x = int(center[1] - 360/2)
        y = int(center[0] - 360/2)
        rgb = rgb[y:y+360, x-55:x+305]
        depth = depth[y:y+360, x:x+360]

        cv2.imwrite("/home/hilbertxu/dataset/tubes/mix/rgb_{:0>3d}.png".format(self.count), cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        cv2.imwrite("/home/hilbertxu/dataset/tubes/mix/depth_{:0>3d}.png".format(self.count), depth)
        self.count += 1

        # Normalise images
        rgb_norm = deepcopy(rgb).astype(np.float32) / 255.0
        rgb_norm -= rgb_norm.mean()
        rgb_norm = cv2.resize(rgb_norm, (224,224))

        # # max_val = np.max(depth)
        depth_norm = deepcopy(depth)
        depth_norm = depth_norm / np.max(depth_norm)
        depth_norm = np.clip((depth_norm - depth_norm.mean()), -1, 1) 
        depth_norm = cv2.resize(depth_norm, (224,224))
        depth_norm = np.expand_dims(depth_norm, axis=-1)

        rgbd = np.expand_dims(np.concatenate([rgb_norm, depth_norm], axis=-1), axis=0)
        rgbd_tensor = torch.from_numpy(rgbd).permute(0,3,1,2).float().cuda()

        return rgbd_tensor, rgb, depth



    def imageCallback(self, rgb_msg, depth_msg):
        inp_tensor, rgb, depth = self.preprocess(rgb_msg, depth_msg)

        # with torch.no_grad():
        #     pred = self.net.predict(inp_tensor)
        #     q_img, ang_img, width_img = post_process_output(pred['pos'],
        #                                                     pred['cos'],
        #                                                     pred['sin'],
        #                                                     pred['width'])
        # fig = plt.figure(figsize=(10, 10))                                     
        # # img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # plot = plot_results(fig,
        #                     rgb_img=rgb,
        #                     grasp_q_img=q_img,
        #                     grasp_angle_img=ang_img,
        #                     depth_img=depth*255,
        #                     no_grasps=5,
        #                     grasp_width_img=width_img)
        
        # print("Save results as picture")
        # time = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        # save_name = '/home/hilbertxu/air_ws/robotic-grasping/figures/{}-results-small-30'.format(time)
        # plot.savefig(save_name + '.png')
        # #plot.clf()
        # print("Finish")



if __name__ == "__main__":
    test = GRConvNetROS()



