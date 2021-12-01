from matplotlib.pyplot import axis
import numpy
import torch
from inference.models import get_network
from utils.dataset_processing import image
from inference.post_process import post_process_output

import cv2
import matplotlib.pyplot as plt 
import os
from datetime import datetime

from utils.visualisation.plot import plot_results

network = get_network('grconvnet3')
net = network(
    input_channels=4,
    dropout=True,
    prob=0.1,
    channel_size=32
)

import numpy as np
from graspnetAPI import GraspNet

def inpaint(img, missing_value=0):
    """
    Inpaint missing values in depth image.
    :param missing_value: Value to fill in teh depth image.
    """
    # cv2 inpainting doesn't handle the border properly
    # https://stackoverflow.com/questions/25974033/inpainting-depth-map-still-a-black-image-border
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    mask = (img == missing_value).astype(np.uint8)

    # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
    scale = np.abs(img).max()
    img = img.astype(np.float32) / scale  # Has to be float32, 64 not supported.
    img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)

    # Back to original size and value range.
    img = img[1:-1, 1:-1]
    img = img * scale

    return img

# g = GraspNet('/home/hilbertxu/dataset/graspnet', camera="realsense", split='train')
# rgb = g.loadBGR(sceneId=0, camera="realsense", annId=0)
# depth = g.loadDepth(sceneId=0, camera="realsense", annId=0).reshape(720,1280,1)
# max_val = np.max(depth)
# depth = depth * (255 / max_val)
# depth = np.clip((depth - depth.mean())/175, -1, 1)

# # center_crop:
# rgb = rgb[:, 0:720,:]
# depth = depth[:, 0:720,:]



# rgb = cv2.imread("/home/hilbertxu/air_ws/visionBasedManipulation/dataset/2021-11-17 17-46-39-rgb.png")
# depth = cv2.cvtColor(cv2.imread("/home/hilbertxu/air_ws/visionBasedManipulation/dataset/2021-11-17 17-46-39-depth.png"), cv2.COLOR_BGR2GRAY)

# print(rgb.shape, depth.shape)

# rgb = cv2.resize(rgb, (224,224))
# depth = cv2.resize(depth, (224,224))
# max_val = np.max(depth)
# depth = depth * (255 / max_val)
# depth = np.clip((depth - depth.mean())/175, -1, 1)

# print(rgb.shape, depth.shape)

# width = 224
# height = 224

# output_size = 224

# left = (width - output_size) // 2
# top = (height - output_size) // 2
# right = (width + output_size) // 2
# bottom = (height + output_size) // 2

# bottom_right = (bottom, right)
# top_left = (top, left)

# print(bottom_right, top_left)

# print(rgb.shape)
# print(depth.shape)

# depth_img = image.DepthImage(np.expand_dims(depth, -1))
# depth_img.crop(bottom_right=bottom_right, top_left=top_left)
# print(depth_img.img.shape)
# depth_img.normalise()
# depth_img.img = depth_img.img.transpose((2, 0, 1))


# rgb_img = image.Image(rgb)
# rgb_img.crop(bottom_right=bottom_right, top_left=top_left)
# print(rgb_img.img.shape)
# rgb_img.normalise()
# rgb_img.img = rgb_img.img.transpose((2, 0, 1))


rgb_img = cv2.imread("/home/hilbertxu/dataset/tubes/mix/rgb_000.png")

# cv2.imshow("rgb", rgb_img)
# cv2.waitKey(0)

rgb_img = rgb_img / 255.0
rgb_img -= rgb_img.mean()

depth_img = cv2.imread("/home/hilbertxu/dataset/tubes/mix/depth_000.png", cv2.COLOR_BGR2GRAY)
depth_img = inpaint(depth_img)
depth_img = depth_img / np.max(depth_img)
depth_img -= depth_img.mean()


print(depth_img)

rgb_img = rgb_img[37:299, 141:347]
depth_img = np.expand_dims(depth_img[37:299, 141:347],-1)

print(rgb_img.shape)
print(depth_img.shape)



# crop for 30 large tubes (137, 93) (307, 240)
# crop for 15 large tubes (150, 91) (312, 230)
# crop for 8 large tubes (111, 109) (281, 242)
# crop for 8 small tubes (125, 104) (284, 239)
# crop for 15 small tubes (128, 103) (268, 225)
# crop for all small tubes (138, 112) (305, 220)
# crop for mix tubes (141,37) (347, 299)




x = torch.from_numpy(
    np.concatenate([
        np.expand_dims(rgb_img.transpose(2,0,1), 0), 
        np.expand_dims(depth_img.transpose(2,0,1), 0)
    ], axis=1)
).cuda().float()

print(rgb_img.shape)
print(depth_img.shape)


net = torch.load("/home/hilbertxu/air_ws/robotic-grasping/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98")
net = net.cuda()

with torch.no_grad():

    pred = net.predict(x)

q_img, ang_img, width_img = post_process_output(pred['pos'],
                                                pred['cos'],
                                                pred['sin'],
                                                pred['width'])

fig = plt.figure(figsize=(10, 10))

im_bgr = cv2.cvtColor(cv2.imread("/home/hilbertxu/dataset/tubes/mix/rgb_000.png"), cv2.COLOR_RGB2BGR)[93:240, 137:307]
plot = plot_results(fig,
                    rgb_img=im_bgr,
                    grasp_q_img=q_img,
                    grasp_angle_img=ang_img,
                    depth_img=depth_img,
                    no_grasps=10,
                    grasp_width_img=width_img)

time = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
save_name = './{}'.format(time)
plot.savefig(save_name + '.png')
plot.clf()
