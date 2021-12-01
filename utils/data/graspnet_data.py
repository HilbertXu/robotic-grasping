import random
import glob
import os
from cv2 import sort
import random
from graspnetAPI.grasp import RectGrasp

import numpy as np
from numpy.lib.type_check import imag
import torch
import torch.utils.data

from graspnetAPI import GraspNet

from utils.dataset_processing import grasp, image
from detectron2.layers import nms_rotated

class GraspNetDataBase(object):
    def __init__(self, file_path="/home/hilbertxu/dataset/graspnet/scenes", camera="realsense", random_crop=True) -> None:
        super().__init__()

        self.camera = camera
        self.random_crop = random_crop
        self.g = GraspNet('/home/hilbertxu/dataset/graspnet', camera=camera, split='train')

        self.output_size = 720
        
        self.grasp_files = glob.glob(os.path.join(file_path, "*", camera, "rect","*.npy"))
        self.grasp_files = glob.glob(os.path.join(file_path, "*", camera, "rect","*.npy"))
        self.grasp_files.sort()
        self.length = len(self.grasp_files)

        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        self.depth_files = [f.replace('.npy', '.png').replace('rect', 'depth') for f in self.grasp_files]
        
        
        self.rgb_files = [f.replace('.npy', '.png').replace('rect', 'rgb') for f in self.grasp_files]
    
    def _apply_nms(self, rects):
        center_x, center_y, open_x, open_y, height, scores, cls_ids = rects[:,0], rects[:,1], rects[:,2], rects[:,3], rects[:,4], rects[:,5], rects[:,6]
        
        # Calculate rotated bbox and apply NMS
        tmp = np.array([open_x - center_x, open_y - center_y])
        width = (np.sqrt(np.sum(np.square(tmp), axis=0)) * 2).reshape(-1,1)

        center = np.array([center_x, center_y])
        left = np.array([open_x, open_y])
        axis = left - center
        normal = np.array([-axis[1], axis[0]])
        # Calculate normal to determine the angle of bbox
        normal = normal / np.linalg.norm(normal) * height / 2
        tan = normal[0] / normal[1]
        # angle: [-pi/2, pi/2]
        angle = np.arctan(tan) * 180.0 / 3.14159

        r_box = np.concatenate([
            center_x.reshape(-1,1), center_y.reshape(-1,1), 
            width.reshape(-1,1), height.reshape(-1,1),
            angle.reshape(-1,1)
        ], axis=1)

        keep = nms_rotated(torch.from_numpy(r_box), torch.from_numpy(scores), 0.05).cpu().numpy()
        
        return keep
    
    def _load_from_graspnet_file(self, idx):
        grasp_file = self.grasp_files[idx]
        grasp_file_split = grasp_file.split('/')
        scene_id, ann_id = int(grasp_file_split[-4][-4:]), int(grasp_file_split[-1][:-4]) 
        rect_grasp_group = self.g.loadGrasp(sceneId=scene_id, camera=self.camera, annId=ann_id, fric_coef_thresh=0.2, format='rect')
        rects = rect_grasp_group.rect_grasp_group_array
        
        keep = self._apply_nms(rect_grasp_group.rect_grasp_group_array)


        grs = []

        # print("normal", normal)
        for i in list(keep):
            center_x, center_y, open_x, open_y, height, score, object_id = rects[i,:]
            center = np.array([center_x, center_y])
            left = np.array([open_x, open_y])
            axis = left - center
            normal = np.array([-axis[1], axis[0]])
            normal = normal / np.linalg.norm(normal) * height / 2
            p1 = center + normal + axis
            p2 = center + normal - axis
            p3 = center - normal - axis
            p4 = center - normal + axis

            gr = np.array([
                (int(p1[1]), int(p1[0])),
                (int(p2[1]), int(p2[0])),
                (int(p3[1]), int(p3[0])),
                (int(p4[1]), int(p4[0]))
            ])

            grs.append(grasp.GraspRectangle(gr))
        
        return grasp.GraspRectangles(grs)
    

    def _mask_bb_out_of_range(self, gtbbs, crop_range=None):
        print(crop_range)
        keep = []
        for gr in gtbbs.grs:
            points = gr.points
            if ((points[:,0] <= crop_range[1][0]).all() and (points[:,0] >= crop_range[0][0]).all()) & ((points[:,1] <=crop_range[1][1]).all() and (points[:,1] >= crop_range[0][1]).all()):
                keep.append(gr)
        
        return grasp.GraspRectangles(keep)
        
    def _get_crop_attrs(self, idx):
        gtbbs = self._load_from_graspnet_file(idx)
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 1280 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 720 - self.output_size))

        return center, left, top

    
    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = self._load_from_graspnet_file(idx)
        center, left, top = self._get_crop_attrs(idx)

        crop_range = [(0, 0), (720, 720)]
        gtbbs.rotate(rot, center)
        gtbbs.offset((-top, -left))
        gtbbs.zoom(zoom, (self.output_size // 2, self.output_size // 2))
        masked_gtbbs = self._mask_bb_out_of_range(gtbbs, crop_range)

        return masked_gtbbs


    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_file = self.rgb_files[idx]
        rgb_file_split = rgb_file.split('/')
        scene_id, ann_id = int(rgb_file_split[-4][-4:]), int(rgb_file_split[-1][:-4])
        rgb = self.g.loadRGB(sceneId=scene_id, camera=self.camera, annId=ann_id) 
        print(rgb.shape)
        rgb_img = image.Image(rgb)
        center, left, top = self._get_crop_attrs(idx)
        rgb_img.rotate(rot, center)
        rgb_img.crop((top, left), (min(720, top + self.output_size), min(1280, left + self.output_size)))
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_file = self.depth_files[idx]
        depth_file_split = depth_file.split('/')
        scene_id, ann_id = int(depth_file_split[-4][-4:]), int(depth_file_split[-1][:-4])
        depth = self.g.loadDepth(sceneId=scene_id, camera=self.camera, annId=ann_id) 

        depth_img = image.DepthImage(depth)
        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img
        

if __name__ == "__main__":

    import matplotlib.pyplot as plt


    dataset = GraspNetDataBase()

    gtbb = dataset.get_gtbb(0)

    rgb = dataset.get_rgb(0, normalise=False)

    depth = dataset.get_depth(0)

    fig, ax = plt.subplots(figsize=(7,7))
    ax.imshow(rgb)

    gtbb.show(ax=ax)

    plt.show()

    # file_path="/home/hilbertxu/dataset/graspnet/scenes"
    # camera="realsense"
    # grasp_files = glob.glob(os.path.join(file_path, "*", camera, "rect","*.npy"))
    # grasp_files.sort()
    # print(grasp_files)
    # depth_files = [f.replace('.npy', '.png').replace('rect', 'depth') for f in grasp_files]
    # rgb_files = [f.replace('.npy', '.png').replace('rect', 'rgb') for f in grasp_files]
    # print(len(grasp_files))
    # print(len(depth_files))
    # print(len(rgb_files))

    # # Grasp: [Center-x, center-y, open-x, open-y, height, score, class_id]

    # import cv2
    # from graspnetAPI import GraspNet
    # import math

    # g = GraspNet('/home/hilbertxu/dataset/graspnet', camera=camera, split='train')

    # print(grasp_files[0])

    # # g.showSceneGrasp(sceneId=0, camera=camera, annId=0, format='rect', class_wise_topk_k=3, threshold=0.5)
    # # a = "/home/hilbertxu/dataset/graspnet/scenes/scene_0090/realsense/rect/0101.npy"
    # # print(a.split("/"))

    # # a_split = a.split('/')
    # # print(a_split[-4][-4:])
    # # print(a_split[-1][:-4])
    # # scene_id, ann_id = int(a_split[-4][-4:]), int(a_split[-1][:-4])
    # # print(scene_id, ann_id)
    # # rects = np.load(grasp_files[0])
    # # scores = rects[:, 5]
    # # print(np.sort(scores))
    # # print(scores.shape)
    # # sort_idx = np.argsort(scores)
    # # print(sort_idx)
    # # rects = rects[sort_idx,:]
    # # bgr = cv2.imread(grasp_files[0])

    # # print(rects.shape)
    # bgr = g.loadBGR(sceneId=0, camera=camera, annId=0)
    # # _6d_grasp_group = g.loadGrasp(sceneId=0, camera=camera, annId=0, fric_coef_thresh=0.2, format='6d')
    # # nms_grasp = _6d_grasp_group.nms(translation_thresh=0.1, rotation_thresh= 30 / 180.0 * 3.1416)
    # # nms_rect_group = nms_grasp.to_rect_grasp_group(camera)

    # rect_grasp_group = g.loadGrasp(sceneId=0, camera=camera, annId=0, fric_coef_thresh=0.2, format='rect')



    # # print(nms_rect_group.rect_grasp_group_array)

    # _rect_grasps = rect_grasp_group.rect_grasp_group_array

    # print(_rect_grasps.shape)

    # center_x, center_y, open_x, open_y, height = _rect_grasps[:, 0], _rect_grasps[:, 1], _rect_grasps[:, 2], _rect_grasps[:,3], _rect_grasps[:,4]

    # tmp = np.array([open_x - center_x, open_y - center_y])
    # width = (np.sqrt(np.sum(np.square(tmp), axis=0)) * 2).reshape(-1,1)
    # print(width.shape)

    # # print(width)
    # print(center_x.reshape(-1,1).shape)
    # box = np.concatenate([center_x.reshape(-1,1), center_y.reshape(-1,1), width.reshape(-1,1), height.reshape(-1,1)], axis=1)


    # center = np.array([center_x, center_y])
    # left = np.array([open_x, open_y])
    # axis = left - center
    # normal = np.array([-axis[1], axis[0]])
    # normal = normal / np.linalg.norm(normal) * height / 2

    # tan = normal[0] / normal[1]
    # angle = np.arctan(tan) * 180.0 / 3.14159
    
    # r_box = np.concatenate([box, angle.reshape(-1,1)], axis=1)

    # print(r_box.shape)

    # scores = _rect_grasps[:,5]

    # from detectron2.layers import nms_rotated

    # keep = nms_rotated(torch.from_numpy(r_box), torch.from_numpy(scores), 0.05).cpu().numpy()
    
    # print(keep.shape, keep)

    # rect_grasp_group.rect_grasp_group_array = rect_grasp_group.rect_grasp_group_array[keep, :]
    
    # img = rect_grasp_group.to_opencv_image(bgr)
    # # 
    # #  cv2.circle(bgr, (int(center_x), int(center_y)), 3, (255,255,255), 4)
    # # cv2.circle(bgr, (int(open_x), int(open_y)), 3, (0, 255, 0), 4)

    # # center = np.array([center_x, center_y])
    # # left = np.array([open_x, open_y])
    # # axis = left - center
    # # print(axis)
    # # normal = np.array([-axis[1], axis[0]])
    # # print(normal)
    # # normal = normal / np.linalg.norm(normal) * height / 2
    # # print(height, normal)

    # # img = nms_rect_group.to_opencv_image(bgr)

    # cv2.imshow('rect grasps', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    

