from numpy.core.fromnumeric import shape
from utils.data import get_dataset
import matplotlib.pyplot as plt
import cv2
import numpy as np

dataset = get_dataset("cornell")(
    "/home/hilbertxu/dataset/cornel-grasp",
    output_size=224,
    ds_rotate=0,
    random_rotate=True,
    random_zoom=True,
    include_depth=1,
    include_rgb=1
)
rgb = dataset.get_rgb(0, normalise=False)



fig, ax = plt.subplots(figsize=(7,7))
ax.imshow(rgb)


gtbb = dataset.get_gtbb(0)
gtbb.show(ax=ax)


x, (pos, cos, sin, width), _, _, _ = dataset[0]


plt.figure(figsize=(10,10))

ax1 = plt.subplot(151)
ax1.imshow(rgb)
ax1.set_title("RGB")


ax2 = plt.subplot(152)
ax2.imshow(pos.permute(1,2,0).cpu().numpy()*255)
ax2.set_title("pos")

ax3 = plt.subplot(153)
ax3.imshow(cos.permute(1,2,0).cpu().numpy()*255)
ax3.set_title("cos")

ax4 = plt.subplot(154)
ax4.imshow(sin.permute(1,2,0).cpu().numpy()*255)
ax4.set_title("sin")

ax5 = plt.subplot(155)
ax5.imshow(width.permute(1,2,0).cpu().numpy()*255)
ax5.set_title("width")

plt.show()




