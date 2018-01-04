
import cv2 as cv
import numpy as np
import os
import sys
EPSILON = sys.float_info.epsilon  # smallest possible difference

def convert_to_rgb(minval, maxval, val, colors):
    fi = float(val-minval) / float(maxval-minval) * (len(colors)-1)
    i = int(fi)
    f = fi - i
    if f < EPSILON:
        return colors[i]
    else:
        (r1, g1, b1), (r2, g2, b2) = colors[i], colors[i+1]
        return int(r1 + f*(r2-r1)), int(g1 + f*(g2-g1)), int(b1 + f*(b2-b1))
# Will use matplotlib for showing the image
from matplotlib import pyplot as plt
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
dir = "/home/niranjan/Projects/pytorch-CycleGAN-and-pix2pix/datasets/block/foldB_2"
dir2 = "/home/niranjan/Projects/pytorch-CycleGAN-and-pix2pix/datasets/block/foldB_3"

images = []
dest = []
assert os.path.isdir(dir), '%s is not a valid directory' % dir
m = []

with open('/home/niranjan/Projects/pydart2/Force1.txt') as input_file:
    for line in input_file:
        x, y, m1 = (
            item.strip() for item in line.split(' ', 2))
        m.append(float(m1))


for root, _, fnames in sorted(os.walk(dir)):
    for fname in fnames:
        path = os.path.join(root, fname)
        path2 = os.path.join(dir2, fname)
        images.append(path)
        dest.append(path2)
print(len(images))
temp = cv.imread(os.path.join(root,'obj8_3__B0.png'))
img_default = (np.sum(temp, 2)).astype(dtype='uint8')
idx = img_default[img_default > 0]
print(idx.shape)
#mask = img_default*0
mask = np.where(img_default > 0, 1, img_default)
images = sorted(images)
#print(dest[2])
for i in range(len(images)):
    # print(images[i])
    #if "1__B0" in images[i]:
    if True:
        img2 = cv.imread(os.path.join(root, 'obj'+str(int(i/3)+1)+'_'+str(int(i%3)+1)+'__B0.png'))
        #print(os.path.join(root, 'obj'+str(int(i/3)+1)+'_'+str(int(i%3)+1)+'__B0.png'),m[i])
        img_tem = (np.sum(img2, 2)).astype(dtype='uint8')
        mask = np.where(img_tem > 0, 1, img_tem)
        r, g, b = convert_to_rgb(min(m), max(m), m[i], colors)
        img = np.zeros_like(img2)
        img[:, :, 0] = mask * np.max(r)
        img[:, :, 1] = mask * np.max(g)
        img[:, :, 2] = mask * np.max(b)
        #print((np.max(np.max(img[:, :, 1]))))
        #print(images[i], dest[i])
        print(os.path.join(root, 'obj' + str(int(i / 3) + 1) + '_' + str(int(i % 3) + 1) + '__B0.png'), m[i],r,g,b)
        cv.imwrite(os.path.join(dir2, 'obj' + str(int(i / 3) + 1) + '_' + str(int(i % 3) + 1) + '__B0.png'), img)
        # cv.imshow("win1", img.astype(dtype='uint8'))
        # cv.waitKey()