import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from imageio import imread
from skimage.color import rgb2gray
from .utils import create_mask
import cv2
from skimage.feature import canny
import torchvision.transforms as transforms
from random import randint

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, landmark_flist, mask_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.config = config
        self.augment = augment
        self.training = training

        self.data = self.load_flist(flist)
        self.mask_dir = mask_flist
        self.keypoints_dir = landmark_flist
        #self.mask_data = self.load_flist(mask_flist)
        #self.landmark_data = self.load_flist_text(landmark_flist)

        self.input_size = config.INPUT_SIZE
        self.mask = config.MASK

        # in test mode, there's a one-to-one relationship between mask and image
        # masks are loaded non random
       

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        item = self.load_item(index)
        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.input_size

        img_name = self.load_name(index)

        keypoints_file_name = img_name.split('.')[0] + '_keypoints.txt'



        keypoints_file_path = os.path.join(self.keypoints_dir, keypoints_file_name)
        #print (mask_path)
        #print (keypoints_file_path)

        # load image
        img = imread(self.data[index])

        if self.config.MODEL == 2 and self.config.MODE == 1:
            landmark = self.load_lmk([size, size], keypoints_file_path, img.shape)

        # resize/crop if needed
        if size != 0:
            img = self.resize(img, size, size, centerCrop=True)

        # load img, landmark, mask for training
        if self.mask == 6:
            #mask is not a tensor in fixed mask reading
            mask_path = os.path.join(self.mask_dir, img_name)
            mask = self.load_mask(img, mask_path)

            if self.config.MODEL == 2 and self.config.MODE == 1:
                return self.to_tensor(img), torch.from_numpy(landmark).long(), self.to_tensor(mask)

            #load img and mask for testing
            if self.config.MODEL == 2 and self.config.MODE == 2:
                return self.to_tensor(img), [], self.to_tensor(mask)


        elif self.mask == 8:
            #mask is already a tensor in freeform mask
            mask = self.load_mask(img, None)

            if self.config.MODEL == 2 and self.config.MODE == 1:
                return self.to_tensor(img), torch.from_numpy(landmark).long(), mask

            # load img and mask for testing
            if self.config.MODEL == 2 and self.config.MODE == 2:
                return self.to_tensor(img), [], mask



    def load_lmk(self, target_shape, keypoints_file_path, size_before, center_crop = True):
        imgh,imgw = target_shape[0:2]

        landmarks = np.genfromtxt(keypoints_file_path)
        landmarks = landmarks.reshape(self.config.LANDMARK_POINTS, 2)
        #landmarks = (landmarks/4)

        if self.input_size != 0:
            if center_crop:
                side = np.minimum(size_before[0],size_before[1])
                i = (size_before[0] - side) // 2
                j = (size_before[1] - side) // 2
                landmarks[0:self.config.LANDMARK_POINTS , 0] -= j
                landmarks[0:self.config.LANDMARK_POINTS , 1] -= i

            landmarks[0:self.config.LANDMARK_POINTS ,0] *= (imgw/side)
            landmarks[0:self.config.LANDMARK_POINTS ,1] *= (imgh/side)
        landmarks = (landmarks+0.5).astype(np.int16)

        return landmarks


    def load_mask(self, img, mask_path):
        imgh, imgw = img.shape[0:2]
        mask_type = self.mask

        # 50% no mask, 25% random block mask, 25% external mask, for landmark predictor training.
        if mask_type == 5:
            mask_type = 0 if np.random.uniform(0,1) >= 0.5 else 4

        # no mask
        if mask_type == 0:
            return np.zeros((self.config.INPUT_SIZE,self.config.INPUT_SIZE))

        # external + random block
        if mask_type == 4:
            mask_type = 1 if np.random.binomial(1, 0.5) == 1 else 3

        # random block
        if mask_type == 1:
            return create_mask(imgw, imgh, imgw // 2, imgh // 2)

        # center mask
        if mask_type == 2:
            return create_mask(imgw, imgh, imgw//2, imgh//2, x = imgw//4, y = imgh//4)

        # external
        if mask_type == 3:
            #mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = imread(mask_path)
            mask = self.resize(mask, imgh, imgw)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask

        # test mode: load mask non random
        if mask_type == 6:
            mask = imread(mask_path)
            mask = self.resize(mask, imgh, imgw, centerCrop=False)

            mask =  cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            mask = (mask > 50).astype(np.uint8) * 255

            return mask
        # random mask
        if mask_type ==7:
                mask = 1 - generate_stroke_mask([imgh, imgw])
                mask = (mask > 0).astype(np.uint8) * 255
                mask = self.resize(mask, imgh, imgw, centerCrop=False)
                return mask

        if mask_type ==8:
            random_num =randint(0, 1)
            if random_num == 0:
                mask = self.random_irregular_mask(img)
            elif random_num == 1:
                mask = self.random_freefrom_mask(img)
            return mask

    def random_irregular_mask(self, img):
        img = self.to_tensor(img)
        """Generates a random irregular mask with lines, circles and elipses"""
        transform = transforms.Compose([transforms.ToTensor()])
        # mask = torch.ones_like(img)
        size = img.size()
        mask = torch.ones(1, size[1], size[2])
        img = np.zeros((size[1], size[2], 1), np.uint8)

        # Set size scale
        max_width = 20
        if size[1] < 64 or size[2] < 64:
            raise Exception("Width and Height of mask must be at least 64!")

        number = random.randint(16, 64)
        for _ in range(number):
            model = random.random()
            if model < 0.6:
                # Draw random lines
                x1, x2 = randint(1, size[1]), randint(1, size[1])
                y1, y2 = randint(1, size[2]), randint(1, size[2])
                thickness = randint(4, max_width)
                cv2.line(img, (x1, y1), (x2, y2), (1, 1, 1), thickness)

            elif model > 0.6 and model < 0.8:
                # Draw random circles
                x1, y1 = randint(1, size[1]), randint(1, size[2])
                radius = randint(4, max_width)
                cv2.circle(img, (x1, y1), radius, (1, 1, 1), -1)

            elif model > 0.8:
                # Draw random ellipses
                x1, y1 = randint(1, size[1]), randint(1, size[2])
                s1, s2 = randint(1, size[1]), randint(1, size[2])
                a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
                thickness = randint(4, max_width)
                cv2.ellipse(img, (x1, y1), (s1, s2), a1, a2, a3, (1, 1, 1), thickness)

        img = img.reshape(size[2], size[1])
        img = Image.fromarray(img * 255)
        #img.save('testg.jpg')
        img_mask = transform(img)
        #ask[0, :, :] = img_mask < 1

        return img_mask

    def random_freefrom_mask(self, img, mv=5, ma=4.0, ml=40, mbw=10):
        transform = transforms.Compose([transforms.ToTensor()])

        img = self.to_tensor(img)
        size = img.size()
        mask = torch.ones(1, size[1], size[2])
        img = np.zeros((size[1], size[2], 1), np.uint8)
        num_v = 12 + np.random.randint(mv)  # tf.random_uniform([], minval=0, maxval=config.MAXVERTEX, dtype=tf.int32)

        for i in range(num_v):
            start_x = np.random.randint(size[1])
            start_y = np.random.randint(size[2])
            for j in range(1 + np.random.randint(5)):
                angle = 0.01 + np.random.randint(ma)
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10 + np.random.randint(ml)
                brush_w = 10 + np.random.randint(mbw)
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)

                cv2.line(img, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y

        img = img.reshape(size[2], size[1])
        img = Image.fromarray(img * 255)

        img_mask = transform(img)
        #mask[0, :, :] = img_mask < 1

        return img_mask
    def to_tensor(self, img):
        img = Image.fromarray(img)

        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = np.array(Image.fromarray(img).resize((height, width)))
        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png')) + list(glob.glob(flist + '/*.jpeg'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except Exception as e:
                    print(e)
                    return [flist]

        return []

    def load_flist_text(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.txt'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except Exception as e:
                    print(e)
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

    def shuffle_lr(self, parts, pairs=None):
        """Shuffle the points left-right according to the axis of symmetry
        of the object.
        Arguments:
            parts {torch.tensor} -- a 3D or 4D object containing the
            heatmaps.
        Keyword Arguments:
            pairs {list of integers} -- [order of the flipped points] (default: {None})
        """

        if pairs is None:
            if self.config.LANDMARK_POINTS == 68:
                pairs = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                     26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 27, 28, 29, 30, 35,
                     34, 33, 32, 31, 45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41,
                     40, 54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55, 64, 63,
                     62, 61, 60, 67, 66, 65]
            elif self.config.LANDMARK_POINTS == 98:
                pairs = [32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9,
                         8, 7, 6, 5, 4, 3, 2, 1, 0, 46, 45, 44, 43, 42, 50, 49, 48, 47, 37, 36, 35, 34, 33, 41, 40, 39,
                         38, 51, 52, 53, 54, 59, 58, 57, 56, 55, 72, 71, 70, 69, 68, 75, 74, 73, 64, 63, 62, 61, 60, 67,
                         66, 65, 82, 81, 80, 79, 78, 77, 76, 87, 86, 85, 84, 83, 92, 91, 90, 89, 88, 95, 94, 93, 97, 96]

        if len(parts.shape) == 3:
            parts = parts[:,pairs,...]
        else:
            parts = parts[pairs,...]

        return parts

def generate_stroke_mask(im_size, max_parts=15, maxVertex=25, maxLength=100, maxBrushWidth=24, maxAngle=360):
    mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
    parts = random.randint(1, max_parts)
    for i in range(parts):
        mask = mask + np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0], im_size[1])
    mask = np.minimum(mask, 1.0)

    return mask

def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)
        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)
        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask