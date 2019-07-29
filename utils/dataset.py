from torch.utils.data import Dataset
from tools import *
import random
import elasticdeform

class BaseDataset3D(Dataset):
    def __init__(self, ids, patch_size, force_fg=2 / 3, pre_load=False, return_patch=True, aug=0):
        self.ids = ids
        self.patch_size = patch_size
        self.force_fg = force_fg
        self.pre_load = pre_load
        self.return_patch = return_patch
        self.aug = aug
        if pre_load:
            self.__pre_load__()

    def __getitem__(self, index):
        if self.pre_load:
            img = self.imgs[index]
            seg = self.segs[index]
            bboxes = self.bboxes[index]
        else:
            h5_id = self.ids[index]
            f = h5py.File(h5_id, 'r')
            img = np.array(f['data'])
            if 'seg' in f.keys():
                seg = np.array(f['seg'])
                bboxes = np.array(f['bboxes'])
            else:
                seg = None
            f.close()

        if not self.return_patch:
            if seg is None:
                return img[np.newaxis]
            else:
                return img[np.newaxis], seg.astype(np.int64)
        
        if random.random() < self.aug:
            img, seg = self.elastic_deform(img, seg)

        if random.random() < self.force_fg:
            img, seg = self.crop(img, seg, bboxes)
        else:
            img, seg = self.random_crop(img, seg)

        return img[np.newaxis], seg.astype(np.int64)

    def __len__(self):
        return len(self.ids)
    
    def crop(self, img, seg, bboxes):
        '''
        TO DO : i需要自动确定
        '''
        i = random.randint(0, len(bboxes) - 1)
        cen = self.get_cen(img.shape, bboxes, i)
        # print(cen)
        crop_img = img[cen[0] - self.patch_size[0] // 2:cen[0] + self.patch_size[0] // 2,
                       cen[1] - self.patch_size[1] // 2:cen[1] + self.patch_size[1] // 2,
                       cen[2] - self.patch_size[2] // 2:cen[2] + self.patch_size[2] // 2]
        crop_seg = seg[cen[0] - self.patch_size[0] // 2:cen[0] + self.patch_size[0] // 2,
                       cen[1] - self.patch_size[1] // 2:cen[1] + self.patch_size[1] // 2,
                       cen[2] - self.patch_size[2] // 2:cen[2] + self.patch_size[2] // 2]
        return crop_img, crop_seg
    
    def random_crop(self, img, seg):
        
        # python自带的random包含high,与numpy的random不同
        d_s = random.randint(0, img.shape[0] - self.patch_size[0])
        h_s = random.randint(0, img.shape[1] - self.patch_size[1])
        w_s = random.randint(0, img.shape[2] - self.patch_size[2])
    
        crop_img = img[d_s:d_s + self.patch_size[0], h_s:h_s + self.patch_size[1], w_s:w_s + self.patch_size[2]]
        crop_seg = seg[d_s:d_s + self.patch_size[0], h_s:h_s + self.patch_size[1], w_s:w_s + self.patch_size[2]]
        return crop_img, crop_seg
    
    def get_cen(self, img_shape, bboxes, i):
        cen = np.zeros(3, dtype=np.int32)
        cen[0] = random.randint(bboxes[i, 0], bboxes[i, 3])
        cen[1] = random.randint(bboxes[i, 1], bboxes[i, 4])
        cen[2] = random.randint(bboxes[i, 2], bboxes[i, 5])

        cen[0] = min(max(0 + self.patch_size[0] // 2, cen[0]), img_shape[0] - self.patch_size[0] // 2)
        cen[1] = min(max(0 + self.patch_size[1] // 2, cen[1]), img_shape[1] - self.patch_size[1] // 2)
        cen[2] = min(max(0 + self.patch_size[2] // 2, cen[2]), img_shape[2] - self.patch_size[2] // 2)
        return cen

    def random_flip(self, img, seg):
        flip_num = random.randint(1, 7)
        
        if flip_num == 1:
            img = np.flip(img, 0)
            seg = np.flip(seg, 0)
        elif flip_num == 2:
            img = np.flip(img, 1)
            seg = np.flip(seg, 1)
        elif flip_num == 3:
            img = np.flip(img, 2)
            seg = np.flip(seg, 2)
        elif flip_num == 4:
            img = np.flip(img, (0, 1))
            seg = np.flip(seg, (0, 1))
        elif flip_num == 5:
            img = np.flip(img, (0, 2))
            seg = np.flip(seg, (0, 2))
        elif flip_num == 6:
            img = np.flip(img, (1, 2))
            seg = np.flip(seg, (1, 2))
        elif flip_num == 7:
            img = np.flip(img, (0, 1, 2))
            seg = np.flip(seg, (0, 1, 2))
        return img.copy(), seg.copy()
    
    def elastic_deform(self, img, seg):
        i = random.randint(0, 2)
        if i == 0:
            img, seg = elasticdeform.deform_random_grid([img, seg], order=[3, 0], sigma=7, points=3, axis=[(0, 1), (0, 1)])
        elif i == 1:
            img, seg = elasticdeform.deform_random_grid([img, seg], order=[3, 0], sigma=7, points=3, axis=[(1, 2), (1, 2)])
        else:
            img, seg = elasticdeform.deform_random_grid([img, seg], order=[3, 0], sigma=7, points=3, axis=[(0, 2), (0, 2)])
        return img, seg

    def __pre_load__(self):
        self.imgs = []
        self.segs = []
        self.bboxes = []
        for id in self.ids:
            f = h5py.File(id, 'r')
            self.imgs.append(np.array(f['data']))
            self.segs.append(np.array(f['seg']))
            self.bboxes.append(np.array(f['bboxes']))
            f.close()

class BaseDataset2D(Dataset):  # 如果不预先加载到内存，速度会慢很多
    def __init__(self, ids, patch_size, force_fg=1 / 3, pre_load=False, return_patch=True):
        self.ids = ids
        self.patch_size = patch_size
        self.force_fg = force_fg
        self.pre_load = pre_load
        self.return_patch = return_patch
        if pre_load:
            self.__pre_load__()

    def __getitem__(self, index):
        if self.pre_load:
            img = self.imgs[index]
            seg = self.segs[index]
            bboxes = self.bboxes[index]
        else:
            h5_id = self.ids[index]
            f = h5py.File(h5_id, 'r')
            img = np.array(f['data'])
            if 'seg' in f.keys():
                seg = np.array(f['seg'])
                bboxes = np.array(f['bboxes'])
            else:
                seg = None
            f.close()

        if not self.return_patch:
            if seg is None:
                return img
            else:
                return img, seg.astype(np.int64)

        if random.random() < self.force_fg:  # 这里控制随机数的比例会对结果产生影响
            img, seg = self.crop(img, seg, bboxes)
        else:
            img, seg = self.random_crop(img, seg)
        return img[np.newaxis], seg.astype(np.int64)
    
    def __len__(self):
        return len(self.ids)

    def crop(self, img, seg, bboxes):
        i = random.randint(0, len(bboxes) - 1)
        cen = self.get_cen(img.shape, bboxes, i)
        crop_img = img[cen[0],
                       cen[1] - self.patch_size[0] // 2:cen[1] + self.patch_size[0] // 2,
                       cen[2] - self.patch_size[1] // 2:cen[2] + self.patch_size[1] // 2]
        crop_seg = seg[cen[0],
                       cen[1] - self.patch_size[0] // 2:cen[1] + self.patch_size[0] // 2,
                       cen[2] - self.patch_size[1] // 2:cen[2] + self.patch_size[1] // 2]
        return crop_img, crop_seg
    
    def random_crop(self, img, seg):
        
        # python自带的random包含high,与numpy的random不同
        d_s = random.randint(0, img.shape[0] - 1)
        h_s = random.randint(0, img.shape[1] - self.patch_size[0])
        w_s = random.randint(0, img.shape[2] - self.patch_size[1])
    
        crop_img = img[d_s, h_s:h_s + self.patch_size[0], w_s:w_s + self.patch_size[1]]
        crop_seg = seg[d_s, h_s:h_s + self.patch_size[0], w_s:w_s + self.patch_size[1]]
        return crop_img, crop_seg
    
    def get_cen(self, img_shape, bboxes, i):
        cen = np.zeros(3, dtype=np.int32)
        cen[0] = random.randint(bboxes[i, 0], bboxes[i, 3])
        cen[1] = random.randint(bboxes[i, 1], bboxes[i, 4])
        cen[2] = random.randint(bboxes[i, 2], bboxes[i, 5])
        cen[0] = min(cen[0], img_shape[0] - 1)
        cen[1] = min(max(0 + self.patch_size[0] // 2, cen[1]), img_shape[1] - self.patch_size[0] // 2)
        cen[2] = min(max(0 + self.patch_size[1] // 2, cen[2]), img_shape[2] - self.patch_size[1] // 2)
        return cen

    def random_flip(self, img, seg):
        flip_num = random.randint(1, 4)
        
        if flip_num == 1:
            img = np.flip(img, 0)
            seg = np.flip(seg, 0)
        elif flip_num == 2:
            img = np.flip(img, 1)
            seg = np.flip(seg, 1)
        elif flip_num == 3:
            img = np.flip(img, (0, 1))
            seg = np.flip(seg, (0, 1))
        
        return img.copy(), seg.copy()


    def __pre_load__(self):
        self.imgs = []
        self.segs = []
        self.bboxes = []
        for id in self.ids:
            f = h5py.File(id, 'r')
            self.imgs.append(np.array(f['data']))
            self.segs.append(np.array(f['seg']))
            self.bboxes.append(np.array(f['bboxes']))
            f.close()

