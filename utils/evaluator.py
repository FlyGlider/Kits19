import sys
sys.path.append('.')
sys.path.append('./models')
sys.path.append('./utils/')
import argparse
import torch
import torch.nn.functional as F
from models.unet import UNet
from models.resunet import ResUNet
from sklearn.model_selection import KFold
from tools import *
from dataset import *
from torch.utils.data import DataLoader
from scipy.ndimage.filters import gaussian_filter
from medpy import metric


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--val', dest='val', default=0,
                        type=int, help='choose which validation')
    parser.add_argument('-s', '--stage', dest='stage', default='val',
                        type=str, help='choose the best model in which stage')
    args = parser.parse_args()
    return args

class Evaluator():
    def __init__(self, model, dir_h5_train, dir_checkpoint, dir_prediction, args):
        self.dataset_props = load_pickle('dataset_props.pkl')
        self.args = args
        self.dir_h5_train = dir_h5_train
        self.dir_checkpoint = dir_checkpoint
        self.dir_prediction = dir_prediction
        self.device = torch.device('cuda')
        self.model = self.get_pretrain_model(model)
        self.val_ids, self.val_loader, self.size, self.spacing = self.split_val()
        if self.model.mode == '2D':
            self.patch_size = self.dataset_props['plan_2d']['patch_size']
        elif self.model.mode == '3D':
            self.patch_size = self.dataset_props['plan_3d']['patch_size']
        elif self.model.mode == '3D_LOW':
            self.patch_size = self.dataset_props['plan_3d_low']['patch_size']
        else:
            raise NotImplementedError('mode [%s] is not found' % self.model.mode)
        self.num_class = len(self.dataset_props['label'].keys())

        create_dir(dir_prediction)
        
    
    def run(self):
        res_list = []
        for id, (data, seg), size, spacing in zip(self.val_ids, self.val_loader, self.size, self.spacing):
            data = data.to(self.device)
            seg = seg.cpu().numpy().squeeze()
            score_map = self.generate_score_map(data, self.patch_size, self.num_class, use_gaussian=False)

            # f = h5py.File('{}{}'.format(self.dir_prediction, os.path.split(id)[1]), 'w')
            # f.create_dataset(name='score_map', shape=score_map.shape, data=score_map, chunks=True, compression=9)
            
            res, prediction = self.evaluate(score_map, seg, size, spacing, 1, False)
            res_list.append(res)
            print(id, res)

            # f.create_dataset(name='prediction', shape=prediction.shape, data=prediction, chunks=True, compression=9)
            # f.close()
        print()

    
    def split_val(self):
        # 确定下标
        all_ids = get_ids(self.dir_h5_train)
        kf = KFold(n_splits=5, shuffle=True, random_state=125)
        _, val_index = list(kf.split(all_ids))[self.args.val]
        val_ids = [all_ids[x] for x in val_index]
        original_size = self.dataset_props['original_size']
        original_size = [original_size[x] for x in val_index]
        original_spacing = self.dataset_props['original_spacing']
        original_spacing = [original_spacing[x] for x in val_index]
        
        # 根据模式选择数据库
        if self.model.mode == '2D':
            val_dataset = BaseDataset2D(ids=val_ids, patch_size=None, force_fg=0, pre_load=False, return_patch=False)
        else:
            val_dataset = BaseDataset3D(ids=val_ids, patch_size=None, force_fg=0, pre_load=False, return_patch=False)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        return val_ids, val_loader, original_size, original_spacing
    
    def get_pretrain_model(self, model):
        model = model.to(self.device)
        model.load_state_dict(torch.load(
            '{}model_best_{}_{}.pth'.format(self.dir_checkpoint, self.args.stage, self.args.val)
        )['state_dict'])
        model.eval()
        return model
    
    def generate_score_map(self, data, patch_size, num_class, use_gaussian):
        if self.model.mode == '2D':
            patch_size = np.append([1], patch_size)
            ss = data.shape[1:]
        else:
            ss = data.shape[2:]
            
        score = torch.zeros((num_class,) + ss, dtype=torch.float).to(self.device)
        count = torch.zeros(ss, dtype=torch.float).to(self.device)
        idx_d = np.append(np.arange(0, ss[0] - patch_size[0], int(np.ceil(patch_size[0] / 2))), ss[0] - patch_size[0])
        idx_h = np.append(np.arange(0, ss[1] - patch_size[1], int(np.ceil(patch_size[1] / 2))), ss[1] - patch_size[1])
        idx_w = np.append(np.arange(0, ss[2] - patch_size[2], int(np.ceil(patch_size[2] / 2))), ss[2] - patch_size[2])
        with torch.no_grad():
            for i in idx_d:
                for j in idx_h:
                    for k in idx_w:
                        if self.model.mode == '2D':
                            patch = data[:, i: i + patch_size[0], j: j + patch_size[1], k:k + patch_size[2]]
                        else:
                            patch = data[:, :, i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]]
                        result_patch = F.softmax(self.model(patch), dim=1)
                        if use_gaussian:
                            tmp = np.zeros(patch_size)
                            center_coords = tuple([i // 2 for i in patch_size])
                            sigma = [i // 8 for i in patch_size]
                            tmp[center_coords] = 1
                            tmp = gaussian_filter(input=tmp, sigma=sigma, order=0, mode='constant', cval=0)
                            tmp = tmp / tmp.max() * 1 + 1e-8
                            tmp = torch.from_numpy(tmp.astype('float32')).to(self.device)
                            count[i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]] += tmp
                        else:
                            tmp = 1
                            count[i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]] += tmp
                        if self.model.mode == '2D':
                            score[:, i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]] += result_patch.transpose(0, 1) * tmp
                        else:
                            score[:, i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]] += result_patch.squeeze() * tmp
        score_map = score / count.repeat(num_class, 1, 1, 1)
        return score_map.cpu().numpy()
    
    def calculate_dice(self, score_map, seg):
        prediction = torch.argmax(score_map, axis=0)
        res = dict()
        for j in range(1, self.num_class):
            p = prediction == j
            q = seg == j
            p, q = p.float(), q.float()
            dice = (2 * (p * q) + 1e-08) / (p.sum() + q.sum() + 1e-08)
            res[j] = dice
        return res, prediction.cpu().numpy()
    
    def evaluate(self, score_map, seg, size, spacing, connectivity=1, use_post_process=True):  
        seg = resize(seg, size, order=0, mode='edge', cval=0, clip=True, preserve_range=True, anti_aliasing=False).astype(np.int8)
        seg = self.seg2one_hot(seg, self.num_class)
        score_map = np.vstack([resize(score_map[i], size, order=3, mode='reflect', cval=0,
                               clip=True, preserve_range=True, anti_aliasing=False).astype('float32')[np.newaxis] for i in range(len(score_map))])
        res = dict()
        if use_post_process:
            prediction = post_process(score_map)
        else:
            prediction = np.argmax(score_map, axis=0)
        prediction = self.seg2one_hot(prediction, self.num_class)
        for i in range(1, self.num_class):
            res[i] = dict()
            if np.any(prediction[i]) and np.any(seg[i]):
                # res[i]['jaccard'] = metric.jc(prediction[i], seg[i])
                res[i]['dice'] = metric.dc(prediction[i], seg[i])
                # res[i]['hausdorff'] = metric.hd(prediction[i], seg[i], voxelspacing=spacing, connectivity=connectivity)
                # res[i]['hausdorff95'] = metric.hd95(prediction[i], seg[i], voxelspacing=spacing, connectivity=connectivity)
                # res[i]['mean_surface_distance'] = metric.asd(prediction[i], seg[i], voxelspacing=spacing, connectivity=connectivity)
            elif not (np.any(prediction[i]) or np.any(seg[i])):
                # res[i]['jaccard'] = 1
                res[i]['dice'] = 1
                # res[i]['hausdorff'] = 0
                # res[i]['hausdorff95'] = 0
                # res[i]['mean_surface_distance'] = 0
            else:
                # res[i]['jaccard'] = 0
                res[i]['dice'] = 0
                # res[i]['hausdorff'] = np.inf
                # res[i]['hausdorff95'] = np.inf
                # res[i]['mean_surface_distance'] = np.inf
        prediction = np.argmax(prediction, axis=0)
        return res, prediction

    def seg2one_hot(self, seg, num_class):
        '''
        将分割图(seg)转化为one——hot编码,比如seg大小为(512, 512),3个类别,转换为one——hot大小(3, 512, 512)
        '''
        one_hot = np.stack([seg == c for c in range(num_class)])
        return one_hot
    
    def post_process(self, score_map):
        pass
    

if __name__ == '__main__':
    dir_h5_train = 'h5_data_train_3d/'
    dir_checkpoint = 'checkpoints/3d/'
    dir_prediction = 'prediction/3d/'
    create_dir(dir_prediction)
    dataset_props = load_pickle('dataset_props.pkl')
    pool_layer_kernel_sizes = dataset_props['plan_3d']['pool_layer_kernel_sizes']
    args = get_args()
    model = ResUNet(in_ch=1, base_num_features=30, num_classes=3, norm_type='batch', nonlin_type='relu', pool_type='max',
                    pool_layer_kernel_sizes=pool_layer_kernel_sizes, deep_supervision=False, mode='3D')
    evaluator = Evaluator(model, dir_h5_train, dir_checkpoint, dir_prediction, args)
    evaluator.run()

