import sys
sys.path.append('.')
sys.path.append('./models')
sys.path.append('./utils/')
import argparse
from models.unet import UNet
from models.resunet import ResUNet
from sklearn.model_selection import KFold
from tools import *
from dataset import *
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from utils.loss import *
from scipy.ndimage.filters import gaussian_filter
import torch.optim as optim
from torch.nn import init

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--stage', dest='stage', default='val',
                        type=str, help='choose the best model in which stage')
    args = parser.parse_args()
    return args

class Tester():
    def __init__(self, models, dir_h5_test, dir_prediction, dir_checkpoint, args):
        self.args = args
        self.dataset_props = load_pickle('dataset_props.pkl')
        self.dir_h5_test = dir_h5_test
        self.dir_prediction = dir_prediction
        self.dir_checkpoint = dir_checkpoint
        self.num_class = len(self.dataset_props['label'].keys())
        self.device = torch.device('cuda')
        self.models = self.get_pretrain_models(models)
        self.test_ids, self.test_loader = self.get_test_loader()
        self.original_size = self.dataset_props['original_size_test'][42:]
        # self.original_size.reverse()  # ------------------------------------------------------------
        if models[0].mode == '2D':
            self.patch_size = self.dataset_props['plan_2d']['patch_size']
        elif models[0].mode == '3D':
            self.patch_size = self.dataset_props['plan_3d']['patch_size']
        elif models[0].mode == '3D_LOW':
            self.patch_size = self.dataset_props['plan_3d_low']['patch_size']
        else:
            raise NotImplementedError('mode [%s] is not found' % self.model.mode)


    def run(self):
        for id, data, size in zip(self.test_ids, self.test_loader, self.original_size):
            data = data.to(self.device)
            score_map = 0
            for model in self.models:
                score_map += self.generate_score_map(data, model, patch_size=self.patch_size,
                                                     num_class=self.num_class, use_gaussian=True, mode=models[0].mode)
            score_map /= len(self.models)

            f = h5py.File('{}{}'.format(self.dir_prediction, os.path.split(id)[1]), 'w')
            f.create_dataset(name='score_map', shape=score_map.shape, data=score_map, chunks=True, compression=9)

            # score_map = np.vstack([resize(score_map[i], size, order=3, mode='reflect', cval=0,
            #                               clip=True, preserve_range=True, anti_aliasing=False).astype('float32')[np.newaxis] for i in range(len(score_map))])
            # prediction = np.argmax(score_map, axis=0)
            # for i in range(len(prediction)):
            #     plt.imshow(prediction[i])
            #     plt.show()
            # f.create_dataset(name='prediction', shape=prediction.shape, data=prediction, chunks=True, compression=9)
            f.close()

    def get_test_loader(self):
        test_ids = get_ids(self.dir_h5_test)[42:]
        # test_ids.reverse()  # -------------------------------------------------------------------------
        if self.models[0].mode == '3D':
            test_dataset = BaseDataset3D(ids=test_ids, patch_size=None, force_fg=0, pre_load=False, return_patch=False)
        else:
            test_dataset = BaseDataset2D(ids=test_ids, patch_size=None, force_fg=0, pre_load=False, return_patch=False)
        test_loader = DataLoader(test_dataset)
        return test_ids, test_loader
    
    def get_pretrain_models(self, models):
        i = 0
        for state_dict in sorted(os.listdir(self.dir_checkpoint)):
            if state_dict.find(self.args.stage) != -1:
                models[i].load_state_dict(torch.load(self.dir_checkpoint + state_dict)['state_dict'])
                models[i].to(self.device)
                models[i].eval()
                i = i + 1
        return models

    def generate_score_map(self, data, model, patch_size, num_class, use_gaussian, mode):
        if mode == '2D':
            patch_size = np.append([1], patch_size)

        score = torch.zeros((num_class,) + data.size()[2:], dtype=torch.float).to(self.device)
        count = torch.zeros(data.size()[2:], dtype=torch.float).to(self.device)
        ss = data.size()[2:]
        idx_d = np.append(np.arange(0, ss[0] - patch_size[0], int(np.ceil(patch_size[0] / 2))), ss[0] - patch_size[0])
        idx_h = np.append(np.arange(0, ss[1] - patch_size[1], int(np.ceil(patch_size[1] / 2))), ss[1] - patch_size[1])
        idx_w = np.append(np.arange(0, ss[2] - patch_size[2], int(np.ceil(patch_size[2] / 2))), ss[2] - patch_size[2])
        with torch.no_grad():
            for i in idx_d:
                for j in idx_h:
                    for k in idx_w:
                        if mode == '2D':
                            patch = data[:, i: i + patch_size[0], j: j + patch_size[1], k:k + patch_size[2]]
                        else:
                            patch = data[:, :, i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]]
                        result_patch = F.softmax(model(patch), dim=1)
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
                        if mode == '2D':
                            score[:, i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]] += result_patch.transpose(0, 1) * tmp
                        else:
                            score[:, i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]] += result_patch.squeeze() * tmp
        score_map = score / count.repeat(num_class, 1, 1, 1)
        return score_map.cpu().numpy()

if __name__ == '__main__':
    dir_checkpoint = 'checkpoints/3d/'
    dir_h5_test = 'h5_data_test_3d/'
    dir_prediction = 'prediction/test_3d_val/'
    create_dir(dir_prediction)
    dataset_props = load_pickle('dataset_props.pkl')
    pool_layer_kernel_sizes = dataset_props['plan_3d']['pool_layer_kernel_sizes']
    args = get_args()
    models = []
    for i in range(5):
        models.append(ResUNet(in_ch=1, base_num_features=30, num_classes=3, norm_type='batch', nonlin_type='relu', pool_type='max',
                      pool_layer_kernel_sizes=pool_layer_kernel_sizes, deep_supervision=False, mode='3D'))
    tester = Tester(models, dir_h5_test, dir_prediction, dir_checkpoint, args)
    tester.run()
