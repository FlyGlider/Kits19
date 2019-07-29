import sys
sys.path.append('.')
sys.path.append('./models')
sys.path.append('./utils/')
import argparse
import SimpleITK as sitk
from models.unet import UNet
from sklearn.model_selection import KFold
from tools import *
from medpy import metric
from skimage.io import imsave


class ModelEnsembler():
    def __init__(self, dir_original, dir_predictions, dir_ensemble, mode='test'):  # mode='evaluate'
        """
        测试模式，dir_original为原始文件夹，评估模式，dir_original为h5文件夹
        """
        self.dataset_props = load_pickle('dataset_props.pkl')
        self.dir_original = dir_original
        self.dir_predictions = dir_predictions
        self.dir_ensemble = dir_ensemble
        self.ids = sorted(os.listdir(dir_predictions[0]))
        if mode == 'evaluate':
            self.original_size = self.dataset_props['original_size']
            self.original_spacing = self.dataset_props['original_spacing']
        else:
            self.original_size = self.dataset_props['original_size_test']
            self.original_spacing = self.dataset_props['original_spacing_test']
        self.num_class = len(self.dataset_props['label'].keys())
        self.mode = mode
    
    def run(self):
        res_list = []
        for id, size, spacing in zip(self.ids, self.original_size, self.original_spacing):
            score_map = 0
            for dir_prediction in self.dir_predictions:
                f = h5py.File('{}{}'.format(dir_prediction, id), 'r')
                score_map += np.array(f['score_map'])
                f.close()
            score_map /= len(self.dir_predictions)
            score_map = score_map = np.vstack([resize(score_map[i], size, order=3, mode='reflect', cval=0,
                                                      clip=True, preserve_range=True, anti_aliasing=False).astype('float32')[np.newaxis] for i in range(len(score_map))])

            if self.mode == 'evaluate':
                f = h5py.File('{}{}'.format(self.dir_original, id), 'r')
                seg = np.array(f['seg'])
                f.close()
                res, prediction = self.evaluate(score_map, seg, spacing, connectivity=1, use_post_process=False)
                print(res)
                res_list.append(res)
            else:
                prediction = np.argmax(score_map, axis=0)
                self.submit(id, prediction)
                # f = h5py.File('{}{}'.format(self.dir_ensemble, id), 'w')
                # f.create_dataset(name='score_map', shape=score_map.shape, data=score_map, chunks=True, compression=9)
                # f.create_dataset(name='prediction', shape=prediction.shape, data=prediction, chunks=True, compression=9)
                # f.close()
        print()

    
    def evaluate(self, score_map, seg, spacing, connectivity=1, use_post_process=False):
        seg = self.seg2one_hot(seg, self.num_class)
        res = dict()
        if use_post_process:
            prediction = self.post_process(score_map)
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
    
    def submit(self, id, prediction):
        img = sitk.ReadImage('{}case_{:05d}/imaging.nii.gz'.format(self.dir_original, int(id.split('.')[0])))
        prediction = prediction.transpose(2, 1, 0)
        prediction = sitk.GetImageFromArray(prediction)
        prediction.SetOrigin(img.GetOrigin())
        prediction.SetDirection(img.GetDirection())
        prediction.SetSpacing(img.GetSpacing())
        sitk.WriteImage(prediction, '{}prediction_{:05d}.nii.gz'.format(self.dir_ensemble, int(id.split('.')[0])))


    def seg2one_hot(self, seg, num_class):
        '''
        将分割图(seg)转化为one hot编码,比如seg大小为(512, 512),3个类别,转换为one——hot大小(3, 512, 512)
        '''
        one_hot = np.stack([seg == c for c in range(num_class)])
        return one_hot
    
    def post_process(self):
        pass

if __name__ == '__main__':
    dir_original = 'data_test/'
    dir_predictions = ['prediction/test_3d_val/', 'prediction/test_3d_train/']
    dir_ensemble = 'prediction/test_ensemble/'
    create_dir(dir_ensemble)
    ensembler = ModelEnsembler(dir_original, dir_predictions, dir_ensemble, mode='test')
    ensembler.run()