import sys
sys.path.append('utils')
import argparse
import glob
import os

import SimpleITK as sitk
from utils.tools import *
from utils.preprocesser import Preprocesser
from utils.cropper import Cropper
from utils.dataset_analyzer import DatasetAnalyzer
from utils.experiment_planner import ExperimentPlanner


class H5DataPreparer():
    def __init__(self, dir_original, dir_h5_original, dirs_h5_train, num_process=2, mode='train'):
        self.dir_original = dir_original
        self.dir_h5_original = dir_h5_original
        self.dirs_h5_train = dirs_h5_train
        self.num_process = num_process
        self.mode = mode
        
    
    # 图像用float32,seg用int8
    def read_data(self):
        create_dir(self.dir_h5_original)
        original_size = []
        original_spacing = []
        for root, _, files in sorted(os.walk(self.dir_original)):
            if 'imaging.nii.gz' not in files:
                continue
            files.sort()  # windows和linux顺序不一样
            img = sitk.ReadImage(os.path.join(root, files[0]))
            img_array = sitk.GetArrayFromImage(img).transpose(2, 1, 0).astype('float32')  # 轴置换也可以自动 最大spacing的轴放在前面

            spacing = np.array(img.GetSpacing())

            original_size.append(img_array.shape)
            original_spacing.append(spacing)
            
            f = h5py.File('{}{}.h5'.format(self.dir_h5_original, root[-3:]), 'w')
            f.create_dataset(name='data', shape=img_array.shape, data=img_array, chunks=True)
            f.create_dataset(name='spacing', shape=spacing.shape, data=spacing, chunks=True, compression=9)

            if 'segmentation.nii.gz' in files:
                seg_array = sitk.GetArrayFromImage(seg).transpose(2, 1, 0).astype(np.int8)
                seg = sitk.ReadImage(os.path.join(root, files[1]))
                f.create_dataset(name='seg', shape=seg_array.shape, data=seg_array, chunks=True, compression=9)
            f.close()
            print(root)

        if self.mode == 'train':
            dataset_props = {}
            dataset_props['label'] = {'0': 'background', '1': 'kidney', '2': 'tumor'}
            dataset_props['original_size'] = original_size
            dataset_props['original_spacing'] = original_spacing
            dataset_props['modality'] = {'0': 'CT'}

            save_pickle(dataset_props, 'dataset_props.pkl')
        else:
            dataset_props = load_pickle('dataset_props.pkl')
            dataset_props['original_size_test'] = original_size
            dataset_props['original_spacing_test'] = original_spacing
            # 3d
            target_spacing = dataset_props['plan_3d']['target_spacing']
            target_size = [np.round(np.array(i) * j / target_spacing) for i, j in zip(original_size, original_spacing)]
            dataset_props['plan_3d']['target_size_test'] = target_size
            # 2d
            patch_size = dataset_props['plan_2d']['patch_size']
            target_size = [i[:1] + tuple(patch_size) for i in original_size]
            dataset_props['plan_2d']['target_size_test'] = target_size
            save_pickle(dataset_props, 'dataset_props.pkl')


    def process_h5_data(self):
        if self.mode == 'train':
            cropper = Cropper(self.dir_h5_original)
            cropper.run()
            analyzer = DatasetAnalyzer(self.dir_h5_original, has_seg=True, num_process=self.num_process)
            analyzer.run()
            planner = ExperimentPlanner()
            planner.run()
        preprocesser = Preprocesser(dir_h5_original=self.dir_h5_original, dirs_h5_train=self.dirs_h5_train, num_process=self.num_process, mode=self.mode)
        preprocesser.run()
    
    
if __name__ == '__main__':
    dir_original = 'data_test'
    dir_h5_original = 'h5_data_test_original/'
    dirs_h5_train = ['h5_data_test_3d/', 'h5_data_test_2d/']  # ['h5_data_train_3d/', 'h5_data_train_2d/', 'h5_data_train_3d_low/']
    PHD = H5DataPreparer(dir_original, dir_h5_original, dirs_h5_train, num_process=2, mode='test')
    # PHD.read_data()
    PHD.process_h5_data()
