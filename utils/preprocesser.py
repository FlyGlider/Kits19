from tools import *


class Preprocesser():
    def __init__(self, dir_h5_original, dirs_h5_train, num_process, mode='train'):
        self.dataset_props = load_pickle('dataset_props.pkl')
        self.dir_h5_original = dir_h5_original
        self.dirs_h5_train = dirs_h5_train
        self.num_process = num_process
        self.ids = get_ids(dir_h5_original)
        if mode == 'train':
            self.target_sizes = [self.dataset_props['plan_3d']['target_size'], self.dataset_props['plan_2d']['target_size']]
        else:
            self.target_sizes = [self.dataset_props['plan_3d']['target_size_test'], self.dataset_props['plan_2d']['target_size_test']]
        # if 'plan_3d_low' in self.dataset_props.keys():
        #     self.target_sizes.append(self.dataset_props['plan_3d_low']['target_size'])
        for dir_h5_train in self.dirs_h5_train:
            create_dir(dir_h5_train)
    
    def run(self):
        for i, target_size in enumerate(self.target_sizes):
            p = Pool(self.num_process)
            res = p.map(self.resample_and_normalize, zip(self.ids, target_size, [i] * len(self.ids)))
            p.close()
            p.join()
        print('resample and normalization are done')
    
    def resample_and_normalize(self, args):
        id, target_size, i = args
        f1 = h5py.File(id, 'r')
        f2 = h5py.File('{}{}'.format(self.dirs_h5_train[i], os.path.split(id)[1]), 'w')
        data = np.array(f1['data'])
        # resample
        data = resize(data, target_size, order=3, mode='reflect', cval=0, clip=True, preserve_range=True, anti_aliasing=False).astype('float32')  # 不使用模糊的话平均值和方差基本不变
        # normalize
        if self.dataset_props['modality']['0'] == 'CT':
            low_bound = self.dataset_props['percentile_00_5'] - 100
            upper_bound = self.dataset_props['percentile_99_5']
            data = np.clip(data, low_bound, upper_bound)
        data = (data - data.mean()) / data.std()
        
        f2.create_dataset(name='data', shape=data.shape, data=data, chunks=True, compression=9)
        if 'seg' in list(f1.keys()):
            seg = np.array(f1['seg'])
            seg = resize(seg, target_size, order=0, mode='edge', cval=0, clip=True, preserve_range=True, anti_aliasing=False).astype(np.int8)
            bboxes = self.get_bbox(seg)
            f2.create_dataset(name='seg', shape=seg.shape, data=seg, chunks=True, compression=9)
            f2.create_dataset(name='bboxes', shape=bboxes.shape, data=bboxes, chunks=True, compression=9)
        f1.close()
        f2.close()
    
    def get_bbox(self, seg):
        props = regionprops(seg)
        bboxes = [x['bbox'] for x in props]
        bboxes = np.array(bboxes)
        return bboxes

if __name__ == '__main__':
    dir_h5_original = 'h5_data_original/'
    dirs_h5_train = ['h5_data_train_3d/', 'h5_data_train_2d/', 'h5_data_train_3d_low/']
    preprocesser = Preprocesser(dir_h5_original, dirs_h5_train, num_process=2)
    preprocesser.run()