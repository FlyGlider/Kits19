from tools import *


class DatasetAnalyzer():
    def __init__(self, dir_h5_original, has_seg=True, num_process=2):
        self.num_process = num_process
        self.ids = get_ids(dir_h5_original)
        self.dataset_props = load_pickle('dataset_props.pkl')
        self.num_class = len(self.dataset_props['label'].keys())
        self.regions = list(range(1, self.num_class))  # 没有算背景
        self.regions.append(list(range(1, self.num_class)))  # [1, 2, 3, [1, 2, 3]]
        self.size = self.dataset_props['original_size']
        self.spacing = self.dataset_props['original_spacing']
        self.has_seg = has_seg
    
    def run(self):
        if self.has_seg:
            p = Pool(self.num_process)
            res = p.map(self.analyze_seg, zip(self.ids, self.spacing))
            p.close()
            p.join()
            self.dataset_props['props_per_patient'] = dict()
            for id, (class_in_this_seg, all_in_one_region, vol_per_class, region_vol_per_class) in zip(self.ids, res):
                self.dataset_props['props_per_patient'][os.path.split(id)[1]] = dict()
                self.dataset_props['props_per_patient'][os.path.split(id)[1]]['class_in_this_seg'] = class_in_this_seg
                self.dataset_props['props_per_patient'][os.path.split(id)[1]]['all_in_one_region'] = all_in_one_region
                self.dataset_props['props_per_patient'][os.path.split(id)[1]]['vol_per_class'] = vol_per_class
                self.dataset_props['props_per_patient'][os.path.split(id)[1]]['region_vol_per_class'] = region_vol_per_class

        p = Pool(self.num_process)
        res = p.map(self.analyze_img, self.ids)
        p.close()
        p.join()
        total_voxels = np.zeros(0)
        for id, (voxels_in_foreground, median, mean, sd, mn, mx, percentile_99_5, percentile_00_5) in zip(self.ids, res):
            total_voxels = np.hstack((total_voxels, voxels_in_foreground))
            self.dataset_props['props_per_patient'][os.path.split(id)[1]]['median'] = median
            self.dataset_props['props_per_patient'][os.path.split(id)[1]]['mean'] = mean
            self.dataset_props['props_per_patient'][os.path.split(id)[1]]['sd'] = sd
            self.dataset_props['props_per_patient'][os.path.split(id)[1]]['mn'] = mn
            self.dataset_props['props_per_patient'][os.path.split(id)[1]]['mx'] = mx
            self.dataset_props['props_per_patient'][os.path.split(id)[1]]['percentile_99_5'] = percentile_99_5
            self.dataset_props['props_per_patient'][os.path.split(id)[1]]['percentile_00_5'] = percentile_00_5
        median, mean, sd, mn, mx, percentile_99_5, percentile_00_5 = self.compute_stats(total_voxels)
        self.dataset_props['median'] = median
        self.dataset_props['mean'] = mean
        self.dataset_props['sd'] = sd
        self.dataset_props['mn'] = mn
        self.dataset_props['mx'] = mx
        self.dataset_props['percentile_99_5'] = percentile_99_5
        self.dataset_props['percentile_00_5'] = percentile_00_5

        save_pickle(self.dataset_props, 'dataset_props.pkl')
        
    def analyze_img(self, img_id):
        voxels_in_foreground = self.get_voxels_in_foreground(img_id)
        median, mean, sd, mn, mx, percentile_99_5, percentile_00_5 = self.compute_stats(voxels_in_foreground)
        return voxels_in_foreground, median, mean, sd, mn, mx, percentile_99_5, percentile_00_5

    def analyze_seg(self, args):
        seg_id, spacing = args
        f = h5py.File(seg_id, 'r')
        seg = np.array(f['seg'])
        f.close()
        vol_per_voxel = np.prod(spacing)
        class_in_this_seg = np.unique(seg)
        all_in_one_region = self.check_if_all_in_one_region(seg, self.regions)
        vol_per_class, region_vol_per_class = self.collect_class_and_region_size(seg, self.num_class, vol_per_voxel)
        return class_in_this_seg, all_in_one_region, vol_per_class, region_vol_per_class

    def get_voxels_in_foreground(self, img_id):
        f = h5py.File(img_id, 'r')
        img = np.array(f['data'])
        seg = np.array(f['seg'])
        f.close()
        voxels = img[seg > 0][::10]  # no need to take every voxel
        return voxels

    def compute_stats(self, voxels):
        '''
        输入前景体素
        返回中位数，均值，方差，最小值，最大值，99.5%体素值，0.5体素值
        '''
        if len(voxels) == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        median = np.median(voxels)
        mean = np.mean(voxels)
        sd = np.std(voxels)
        mn = np.min(voxels)
        mx = np.max(voxels)
        percentile_99_5 = np.percentile(voxels, 99.5)
        percentile_00_5 = np.percentile(voxels, 00.5)
        return median, mean, sd, mn, mx, percentile_99_5, percentile_00_5

    def check_if_all_in_one_region(self, seg, regions):
        '''
        检查每个类别是否在同一区域以及所有类别是否在同一区域
        '''
        all_in_one_region = []
        for r in regions[:-1]:
            new_seg = np.zeros_like(seg)
            new_seg[new_seg == r] = 1
            _, num_label = label(new_seg, return_num=True)
            all_in_one_region.append(num_label)
        new_seg = np.zeros_like(seg)
        for r in regions[-1]:
            new_seg[seg == r] = 1
        _, num_label = label(new_seg, return_num=True)
        all_in_one_region.append(num_label)
        all_in_one_region = [True if x == 1 else False for x in all_in_one_region]  # [True, False, False, False]
        return all_in_one_region
    
    def collect_class_and_region_size(self, seg, num_class, vol_per_voxel):
        '''
        每个类别所占的体积和每个类别的每个区域所占的体积(不计算背景)
        '''
        vol_per_class = []
        region_vol_per_class = dict()
        for c in range(1, num_class):
            vol_per_class.append(np.sum(seg == c) * vol_per_voxel)
            region_vol_per_class[c] = []
            label_map, num_label = label(seg == c, return_num=True)
            for l in range(1, num_label + 1):
                region_vol_per_class[c].append(np.sum((label_map == l)) * vol_per_voxel)
        return vol_per_class, region_vol_per_class
    

if __name__ == '__main__':
    dir_h5_original = 'h5_data_original/'
    dataset_analyzer = DatasetAnalyzer(dir_h5_original)
    dataset_analyzer.run()
