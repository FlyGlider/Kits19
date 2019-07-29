from tools import *


class ExperimentPlanner():
    max_size = np.array([128, 128, 128])

    def __init__(self):
        self.dataset_props = load_pickle('dataset_props.pkl')
        self.original_size = self.dataset_props['original_size']
    
    def run(self):
        self.determine_whether_to_use_mask_for_norm()

        target_spacing, target_size, patch_size, num_pool_per_axis, pool_layer_kernel_sizes, need_low = self.get_network_plan_3d(min_feature_map_size=4)
        self.dataset_props['plan_3d'] = dict()
        self.dataset_props['plan_3d']['target_spacing'] = target_spacing
        self.dataset_props['plan_3d']['target_size'] = target_size
        self.dataset_props['plan_3d']['patch_size'] = patch_size
        self.dataset_props['plan_3d']['num_pool_per_axis'] = num_pool_per_axis
        self.dataset_props['plan_3d']['pool_layer_kernel_sizes'] = pool_layer_kernel_sizes

        if need_low:
            self.dataset_props['plan_3d_low'] = dict()
            self.dataset_props['plan_3d_low']['target_size'] = [self.max_size] * len(target_size)
            self.dataset_props['plan_3d_low']['patch_size'] = self.max_size.tolist()
            self.dataset_props['plan_3d_low']['num_pool_per_axis'] = [5] * 3
            self.dataset_props['plan_3d_low']['pool_layer_kernel_sizes'] = [[2, 2, 2]] * 5

        # 这里将图像采样到xy平面固定大小
        target_spacing, target_size, patch_size, num_pool_per_axis, pool_layer_kernel_sizes = self.get_network_plan_2d(min_feature_map_size=6)
        self.dataset_props['plan_2d'] = dict()
        self.dataset_props['plan_2d']['target_spacing'] = target_spacing
        self.dataset_props['plan_2d']['target_size'] = target_size
        self.dataset_props['plan_2d']['patch_size'] = patch_size
        self.dataset_props['plan_2d']['num_pool_per_axis'] = num_pool_per_axis
        self.dataset_props['plan_2d']['pool_layer_kernel_sizes'] = pool_layer_kernel_sizes

        save_pickle(self.dataset_props, 'dataset_props.pkl')
    
    def determine_whether_to_use_mask_for_norm(self):
        """
        照这个意思，只有size_reduction到一定程度才会为True
        """
        self.dataset_props['use_nonzero_mask_for_norm'] = dict()
        modality = self.dataset_props['modality']
        for k, v in modality.items():
            if v == 'CT':
                self.dataset_props['use_nonzero_mask_for_norm'][k] = False
            elif np.median(self.dataset_props['size_reduction']) > 4 / 3:
                self.dataset_props['use_nonzero_mask_for_norm'][k] = True
            else:
                self.dataset_props['use_nonzero_mask_for_norm'][k] = False
        

    def get_network_plan_3d(self, min_feature_map_size):
        """
        [池化层是根据patchsize的大小确定，还是根据spacing大小确定？]
        
        Parameters
        ----------
        patch_size : [type]
            [description]
        min_featrue_map_size : [type]
            [description]
        spacing : [type]
            [description]
        
        Returns
        -------
        [type]
            [description]
        """
        original_spacing = self.dataset_props['original_spacing']
        original_size = self.dataset_props['original_size']
        target_spacing = np.median(np.vstack(original_spacing), axis=0)
        target_size = [np.round(np.array(i) * j / target_spacing) for i, j in zip(original_size, original_spacing)]

        target_size_median = np.median(np.vstack(target_size), axis=0)
        patch_size = target_size_median.copy()
        target_size_min = np.min(np.vstack(target_size), axis=0)
        while np.any(patch_size > target_size_min):
            patch_size = np.floor(patch_size / 1.1)
        
        while np.prod(patch_size) > np.prod(self.max_size):
            patch_size[patch_size == max(patch_size)] = np.floor(patch_size[patch_size == max(patch_size)] / 1.1)
        
        if np.prod(target_size_median / patch_size) > 4:
            need_low = True
        else:
            need_low = False
            
        num_pool_per_axis = np.floor([np.log(i / min_feature_map_size) / np.log(2) for i in patch_size]).astype(np.int8)
        num_pool_max = max(num_pool_per_axis)
        pool_layer_kernel_sizes = np.ones((num_pool_max, len(patch_size)), dtype=np.int8) * 2
        for i, j in enumerate(num_pool_max - num_pool_per_axis):
            pool_layer_kernel_sizes[:j, i] = 1
        shape_must_be_divisible_by = np.power(2, num_pool_per_axis)
        patch_size = [int(i - i % j) for i, j in zip(patch_size, shape_must_be_divisible_by)]

        return target_spacing, target_size, patch_size, num_pool_per_axis, pool_layer_kernel_sizes.tolist(), need_low
    

    def get_network_plan_2d(self, min_feature_map_size):
        original_size = self.dataset_props['original_size']
        original_spacing = self.dataset_props['original_spacing']
        patch_size = np.median(np.vstack(original_size), axis=0)[1:].astype(int)
        target_size = [i[:1] + tuple(patch_size) for i in original_size]
        target_spacing = [np.array(i) * j / np.array(k) for i, j, k in zip(original_size, original_spacing, target_size)]
        
        num_pool_per_axis = np.floor([np.log(i / min_feature_map_size) / np.log(2) for i in patch_size]).astype(np.int8)
        num_pool_max = max(num_pool_per_axis)
        pool_layer_kernel_sizes = np.ones((num_pool_max, len(patch_size)), dtype=np.int8) * 2
        for i, j in enumerate(num_pool_max - num_pool_per_axis):
            pool_layer_kernel_sizes[:j, i] = 1
        shape_must_be_divisible_by = np.power(2, num_pool_per_axis)
        patch_size = [int(i - i % j) for i, j in zip(patch_size, shape_must_be_divisible_by)]
        return target_spacing, target_size, patch_size, num_pool_per_axis, pool_layer_kernel_sizes.tolist()

if __name__ == '__main__':
    EP = ExperimentPlanner()
    EP.run()

