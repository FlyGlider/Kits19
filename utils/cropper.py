from tools import *

# 需要记录bbox
class Cropper():
    def __init__(self, dir_h5_original):
        self.ids = get_ids(dir_h5_original)
        self.dataset_props = load_pickle('dataset_props.pkl')
        self.dataset_props['size_after_crop'] = []
        self.use_crop = False

    def run(self):
        if self.use_crop:
            for id in self.ids:
                f = h5py.File(id, 'r+')
                img = np.array(f['data'])
                nonzero_mask = self.create_nonzero_mask(img)
                bbox = self.get_bbox_from_mask(nonzero_mask)
                img = img[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
                f.create_dataset(name='data', shape=img.shape, data=img, chunks=True, compression=9)
                f.create_dataset(name='crop_bbox', shape=bbox.shape, data=bbox, chunks=True, compression=9)
                if 'seg' in list(f.keys()):
                    seg = np.array(f['seg'])
                    seg = seg[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
                    f.create_dataset(name='seg', shape=seg.shape, data=seg, chunks=True, compression=9)
                f.close()
                self.dataset_props['size_after_crop'].append(img.shape)  # 这里要是开多进程的话要改
        else:
            self.dataset_props['size_after_crop'] = self.dataset_props['original_size']
        
        self.dataset_props['size_reduction'] = [np.prod(size_original) / np.prod(size_after_crop) for size_original, size_after_crop in zip(
            self.dataset_props['original_size'], self.dataset_props['size_after_crop']
        )]
        save_pickle(self.dataset_props, 'dataset_props.pkl')
    
    def create_nonzero_mask(self, img):
        nonzero_mask = img != 0
        nonzero_mask = binary_fill_holes(nonzero_mask)
        return nonzero_mask.astype(np.int8)
    
    def get_bbox_from_mask(self, mask):
        props = regionprops(mask)
        bbox = props[0]['bbox']
        return np.array(bbox)

if __name__ == '__main__':
    dir_h5_original = 'h5_data_original/'
    cropper = Cropper(dir_h5_original)
    cropper.run()
