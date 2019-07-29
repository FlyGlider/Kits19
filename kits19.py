from utils.trainer import *
from utils.evaluator import Evaluator
# from utils.tester import Tester

plt.switch_backend('agg')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', dest='epoch', default=100, type=int,
                        help='number of epochs')
    parser.add_argument('-b', '--batch-size', dest='batchsize', default=1,
                        type=int, help='batch size')
    parser.add_argument('-l', '--learning-rate', dest='lr', default=0.0003,
                        type=float, help='learning rate')
    parser.add_argument('-g', '--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('-v', '--val', dest='val', default=0,
                        type=int, help='choose which validation')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH', 
                        help='path to latest checkpoint (default: none)')  # checkpoints_2d/model_best_0_200.pth
    parser.add_argument('-w', '--num_workers', dest='num_workers', default=0,
                        type=int, help='how many subprocesses to use for data loading')
    parser.add_argument('-p', '--pre_load', dest='pre_load', default=False,
                        type=bool, help='whether to pre-load dataset')  # 实际上只要输入就是True
    parser.add_argument('--ts', dest='train_samples', default=1000,
                        type=int, help='how many train sample in one epoch')
    parser.add_argument('--vs', dest='val_samples', default=100,
                        type=int, help='how many val sample in one epoch')
    parser.add_argument('-s', '--stage', dest='stage', default='train',
                        type=str, help='choose the best model in which stage')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    dir_h5_train = 'h5_data_train_2d/'
    dir_checkpoint = 'checkpoints/2d/'
    dir_prediction = 'predictions/2d/'
    create_dir(dir_checkpoint)
    create_dir(dir_prediction)
    dataset_props = load_pickle('dataset_props.pkl')
    pool_layer_kernel_sizes = dataset_props['plan_2d']['pool_layer_kernel_sizes']
    args = get_args()
    model = ResUNet(in_ch=1, base_num_features=30, num_classes=3, norm_type='batch', nonlin_type='relu', pool_type='max',
                    pool_layer_kernel_sizes=pool_layer_kernel_sizes, deep_supervision=True, mode='2D')
    trainer = Trainer(model, dir_h5_train, dir_checkpoint, args)
    trainer.run()
    model = ResUNet(in_ch=1, base_num_features=30, num_classes=3, norm_type='batch', nonlin_type='relu', pool_type='max',
                    pool_layer_kernel_sizes=pool_layer_kernel_sizes, deep_supervision=False, mode='2D')
    evaluator = Evaluator(model, dir_h5_train, dir_checkpoint, dir_prediction, args)
    evaluator.run()