import sys
sys.path.append('.')
sys.path.append('./models')
sys.path.append('./utils/')
import argparse
from models.unet import UNet
from models.resunet import ResUNet
from models.pspunet import PSPUNet
from models.asppunet import ASPPUNet
from sklearn.model_selection import KFold
from tools import *
from dataset import *
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from utils.loss import *
import torch.optim as optim
from torch.nn import init


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', dest='epoch', default=100, type=int,
                        help='number of epochs')
    parser.add_argument('-b', '--batch-size', dest='batchsize', default=2,
                        type=int, help='batch size')
    parser.add_argument('-l', '--learning-rate', dest='lr', default=0.0003,
                        type=float, help='learning rate')
    parser.add_argument('-g', '--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('-v', '--val', dest='val', default=0,
                        type=int, help='choose which validation')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')  # checkpoints_2d/model_best_0.pth
    parser.add_argument('-w', '--num_workers', dest='num_workers', default=0,
                        type=int, help='how many subprocesses to use for data loading')
    parser.add_argument('-p', '--pre_load', dest='pre_load', default=False,
                        type=bool, help='whether to pre-load dataset')  # 实际上只要输入就是True
    parser.add_argument('--ts', dest='train_samples', default=1000,
                        type=int, help='how many train sample in one epoch')
    parser.add_argument('--vs', dest='val_samples', default=100,
                        type=int, help='how many val sample in one epoch')
    args = parser.parse_args()
    return args

class Trainer():
    def __init__(self, model, dir_h5_train, dir_checkpoint, args):
        self.dataset_props = load_pickle('dataset_props.pkl')
        self.dir_h5_train = dir_h5_train
        self.dir_checkpoint = dir_checkpoint
        self.args = args
        self.device = torch.device('cuda')
        self.model = model.to(self.device)
        self.init_weight(self.model)
        self.criterions = [nn.CrossEntropyLoss().to(self.device), Criterion_dice(do_bg=False).to(self.device)]
        self.criterions_item = [AverageMeter(), AverageMeter(), AverageMeter()]  # ce, dice, ce+dice
        self.criterions_item_epoch = [np.zeros(args.epoch), np.zeros(args.epoch), np.zeros(args.epoch)]  # ce, dice, ce+dice
        self.criterions_item_epoch_val = [np.zeros(args.epoch), np.zeros(args.epoch), np.zeros(args.epoch)]  # ce, dice, ce+dice
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2, patience=20,
                                                                 verbose=True, threshold=1e-3, threshold_mode='abs')
        self.best_train_loss = None
        self.best_val_loss = None
        self.train_loader, self.val_loader = self.split_train_val()
        
        if args.resume != '':
            self.start_epoch = self.resume()
        else:
            self.start_epoch = 0
        
        print(args)

    def run(self):
        for epoch in range(self.start_epoch, self.args.epoch):
            train_loss = self.train(epoch)
            val_loss = self.validate(epoch)
            if self.best_train_loss is None:
                self.best_train_loss = train_loss
            elif train_loss < self.best_train_loss:
                self.best_train_loss = train_loss
                self.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'best_loss': self.best_train_loss,
                    'optimizer': self.optimizer.state_dict()
                }, 'train')
            if self.best_val_loss is None:
                self.best_val_loss = val_loss
            elif val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'best_loss': self.best_val_loss,
                    'optimizer': self.optimizer.state_dict()
                }, 'val')
            self.lr_scheduler.step(train_loss)
            if self.optimizer.param_groups[0]['lr'] < 1e-6:
                break
        print(self.best_train_loss, self.best_val_loss)
        np.save('{}train_{}.npy'.format(self.dir_checkpoint, self.args.val), np.array(self.criterions_item_epoch))
        np.save('{}val_{}.npy'.format(self.dir_checkpoint, self.args.val), np.array(self.criterions_item_epoch_val))
        # self.plot_curves(self.criterions_item_epoch[0], 'train_ce')
        # self.plot_curves(self.criterions_item_epoch[1], 'train_dice')
        # self.plot_curves(self.criterions_item_epoch[2], 'train_ce_dice')
        # self.plot_curves(self.criterions_item_epoch_val[0], 'val_ce')
        # self.plot_curves(self.criterions_item_epoch_val[1], 'val_dice')
        # self.plot_curves(self.criterions_item_epoch_val[2], 'val_ce_dice')
    
    def train(self, epoch):
        for criterion_item in self.criterions_item:
            criterion_item.reset()
        self.model.train()
        for i, (data, seg) in enumerate(self.train_loader):
            data = data.to(self.device)
            seg = seg.to(self.device)
            outputs = self.model(data)
            total_loss = 0
            for criterion, criterion_item in zip(self.criterions, self.criterions_item[:-1]):
                loss = 0
                for output in outputs:
                    loss += criterion(output, seg)
                loss /= len(outputs)
                criterion_item.update(loss.item())
                total_loss += loss
            self.criterions_item[-1].update(total_loss.item())
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            if (i + 1) % (self.args.train_samples // 10) == 0:
                print('Epoch: [{}][{}/{}]\t ce:{:.4f}\t dice:{:.4f}\t ce+dice:{:.4f}'.format(epoch + 1, i + 1, len(self.train_loader),
                      self.criterions_item[0].val, self.criterions_item[1].val, self.criterions_item[2].val))
        for criterion_item_epoch, criterion_item in zip(self.criterions_item_epoch, self.criterions_item):
            criterion_item_epoch[epoch] = criterion_item.avg
        
        print('Train Epoch: [{}]\t avg ce:{:.4f}\t dice:{:.4f}\t ce+dice:{:.4f}'.format(epoch + 1,
              self.criterions_item[0].avg, self.criterions_item[1].avg, self.criterions_item[2].avg))

        self.criterions_item[-1].update_ma()
        return self.criterions_item[-1].ma
    
    def validate(self, epoch):
        for criterion_item in self.criterions_item:
            criterion_item.reset()
        self.model.eval()
        with torch.no_grad():
            for data, seg in self.val_loader:
                data = data.to(self.device)
                seg = seg.to(self.device)
                outputs = self.model(data)
                total_loss = 0
                for criterion, criterion_item in zip(self.criterions, self.criterions_item[:-1]):
                    loss = 0
                    for output in outputs:
                        loss += criterion(output, seg)
                    loss /= len(outputs)
                    criterion_item.update(loss.item())
                    total_loss += loss
                self.criterions_item[-1].update(total_loss.item())
            for criterion_item_epoch_val, criterion_item in zip(self.criterions_item_epoch_val, self.criterions_item):
                criterion_item_epoch_val[epoch] = criterion_item.avg
            
            print('Val Epoch: [{}]\t avg ce:{:.4f}\t dice:{:.4f}\t ce+dice:{:.4f}'.format(epoch + 1,
                  self.criterions_item[0].avg, self.criterions_item[1].avg, self.criterions_item[2].avg))
            
        self.criterions_item[-1].update_ma_val()
        return self.criterions_item[-1].avg

    def split_train_val(self):
        # 确定下标
        all_ids = get_ids(self.dir_h5_train)
        kf = KFold(n_splits=5, shuffle=True, random_state=125)
        train_index, val_index = list(kf.split(all_ids))[self.args.val]
        train_ids = [all_ids[x] for x in train_index]
        val_ids = [all_ids[x] for x in val_index]
        
        # 根据模式选择数据库
        if self.model.mode == '2D':
            train_dataset = BaseDataset2D(ids=train_ids, patch_size=self.dataset_props['plan_2d']['patch_size'],
                                          force_fg=1 / 3, pre_load=self.args.pre_load, return_patch=True)
            val_dataset = BaseDataset2D(ids=val_ids, patch_size=self.dataset_props['plan_2d']['patch_size'],
                                        force_fg=1, pre_load=self.args.pre_load, return_patch=True)
            
        elif self.model.mode == '3D':
            train_dataset = BaseDataset3D(ids=train_ids, patch_size=self.dataset_props['plan_3d']['patch_size'],
                                          force_fg=1 / 3, pre_load=self.args.pre_load, return_patch=True)
            val_dataset = BaseDataset3D(ids=val_ids, patch_size=self.dataset_props['plan_3d']['patch_size'],
                                        force_fg=1, pre_load=self.args.pre_load, return_patch=True)
        elif self.model.mode == '3D_LOW':
            train_dataset = BaseDataset3D(ids=train_ids, patch_size=self.dataset_props['plan_3d_low']['patch_size'],
                                          force_fg=1 / 3, pre_load=self.args.pre_load, return_patch=True, aug=1 / 2)
            val_dataset = BaseDataset3D(ids=val_ids, patch_size=self.dataset_props['plan_3d_low']['patch_size'],
                                        force_fg=1, pre_load=self.args.pre_load, return_patch=True)
        else:
            raise NotImplementedError('mode [%s] is not found' % self.model.mode)

        train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=self.args.batchsize * self.args.train_samples)
        train_loader = DataLoader(
            train_dataset, batch_size=self.args.batchsize, sampler=train_sampler, num_workers=self.args.num_workers)
        val_sampler = RandomSampler(val_dataset, replacement=True, num_samples=self.args.batchsize * self.args.val_samples)
        val_loader = DataLoader(
            val_dataset, batch_size=self.args.batchsize, sampler=val_sampler, num_workers=self.args.num_workers)

        return train_loader, val_loader

    
    def resume(self):
        if os.path.isfile(self.args.resume):
            checkpoint = torch.load(self.args.resume)
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}) (best_loss:{})"
                  .format(self.args.resume, start_epoch, best_loss))
            return start_epoch
        else:
            raise RuntimeError(
                "=> no checkpoint found at '{}'".format(args.resume))

                
    def init_weight(self, model, init_type='xavier', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm') != -1:
                init.normal_(m.weight, 1.0, gain)
                init.constant_(m.bias, 0.0)

        print('initialize network with %s' % init_type)
        model.apply(init_func)
    
    def save_checkpoint(self, state, stage):
        filename = '{}model_best_{}_{}.pth'.format(self.dir_checkpoint, stage, self.args.val)
        torch.save(state, filename)
    
    def plot_curves(self, array, label):
        plt.plot(np.arange(self.args.epoch), array, 'r', label=label)
        plt.legend(loc='best')
        plt.savefig('{}{}_{}.png'.format(self.dir_checkpoint, label, self.args.val))
        plt.cla()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()
        self.ma = None
        self.ma_val = None
        self.alpha = 0.93

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def update_ma(self):
        if self.ma is None:
            self.ma = self.avg
        else:
            self.ma = self.alpha * self.ma + (1 - self.alpha) * self.avg
    def update_ma_val(self):
        if self.ma_val is None:
            self.ma_val = self.avg
        else:
            self.ma_val = self.alpha * self.ma_val + (1 - self.alpha) * self.avg

if __name__ == '__main__':
    dir_h5_train = 'h5_data_train_3d_low/'
    dir_checkpoint = 'checkpoints/3d_low/'
    create_dir(dir_checkpoint)
    dataset_props = load_pickle('dataset_props.pkl')
    pool_layer_kernel_sizes = dataset_props['plan_3d_low']['pool_layer_kernel_sizes']
    model = ResUNet(in_ch=1, base_num_features=30, num_classes=3, norm_type='batch', nonlin_type='relu', pool_type='max',
                    pool_layer_kernel_sizes=pool_layer_kernel_sizes, deep_supervision=True, mode='3D_LOW')
    args = get_args()
    trainer = Trainer(model, dir_h5_train, dir_checkpoint, args)
    trainer.run()
