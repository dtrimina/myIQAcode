import pandas as pd
import numpy as np
import torchvision, torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn.functional as F


class TrainPatchDataset(Dataset):

    def __init__(self, root, info, num_patches_per_img, Tsal, patch_size=32):
        super(TrainPatchDataset, self).__init__()
        self.root = root
        self.info = info
        self.num_patches_per_img = num_patches_per_img
        self.Tsal = Tsal
        self.patch_size = patch_size

        self.LRimgs, self.saliencys, self.dmoses, self.len = self._get_patches()

    def __len__(self):
        return self.len

    @property
    def _saliency_filter(self):
        size = self.patch_size
        return torch.ones((1, 1, size, size)) / (size * size)

    def _get_patches(self):

        totensor = torchvision.transforms.ToTensor()

        LRimgs = []
        saliencys = []
        dmoses = []

        for i in range(len(self.info)):
            info_i = self.info.iloc[i]
            disName = info_i['disName']
            disType = info_i['disType']
            L = totensor(Image.open(f'{self.root}/dis_imgs/{disType}/L/{disName}')).unsqueeze(0)  # size=[1, 3, 360, 480]
            R = totensor(Image.open(f'{self.root}/dis_imgs/{disType}/R/{disName}')).unsqueeze(0)
            LRimg = torch.cat([L, R], dim=0)  # [2, 3, 360, 480]

            L_saliency = totensor(Image.open(f'{self.root}/saliency/GBVS/{disType}/L/{disName}')).unsqueeze(0)  # size=[1, 1, 360, 640]
            R_saliency = totensor(Image.open(f'{self.root}/saliency/GBVS/{disType}/R/{disName}')).unsqueeze(0)

            L_sal_mean = F.conv2d(L_saliency, self._saliency_filter)
            R_sal_mean = F.conv2d(R_saliency, self._saliency_filter)
            patch_sal = ((L_sal_mean + R_sal_mean) / 2).squeeze()  # torch.Size([329, 609])

            satisfied_Tsal_indexs = torch.nonzero(patch_sal >= self.Tsal)
            num_satisfied = satisfied_Tsal_indexs.shape[0]
            satisfied_num_patches_per_img = min(self.num_patches_per_img, num_satisfied)

            random_indexes = torch.randint(0, num_satisfied, size=(satisfied_num_patches_per_img,))
            choosed_indexes = satisfied_Tsal_indexs[random_indexes, ...]

            for j in range(satisfied_num_patches_per_img):
                x, y = choosed_indexes[j]
                saliencys.append(patch_sal[x, y])
                LRimgs.append(LRimg[:, :, x:x+32, y:y+32].unsqueeze(0))
                dmoses.append(info_i['dmos'])

        LRimgs = torch.cat(LRimgs, dim=0)
        saliencys = torch.tensor(saliencys)
        dmoses = torch.tensor(dmoses)
        return LRimgs, saliencys, dmoses, dmoses.shape[0]

    def __getitem__(self, index):

        return {
            'LRimg': self.LRimgs[index],
            'dmos': self.dmoses[index]
        }


class TestPatchDataset(Dataset):

    def __init__(self, root, info, use_sal, patch_size=32):
        super(TestPatchDataset, self).__init__()
        self.root = root
        self.info = info
        self.use_sal = use_sal
        self.patch_size = patch_size

        self.LRimgs, self.saliencys, self.dmoses, self.len = self._test_processing()

    def __len__(self):
        return self.len

    @property
    def _saliency_filter(self):
        size = self.patch_size
        return torch.ones((1, 1, size, size)) / (size * size)

    def _test_processing(self):

        totensor = torchvision.transforms.ToTensor()

        LRimgs = []
        saliencys = []
        dmoses = []

        for i in range(len(self.info)):
            info_i = self.info.iloc[i]
            disName = info_i['disName']
            disType = info_i['disType']
            L = totensor(Image.open(f'{self.root}/dis_imgs/{disType}/L/{disName}')).unsqueeze(0)  # size=[1, 3, 360, 480]
            R = totensor(Image.open(f'{self.root}/dis_imgs/{disType}/R/{disName}')).unsqueeze(0)
            LRimg = torch.cat([L, R], dim=0)  # [2, 3, 360, 480]

            if self.use_sal:

                L_saliency = totensor(Image.open(f'{self.root}/saliency/GBVS/{disType}/L/{disName}')).unsqueeze(0)  # size=[1, 1, 360, 640]
                R_saliency = totensor(Image.open(f'{self.root}/saliency/GBVS/{disType}/R/{disName}')).unsqueeze(0)

                L_sal_mean = F.conv2d(L_saliency, self._saliency_filter, stride=2)
                R_sal_mean = F.conv2d(R_saliency, self._saliency_filter, stride=2)
                patch_sal = (L_sal_mean + R_sal_mean) / 2  # torch.Size([1, 1, 329, 609])
            else:
                patch_sal = torch.ones(size=(1, 1, IQADataset.H-self.patch_size+1, IQADataset.W-self.patch_size+1))

            LRimgs.append(LRimg.unsqueeze(0))
            saliencys.append(patch_sal)
            dmoses.append(info_i['dmos'])

        LRimgs = torch.cat(LRimgs, dim=0)
        saliencys = torch.cat(saliencys)
        dmoses = torch.tensor(dmoses)
        return LRimgs, saliencys, dmoses, dmoses.shape[0]

    def __getitem__(self, index):

        return {
            'LRimg': self.LRimgs[index],
            'saliency': self.saliencys[index],
            'dmos': self.dmoses[index]
        }


class IQADataset(object):

    # default paramenters for Live3D datasets
    H, W = 360, 640

    def __init__(self, root,
                 dataset,
                 randomstate,
                 splite_by='distype',
                 use_sal = True,
                 num_patches_per_img=50,
                 Tsal=0, trainratio=0.8, patch_size=32):

        self.root = f'{root}/{dataset}'
        self.dataset = dataset
        self.randomstate = randomstate
        self.splite_by = splite_by
        self.use_sal = use_sal
        self.num_patches_per_image = num_patches_per_img
        self.Tsal = Tsal
        self.trainratio = trainratio
        self.patch_size = patch_size

        self.distypes = ['ff', 'gb', 'jg', 'jk', 'wn']
        self.distype_trains = ['ff_train', 'gb_train', 'jg_train', 'jk_train', 'wn_train']
        self.distype_tests = ['ff_test', 'gb_test', 'jg_test', 'jk_test', 'wn_test']

        self.infos = self._splite(randomstate, trainratio)

    @property
    def _info(self):
        return pd.read_csv(f'{self.root}/{self.dataset}.csv')

    def __len__(self):
        return len(self._info)

    def _splite(self, randomstate, trainratio):
        info = self._info

        if self.splite_by == 'random':
            traininfo, testinfo = train_test_split(info, train_size=trainratio, random_state=randomstate)
            ff_train, ff_test = traininfo[traininfo['disType'] == 'ff'], testinfo[testinfo['disType'] == 'ff']
            gb_train, gb_test = traininfo[traininfo['disType'] == 'gblur'], testinfo[testinfo['disType'] == 'gblur']
            jg_train, jg_test = traininfo[traininfo['disType'] == 'jpeg'], testinfo[testinfo['disType'] == 'jpeg']
            jk_train, jk_test = traininfo[traininfo['disType'] == 'jp2k'], testinfo[testinfo['disType'] == 'jp2k']
            wn_train, wn_test = traininfo[traininfo['disType'] == 'wn'], testinfo[testinfo['disType'] == 'wn']

        elif self.splite_by == 'distype':
            ff_train, ff_test = train_test_split(info[info['disType'] == 'ff'], train_size=trainratio, random_state=randomstate)
            gb_train, gb_test = train_test_split(info[info['disType'] == 'gblur'], train_size=trainratio, random_state=randomstate)
            jg_train, jg_test = train_test_split(info[info['disType'] == 'jpeg'], train_size=trainratio, random_state=randomstate)
            jk_train, jk_test = train_test_split(info[info['disType'] == 'jp2k'], train_size=trainratio, random_state=randomstate)
            wn_train, wn_test = train_test_split(info[info['disType'] == 'wn'], train_size=trainratio, random_state=randomstate)
            traininfo = pd.concat([ff_train, gb_train, jg_train, jk_train, wn_train], axis=0)
            testinfo = pd.concat([ff_test, gb_test, jg_test, jk_test, wn_test], axis=0)
        else:
            raise ValueError('splite_by should be "random" or "distype"')

        infos = {
            'ff_train': ff_train, 'ff_test': ff_test,
            'gb_train': gb_train, 'gb_test': gb_test,
            'jg_train': jg_train, 'jg_test': jg_test,
            'jk_train': jk_train, 'jk_test': jk_test,
            'wn_train': wn_train, 'wn_test': wn_test,
            'traininfo': traininfo, 'testinfo': testinfo,
        }

        return infos

    def train_test_datasets(self):
        return {
            'trainset': TrainPatchDataset(self.root, self.infos['traininfo'], self.num_patches_per_image, self.Tsal, self.patch_size),
            'testset': TestPatchDataset(self.root, self.infos['testinfo'], self.use_sal, self.patch_size)
        }

    def train_test_datasets_for_distypes(self):

        datasets = {}
        for train in self.distype_trains:
            datasets[train] = TrainPatchDataset(self.root, self.infos[train], self.num_patches_per_image, self.Tsal, self.patch_size)
        for test in self.distype_tests:
            datasets[test] = TestPatchDataset(self.root, self.infos[test], self.use_sal, self.patch_size)

        return datasets

    def load_loader(self, trainbs, testbs, shuffle=True, pin_memory=True, num_workers=8):
        datasets = self.train_test_datasets()
        train_loader = DataLoader(datasets['trainset'], trainbs, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)
        test_loader = DataLoader(datasets['testset'], testbs, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)
        return train_loader, test_loader

    def load_loader_for_distypes(self, trainbs, testbs, shuffle=True, pin_memory=True, num_workers=8):
        dataloaders = {}
        datasets = self.train_test_datasets_for_distypes()

        for train in self.distype_trains:
            dataloaders[train] = DataLoader(datasets[train], trainbs, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)
        for test in self.distype_tests:
            dataloaders[test] = DataLoader(datasets[test], testbs, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)

        return dataloaders


if __name__ == '__main__':
    root = '/home/dtrimina/disk-1T/3D'
    dataset = 'LivePhase2'
    randomstate = 999

    iqadataset = IQADataset(root, dataset, randomstate)
    print(len(iqadataset))

    tr_loader, test_loader = iqadataset.load_loader(128, 8)
    print(len(tr_loader), len(test_loader))
    for sample in tr_loader:
        LRimg = sample['LRimg']
        dmos = sample['dmos']
        print(LRimg.shape, dmos.shape)

    for sample in test_loader:
        LRimg = sample['LRimg']
        saliency = sample['saliency']
        dmos = sample['dmos']
        print(LRimg.shape, saliency.shape, dmos.shape)

    dataloaders = iqadataset.load_loader_for_distypes(128, 8)
    for key, value in enumerate(dataloaders):
        print(len(value))




