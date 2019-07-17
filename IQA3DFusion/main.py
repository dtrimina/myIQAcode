import torch

from utiles.train_and_eval import train_3D
from datasets import IQADataset
import warnings

from models.weight_score_fusion import Proposed
from models.net import Net
from models.fusion import FusionNet

warnings.filterwarnings('ignore')


def train_model(args):
    if args.net_name == 'FusionNet':
        model = FusionNet()
    else:
        raise ('no such model')

    if args.transfer_path is not None:
        print('load pre trained model from {}.'.format(args.transfer_path))
        model.load_state_dict(torch.load(args.transfer_path))

    lr = 0.01
    n_ep = 100

    root = '/home/dtrimina/disk-1T/3D'
    dataset = 'LivePhase2'
    randomstate = 9997
    splite_by = 'distype'
    use_sal = args.test_use_saliency
    num_patches_per_img = 500
    Tsal = 0.3
    trainratio = 0.8
    patch_size = 32

    iqaset = IQADataset(
        root=root,
        dataset=dataset,
        randomstate=randomstate,

        splite_by=splite_by,
        use_sal=use_sal,
        num_patches_per_img=num_patches_per_img,
        Tsal=Tsal,
        trainratio=trainratio,
        patch_size=patch_size,
    )

    savename = f'{args.net_name}_{dataset}_random_{randomstate}_Tsal_{Tsal}'

    print(savename)
    max_plcc, max_srocc = train_3D(model, n_ep, savename, lr, iqaset, args)

    print(f'random_state={randomstate} max_plcc/srocc={max_plcc:.4f}/{max_srocc:.4f}')

    return 0


if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-net_name', type=str, default='FusionNet')
    parser.add_argument('-transfer_path', type=str, default=None)

    parser.add_argument('-gpu_id', type=str, default='0')
    parser.add_argument('-test_use_saliency', type=bool, default=True)
    # parser.add_argument('-Tsal', type=float, default=0)

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    train_model(args)
