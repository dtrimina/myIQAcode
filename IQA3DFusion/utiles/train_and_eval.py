import torch
import pandas as pd
import numpy as np
import time
from utiles.metric import eval4metric
import torch.nn as nn


def eval_3D(model, val_loader, loss_func, args):

    dmos = []
    dmos_pre = []

    val_loss = 0
    for _, sample in enumerate(val_loader):
        LRimg = sample['LRimg'].cuda()
        saliency = sample['saliency'].cuda()
        score = model(LRimg)

        if args.test_use_saliency:
            score = torch.sum(saliency * score, dim=[1, 2, 3]) / torch.sum(saliency, dim=[1, 2, 3])
        else:
            score = torch.mean(score, dim=[1, 2, 3])

        loss = loss_func(score, sample['dmos'].float().cuda())
        val_loss += loss.item()*LRimg.size(0)

        dmos.append(sample['dmos'])
        dmos_pre.append(score.detach().cpu())
    val_loss = val_loss / len(val_loader.dataset)

    dmos_pre = torch.cat(dmos_pre).numpy()
    dmos = torch.cat(dmos).numpy()

    plcc, srocc, *_ = eval4metric(dmos, dmos_pre)

    return val_loss, plcc, srocc


def train_3D(model, n_ep, savename, lr, iqaset, args):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), weight_decay=0.0005, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    model.cuda()

    logs = pd.DataFrame(columns=['ep', 'train_loss', 'eval_loss', 'plcc', 'srocc', 'time'])

    tr_minloss = np.Inf
    max_srocc = 0
    max_plcc = 0
    critition = nn.MSELoss().cuda()
    t_start = time.time()

    for i_ep in range(1, n_ep + 1):

        t0 = time.time()
        tr_loader, val_loader = iqaset.load_loader(trainbs=128, testbs=2)

        # print('training ...')
        model.train()
        train_loss = 0
        for _, sample in enumerate(tr_loader):
            optimizer.zero_grad()
            LRimg = sample['LRimg'].cuda()

            score = model(LRimg)

            loss = critition(score.view(-1), sample['dmos'].float().cuda())

            train_loss += loss.item() * LRimg.size(0)
            loss.backward()
            optimizer.step()
        train_loss = train_loss / len(tr_loader.dataset)
        scheduler.step()

        if train_loss < tr_minloss:
            tr_minloss = train_loss
            # torch.save(model.state_dict(), f'ckpt/{savename}_trminloss.pkl')

        # print('evaluating ...')
        model.eval()
        eval_loss, plcc, srocc = eval_3D(model, val_loader, critition, args)

        if srocc > max_srocc:
            max_srocc = srocc
            max_plcc = plcc
            torch.save(model.state_dict(), f'results/{savename}_maxsrocc.pkl')

        t_1_epoch = (time.time() - t0) / 60  # min
        t_end = (time.time() - t_start) / 60

        print(
            f'ep_{i_ep} {t_1_epoch:.2f}/{t_end:.2f}min: tr/val_loss={train_loss:.4f}/{eval_loss:.4f}, plcc/srocc={plcc:.4f}/{srocc:.4f}')

        log = pd.DataFrame([[i_ep, train_loss, eval_loss, plcc, srocc, t_end]],
                           columns=['ep', 'train_loss', 'eval_loss', 'plcc', 'srocc', 'time'])
        logs = pd.concat([logs, log])
        logs.to_csv(f'results/{savename}_infos.csv', index=False)

    return max_plcc, max_srocc
