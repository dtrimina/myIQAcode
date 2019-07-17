import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


infos = {

    # 'haha': pd.read_csv('ckpt/saliency=0.3__ratio=0.3611__test_use_saliency=True_infos.csv'),

    # 'use s=0.1 train_ratio=0.7416': pd.read_csv('ckpt/saliency_phase2_9997random/saliency=0.1__ratio=0.7416__test_use_saliency=True_infos.csv'),
    # 'use s=0.2 train_ratio=0.5104': pd.read_csv('ckpt/saliency_phase2_9997random/saliency=0.2__ratio=0.5104__test_use_saliency=True_infos.csv'),
    # 'use s=0.3 train_ratio=0.3608': pd.read_csv('ckpt/saliency_phase2_9997random/saliency=0.3__ratio=0.3608__test_use_saliency=True_infos.csv'),
    # 'use s=0.4 train_ratio=0.2425': pd.read_csv('ckpt/saliency_phase2_9997random/saliency=0.4__ratio=0.2425__test_use_saliency=True_infos.csv'),
    # 'use s=0.5 train_ratio=0.1476': pd.read_csv('ckpt/saliency_phase2_9997random/saliency=0.5__ratio=0.1476__test_use_saliency=True_infos.csv'),
    # 'use s=0.6 train_ratio=0.0731': pd.read_csv('ckpt/saliency_phase2_9997random/saliency=0.6__ratio=0.0731__test_use_saliency=True_infos.csv'),
    # 'use s=0.7 train_ratio=0.0245': pd.read_csv('ckpt/saliency_phase2_9997random/saliency=0.7__ratio=0.0245__test_use_saliency=True_infos.csv'),
    # 'use s=0.8 train_ratio=0.0029': pd.read_csv('ckpt/saliency_phase2_9997random/saliency=0.8__ratio=0.0029__test_use_saliency=True_infos.csv'),

    # 'no use train_ratio=0.7416': pd.read_csv('ckpt/no_saliency_phase2_9997random/saliency=0.1__ratio=0.7416__test_use_saliency=False_infos.csv'),
    # 'no use train_ratio=0.5104': pd.read_csv('ckpt/no_saliency_phase2_9997random/saliency=0.2__ratio=0.5104__test_use_saliency=False_infos.csv'),
    # 'no use train_ratio=0.3608': pd.read_csv('ckpt/no_saliency_phase2_9997random/saliency=0.3__ratio=0.3608__test_use_saliency=False_infos.csv'),
    # 'no use train_ratio=0.2425': pd.read_csv('ckpt/no_saliency_phase2_9997random/saliency=0.4__ratio=0.2425__test_use_saliency=False_infos.csv'),
    # 'no use train_ratio=0.1476': pd.read_csv('ckpt/no_saliency_phase2_9997random/saliency=0.5__ratio=0.1476__test_use_saliency=False_infos.csv'),
    # 'no use train_ratio=0.0731': pd.read_csv('ckpt/no_saliency_phase2_9997random/saliency=0.6__ratio=0.0731__test_use_saliency=False_infos.csv'),
    # 'no use train_ratio=0.0245': pd.read_csv('ckpt/no_saliency_phase2_9997random/saliency=0.7__ratio=0.0245__test_use_saliency=False_infos.csv'),
    # 'no use train_ratio=0.0029': pd.read_csv('ckpt/no_saliency_phase2_9997random/saliency=0.8__ratio=0.0029__test_use_saliency=False_infos.csv'),

    # 'w/o saliency selection': pd.read_csv(
    #     'ckpt/no_saliency_phase2_9997random/saliency=0.3__ratio=0.3608__test_use_saliency=False_infos.csv'),
    # 'w   saliency selection': pd.read_csv(
    #     'ckpt/saliency_phase2_9997random/saliency=0.3__ratio=0.3608__test_use_saliency=True_infos.csv'),

    # 'w    saliency selection_446': pd.read_csv(
    #     'ckpt/random_state=446__saliency=0.3__test_use_saliency=True_infos.csv'),
    # 'w/o saliency selection_446': pd.read_csv(
    #     'ckpt/random_state=446__saliency=0.3__test_use_saliency=False_infos.csv'),

    # 'w    saliency selection_7666': pd.read_csv(
    #     'ckpt/random_state=7666__saliency=0.3__test_use_saliency=True_infos.csv'),
    # 'w/o saliency selection_7666': pd.read_csv(
    #     'ckpt/random_state=7666__saliency=0.3__test_use_saliency=False_infos.csv'),

    # 'w    saliency selection_2811': pd.read_csv(
    #     'ckpt/random_state=2811__saliency=0.3__test_use_saliency=True_infos.csv'),
    'w/o saliency selection': pd.read_csv(
        'ckpt/random_state=2811__saliency=0.3__test_use_saliency=False_infos.csv'),
'w    saliency selection': pd.read_csv(
        'ckpt/random_state=446__saliency=0.3__test_use_saliency=True_infos.csv'),

}

first_ep = 20

plt.figure(1)
for key, info in infos.items():
    ep = info['ep'][first_ep:]
    srocc = np.abs(info['srocc'])[first_ep:]
    print(key, srocc.max())
    plt.plot(ep, srocc, label=key)

plt.xlabel('epoch')
plt.ylabel('srocc')
plt.xlim(first_ep, 1000)
plt.xticks([first_ep, 200, 400, 600, 800, 1000])
plt.legend()

plt.figure(2)
for key, info in infos.items():
    time = info['time'][info['ep'] > first_ep]
    srocc = np.abs(info['srocc'])[first_ep:]
    plt.plot(time, srocc, label=key)

plt.xlabel('time')
plt.ylabel('srocc')
plt.legend()

plt.show()


# sal = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
# srocc = [
# 0.9715789473684208,
# 0.9713767452361031,
# 0.9721194879089616,
# 0.9686201991465148,
# 0.9653854907539118,
# 0.9652062588904696,
# 0.9572403982930298,
# ]
# ratio = [
# 0.7416,
# 0.5104,
# 0.3608,
# 0.2425,
# 0.1476,
# 0.0731,
# 0.0245,
# ]
#
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(sal, srocc)
# ax1.set_ylabel('srocc')
# ax1.set_xlabel(r'$T_{sal}$')
#
#
# ax2 = ax1.twinx()
# ax2.plot(sal, ratio, 'r')
# ax2.set_xlim([0.05, 0.75])
# ax2.set_ylabel('ratio of selected training data')
# # ax2.set_xlabel(r'$\T_sal$')
# # plt.legend()
#
# plt.show()



