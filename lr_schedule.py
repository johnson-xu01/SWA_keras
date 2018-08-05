from keras.callbacks import LearningRateScheduler


def descentLR(epoch, SWA_START, lr_start, lr_end):
    """
    :param epoch: epoch index (integer, indexed from 0)
    :param SWA_START: the number of epoch after which SWA will start to average models
    :param lr_start: initial learning rate
    :param lr_end: SWA learning rate
    :return: new learning rate (float)
    """
    t = epoch / SWA_START
    lr_ratio = lr_end / lr_start
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return lr_start * factor


def cyclicLR(epoch, c, lr_start, lr_end):
    """
    :param epoch:  epoch index (integer, indexed from 0)
    :param c:  the cycle length (the hyper-parameters of the method)
    :param lr_start: initial learning rate
    :param lr_end: SWA learning rate
    :return: new learning rate (float)
    """
    t_epoch = (epoch % c + 1) / c
    lr_now = (1 - t_epoch) * lr_start + t_epoch * lr_end
    return lr_now


def LR_schedule(epoch, flag=False, SWA_START=150, lr_start=0.1, lr_end=0.05, c=10):
    """
    :param epoch: epoch index (integer, indexed from 0)
    :param flag: whether use cyclicLR or not (default: False, use descentLR)
    :param lr: current learning rate (float, not use here)
    :param SWA_START: the number of epoch after which SWA will start to average models (default: 150)
    :param lr_start: initial learning rate (default: 0.1)
    :param lr_end: SWA learning rate (default: 0.05)
    :return: new learning rate (float)
    """
    if flag:
        return cyclicLR(epoch, c, lr_start, lr_end)
    else:
        return descentLR(epoch, SWA_START, lr_start, lr_end)


if __name__ == '__main__':
    schedule = lambda epoch: LR_schedule(epoch, SWA_START=126, lr_start=0.1, lr_end=0.05)
    # schedule = lambda epoch: LR_schedule(epoch, flag=True, c=30, lr_start=0.1, lr_end=0.05)
    LearningRateScheduler(schedule=schedule)
    import numpy as np
    import matplotlib.pyplot as plt

    epochs = np.arange(150)
    lrs = []
    for epoch in epochs:
        lrs.append(schedule(epoch))
    lrs = np.array(lrs)
    plt.plot(lrs)
    plt.show()
