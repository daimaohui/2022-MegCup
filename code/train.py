import megengine as mge
import megengine.module as M
import megengine.functional as F
import tqdm
import numpy as np
from megengine.utils.module_stats import module_stats
import matplotlib.pyplot as plt
import os
from U-net import *
patchsz = 256
batchsz = 16

def cal_score(pred, gt, sum):
    content = open(pred, 'rb').read()
    samples_pred = np.float32(np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256)))
    content = open(gt, 'rb').read()
    samples_gt = np.float32(np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256)))
    content = open("input.bin", 'rb').read()
    input = np.float32(np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256)))
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(input[sum % len(input)])
    plt.subplot(1, 3, 2)
    plt.imshow(samples_pred[sum % len(input)])
    plt.subplot(1, 3, 3)
    plt.imshow(samples_gt[sum % len(input)])
    plt.show()
    means = samples_gt.mean(axis=(1, 2))
    weight = (1 / means) ** 0.5
    diff = np.abs(samples_pred - samples_gt).mean(axis=(1, 2))
    diff = diff * weight
    score = diff.mean()

    score = np.log10(100 / score) * 5

    print('score', score)

    return score


if __name__ == '__main__':
    sum = 0
    file="workspace/model_8.pkl"
    if os.path.exists(file):
        net = mge.load(file)
    else:
        net = Network()

    input_data = np.random.rand(1, 1, 256, 256).astype("float32")
    total_stats, stats_details = module_stats(
        net,
        inputs=(input_data,),
        cal_params=True,
        cal_flops=True,
        logging_to_stdout=True,
    )
    print("params %.3fK MAC/pixel %.0f" % (
    total_stats.param_dims / 1e3, total_stats.flops / input_data.shape[2] / input_data.shape[3]))
    print('loading data')
    content = open('dataset/burst_raw/competition_train_input.0.2.bin', 'rb').read()
    samples_ref = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))
    content = open('dataset/burst_raw/competition_train_gt.0.2.bin', 'rb').read()
    samples_gt = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))

    train_steps = 0
    opt = mge.optimizer.Adam(net.parameters(), lr=1e-3)
    gm = mge.autodiff.GradManager().attach(net.parameters())
    min_score = 0
    losses = []
    print('training')
    train_samples_ref = samples_ref[:]
    test_samples_ref = samples_ref[7168:]
    train_samples_gt = samples_gt[:]
    test_samples_gt = samples_gt[7168:]
    fout = open('input.bin', 'wb')
    for i in tqdm.tqdm(range(0, len(test_samples_ref), batchsz)):
        i_end = min(i + batchsz, len(test_samples_ref))
        pred = test_samples_ref[i:i_end, None, :, :]
        fout.write(pred.tobytes())
    fout.close()
    fout = open('result.bin', 'wb')
    for i in tqdm.tqdm(range(0, len(test_samples_gt), batchsz)):
        i_end = min(i + batchsz, len(test_samples_gt))
        pred = test_samples_gt[i:i_end, None, :, :]
        fout.write(pred.tobytes())
    fout.close()
    for it in range(0, train_steps):
        if it%10==0 and it!=0:
            for g in opt.param_groups:
                g['lr'] = g['lr']/5*4
        permutation = np.random.permutation(train_samples_ref.shape[0])
        train_samples_ref = train_samples_ref[permutation, :, :]
        train_samples_gt = train_samples_gt[permutation, :, :]
        for t in range(int(train_samples_ref.shape[0] / batchsz)):
            if t % 100 == 0:
                # 测试
                fout = open('pred.bin', 'wb')
                for i in tqdm.tqdm(range(0, len(test_samples_ref), batchsz)):
                    i_end = min(i + batchsz, len(test_samples_ref))
                    batch_inp = mge.tensor(np.float32(test_samples_ref[i:i_end, None, :, :]) * np.float32(1 / 65536))
                    pred = net(batch_inp)
                    pred = (pred.numpy()[:, :, :] * 65536).clip(0, 65535).astype('uint16')
                    fout.write(pred.tobytes())
                fout.close()
                score = cal_score("pred.bin", "result.bin", sum)
                sum += 1
                if score > min_score:
                    min_score = score
                    print("save model!")
                    mge.save(net, file)
            batch_inp = mge.tensor(train_samples_ref[t * batchsz:(t + 1) * batchsz] * np.float32(1. / 65536))
            batch_out = mge.tensor(train_samples_gt[t * batchsz:(t + 1) * batchsz] * np.float32(1. / 65536))
            opt.zero_grad()
            with gm:
                pred = net(batch_inp)
                loss = F.nn.l1_loss(pred, batch_out)+F.nn.square_loss(pred,batch_out)*3.5
#                 loss = F.nn.l1_loss(pred, batch_out)
                gm.backward(loss)
                opt.step().clear_grad()
            loss = float(loss.numpy())
            losses.append(loss)
            if t % 10 == 0:
                print('it', it, 'loss', loss * 100000, 'mean', np.mean(losses[-100:]) * 100000)

    print('prediction')
    net = mge.load(file)
    content = open('dataset/burst_raw/competition_test_input.0.2.bin', 'rb').read()
    samples_ref = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))
    fout = open('workspace/result.bin', 'wb')

    for i in tqdm.tqdm(range(0, len(samples_ref), batchsz)):
        i_end = min(i + batchsz, len(samples_ref))
        batch_inp = mge.tensor(np.float32(samples_ref[i:i_end, None, :, :]) * np.float32(1 / 65536))
        pred = net(batch_inp)
        pred = (pred.numpy()[:, :, :] * 65536).clip(0, 65535).astype('uint16')
        fout.write(pred.tobytes())


    fout.close()