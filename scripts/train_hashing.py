import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from pprint import pprint

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import configs
from functions.hashing import get_hamm_dist, calculate_mAP
from utils import io
from utils.misc import AverageMeter, Timer


def get_hd(a, b):
    return 0.5 * (a.size(0) - a @ b.t()) / a.size(0)


def get_centroid(nclass, nbit, maxtries=10000, initdist=0.61, mindist=0.2, reducedist=0.01):
    """
    brute force to find centroid with furthest distance
    :param nclass:
    :param nbit:
    :param maxtries:
    :param initdist:
    :param mindist:
    :param reducedist:
    :return:
    """
    centroid = torch.zeros(nclass, nbit)
    i = 0
    count = 0
    currdist = initdist
    while i < nclass:
        print(i, end='\r')
        c = torch.randn(nbit).sign()
        nobreak = True
        for j in range(i):
            if get_hd(c, centroid[j]) < currdist:
                i -= 1
                nobreak = False
                break
        if nobreak:
            centroid[i] = c
        else:
            count += 1

        if count >= maxtries:
            count = 0
            currdist -= reducedist
            print('reduce', currdist, i)
            if currdist < mindist:
                raise ValueError('cannot find')

        i += 1
    centroid = centroid[torch.randperm(nclass)]
    return centroid


def batch_sign_dist(inputs, centroids, margin=0, bs=32):
    # to avoid out of memory when inputs is large
    n, b1 = inputs.size()
    nclass, b2 = centroids.size()
    assert b1 == b2, 'inputs and centroids must have same number of bit'

    ds = configs.tensor_to_dataset(inputs)
    dl = DataLoader(ds, bs, shuffle=False, drop_last=False)

    out = []

    for batch_inputs in dl:
        # sl = relu(margin - x*y)
        size = batch_inputs.size(0)
        batch_out = batch_inputs.view(size, 1, b1) * centroids.sign().view(1, nclass, b1)
        batch_out = torch.relu(margin - batch_out)  # (n, nclass, nbit)
        batch_out = batch_out.sum(2)  # (n, nclass)
        out.append(batch_out)

    out = torch.cat(out, dim=0)

    return out


def update_centroid(codes, labels, centroids, params):
    new_centroids = torch.zeros_like(centroids)

    train_codes = codes.sign()  # (N, nbit)
    train_labels = labels.argmax(dim=1)

    for c in range(centroids.size(0)):  # nclass
        code_from_label = train_codes[train_labels == c]
        if params['update_rate'] != 0:
            topn = int(params['update_rate'] * code_from_label.size(0))

            dist = batch_sign_dist(code_from_label,
                                   centroids[c].view(1, -1),
                                   params['margin'],
                                   params['batch_size']).view(code_from_label.size(0))
            code_from_label = code_from_label[torch.argsort(dist)[:topn]]

        code_for_class = code_from_label.mean(0).sign()

        new_centroids[c].data.copy_(code_for_class)

    return new_centroids


def calculate_accuracy(logits, hamm_dist, labels, loss_param):
    if loss_param['multiclass']:
        pred = logits.topk(5, 1, True, True)[1].t()
        correct = pred.eq(labels.argmax(1).view(1, -1).expand_as(pred))
        acc = correct[:5].view(-1).float().sum(0, keepdim=True) / logits.size(0)

        pred = hamm_dist.topk(5, 1, False, True)[1].t()
        correct = pred.eq(labels.argmax(1).view(1, -1).expand_as(pred))
        slacc = correct[:5].view(-1).float().sum(0, keepdim=True) / hamm_dist.size(0)
    else:
        acc = (logits.argmax(1) == labels.argmax(1)).float().mean()
        slacc = (hamm_dist.argmin(1) == labels.argmax(1)).float().mean()

    return acc, slacc


def calculate_loss(logits, codes, centroids, labels, loss_param):
    loss_sl = torch.tensor(0.).to(loss_param['device'])
    loss_ce = torch.tensor(0.).to(loss_param['device'])

    if loss_param['ce'] != 0:
        if loss_param['multiclass']:
            loss_ce = F.binary_cross_entropy_with_logits(logits, labels.float())
        else:
            loss_ce = F.cross_entropy(logits, labels.argmax(1))

    if loss_param['sl'] != 0:
        if loss_param['multiclass']:
            bs, nbit = codes.size()
            nclass = centroids.size(0)

            loss_sl = torch.relu(loss_param['margin'] - codes.view(bs, 1, nbit) * centroids.view(1, nclass, nbit))
            loss_sl = loss_sl.sum(2)  # (bs, nclass, nbit) -> (bs, nclass)

            loss_sl = loss_sl * labels.float()
            loss_sl = loss_sl.sum(1).mean()  # (bs, nclass) -> (bs,) -> ()
        else:
            loss_sl = torch.relu(loss_param['margin'] - codes * centroids[labels.argmax(1)])
            loss_sl = loss_sl.sum(dim=1).mean()

    return loss_ce, loss_sl


def train_hashing(optimizer, model, centroids, train_loader, loss_param, uc=False):
    model.train()
    device = loss_param['device']
    meters = defaultdict(AverageMeter)
    total_timer = Timer()
    timer = Timer()

    total_timer.tick()

    train_codes = []
    train_labels = []

    for i, (data, labels) in enumerate(train_loader):
        timer.tick()

        # clear gradient
        optimizer.zero_grad()

        data, labels = data.to(device), labels.to(device)
        logits, codes = model(data)

        bs, nbit = codes.size()
        nclass = labels.size(1)

        loss_ce, loss_sl = calculate_loss(logits, codes, centroids, labels, loss_param)
        loss_reg = torch.pow(codes, 2).mean()
        loss_total = loss_param['ce'] * loss_ce + loss_param['sl'] * loss_sl + loss_param['reg'] * loss_reg

        if uc:
            train_codes.append(codes.detach())
            train_labels.append(labels)

        # backward and update
        loss_total.backward()
        optimizer.step()

        hamm_dist = get_hamm_dist(codes, centroids, normalize=True)
        acc, slacc = calculate_accuracy(logits, hamm_dist, labels, loss_param)

        timer.toc()
        total_timer.toc()

        # store results
        meters['loss_total'].update(loss_total.item(), data.size(0))
        meters['loss_sl'].update(loss_sl.item(), data.size(0))
        meters['loss_ce'].update(loss_ce.item(), data.size(0))
        meters['loss_reg'].update(loss_reg.item(), data.size(0))
        meters['acc'].update(acc.item(), data.size(0))
        meters['slacc'].update(slacc.item(), data.size(0))

        meters['time'].update(timer.total)

        print(f'Train [{i + 1}/{len(train_loader)}] '
              f'CE: {meters["loss_ce"].avg:.4f} '
              f'SL: {meters["loss_sl"].avg:.4f} '
              f'REG: {meters["loss_reg"].avg:.4f} '
              f'T: {meters["loss_total"].avg:.4f} '
              f'A(CE): {meters["acc"].avg:.4f} '
              f'A(SL): {meters["slacc"].avg:.4f} '
              f'({timer.total:.2f}s / {total_timer.total:.2f}s)', end='\r')

    print()
    total_timer.toc()

    if uc:
        logging.info('Update Centroids')
        train_codes = torch.cat(train_codes)
        train_labels = torch.cat(train_labels)
        centroids.data.copy_(update_centroid(train_codes, train_labels, centroids, loss_param))

    meters['total_time'].update(total_timer.total)

    return meters


def test_hashing(model, centroids, test_loader, loss_param, return_codes=False):
    model.eval()
    device = loss_param['device']
    meters = defaultdict(AverageMeter)
    total_timer = Timer()
    timer = Timer()

    total_timer.tick()

    ret_codes = []
    ret_labels = []

    for i, (data, labels) in enumerate(test_loader):
        timer.tick()

        with torch.no_grad():
            data, labels = data.to(device), labels.to(device)
            logits, codes = model(data)

            loss_ce, loss_sl = calculate_loss(logits, codes, centroids, labels, loss_param)
            loss_total = loss_param['ce'] * loss_ce + loss_param['sl'] * loss_sl

            hamm_dist = get_hamm_dist(codes, centroids, normalize=True)
            acc, slacc = calculate_accuracy(logits, hamm_dist, labels, loss_param)

            if return_codes:
                ret_codes.append(codes)
                ret_labels.append(labels)

        timer.toc()
        total_timer.toc()

        # store results
        meters['loss_total'].update(loss_total.item(), data.size(0))
        meters['loss_sl'].update(loss_sl.item(), data.size(0))
        meters['loss_ce'].update(loss_ce.item(), data.size(0))
        meters['acc'].update(acc.item(), data.size(0))
        meters['slacc'].update(slacc.item(), data.size(0))

        meters['time'].update(timer.total)

        print(f'Test [{i + 1}/{len(test_loader)}] '
              f'CE: {meters["loss_ce"].avg:.4f} '
              f'SL: {meters["loss_sl"].avg:.4f} '
              f'T: {meters["loss_total"].avg:.4f} '
              f'A(CE): {meters["acc"].avg:.4f} '
              f'A(SL): {meters["slacc"].avg:.4f} '
              f'({timer.total:.2f}s / {total_timer.total:.2f}s)', end='\r')

    print()
    meters['total_time'].update(total_timer.total)

    if return_codes:
        res = {
            'codes': torch.cat(ret_codes),
            'labels': torch.cat(ret_labels)
        }
        return meters, res

    return meters


def prepare_dataloader(config):
    logging.info('Creating Datasets')
    train_dataset = configs.dataset(config, filename='train.txt', transform_mode='train')

    separate_multiclass = config['dataset_kwargs'].get('separate_multiclass', False)
    config['dataset_kwargs']['separate_multiclass'] = False
    test_dataset = configs.dataset(config, filename='test.txt', transform_mode='test')
    db_dataset = configs.dataset(config, filename='database.txt', transform_mode='test')
    config['dataset_kwargs']['separate_multiclass'] = separate_multiclass  # during mAP, no need to separate

    logging.info(f'Number of DB data: {len(db_dataset)}')
    logging.info(f'Number of Train data: {len(train_dataset)}')

    train_loader = configs.dataloader(train_dataset, config['batch_size'])
    test_loader = configs.dataloader(test_dataset, config['batch_size'], shuffle=False, drop_last=False)
    db_loader = configs.dataloader(db_dataset, config['batch_size'], shuffle=False, drop_last=False)

    return train_loader, test_loader, db_loader


def prepare_model(config, device):
    logging.info('Creating Model')
    model = configs.arch(config)
    extrabit = model.extrabit
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    model = model.to(device)
    return model, extrabit


def main(config):
    device = torch.device(config.get('device', 'cuda:0'))

    io.init_save_queue()

    start_time = time.time()
    configs.seeding(config['seed'])

    logdir = config['logdir']
    assert logdir != '', 'please input logdir'

    pprint(config)

    os.makedirs(f'{logdir}/models', exist_ok=True)
    os.makedirs(f'{logdir}/optims', exist_ok=True)
    os.makedirs(f'{logdir}/outputs', exist_ok=True)
    json.dump(config, open(f'{logdir}/config.json', 'w+'), indent=4, sort_keys=True)

    train_loader, test_loader, db_loader = prepare_dataloader(config)
    model, extrabit = prepare_model(config, device)

    optimizer = configs.optimizer(config, model.parameters())
    scheduler = configs.scheduler(config, optimizer)

    nclass = config['arch_kwargs']['nclass']
    nbit = config['arch_kwargs']['nbit'] + extrabit

    logging.info(f'Total Bit: {nbit}')
    if config['centroid_generation'] == 'N':  # normal
        centroids = torch.randn(nclass, nbit)
    elif config['centroid_generation'] == 'B':  # bernoulli
        prob = torch.ones(nclass, nbit) * 0.5
        centroids = torch.bernoulli(prob) * 2. - 1.
    else:  # O: optim
        centroids = get_centroid(nclass, nbit)

    centroids = centroids.sign().to(device)
    io.fast_save(centroids, f'{logdir}/outputs/centroids.pth')

    train_history = []
    test_history = []

    loss_param = config.copy()
    loss_param.update({
        'ucnow': 0,
        'device': device
    })

    best = 0
    curr_metric = 0

    nepochs = config['epochs']
    neval = config['eval_interval']

    logging.info('Training Start')

    for ep in range(nepochs):
        logging.info(f'Epoch [{ep + 1}/{nepochs}]')
        res = {'ep': ep + 1}

        uc = loss_param['update_time'] != 0 and (ep + 1) % loss_param['update_time'] == 0
        train_meters = train_hashing(optimizer, model, centroids, train_loader, loss_param, uc=uc)
        scheduler.step()

        for key in train_meters: res['train_' + key] = train_meters[key].avg
        train_history.append(res)
        # train_outputs.append(train_out)

        eval_now = (ep + 1) == nepochs or (neval != 0 and (ep + 1) % neval == 0)
        if eval_now:
            res = {'ep': ep + 1}

            test_meters, test_out = test_hashing(model, centroids, test_loader, loss_param, True)
            db_meters, db_out = test_hashing(model, centroids, db_loader, loss_param, True)

            for key in test_meters: res['test_' + key] = test_meters[key].avg
            for key in db_meters: res['db_' + key] = db_meters[key].avg

            res['mAP'] = calculate_mAP(db_out['codes'], db_out['labels'],
                                       test_out['codes'], test_out['labels'],
                                       loss_param['R'])
            logging.info(f'mAP: {res["mAP"]:.6f}')

            curr_metric = res['mAP']
            test_history.append(res)
            # test_outputs.append(outs)

        json.dump(train_history, open(f'{logdir}/train_history.json', 'w+'), indent=True, sort_keys=True)
        # io.fast_save(train_outputs, f'{logdir}/outputs/train_last.pth')

        if len(test_history) != 0:
            json.dump(test_history, open(f'{logdir}/test_history.json', 'w+'), indent=True, sort_keys=True)
            # io.fast_save(test_outputs, f'{logdir}/outputs/test_last.pth')

        modelsd = model.state_dict()
        # optimsd = optimizer.state_dict()
        # io.fast_save(modelsd, f'{logdir}/models/last.pth')
        # io.fast_save(optimsd, f'{logdir}/optims/last.pth')
        save_now = config['save_interval'] != 0 and (ep + 1) % config['save_interval'] == 0
        if save_now:
            io.fast_save(modelsd, f'{logdir}/models/ep{ep + 1}.pth')
            # io.fast_save(optimsd, f'{logdir}/optims/ep{ep + 1}.pth')
            # io.fast_save(train_outputs, f'{logdir}/outputs/train_ep{ep + 1}.pth')

        if best < curr_metric:
            best = curr_metric
            io.fast_save(modelsd, f'{logdir}/models/best.pth')

    modelsd = model.state_dict()
    io.fast_save(modelsd, f'{logdir}/models/last.pth')
    total_time = time.time() - start_time
    io.join_save_queue()
    logging.info(f'Training End at {datetime.today().strftime("%Y-%m-%d %H:%M:%S")}')
    logging.info(f'Total time used: {total_time / (60 * 60):.2f} hours')
    logging.info(f'Best mAP: {best:.6f}')
    logging.info(f'Done: {logdir}')

    return logdir
