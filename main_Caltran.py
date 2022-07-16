from __future__ import print_function
import argparse
import numpy as np
import os
import torch
import torch.optim as optim
import copy
import pickle
import higher

from utils.save_data import save_data
from utils.load_data import load_source, load_target, load_test
from utils.batch_generator import batch_generator
from models.L2E import L2EModel

# Command setting
parser = argparse.ArgumentParser(description='Learning to Evolve (L2E)')
parser.add_argument('-model_name', type=str, default='L2E', help='model name')
parser.add_argument('-disc', type=str, default='JS-divergence', help='MMD|JS-divergence|C-divergence')
parser.add_argument('-batch_size', type=int, default=32, help='batch size')
parser.add_argument('-root_dir', type=str, default='../data/caltran_continuous/')
parser.add_argument('-source', type=str, default='caltran')
parser.add_argument('-target', type=str, default='caltran')
parser.add_argument('-num_classes', type=int, default=2)
parser.add_argument('-meta_epochs', type=int, default=5)
parser.add_argument('-inner_epochs', type=int, default=1)
parser.add_argument('-meta_lr', type=float, default=0.5)
parser.add_argument('-update_lr', type=float, default=0.5)
parser.add_argument('-cuda', type=int, default=0, help='cuda id')
parser.add_argument('-seed', type=int, default=0, help='Random seed')
args = parser.parse_args()
device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')


def get_optimizer(model, learning_rate):
    param_group = []
    for k, v in model.named_parameters():
        if k.__contains__('base_network'):
            param_group += [{'name': k, 'params': v, 'lr': learning_rate / 10}]
        else:
            param_group += [{'name': k, 'params': v, 'lr': learning_rate}]
    optimizer = optim.Adadelta(param_group, lr=learning_rate)
    return optimizer


def train(all_src_data):
    if len(all_src_data) == 1:
        all_src_data = all_src_data * 2
    print("Meta-training...")
    model = L2EModel(num_classes=args.num_classes, disc=args.disc).to(device)
    meta_optim = get_optimizer(model, args.meta_lr)

    for m_epoch in range(args.meta_epochs):
        print("Meta epoch: [{:02d}/{:02d}]".format(m_epoch + 1, args.meta_epochs))
        optimizer = get_optimizer(model, args.update_lr)
        meta_optim.zero_grad()

        for i in range(len(all_src_data)-1):
            s_data = all_src_data[i]
            t_data = all_src_data[i+1]

            s_msk = np.random.rand(s_data['X'].shape[0]) < 0.9
            t_msk = np.random.rand(t_data['X'].shape[0]) < 0.9
            s_train_data, t_train_data = {}, {}
            s_train_data['X'], s_train_data['Y'] = s_data['X'][s_msk], s_data['Y'][s_msk]
            t_train_data['X'], t_train_data['Y'] = t_data['X'][t_msk], t_data['Y'][t_msk]
            s_generator = batch_generator(s_train_data, args.batch_size)
            t_generator = batch_generator(t_train_data, args.batch_size)
            num_batch = t_train_data['X'].shape[0] // args.batch_size
            with higher.innerloop_ctx(model, optimizer, copy_initial_weights=False) as (fnet, diffopt):
                for k in range(args.inner_epochs*num_batch):
                    model.train()
                    sinputs, slabels = next(s_generator)
                    tinputs, _ = next(t_generator)
                    sinputs = torch.tensor(sinputs, requires_grad=False).to(device)
                    slabels = torch.tensor(slabels, requires_grad=False, dtype=torch.long).to(device)
                    tinputs = torch.tensor(tinputs, requires_grad=False).to(device)
                    loss = model(sinputs, slabels, tinputs)
                    diffopt.step(loss)

                meta_sinputs = torch.tensor(s_data['X'][~s_msk], requires_grad=False).to(device)
                meta_tinputs = torch.tensor(t_data['X'][~t_msk], requires_grad=False).to(device)
                meta_tlabels = torch.tensor(t_data['Y'][~t_msk], requires_grad=False, dtype=torch.long).to(device)
                meta_loss = model.meta_loss(meta_sinputs, meta_tinputs, meta_tlabels)
                meta_loss.backward()
            meta_optim.step()
    return model


def fine_tune(model, old_tgt_data, tgt_data):
    print("Fine-tune...")
    s_generator = batch_generator(old_tgt_data, args.batch_size)
    t_generator = batch_generator(tgt_data, args.batch_size)

    optimizer = get_optimizer(model, args.update_lr)
    num_batch = tgt_data['X'].shape[0] // args.batch_size

    for k in range(args.inner_epochs*num_batch):
        model.train()
        sinputs, slabels = next(s_generator)
        tinputs, _ = next(t_generator)
        sinputs = torch.tensor(sinputs, requires_grad=False).to(device)
        slabels = torch.tensor(slabels, requires_grad=False, dtype=torch.long).to(device)
        tinputs = torch.tensor(tinputs, requires_grad=False).to(device)
        loss = model(sinputs, slabels, tinputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def test(model, test_data, save_preds=False, target_index=None):
    all_preds = []
    test_acc = 0.
    t_size = 100
    model.eval()

    output_y = []
    with torch.no_grad():
        t_len = test_data['X'].shape[0] // t_size
        for j in range(t_len):
            x = torch.tensor(test_data['X'][t_size * j:t_size * (j + 1)], requires_grad=False).to(device)
            y = torch.tensor(test_data['Y'][t_size * j:t_size * (j + 1)], requires_grad=False, dtype=torch.long).to(device)
            outputs = model.inference(x)
            preds = torch.max(outputs, 1)[1]
            test_acc += torch.sum(preds == y)

            probs = torch.nn.functional.softmax(outputs, dim=1)
            ent = torch.distributions.Categorical(probs).entropy()
            output_y.append(ent)
            all_preds.append(preds.detach().cpu().numpy())
        if t_len * t_size < test_data['X'].shape[0]:
            x = torch.tensor(test_data['X'][t_size * t_len:], requires_grad=False).to(device)
            y = torch.tensor(test_data['Y'][t_size * t_len:], requires_grad=False, dtype=torch.long).to(device)
            outputs = model.inference(x)
            preds = torch.max(outputs, 1)[1]
            test_acc += torch.sum(preds == y)

            probs = torch.nn.functional.softmax(outputs, dim=1)
            ent = torch.distributions.Categorical(probs).entropy()
            output_y.append(ent)
            all_preds.append(preds.detach().cpu().numpy())
    test_acc = test_acc.double() / test_data['X'].shape[0]

    if save_preds:
        output_y = torch.cat(output_y, dim=0)
        idx = torch.argsort(output_y.flatten(), descending=False).detach().cpu().numpy()
        data = {}
        data['X'] = test_data['X'][idx[:-output_y.shape[0] // 3]]
        data['Y'] = np.concatenate(all_preds, axis=0)[idx[:-output_y.shape[0] // 3]]
        with open("pred_target/" + "Pred_" + args.target + '_' + str(target_index) + ".pkl", "wb") as pkl_file:
            pickle.dump(data, pkl_file)
    return test_acc


if __name__ == '__main__':
    source_id = str(4)
    src_data = []
    if not os.path.isfile("processeData/{}_{}.pkl".format(args.source, source_id)):
        print('*' * 100)
        raw_src_loader = load_source(args.root_dir, source_id, args.batch_size)
        save_data(raw_src_loader, name=args.source + '_{}'.format(source_id))
    src_data.append(pickle.load(open("processeData/{}_{}.pkl".format(args.source, source_id), "rb")))

    for target_id in range(5, 16):
        print("It is the {}-th target task......".format(target_id))
        all_src_data = copy.deepcopy(src_data)
        for i in range(5, target_id):
            all_src_data.append(pickle.load(open("pred_target/" + "Pred_" + args.target + '_' + str(i) + ".pkl", "rb")))

        target_id = str(target_id)
        if not os.path.isfile("processeData/{}_{}.pkl".format(args.target, target_id)):
            tgt_loader = load_target(args.root_dir, target_id, args.batch_size)
            save_data(tgt_loader, name=args.target + '_{}'.format(target_id))
            test_loader = load_test(args.root_dir, target_id, args.batch_size)
            save_data(test_loader, name=args.target + '_test_{}'.format(target_id))
        model = train(all_src_data)

        if int(target_id) < 15:
            tgt_data = pickle.load(open("processeData/{}_{}.pkl".format(args.target, target_id), "rb"))
            model = fine_tune(model, all_src_data[-1], tgt_data)
            acc = test(model, tgt_data, save_preds=True, target_index=target_id)
            print("Meta-Training, Acc=", test(model, pickle.load(open("processeData/{}_test_{}.pkl".format(args.target, target_id), "rb")), save_preds=False, target_index=target_id))
        else:
            h_acc, f_acc = 0, 0
            for k in range(5, 16):
                if k < 20:
                    s = src_data[-1]
                    t = pickle.load(open("processeData/{}_{}.pkl".format(args.target, k), "rb"))
                else:
                    s = pickle.load(open("pred_target/Pred_{}_{}.pkl".format(args.target, k - 1), "rb"))
                    t = pickle.load(open("processeData/{}_{}.pkl".format(args.target, k), "rb"))
                test_data = pickle.load(open("processeData/{}_test_{}.pkl".format(args.target, k), "rb"))
                f_model = fine_tune(copy.deepcopy(model), s, t)
                acc = test(copy.deepcopy(f_model), test_data, save_preds=False)
                print("Final result of {}-th target task: Test acc = {:.4f}".format(k + 1, acc))
                if k < 15:
                    h_acc += acc
                else:
                    f_acc = acc
            print(f_acc, h_acc / 11.0)
