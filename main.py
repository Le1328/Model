import argparse
import torch
import torch.nn as nn
import numpy as np
import json
import warnings
from sklearn.utils import class_weight
from utils import setup_seed, print_metrics_binary, hdl_time, mre_f, device
from model import Model
from data_extraction import data_process_mimic3


def train(train_loader,
          valid_loader,
          demographic_data,
          diagnosis_data,
          idx_list,
          x_dim,
          diag_dim,
          demo_dim,
          h_ts,
          h_diag,
          h_demo,
          proj_dim,
          hidden_dim,
          drop_prob,
          lr,
          task,
          seed,
          epochs,
          file_name,
          device):

    model = Model(x_dim, diag_dim, demo_dim, h_ts, h_diag, h_demo, \
                  proj_dim, hidden_dim, drop_prob, task).to(device)
    opt_model = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(opt_model, milestones=[40, 60, 80, 90], gamma=0.5)

    setup_seed(seed)
    train_loss_ce = []
    train_loss_mae = []
    train_loss_mre = []
    valid_loss_mae = []
    valid_loss_mre = []
    best_epoch = 0
    max_auroc = 0
    min_mae = 9999999

    for each_epoch in range(epochs):
        batch_loss_ce = []
        batch_loss_mae = []
        batch_loss_mre = []
        model.train()

        for step, (batch_x, batch_y, sorted_length, batch_ts, batch_name) in enumerate(train_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.to(device)
            batch_ts = batch_ts.float().to(device)
            time_input = hdl_time(batch_ts)
            time_input = time_input.unsqueeze(2).expand(batch_x.shape[0], batch_x.shape[1], batch_x.shape[2] + 131)

            mask = torch.where(batch_x != -1, torch.ones(batch_x.shape).to(device),
                               torch.zeros(batch_x.shape).to(device))

            batch_demo = []
            batch_diag = []
            for i in range(len(batch_name)):
                cur_id, cur_ep, _ = batch_name[i].split('_', 2)
                cur_idx = cur_id + '_' + cur_ep
                idx = idx_list.index(cur_idx) if cur_idx in idx_list else None

                cur_demo = torch.tensor(demographic_data[idx], dtype=torch.float32)
                cur_diag = torch.tensor(diagnosis_data[idx], dtype=torch.float32)

                batch_demo.append(cur_demo)
                batch_diag.append(cur_diag)

            batch_demo = torch.stack(batch_demo).to(device)
            batch_diag = torch.stack(batch_diag).to(device)
            output = model(batch_x, batch_diag, batch_demo, time_input, sorted_length)

            batch_y = batch_y.long()
            y_out = batch_y.cpu().numpy()
            class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_out),
                                                              y=y_out)
            class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
            ce_f = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
            mae_f = torch.nn.L1Loss(reduction='mean')

            if task == 'Prediction':
                loss_ce = ce_f(output, batch_y)
                batch_loss_ce.append(loss_ce.cpu().detach().numpy())
                loss = loss_ce
            if task == 'Imputation':
                x_hat = output[:, :, :x_dim]
                loss_mae = mae_f(mask * x_hat, mask * batch_x)
                loss_mre = mre_f(mask * x_hat, mask * batch_x)
                batch_loss_mae.append(loss_mae.cpu().detach().numpy())
                batch_loss_mre.append(loss_mre.cpu().detach().numpy())
                loss = loss_mae

            opt_model.zero_grad()
            loss.backward()
            opt_model.step()

        train_loss_ce.append(np.mean(np.array(batch_loss_ce)))
        train_loss_mae.append(np.mean(np.array(batch_loss_mae)))
        train_loss_mre.append(np.mean(np.array(batch_loss_mre)))
        # scheduler.step()

        with torch.no_grad():
            y_true = []
            y_pred = []
            batch_loss_mae = []
            batch_loss_mre = []
            model.eval()

            for step, (batch_x, batch_y, sorted_length, batch_ts, batch_name) in enumerate(valid_loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.to(device)
                batch_ts = batch_ts.float().to(device)
                time_input = hdl_time(batch_ts)
                time_input = time_input.unsqueeze(2).expand(batch_x.shape[0], batch_x.shape[1], batch_x.shape[2] + 131)

                mask = torch.where(batch_x != -1, torch.ones(batch_x.shape).to(device),
                                   torch.zeros(batch_x.shape).to(device))

                batch_demo = []
                batch_diag = []
                for i in range(len(batch_name)):
                    cur_id, cur_ep, _ = batch_name[i].split('_', 2)
                    cur_idx = cur_id + '_' + cur_ep
                    idx = idx_list.index(cur_idx) if cur_idx in idx_list else None

                    cur_demo = torch.tensor(demographic_data[idx], dtype=torch.float32)
                    cur_diag = torch.tensor(diagnosis_data[idx], dtype=torch.float32)

                    batch_demo.append(cur_demo)
                    batch_diag.append(cur_diag)

                batch_demo = torch.stack(batch_demo).to(device)
                batch_diag = torch.stack(batch_diag).to(device)
                output = model(batch_x, batch_diag, batch_demo, time_input, sorted_length)

                if task == 'Prediction':
                    batch_y = batch_y.long()
                    y_pred.append(output)
                    y_true.append(batch_y)
                if task == 'Imputation':
                    x_hat = output[:, :, :x_dim]
                    loss_mae = mae_f(mask * x_hat, mask * batch_x)
                    loss_mre = mre_f(mask * x_hat, mask * batch_x)
                    batch_loss_mae.append(loss_mae.cpu().detach().numpy())
                    batch_loss_mre.append(loss_mre.cpu().detach().numpy())

            if task == 'Prediction':
                y_pred = torch.cat(y_pred, 0)
                y_true = torch.cat(y_true, 0)
                valid_y_pred = y_pred.cpu().detach().numpy()
                valid_y_true = y_true.cpu().detach().numpy()
                ret = print_metrics_binary(valid_y_true, valid_y_pred)
                cur_auroc = ret['auroc']
                if cur_auroc > max_auroc:
                    best_epoch = each_epoch
                    max_auroc = cur_auroc
                    state = {
                        'net': model.state_dict(),
                        'optimizer': opt_model.state_dict(),
                        'epoch': each_epoch
                    }
                    torch.save(state, file_name)

            if task == 'Imputation':
                valid_loss_mae.append(np.mean(np.array(batch_loss_mae)))
                valid_loss_mre.append(np.mean(np.array(batch_loss_mre)))
                cur_mae = valid_loss_mae[-1]
                if cur_mae < min_mae:
                    best_epoch = each_epoch
                    min_mae = cur_mae
                    state = {
                        'net': model.state_dict(),
                        'optimizer': opt_model.state_dict(),
                        'epoch': each_epoch
                    }
                    torch.save(state, file_name)

    return best_epoch


def test(test_loader,
         demographic_data,
         diagnosis_data,
         idx_list,
         x_dim,
         diag_dim,
         demo_dim,
         h_ts,
         h_diag,
         h_demo,
         proj_dim,
         hidden_dim,
         drop_prob,
         task,
         seed,
         file_name,
         device):

    setup_seed(seed)
    model = Model(x_dim, diag_dim, demo_dim, h_ts, h_diag, h_demo, \
                  proj_dim, hidden_dim, drop_prob, task).to(device)
    checkpoint = torch.load(file_name)
    model.load_state_dict(checkpoint['net'])
    model.eval()

    batch_loss_mae = []
    batch_loss_mre = []
    test_loss_mae = []
    test_loss_mre = []
    y_true = []
    y_pred = []
    for step, (batch_x, batch_y, sorted_length, batch_ts, batch_name) in enumerate(test_loader):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.to(device)
        batch_ts = batch_ts.float().to(device)
        time_input = hdl_time(batch_ts)
        time_input = time_input.unsqueeze(2).expand(batch_x.shape[0], batch_x.shape[1], batch_x.shape[2] + 131)

        mask = torch.where(batch_x != -1, torch.ones(batch_x.shape).to(device),
                           torch.zeros(batch_x.shape).to(device))

        batch_demo = []
        batch_diag = []
        for i in range(len(batch_name)):
            cur_id, cur_ep, _ = batch_name[i].split('_', 2)
            cur_idx = cur_id + '_' + cur_ep
            idx = idx_list.index(cur_idx) if cur_idx in idx_list else None

            cur_demo = torch.tensor(demographic_data[idx], dtype=torch.float32)
            cur_diag = torch.tensor(diagnosis_data[idx], dtype=torch.float32)

            batch_demo.append(cur_demo)
            batch_diag.append(cur_diag)

        batch_demo = torch.stack(batch_demo).to(device)
        batch_diag = torch.stack(batch_diag).to(device)
        output = model(batch_x, batch_diag, batch_demo, time_input, sorted_length)

        if task == 'Prediction':
            batch_y = batch_y.long()
            y_pred.append(output)
            y_true.append(batch_y)
        if task == 'Imputation':
            x_hat = output[:, :, :x_dim]
            mae_f = torch.nn.L1Loss(reduction='mean')
            loss_mae = mae_f(mask * x_hat, mask * batch_x)
            loss_mre = mre_f(mask * x_hat, mask * batch_x)
            batch_loss_mae.append(loss_mae.cpu().detach().numpy())
            batch_loss_mre.append(loss_mre.cpu().detach().numpy())

    if task == 'Prediction':
        y_pred = torch.cat(y_pred, 0)
        y_true = torch.cat(y_true, 0)
        test_y_pred = y_pred.cpu().detach().numpy()
        test_y_true = y_true.cpu().detach().numpy()
        ret = print_metrics_binary(test_y_true, test_y_pred)
        cur_auroc = ret['auroc']
        cur_auprc = ret['auprc']
        results = {'auroc': cur_auroc, 'auprc': cur_auprc}
    if task == 'Imputation':
        test_loss_mae.append(np.mean(np.array(batch_loss_mae)))
        test_loss_mre.append(np.mean(np.array(batch_loss_mre)))
        cur_mae = test_loss_mae[-1]
        cur_mre = test_loss_mre[-1]
        results = {'mae': cur_mae, 'mre': cur_mre}

    return results


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--x_dim", type=int)
    parser.add_argument("--diag_dim", type=int)
    parser.add_argument("--demo_dim", type=int)
    parser.add_argument("--h_ts", type=int)
    parser.add_argument("--h_diag", type=int)
    parser.add_argument("--h_demo", type=int)
    parser.add_argument("--proj_dim", type=int)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--drop_prob", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--time_length", type=int)
    parser.add_argument("--task", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_s_path1", type=str)
    parser.add_argument("--data_s_path2", type=str)
    parser.add_argument("--data_idx_path", type=str)
    parser.add_argument("--file_name", type=str)
    args = parser.parse_args()

    x_dim = args.x_dim
    diag_dim = args.diag_dim
    demo_dim = args.demo_dim
    h_ts = args.h_ts
    h_diag = args.h_diag
    h_demo = args.h_demo
    proj_dim = args.proj_dim
    hidden_dim = args.hidden_dim
    drop_prob = args.drop_prob
    lr = args.lr
    seed = args.seed
    epochs = args.epochs
    time_length = args.time_length
    task = args.task
    data_path = args.data_path
    data_s_path1 = args.data_s_path1
    data_s_path2 = args.data_s_path2
    data_idx_path = args.data_idx_path
    file_name = args.file_name

    train_loader, valid_loader, test_loader = data_process_mimic3(data_path, time_length)
    with open(data_s_path1, 'r') as f:
        demographic_data = json.load(f)
    with open(data_s_path2, 'r') as f:
        diagnosis_data = json.load(f)
    with open(data_idx_path, 'r') as f:
        idx_list = json.load(f)

    best_epoch = train(train_loader, valid_loader, demographic_data, diagnosis_data, idx_list, x_dim, diag_dim, demo_dim, h_ts, h_diag, h_demo, proj_dim, hidden_dim, drop_prob, lr, task, seed, epochs, file_name, device)
    results = test(test_loader, demographic_data, diagnosis_data, idx_list, x_dim, diag_dim, demo_dim, h_ts, h_diag, h_demo, proj_dim, hidden_dim, drop_prob, task, seed, file_name, device)
    print(results)

