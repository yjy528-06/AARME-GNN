import os as _os_det
_os_det.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import sqlite3
import csv
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, auc, roc_auc_score, recall_score, f1_score, precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
import warnings
import os
import random
from modeltask1 import FusionLayer, GNN1, GNN2, GNN3, GNN4

warnings.filterwarnings("ignore")


DATASET_PRESET = {
    1: {"n_drug": 572,  "event_num": 65,  "data_dir": "../dataset1"},
    2: {"n_drug": 1258, "event_num": 100, "data_dir": "../dataset2"},
}
DATA_DIR = DATASET_PRESET[1]["data_dir"]


def configure_dataset(args):

    p = DATASET_PRESET[int(args.dataset)]
    args.n_drug = int(p["n_drug"])
    args.event_num = int(p["event_num"])
    if not hasattr(args, "_data_dir_raw"):
        args._data_dir_raw = getattr(args, "data_dir", None)   # 用户显式指定的值（可为 None）
    args.data_dir = args._data_dir_raw or p["data_dir"]
    args.data_dir = os.path.abspath(args.data_dir)
    return args



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




parser = argparse.ArgumentParser(description='GNN based on the whole datas')
parser.add_argument("--epoches", type=int, choices=[100, 500, 1000, 2000], default=120)
parser.add_argument("--batch_size", type=int, choices=[2048, 1024, 512, 256, 128], default=1024)
parser.add_argument("--weigh_decay", type=float, choices=[1e-1, 1e-2, 1e-3, 1e-4, 1e-8], default=1e-8)
parser.add_argument("--lr", type=float, choices=[1e-3, 1e-4, 1e-5, 4 * 1e-3], default=1 * 1e-2)
parser.add_argument("--neighbor_sample_size", choices=[4, 6, 10, 16], type=int, default=6)
parser.add_argument("--event_num", type=int, default=65)
parser.add_argument("--dataset", type=int, choices=[1, 2], default=1,
                    help="1=Dataset1(572药物/65事件), 2=Dataset2(1258药物/100事件)")
parser.add_argument("--data_dir", type=str, default=None,
                    help="数据目录；不填则按 --dataset 自动选择 ../dataset1 或 ../dataset2")
parser.add_argument("--n_drug", type=int, default=572)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--embedding_num", type=int, choices=[128, 64, 256, 32], default=128)


args = parser.parse_args()
args = configure_dataset(args)

def setup_seed():
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if os.environ.get("DDI_DETERMINISTIC", "1") != "0":
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

def read_dataset(drug_name_id, num):

    kg = defaultdict(list)
    tails = {}
    relations = {}
    drug_list = []
    filename = os.path.join(DATA_DIR, "dataset" + str(num) + ".txt")

    with open(filename, encoding="utf8") as reader:
        for line in reader:
            string = line.rstrip().split('//', 2)
            head = string[0]
            tail = string[1]
            relation = string[2]
            drug_list.append(drug_name_id[head])

            if tail not in tails:
                tails[tail] = len(tails)
            if relation not in relations:
                relations[relation] = len(relations)

            if num == 3:
                kg[drug_name_id[head]].append((drug_name_id[tail], relations[relation]))
                kg[drug_name_id[tail]].append((drug_name_id[head], relations[relation]))
            else:
                kg[drug_name_id[head]].append((tails[tail], relations[relation]))

    return kg, len(tails), len(relations)

def prepare(mechanism, action):
    d_label = {}
    d_event = []
    new_label = []
    for i in range(len(mechanism)):
        d_event.append(mechanism[i] + " " + action[i])
    count = {}
    for i in d_event:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    list1 = sorted(count.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(list1)):
        d_label[list1[i][0]] = i

    for i in range(len(d_event)):
        new_label.append(d_label[d_event[i]])
    return new_label

def l2_re(parameter):
    reg = 0
    for param in parameter:
        reg += 0.5 * (param ** 2).sum()
    return reg

def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)
    def _average_binary_score(binary_metric, y_true, y_score, average):
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)
    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)

def evaluate(pred_type, pred_score, y_test, event_num):
    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    each_eval_type = 6
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    y_one_hot = label_binarize(y_test, classes=np.arange(event_num))
    pred_one_hot = label_binarize(pred_type, classes=np.arange(event_num))

    result_all[0] = accuracy_score(y_test, pred_type)
    result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
    result_all[2] = roc_aupr_score(y_one_hot, pred_score, average='macro')
    result_all[3] = roc_auc_score(y_one_hot, pred_score, average='micro')
    result_all[4] = roc_auc_score(y_one_hot, pred_score, average='macro')
    result_all[5] = f1_score(y_test, pred_type, average='micro')
    result_all[6] = f1_score(y_test, pred_type, average='macro')
    result_all[7] = precision_score(y_test, pred_type, average='micro')
    result_all[8] = precision_score(y_test, pred_type, average='macro')
    result_all[9] = recall_score(y_test, pred_type, average='micro')
    result_all[10] = recall_score(y_test, pred_type, average='macro')

    for i in range(event_num):
        result_eve[i, 0] = accuracy_score(
            y_one_hot.take([i], axis=1).ravel(),
            pred_one_hot.take([i], axis=1).ravel()
        )
        result_eve[i, 1] = roc_aupr_score(
            y_one_hot.take([i], axis=1).ravel(),
            pred_one_hot.take([i], axis=1).ravel(),
            average=None
        )
        result_eve[i, 2] = roc_auc_score(
            y_one_hot.take([i], axis=1).ravel(),
            pred_one_hot.take([i], axis=1).ravel(),
            average=None
        )
        result_eve[i, 3] = f1_score(
            y_one_hot.take([i], axis=1).ravel(),
            pred_one_hot.take([i], axis=1).ravel(),
            average='binary'
        )
        result_eve[i, 4] = precision_score(
            y_one_hot.take([i], axis=1).ravel(),
            pred_one_hot.take([i], axis=1).ravel(),
            average='binary'
        )
        result_eve[i, 5] = recall_score(
            y_one_hot.take([i], axis=1).ravel(),
            pred_one_hot.take([i], axis=1).ravel(),
            average='binary'
        )
    return [result_all, result_eve]

def save_result(filepath, result_type, result):
    os.makedirs(filepath, exist_ok=True)
    filename = os.path.join(filepath, f"{result_type}task1_AARME-GNN1.csv")

    with open(filename, "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if result_type == "all":
            writer.writerow(["Metric", "Value"])
            metric_names = [
                "Accuracy",
                "Micro AUPR",
                "Macro AUPR",
                "Micro AUC",
                "Macro AUC",
                "Micro F1",
                "Macro F1",
                "Micro Precision",
                "Macro Precision",
                "Micro Recall",
                "Macro Recall"
            ]
            for name, value in zip(metric_names, result):
                writer.writerow([name, value[0]])

        elif result_type == "each":
            writer.writerow([
                "Event_ID",
                "Accuracy",
                "AUPR",
                "AUC",
                "F1",
                "Precision",
                "Recall"
            ])
            for event_id, metrics in enumerate(result):
                row = [event_id] + list(metrics)
                writer.writerow(row)
    return 0

def train(train_x, train_y, test_x, test_y, net):
    loss_function = nn.CrossEntropyLoss()
    distillation_loss_function = nn.KLDivLoss(reduction='batchmean')
    opti = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weigh_decay)
    test_loss, test_acc, train_l = 0, 0, 0
    train_a = []
    train_x1 = train_x.copy()
    train_x[:, [0, 1]] = train_x[:, [1, 0]]
    train_x_total = torch.LongTensor(np.concatenate([train_x1, train_x], axis=0))
    train_y = torch.LongTensor(np.concatenate([train_y, train_y]))
    train_data = TensorDataset(train_x_total, train_y)
    train_iter = DataLoader(train_data, args.batch_size, shuffle=True)
    test_list = []
    max_test_output = torch.zeros((0, args.event_num), dtype=torch.float)
    max_modality_weights = None

    lambda_balance = 0.2
    lambda_distill = 0.1
    lambda_uni = 0.1

    for epoch in range(args.epoches):
        test_loss, test_score, train_l = 0, 0, 0
        train_a = []
        net.train()

        for x, y in train_iter:
            opti.zero_grad()
            train_acc = 0
            train_label = torch.LongTensor(y).to(DEVICE)
            x = torch.LongTensor(x).to(DEVICE)

            main_output, auxiliary_outputs, modality_weights = net(x)

            l_main = loss_function(main_output, train_label)
            l_uni = 0
            for modality_key in ['mod1', 'mod2', 'mod3', 'mod4']:
                modality_output = auxiliary_outputs[modality_key]
                l_uni += loss_function(modality_output, train_label)
            l_uni = l_uni / 4

            l_task = l_main + lambda_uni * l_uni
            aux_outputs_stacked = torch.stack([
                auxiliary_outputs['mod1'],
                auxiliary_outputs['mod2'],
                auxiliary_outputs['mod3'],
                auxiliary_outputs['mod4']
            ], dim=1)

            weighted_aux_sum = torch.sum(
                aux_outputs_stacked * modality_weights.unsqueeze(2),
                dim=1
            )

            weighted_aux_distribution = F.softmax(weighted_aux_sum, dim=-1)
            main_distribution = F.log_softmax(main_output, dim=-1)

            l_distill = distillation_loss_function(main_distribution, weighted_aux_distribution.detach())

            entropy_per_sample = -torch.sum(modality_weights * torch.log(modality_weights + 1e-8), dim=1)
            l_entropy = -torch.mean(entropy_per_sample)

            l_balance = l_entropy
            l_total = l_task + lambda_balance * l_balance + lambda_distill * l_distill

            l_total.backward()
            opti.step()

            train_l += l_total.item()
            train_acc = accuracy_score(torch.argmax(main_output, dim=1).detach().cpu().numpy(), train_label.cpu().numpy())
            train_a.append(train_acc)

        net.eval()
        with torch.no_grad():
            test_x_tensor = torch.LongTensor(test_x).to(DEVICE)
            main_output_test, _, modality_weights_test = net(test_x_tensor)
            test_output = F.softmax(main_output_test, dim=1)
            test_label = torch.LongTensor(test_y).to(DEVICE)
            loss = loss_function(test_output, test_label)
            test_loss = loss.item()
            test_score = f1_score(torch.argmax(test_output, dim=1).cpu().numpy(), test_label.cpu().numpy(), average='macro')
            test_acc = accuracy_score(torch.argmax(test_output, dim=1).cpu().numpy(), test_label.cpu().numpy())
            test_list.append(test_score)
            if test_score == max(test_list):
                max_test_output = test_output
                max_modality_weights = modality_weights_test
            print("test_acc:", test_acc, "train_acc:", sum(train_a) / len(train_a), "test_score:", test_score)

        print('epoch [%d] total_train_loss: %.6f testing_loss: %.6f ' % (
            epoch + 1, train_l / len(train_y), test_loss / len(test_y)))
        print('         Loss Components - Main: %.4f, Uni: %.4f, Distill: %.4f, Entropy: %.4f' % (
            l_main.item(), l_uni.item(), l_distill.item(), l_entropy.item()))

    return test_loss / len(test_y), max(test_list), train_l / len(train_y), sum(train_a) / len(
        train_a), test_list, max_test_output

def main():
    global DATA_DIR
    DATA_DIR = args.data_dir
    conn = sqlite3.connect(os.path.join(DATA_DIR, "event.db"))
    df_drug = pd.read_sql('select * from drug;', conn)
    extraction = pd.read_sql('select * from extraction;', conn)

    mechanism = extraction['mechanism']
    action = extraction['action']
    drugA = extraction['drugA']
    drugB = extraction['drugB']

    new_label = prepare(mechanism, action)
    new_label = np.array(new_label)

    dict1 = {}
    for i in df_drug["name"]:
        dict1[i] = len(dict1)

    drug_name = [dict1[i] for i in df_drug["name"]]
    drugA_id = [dict1[i] for i in drugA]
    drugB_id = [dict1[i] for i in drugB]

    dataset1_kg, dataset1_tail_len, dataset1_relation_len = read_dataset(dict1, 1)
    dataset2_kg, dataset2_tail_len, dataset2_relation_len = read_dataset(dict1, 2)
    dataset3_kg, dataset3_tail_len, dataset3_relation_len = read_dataset(dict1, 3)
    dataset4_kg, dataset4_tail_len, dataset4_relation_len = read_dataset(dict1, 4)

    x_datasets = {"drugA": drugA_id, "drugB": drugB_id}
    x_datasets = pd.DataFrame(data=x_datasets)
    x_datasets = x_datasets.to_numpy()

    dataset = {}
    dataset["dataset1"], dataset["dataset2"], dataset["dataset3"], dataset[
        "dataset4"] = dataset1_kg, dataset2_kg, dataset3_kg, dataset4_kg

    tail_len = {}
    tail_len["dataset1"], tail_len["dataset2"], tail_len["dataset3"], tail_len[
        "dataset4"] = dataset1_tail_len, dataset2_tail_len, dataset3_tail_len, dataset4_tail_len

    relation_len = {}
    relation_len["dataset1"], relation_len["dataset2"], relation_len["dataset3"], relation_len[
        "dataset4"] = dataset1_relation_len, dataset2_relation_len, dataset3_relation_len, dataset4_relation_len

    train_sum, test_sum = 0, 0
    y_true = np.array([])
    y_score = np.zeros((0, args.event_num), dtype=float)
    y_pred = np.array([])

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    kfold = kf.split(x_datasets, new_label)

    for i, (train_idx, test_idx) in enumerate(kfold):
        net = nn.Sequential(
            GNN1(dataset, tail_len, relation_len, args, dict1, drug_name),
            GNN2(dataset, tail_len, relation_len, args, dict1, drug_name),
            GNN3(dataset, tail_len, relation_len, args, dict1, drug_name),
            GNN4(dataset, tail_len, relation_len, args, dict1, drug_name),
            FusionLayer(args)
        ).to(DEVICE)

        train_x = x_datasets[train_idx]
        train_y = new_label[train_idx]
        test_x = x_datasets[test_idx]
        test_y = new_label[test_idx]

        test_loss, test_acc, train_loss, train_acc, test_list, test_output = train(train_x, train_y, test_x, test_y,
                                                                                   net)

        train_sum += train_acc
        test_sum += test_acc
        pred_type = torch.argmax(test_output, dim=1).cpu().numpy()
        y_pred = np.hstack((y_pred, pred_type))
        y_score = np.row_stack((y_score, test_output.detach().cpu().numpy()))
        y_true = np.hstack((y_true, test_y))

        print('fold %d, test_loss %f, test_acc %f, train_loss %f, train_acc %f' % (
            i, test_loss, test_acc, train_loss, train_acc))

    result_all, result_eve = evaluate(y_pred, y_score, y_true, args.event_num)
    save_result("../result/", "all", result_all)
    save_result("../result/", "each", result_eve)
    # 【返修新增】落盘逐样本概率矩阵：日后重算指标/显著性检验无需重训
    for _n, _a in (("y_true", y_true), ("y_pred", y_pred), ("y_score", y_score)):
        np.save(os.path.join("../result", "%s_AARME_task1.npy" % _n), np.asarray(_a))

    print('%d-fold validation: avg train acc  %f, avg test acc %f' % (i, train_sum / 5, test_sum / 5))
    return

if __name__ == '__main__':
    setup_seed()
    main()