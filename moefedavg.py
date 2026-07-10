"""
聚合逻辑:
  非专家参数: 普通客户端平均 Uniform FedAvg
  专家参数:
    1. uniform: 普通客户端平均
    2. routed_count: 按每个客户端对该专家的激活次数加权平均

架构改动 (Fast Top-2 Sparse MoE):
  1. backbone: ResNet (CIFAR 适配版，无 BatchNorm)
  2. MoE 路由: 标准 Top-K 路由 (无乘法噪声，无负载均衡)
  3. MoE 计算: 真·稀疏按专家遍历
  4. 损失函数: 纯净交叉熵 (CrossEntropy)
  5. 学习率: 固定常数 LR
  6. 训练设置: 关闭 label smoothing，去掉 AutoAugment，仅保留基础增强
"""

import os, copy, argparse, zipfile, urllib.request, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder


# ════════════════════════════════════════════════════════
#  CLI 参数
# ════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser()

    p.add_argument('--dataset', default='cifar10',
                   choices=['cifar10', 'cifar100', 'tinyimagenet', 'femnist'])
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--data_root', default='./data')

    p.add_argument('--num_clients', type=int, default=10)
    p.add_argument('--num_experts', type=int, default=4)
    p.add_argument('--topk', type=int, default=2)

    # 专家聚合方式:
    # uniform: expert 也普通平均
    # routed_count: expert 按激活次数加权平均
    p.add_argument('--expert_agg', default='routed_count',
                   choices=['uniform', 'routed_count'])

    p.add_argument('--rounds', type=int, default=100)
    p.add_argument('--frac', type=float, default=1.0)
    p.add_argument('--local_epochs', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=64)

    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--lr_min', type=float, default=1e-4)        # 保留防止旧命令报错
    p.add_argument('--warmup_rounds', type=int, default=5)      # 保留防止旧命令报错

    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--label_smooth', type=float, default=0.0)

    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto',
                   choices=['auto', 'cpu', 'cuda', 'mps'])

    return p.parse_args()


DATASET_CFG = {
    'cifar10':      {'num_classes': 10,  'in_channels': 3, 'img_size': 32},
    'cifar100':     {'num_classes': 100, 'in_channels': 3, 'img_size': 32},
    'tinyimagenet': {'num_classes': 200, 'in_channels': 3, 'img_size': 64},
    'femnist':      {'num_classes': 62,  'in_channels': 1, 'img_size': 28},
}


# ════════════════════════════════════════════════════════
#  数据集
# ════════════════════════════════════════════════════════

def _prepare_tinyimagenet(root):
    data_path = os.path.join(root, 'tiny-imagenet-200')

    if not os.path.exists(data_path):
        url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        zip_path = os.path.join(root, 'tiny-imagenet-200.zip')

        print('[Dataset] Downloading TinyImageNet...')
        urllib.request.urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(root)

        os.remove(zip_path)

    val_dir = os.path.join(data_path, 'val')
    anno = os.path.join(val_dir, 'val_annotations.txt')
    img_dir = os.path.join(val_dir, 'images')

    if os.path.exists(anno):
        for line in open(anno):
            parts = line.strip().split('\t')
            cls_dir = os.path.join(val_dir, parts[1])
            os.makedirs(cls_dir, exist_ok=True)

            src = os.path.join(img_dir, parts[0])
            if os.path.exists(src):
                os.rename(src, os.path.join(cls_dir, parts[0]))

        os.remove(anno)

        if os.path.isdir(img_dir) and not os.listdir(img_dir):
            os.rmdir(img_dir)

    return data_path


def get_dataset(name, data_root):
    os.makedirs(data_root, exist_ok=True)

    if name == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

        tr = transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        te = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        return (
            torchvision.datasets.CIFAR10(data_root, True, download=True, transform=tr),
            torchvision.datasets.CIFAR10(data_root, False, download=True, transform=te),
        )

    elif name == 'cifar100':
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

        tr = transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        te = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        return (
            torchvision.datasets.CIFAR100(data_root, True, download=True, transform=tr),
            torchvision.datasets.CIFAR100(data_root, False, download=True, transform=te),
        )

    elif name == 'tinyimagenet':
        dp = _prepare_tinyimagenet(data_root)
        mean, std = (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)

        tr = transforms.Compose([
            transforms.RandomCrop(64, 8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        te = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        return (
            ImageFolder(os.path.join(dp, 'train'), transform=tr),
            ImageFolder(os.path.join(dp, 'val'), transform=te),
        )

    else:
        tr = te = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        return (
            torchvision.datasets.EMNIST(
                data_root, split='byclass',
                train=True, download=True, transform=tr
            ),
            torchvision.datasets.EMNIST(
                data_root, split='byclass',
                train=False, download=True, transform=te
            ),
        )


def partition_dirichlet(dataset, num_clients, beta, seed=42):
    rng = np.random.default_rng(seed)
    labels = np.array(dataset.targets if hasattr(dataset, 'targets') else dataset.labels)

    num_classes = int(labels.max()) + 1
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)

        props = rng.dirichlet(np.full(num_clients, beta))
        counts = rng.multinomial(len(idx_c), props)

        cur = 0
        for i, cnt in enumerate(counts):
            client_indices[i].extend(idx_c[cur:cur + cnt].tolist())
            cur += cnt

    for i in range(num_clients):
        rng.shuffle(client_indices[i])

    print(f'\n[Partition] Dirichlet beta={beta}, {num_clients} clients')
    for i, idx in enumerate(client_indices):
        cnt = np.bincount(labels[idx], minlength=num_classes)
        print(f'  Client {i}: {len(idx):>6d} samples | {np.count_nonzero(cnt)}/{num_classes} classes')
    print()

    return client_indices


# ════════════════════════════════════════════════════════
#  ResNet Backbone，无 BatchNorm 版本
# ════════════════════════════════════════════════════════

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            out_ch, out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_ch, out_ch,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        out = out + self.shortcut(x)
        out = self.relu(out)

        return out


class ResNetBackbone(nn.Module):
    def __init__(self, in_channels, img_size):
        super().__init__()

        stem_stride = 1 if img_size <= 32 else 2

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels, 64,
                kernel_size=3,
                stride=stem_stride,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = 512

    @staticmethod
    def _make_layer(in_ch, out_ch, stride):
        return nn.Sequential(
            BasicBlock(in_ch, out_ch, stride=stride),
            BasicBlock(out_ch, out_ch, stride=1),
        )

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)

        return x.flatten(1)


# ════════════════════════════════════════════════════════
#  Fast Sparse MoE (Top-K) 模块
# ════════════════════════════════════════════════════════

class ExpertFFN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TopKGating(nn.Module):
    """
    标准 Top-K 路由：
      - 无负载均衡损失
      - 无乘法噪声
      - topk 权重会重新归一化，让被选中的专家权重和为 1
    """
    def __init__(self, in_dim, num_experts, topk):
        super().__init__()

        self.num_experts = num_experts
        self.topk = min(topk, num_experts)

        self.gate = nn.Linear(in_dim, num_experts, bias=False)

    def forward(self, x):
        logits = self.gate(x)
        probs = torch.softmax(logits.float(), dim=-1)

        topk_vals, topk_idx = probs.topk(self.topk, dim=-1)

        # 关键：Top-K 后重新归一化
        topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-12)

        weights = torch.zeros_like(probs)
        weights.scatter_(1, topk_idx, topk_vals)
        weights = weights.to(x.dtype)

        return weights, topk_idx


class MoELayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_experts, topk):
        super().__init__()

        self.num_experts = num_experts

        self.gating = TopKGating(in_dim, num_experts, topk)

        self.experts = nn.ModuleList([
            ExpertFFN(in_dim, hidden_dim, out_dim)
            for _ in range(num_experts)
        ])

    def forward(self, x, return_counts=False):
        weights, topk_idx = self.gating(x)

        B = x.size(0)
        C = self.experts[0].fc2.out_features

        out = torch.zeros(B, C, device=x.device, dtype=x.dtype)

        for i, expert in enumerate(self.experts):
            expert_mask = (topk_idx == i)
            token_mask = expert_mask.any(dim=-1)

            if not token_mask.any():
                continue

            expert_out = expert(x[token_mask])
            sel_weights = weights[token_mask, i]

            out[token_mask] += expert_out * sel_weights.unsqueeze(-1)

        if return_counts:
            # topk_idx.shape = [batch_size, topk]
            # topk=2 时，每张图会给两个专家各计数 1 次
            expert_counts = torch.bincount(
                topk_idx.reshape(-1),
                minlength=self.num_experts,
            ).to(torch.float32)

            return out, expert_counts

        return out


class MoEFedModel(nn.Module):
    def __init__(self, in_channels, num_classes, img_size, num_experts, topk):
        super().__init__()

        self.backbone = ResNetBackbone(in_channels, img_size)

        feat_dim = self.backbone.feat_dim

        self.moe_head = MoELayer(
            in_dim=feat_dim,
            hidden_dim=512,
            out_dim=num_classes,
            num_experts=num_experts,
            topk=topk,
        )

    def forward(self, x, return_counts=False):
        feat = self.backbone(x)

        if return_counts:
            logits, expert_counts = self.moe_head(feat, return_counts=True)
            return logits, expert_counts

        logits = self.moe_head(feat)

        return logits


# ════════════════════════════════════════════════════════
#  客户端训练
# ════════════════════════════════════════════════════════

def local_train(global_model, loader, device, local_epochs, lr,
                momentum, weight_decay, label_smooth):
    model = copy.deepcopy(global_model).to(device)
    model.train()

    opt = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)

    total_loss = 0.0
    n_processed = 0

    client_sample_count = len(loader.dataset)

    num_experts = len(model.moe_head.experts)
    expert_counts = torch.zeros(num_experts, dtype=torch.float64)

    for _ in range(local_epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()

            logits, batch_expert_counts = model(x, return_counts=True)

            loss = criterion(logits, y)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            opt.step()

            total_loss += loss.item() * x.size(0)
            n_processed += x.size(0)

            expert_counts += batch_expert_counts.cpu().double()

    state = {
        k: v.cpu()
        for k, v in model.state_dict().items()
    }

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    avg_loss = total_loss / max(n_processed, 1)

    return state, client_sample_count, avg_loss, expert_counts.numpy()


# ════════════════════════════════════════════════════════
#  服务端聚合
# ════════════════════════════════════════════════════════

def get_expert_id_from_key(key):
    """
    判断某个参数名是不是 expert 参数，并返回 expert id。

    例如：
      moe_head.experts.0.fc1.weight -> 0
      moe_head.experts.1.fc2.bias   -> 1

    非 expert 参数返回 None。
    """
    parts = key.split('.')

    if len(parts) >= 3 and parts[0] == 'moe_head' and parts[1] == 'experts':
        return int(parts[2])

    return None


def split_expert_fedavg(global_model, client_states, expert_counts,
                        expert_agg='routed_count', eps=1e-12):
    """
    分开聚合：
      1. 非 expert 参数:
           普通 uniform 平均

      2. expert 参数:
           expert_agg == 'uniform':
             普通 uniform 平均

           expert_agg == 'routed_count':
             按该 expert 在每个客户端上的激活次数加权平均

    参数:
      client_states:
        list，每个元素是一个客户端上传的 state_dict

      expert_counts:
        numpy array，shape = [num_selected_clients, num_experts]
        expert_counts[i, e] 表示第 i 个被选客户端对 expert e 的激活次数
    """
    new_state = {}

    num_clients = len(client_states)
    expert_counts = np.asarray(expert_counts, dtype=np.float64)

    for key, g_param in global_model.state_dict().items():
        expert_id = get_expert_id_from_key(key)

        avg_param = torch.zeros_like(
            g_param,
            dtype=torch.float32,
            device='cpu',
        )

        # ==========================================================
        # 1. 非专家参数：普通客户端平均
        #    包括 backbone 和 router/gating
        # ==========================================================
        if expert_id is None:
            for state in client_states:
                avg_param += state[key].float() / num_clients

        # ==========================================================
        # 2. 专家参数：根据 expert_agg 决定聚合方式
        # ==========================================================
        else:
            if expert_agg == 'uniform':
                weights = np.ones(num_clients, dtype=np.float64) / num_clients

            elif expert_agg == 'routed_count':
                counts_e = expert_counts[:, expert_id]
                total_count = counts_e.sum()

                # 如果某个 expert 本轮完全没被用到，退回普通平均
                if total_count <= eps:
                    weights = np.ones(num_clients, dtype=np.float64) / num_clients
                else:
                    weights = counts_e / total_count

            else:
                raise ValueError(f'Unknown expert_agg: {expert_agg}')

            for w, state in zip(weights, client_states):
                avg_param += state[key].float() * float(w)

        new_state[key] = avg_param.to(g_param.device).to(g_param.dtype)

    return new_state


# ════════════════════════════════════════════════════════
#  评估
# ════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        pred = logits.argmax(1)

        correct += (pred == y).sum().item()
        total += y.size(0)

    return 100.0 * correct / max(total, 1)


# ════════════════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════════════════

def main():
    args = get_args()

    if args.device == 'auto':
        device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = DATASET_CFG[args.dataset]

    print(f'\n{"=" * 80}')
    print(f'  MoE-FedAvg No-BN | NonExpert=Uniform | Expert={args.expert_agg} | Top-{args.topk}')
    print(f'  Dataset={args.dataset} | beta={args.beta} | Clients={args.num_clients} | Experts={args.num_experts}')
    print(f'  Device={device} | Rounds={args.rounds} | LR={args.lr} (Constant)')
    print(f'{"=" * 80}\n')

    train_ds, test_ds = get_dataset(args.dataset, args.data_root)

    client_idx = partition_dirichlet(
        train_ds,
        args.num_clients,
        args.beta,
        args.seed,
    )

    client_loaders = [
        DataLoader(
            Subset(train_ds, idx),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        for idx in client_idx
    ]

    test_loader = DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    global_model = MoEFedModel(
        in_channels=cfg['in_channels'],
        num_classes=cfg['num_classes'],
        img_size=cfg['img_size'],
        num_experts=args.num_experts,
        topk=args.topk,
    ).to(device)

    n_params = sum(p.numel() for p in global_model.parameters())
    print(f'[Model] Total params: {n_params:,}\n')

    m = max(1, int(args.num_clients * args.frac))

    best_acc = 0.0
    history_records = []

    print(
        f'{"Round":>5} | {"LR":>7} | {"AvgLoss":>8} | '
        f'{"TestAcc":>8} | {"Best":>8} | {"ExpertRatio":>28}'
    )
    print('-' * 85)

    for rnd in range(1, args.rounds + 1):
        current_lr = args.lr

        chosen = np.random.choice(
            args.num_clients,
            m,
            replace=False,
        ).tolist()

        all_states = []
        all_samples = []
        all_losses = []
        all_expert_counts = []

        for cid in chosen:
            state, n_samples, loss, expert_counts = local_train(
                global_model=global_model,
                loader=client_loaders[cid],
                device=device,
                local_epochs=args.local_epochs,
                lr=current_lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                label_smooth=args.label_smooth,
            )

            all_states.append(state)
            all_samples.append(n_samples)
            all_losses.append(loss)
            all_expert_counts.append(expert_counts)

        all_expert_counts = np.stack(all_expert_counts, axis=0)

        round_expert_counts = all_expert_counts.sum(axis=0)
        round_expert_total = round_expert_counts.sum()
        round_expert_ratio = round_expert_counts / max(round_expert_total, 1.0)

        new_state = split_expert_fedavg(
            global_model=global_model,
            client_states=all_states,
            expert_counts=all_expert_counts,
            expert_agg=args.expert_agg,
        )

        global_model.load_state_dict(new_state)

        avg_loss = float(np.mean(all_losses))

        acc = evaluate(global_model, test_loader, device)
        best_acc = max(best_acc, acc)

        ratio_str = '[' + ', '.join([f'{r:.2f}' for r in round_expert_ratio]) + ']'

        print(
            f'{rnd:>5} | {current_lr:>7.5f} | {avg_loss:>8.4f} | '
            f'{acc:>7.2f}% | {best_acc:>7.2f}% | {ratio_str:>28}'
        )

        record = {
            'Round': rnd,
            'AvgLoss': avg_loss,
            'TestAcc': acc,
            'BestAcc': best_acc,
            'ExpertAgg': args.expert_agg,
        }

        for e in range(args.num_experts):
            record[f'Expert{e}_Count'] = float(round_expert_counts[e])
            record[f'Expert{e}_Ratio'] = float(round_expert_ratio[e])

        history_records.append(record)

        del all_states, all_samples, all_losses, all_expert_counts, new_state
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f'\nDone. Best Acc: {best_acc:.2f}%')

    excel_filename = (
        f'MoEFedAvg_NoBN_nonexpert_uniform_'
        f'expert_{args.expert_agg}_'
        f'{args.dataset}_clients{args.num_clients}_experts{args.num_experts}.xlsx'
    )

    df = pd.DataFrame(history_records)
    df.to_excel(excel_filename, index=False)

    print(f'[Export] 训练数据已成功保存至 Excel 文件: {excel_filename}')


if __name__ == '__main__':
    main()