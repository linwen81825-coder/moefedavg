"""
聚合逻辑:
  非专家参数: 普通客户端平均 Uniform FedAvg
  专家参数:
    1. uniform: 普通客户端平均
    2. routed_count: 按每个客户端对该专家的激活次数加权平均
    3. meta: DeepSet 元网络学习每个专家的客户端聚合权重

架构改动 (Fast Top-2 Sparse MoE):
  1. backbone: ResNet (CIFAR 适配版，GroupNorm)
  2. MoE 路由: 标准 Top-K 路由 (无乘法噪声，无负载均衡)
  3. MoE 计算: 真·稀疏按专家遍历
  4. 损失函数: 纯净交叉熵 (CrossEntropy)
  5. 学习率: 固定常数 LR
  6. 训练设置: 关闭 label smoothing，去掉 AutoAugment，仅保留基础增强
  7. 公平评估: 测试集固定划分 1000 张验证集，其余用于最终测试
  8. 标签噪声: 客户端噪声率独立采样自 Uniform([0,1])，并使用对称标签翻转
"""

import os, sys, copy, argparse, gc, random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torch.func import functional_call



class MetaWeightNet(nn.Module):
    """
    输入形状: [num_experts, num_clients, 2]
    每个二维输入只有: [loss, expert_freq]
    输出形状: [num_experts, num_clients]
    """
    def __init__(self, hidden_dim=16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)

    def forward(self, features):
        encoded = self.encoder(features)
        context = encoded.mean(dim=1, keepdim=True).expand_as(encoded)
        scores = self.scorer(
            torch.cat([encoded, context], dim=-1)
        ).squeeze(-1)
        return torch.softmax(scores, dim=1)


class Tee:
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


class ExperimentLogger:
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, 'w', encoding='utf-8', buffering=1)

    def console(self, message=''):
        print(message)
        self.file.write(message + '\n')

    def detail(self, message=''):
        self.file.write(message + '\n')

    def close(self):
        self.file.close()


# ════════════════════════════════════════════════════════
#  CLI 参数
# ════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser()

    p.add_argument('--dataset', default='cifar10',
                   choices=['cifar10', 'cifar100'])
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--data_root', default='./data')
    p.add_argument('--label_noise', action='store_true')

    p.add_argument('--num_clients', type=int, default=10)
    p.add_argument('--num_experts', type=int, default=4)
    p.add_argument('--topk', type=int, default=2)

    # 专家聚合方式:
    # uniform: expert 也普通平均
    # routed_count: expert 按激活次数加权平均
    # meta: 元网络学习每个专家的客户端聚合权重
    p.add_argument('--expert_agg', default='routed_count',
                   choices=['uniform', 'routed_count', 'meta'])

    p.add_argument('--rounds', type=int, default=100)
    p.add_argument('--frac', type=float, default=1.0)
    p.add_argument('--local_epochs', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--pre_batch_size', type=int, default=128)

    p.add_argument('--lr', type=float, default=0.01)

    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight_decay', type=float, default=1e-4)

    p.add_argument('--meta_lr', type=float, default=1e-3)
    p.add_argument('--meta_hidden_dim', type=int, default=16)
    p.add_argument('--meta_val_batch_size', type=int, default=256)
    p.add_argument('--meta_update_steps', type=int, default=1)

    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto',
                   choices=['auto', 'cpu', 'cuda', 'mps'])

    return p.parse_args()


DATASET_CFG = {
    'cifar10':      {'num_classes': 10,  'in_channels': 3, 'img_size': 32},
    'cifar100':     {'num_classes': 100, 'in_channels': 3, 'img_size': 32},
}


# ════════════════════════════════════════════════════════
#  数据集
# ════════════════════════════════════════════════════════

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

def partition_dirichlet(dataset, num_clients, beta, seed=42, logger=None):
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

    logger.detail(f'[Partition] Dirichlet beta={beta}, clients={num_clients}')
    for i, idx in enumerate(client_indices):
        cnt = np.bincount(labels[idx], minlength=num_classes)
        logger.detail(
            f'  Client {i}: {len(idx):>6d} samples | '
            f'{np.count_nonzero(cnt)}/{num_classes} classes'
        )
    logger.detail()

    return client_indices


def inject_heterogeneous_symmetric_noise(
        dataset, client_indices, num_classes, seed, logger):
    """
    为客户端注入异质对称标签噪声。

    每个客户端 k 的噪声率独立采样：epsilon_k ~ Uniform(0, 1)。
    对每个样本，以 1-epsilon_k 保留原标签；以 epsilon_k 翻转到
    其余 num_classes-1 个类别之一，且错误类别等概率。
    """
    rng = np.random.default_rng(seed)
    clean_targets = np.asarray(dataset.targets, dtype=np.int64)
    noisy_targets = clean_targets.copy()
    noise_rates = rng.uniform(0.0, 1.0, size=len(client_indices))

    logger.detail(
        f'[LabelNoise] heterogeneous symmetric | distribution=Uniform([0,1]) | '
        f'seed={seed}'
    )

    for cid, indices in enumerate(client_indices):
        indices = np.asarray(indices, dtype=np.int64)
        flip_mask = rng.random(indices.size) < noise_rates[cid]
        flip_indices = indices[flip_mask]

        offsets = rng.integers(1, num_classes, size=flip_indices.size)
        noisy_targets[flip_indices] = (
            clean_targets[flip_indices] + offsets
        ) % num_classes

        actual_rate = flip_indices.size / max(indices.size, 1)
        logger.detail(
            f'  Client {cid}: target_rate={noise_rates[cid]:.6f} | '
            f'actual_rate={actual_rate:.6f} | '
            f'flipped={flip_indices.size}/{indices.size}'
        )

    dataset.targets = noisy_targets.tolist()
    logger.detail()

    return noise_rates


def split_validation_test(dataset, seed, logger):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))

    val_indices = indices[:1000].tolist()
    test_indices = indices[1000:].tolist()

    logger.detail(
        f'[TestSplit] validation={len(val_indices)} | '
        f'final_test={len(test_indices)} | seed={seed}'
    )
    logger.detail()

    return Subset(dataset, val_indices), Subset(dataset, test_indices)


# ════════════════════════════════════════════════════════
#  ResNet Backbone，GroupNorm 版本
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
        self.gn1 = nn.GroupNorm(8, out_ch)

        self.conv2 = nn.Conv2d(
            out_ch, out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.gn2 = nn.GroupNorm(8, out_ch)

        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_ch, out_ch,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(8, out_ch),
            )

    def forward(self, x):
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))

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
            nn.GroupNorm(8, 64),
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

def capture_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': None,
    }

    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()

    return state


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])

    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def collect_pre_stats(global_model, pre_loader, device):
    model = copy.deepcopy(global_model).to(device)
    criterion = nn.CrossEntropyLoss()

    rng_state = capture_rng_state()

    model.eval()
    with torch.no_grad():
        x, y = next(iter(pre_loader))
        x = x.to(device)
        y = y.to(device)

        logits, expert_counts = model(x, return_counts=True)
        loss = criterion(logits, y).item()

    restore_rng_state(rng_state)

    expert_counts = expert_counts.cpu().double()
    expert_freq = (
        expert_counts
        / expert_counts.sum().clamp_min(1.0)
        * global_model.moe_head.gating.topk
    )

    del model

    return loss, expert_freq.numpy()


def local_train(global_model, train_loader, device, local_epochs, lr,
                momentum, weight_decay):
    model = copy.deepcopy(global_model).to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    total_batch_loss = 0.0
    num_batches = 0

    num_experts = len(model.moe_head.experts)
    expert_counts = torch.zeros(num_experts, dtype=torch.float64)

    for _ in range(local_epochs):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()

            logits, batch_expert_counts = model(x, return_counts=True)
            loss = criterion(logits, y)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            total_batch_loss += loss.item()
            num_batches += 1
            expert_counts += batch_expert_counts.cpu().double()

    state = {
        k: v.cpu()
        for k, v in model.state_dict().items()
    }

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_loss = total_batch_loss / max(num_batches, 1)
    train_expert_freq = (
        expert_counts
        / expert_counts.sum().clamp_min(1.0)
        * global_model.moe_head.gating.topk
    )

    return (
        state,
        train_loss,
        expert_counts.numpy(),
        train_expert_freq.numpy(),
    )


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
                        expert_agg='routed_count', meta_weights=None):
    """
    分开聚合：
      1. 非 expert 参数:
           普通 uniform 平均

      2. expert 参数:
           expert_agg == 'uniform':
             普通 uniform 平均

           expert_agg == 'routed_count':
             按该 expert 在每个客户端上的激活次数加权平均

           expert_agg == 'meta':
             使用元网络输出的客户端权重

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
            weights = np.ones(num_clients, dtype=np.float64) / num_clients

            if expert_agg == 'routed_count':
                counts_e = expert_counts[:, expert_id]
                if counts_e.sum() > 0:
                    weights = counts_e / counts_e.sum()

            if expert_agg == 'meta':
                weights = meta_weights[expert_id]

            for w, state in zip(weights, client_states):
                avg_param += state[key].float() * float(w)

        new_state[key] = avg_param.to(g_param.device).to(g_param.dtype)

    return new_state


def build_meta_features(losses, expert_freqs, device):
    losses = torch.tensor(losses, dtype=torch.float32, device=device)
    expert_freqs = torch.tensor(
        np.asarray(expert_freqs),
        dtype=torch.float32,
        device=device,
    )

    num_clients, num_experts = expert_freqs.shape

    loss_feature = losses.view(1, num_clients, 1).expand(num_experts, -1, -1)
    freq_feature = expert_freqs.transpose(0, 1).unsqueeze(-1)

    return torch.cat([loss_feature, freq_feature], dim=-1)


def prepare_temporary_parameters(global_model, client_states, device):
    fixed_state = {}
    expert_stacks = {}
    num_clients = len(client_states)

    for key, global_value in global_model.state_dict().items():
        expert_id = get_expert_id_from_key(key)

        if expert_id is None:
            value = torch.zeros_like(
                global_value,
                dtype=torch.float32,
                device=device,
            )
            for state in client_states:
                value += state[key].to(device=device, dtype=torch.float32) / num_clients
            fixed_state[key] = value.to(global_value.dtype)
        else:
            expert_stacks[key] = torch.stack(
                [state[key].to(device=device, dtype=torch.float32)
                 for state in client_states],
                dim=0,
            )

    return fixed_state, expert_stacks


def build_temporary_state(fixed_state, expert_stacks, expert_weights):
    temporary_state = dict(fixed_state)

    for key, stacked_values in expert_stacks.items():
        expert_id = get_expert_id_from_key(key)
        shape = [stacked_values.size(0)] + [1] * (stacked_values.ndim - 1)
        weights = expert_weights[expert_id].view(*shape)
        temporary_state[key] = (stacked_values * weights).sum(dim=0)

    return temporary_state


def update_meta_network(global_model, meta_net, meta_optimizer,
                        client_states, pre_features,
                        validation_loader, device, update_steps):
    criterion = nn.CrossEntropyLoss()
    fixed_state, expert_stacks = prepare_temporary_parameters(
        global_model,
        client_states,
        device,
    )

    global_model.eval()
    meta_net.train()

    total_samples = len(validation_loader.dataset)
    meta_loss = 0.0

    for _ in range(update_steps):
        meta_optimizer.zero_grad()
        total_loss = 0.0

        for x, y in validation_loader:
            x = x.to(device)
            y = y.to(device)

            temporary_weights = meta_net(pre_features)
            temporary_state = build_temporary_state(
                fixed_state,
                expert_stacks,
                temporary_weights,
            )

            logits = functional_call(global_model, temporary_state, (x,))
            loss = criterion(logits, y)

            batch_ratio = x.size(0) / total_samples
            (loss * batch_ratio).backward()

            total_loss += loss.item() * x.size(0)

        meta_optimizer.step()
        meta_loss = total_loss / total_samples

    del fixed_state, expert_stacks

    return meta_loss


# ════════════════════════════════════════════════════════
#  评估
# ════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)
        pred = logits.argmax(1)

        total_loss += loss.item() * y.size(0)
        correct += (pred == y).sum().item()
        total += y.size(0)

    test_loss = total_loss / max(total, 1)
    test_acc = 100.0 * correct / max(total, 1)

    return test_loss, test_acc


# ════════════════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════════════════

def main():
    args = get_args()

    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = (
        f'logs/MoEFedAvg_{args.dataset}_'
        f'expert_{args.expert_agg}_{timestamp}.log'
    )
    logger = ExperimentLogger(log_filename)
    original_stderr = sys.stderr
    sys.stderr = Tee(original_stderr, logger.file)

    if args.device == 'auto':
        device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )
    else:
        device = torch.device(args.device)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = DATASET_CFG[args.dataset]

    logger.detail('[Config]')
    for key, value in vars(args).items():
        logger.detail(f'{key}: {value}')
    logger.detail('meta_val_size: 1000')
    logger.detail(f'train_batch_size: {args.batch_size}')
    logger.detail(f'pre_batch_size: {args.pre_batch_size}')
    logger.detail(f'resolved_device: {device}')
    logger.detail()

    logger.console(f'[Log] {log_filename}')
    logger.console(
        f'MoE-FedAvg | dataset={args.dataset} | agg={args.expert_agg} | '
        f'clients={args.num_clients} | experts={args.num_experts} | rounds={args.rounds}'
    )

    train_ds, full_test_ds = get_dataset(args.dataset, args.data_root)
    validation_ds, test_ds = split_validation_test(full_test_ds, args.seed, logger)

    client_idx = partition_dirichlet(
        train_ds,
        args.num_clients,
        args.beta,
        args.seed,
        logger,
    )

    if args.label_noise:
        inject_heterogeneous_symmetric_noise(
            dataset=train_ds,
            client_indices=client_idx,
            num_classes=cfg['num_classes'],
            seed=args.seed,
            logger=logger,
        )
    else:
        logger.detail('[LabelNoise] disabled')
        logger.detail()

    client_subsets = [Subset(train_ds, idx) for idx in client_idx]

    train_loaders = []
    for cid, client_subset in enumerate(client_subsets):
        generator = torch.Generator()
        generator.manual_seed(args.seed + cid)

        train_loaders.append(
            DataLoader(
                client_subset,
                batch_size=args.batch_size,
                shuffle=True,
                generator=generator,
                num_workers=0,
                pin_memory=True,
            )
        )

    pre_loaders = None
    if args.expert_agg == 'meta':
        pre_loaders = []
        for cid, client_subset in enumerate(client_subsets):
            generator = torch.Generator()
            generator.manual_seed(args.seed + 100000 + cid)

            pre_loaders.append(
                DataLoader(
                    client_subset,
                    batch_size=args.pre_batch_size,
                    shuffle=True,
                    generator=generator,
                    num_workers=0,
                    pin_memory=True,
                )
            )

    validation_loader = DataLoader(
        validation_ds,
        batch_size=args.meta_val_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        num_workers=0,
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
    logger.detail(f'[Model] Total params: {n_params:,}')
    logger.detail()

    meta_net = None
    meta_optimizer = None
    if args.expert_agg == 'meta':
        rng_state = capture_rng_state()
        meta_net = MetaWeightNet(args.meta_hidden_dim).to(device)
        restore_rng_state(rng_state)

        meta_optimizer = optim.Adam(meta_net.parameters(), lr=args.meta_lr)
        meta_params = sum(p.numel() for p in meta_net.parameters())
        logger.detail(
            f'[MetaNet] Total params: {meta_params:,} | '
            f'Optimizer=Adam | LR={args.meta_lr}'
        )
        logger.detail(
            f'[MetaConfig] raw_input=[loss, expert_freq] | input_dim=2 | '
            f'hidden_dim={args.meta_hidden_dim} | '
            f'validation_batch_size={args.meta_val_batch_size}'
        )
        logger.detail(
            f'[MetaConfig] shared_encoder=True | shared_scorer=True | '
            f'pooling=mean | update_steps={args.meta_update_steps}'
        )
        logger.detail('[MetaConfig] softmax_dim=client | scorer_last_zero_init=True')
        logger.detail()

    m = max(1, int(args.num_clients * args.frac))

    best_acc = 0.0
    console_header = (
        f'{"Round":>5} | {"TrainLoss":>9} | {"MetaLoss":>8} | '
        f'{"TestLoss":>8} | {"TestAcc":>8} | {"Best":>8}'
    )
    logger.console()
    logger.console(console_header)
    logger.console('-' * len(console_header))

    logger.detail('[RoundMetrics]')
    logger.detail(
        'round | lr | train_loss | meta_loss | test_loss | test_acc | '
        'best_acc | expert_ratio | chosen_clients'
    )

    for rnd in range(1, args.rounds + 1):
        current_lr = args.lr

        chosen = np.random.choice(
            args.num_clients,
            m,
            replace=False,
        ).tolist()

        all_states = []
        all_losses = []
        all_expert_counts = []
        all_train_expert_freqs = []

        all_pre_losses = []
        all_pre_expert_freqs = []

        for cid in chosen:
            if args.expert_agg == 'meta':
                pre_loss, pre_expert_freq = collect_pre_stats(
                    global_model,
                    pre_loaders[cid],
                    device,
                )
                all_pre_losses.append(pre_loss)
                all_pre_expert_freqs.append(pre_expert_freq)

            (
                state,
                train_loss,
                expert_counts,
                train_expert_freq,
            ) = local_train(
                global_model=global_model,
                train_loader=train_loaders[cid],
                device=device,
                local_epochs=args.local_epochs,
                lr=current_lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )

            all_states.append(state)
            all_losses.append(train_loss)
            all_expert_counts.append(expert_counts)
            all_train_expert_freqs.append(train_expert_freq)

        all_expert_counts = np.stack(all_expert_counts, axis=0)

        round_expert_counts = all_expert_counts.sum(axis=0)
        round_expert_total = round_expert_counts.sum()
        round_expert_ratio = (
            round_expert_counts
            / max(round_expert_total, 1.0)
            * args.topk
        )

        meta_loss = None
        final_meta_weights = None
        temporary_meta_weights = None

        if args.expert_agg == 'meta':
            pre_features = build_meta_features(
                all_pre_losses,
                all_pre_expert_freqs,
                device,
            )
            train_features = build_meta_features(
                all_losses,
                all_train_expert_freqs,
                device,
            )

            with torch.no_grad():
                temporary_meta_weights = meta_net(pre_features).cpu().numpy()

            meta_loss = update_meta_network(
                global_model=global_model,
                meta_net=meta_net,
                meta_optimizer=meta_optimizer,
                client_states=all_states,
                pre_features=pre_features,
                validation_loader=validation_loader,
                device=device,
                update_steps=args.meta_update_steps,
            )

            meta_net.eval()
            with torch.no_grad():
                final_meta_weights = meta_net(train_features).cpu().numpy()

            new_state = split_expert_fedavg(
                global_model=global_model,
                client_states=all_states,
                expert_counts=all_expert_counts,
                expert_agg='meta',
                meta_weights=final_meta_weights,
            )
        else:
            new_state = split_expert_fedavg(
                global_model=global_model,
                client_states=all_states,
                expert_counts=all_expert_counts,
                expert_agg=args.expert_agg,
            )

        global_model.load_state_dict(new_state)

        avg_loss = float(np.mean(all_losses))

        test_loss, acc = evaluate(global_model, test_loader, device)
        best_acc = max(best_acc, acc)

        ratio_str = '[' + ', '.join([f'{r:.2f}' for r in round_expert_ratio]) + ']'

        meta_loss_str = f'{meta_loss:.4f}' if meta_loss is not None else '-'

        logger.console(
            f'{rnd:>5} | {avg_loss:>9.4f} | {meta_loss_str:>8} | '
            f'{test_loss:>8.4f} | {acc:>7.2f}% | {best_acc:>7.2f}%'
        )
        logger.detail(
            f'{rnd} | {current_lr:.6f} | {avg_loss:.6f} | '
            f'{meta_loss_str} | {test_loss:.6f} | {acc:.4f} | '
            f'{best_acc:.4f} | {ratio_str} | {chosen}'
        )

        for i, cid in enumerate(chosen):
            train_freq_text = '[' + ', '.join(
                f'{value:.4f}' for value in all_train_expert_freqs[i]
            ) + ']'

            if args.expert_agg == 'meta':
                pre_freq_text = '[' + ', '.join(
                    f'{value:.4f}' for value in all_pre_expert_freqs[i]
                ) + ']'
                logger.detail(
                    f'[ClientStats] round={rnd} client=C{cid} | '
                    f'pre_loss={all_pre_losses[i]:.4f} | '
                    f'pre_expert_freq={pre_freq_text} | '
                    f'train_loss={all_losses[i]:.4f} | '
                    f'train_expert_freq={train_freq_text}'
                )
            else:
                logger.detail(
                    f'[ClientStats] round={rnd} client=C{cid} | '
                    f'train_loss={all_losses[i]:.4f} | '
                    f'train_expert_freq={train_freq_text}'
                )

        if final_meta_weights is not None:
            for expert_id in range(args.num_experts):
                temp_text = ', '.join(
                    f'C{cid}={temporary_meta_weights[expert_id, i]:.4f}'
                    for i, cid in enumerate(chosen)
                )
                final_text = ', '.join(
                    f'C{cid}={final_meta_weights[expert_id, i]:.4f}'
                    for i, cid in enumerate(chosen)
                )
                logger.detail(f'[MetaTempWeight] round={rnd} expert={expert_id} | {temp_text}')
                logger.detail(f'[MetaFinalWeight] round={rnd} expert={expert_id} | {final_text}')

        del (
            all_states,
            all_losses,
            all_expert_counts,
            all_pre_losses,
            all_pre_expert_freqs,
            all_train_expert_freqs,
            new_state,
        )
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.console(f'\nDone | BestAcc={best_acc:.2f}%')
    logger.console(f'Log saved: {log_filename}')
    sys.stderr = original_stderr
    logger.close()


if __name__ == '__main__':
    main()