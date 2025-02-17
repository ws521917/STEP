import argparse
import os
import time
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from dataloader import MyDataset
from framework import MyModel

from tools import get_config, run_test, train_epoch, get_mapper, update_config, custom_collate

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default= "0")
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--dataset', type=str, default='TC')
parser.add_argument('--test', type=bool, default=False)
parser.add_argument('--dim', type=int, default=32, help='must be a multiple of 4')
parser.add_argument('--encoder', type=str, default='trans', help='encoder type')
parser.add_argument('--batch', type=int, default=128, help='batch size')
parser.add_argument('--epoch', type=int, default= 50, help='epoch num')
args = parser.parse_args()

gpu_list = args.gpu
torch.manual_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
dataset_path = f'./data/{args.dataset}'
timestamp = time.strftime("%Y%m%d%H%M", time.localtime())
dataset_image_path = f'./data/{args.dataset}/image'
save_path = args.dataset
type=args.type
save_dir = f"./saved_models/{save_path}"
config_path = f"{save_dir}/settings.yml"
device = torch.device("cuda")

test_only = args.test


def plot_metrics(losses, acc1, acc10, mrr, args):
    """绘制训练指标折线图，分开保存并在每个点上显示具体值"""
    # 每 5 个 epoch 测试一次
    test_epochs = [i for i in range(1, len(losses) + 1) if i % 10 == 0]

    # 绘制 Loss 图
    plt.figure(figsize=(40, 10))  # 横向加宽
    plt.plot(range(1, len(losses) + 1), losses, label='Loss', color='blue', marker='o')
    for i, loss in enumerate(losses):
        plt.text(i + 1, loss, f'{loss:.2f}', ha='center', va='bottom', fontsize=9)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    loss_path = os.path.join(dataset_image_path, f'loss_plot_{args.dim}_{args.type}.png')
    plt.savefig(loss_path)
    plt.show()

    print(f"Loss plot saved to {loss_path}")

    # 绘制 Accuracy 和 MRR 图
    plt.figure(figsize=(16, 6))  # 横向加宽
    plt.plot(test_epochs, acc1, label='Acc@1', color='green', marker='o')
    plt.plot(test_epochs, acc10, label='Acc@10', color='orange', marker='o')
    plt.plot(test_epochs, mrr, label='MRR', color='red', marker='o')
    for i, (a1, a10, mr) in enumerate(zip(acc1, acc10, mrr)):
        plt.text(test_epochs[i], a1, f'{a1:.2f}', ha='center', va='bottom', fontsize=9, color='green')
        plt.text(test_epochs[i], a10, f'{a10:.2f}', ha='center', va='bottom', fontsize=9, color='orange')
        plt.text(test_epochs[i], mr, f'{mr:.2f}', ha='center', va='bottom', fontsize=9, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics (%)')
    plt.title('Accuracy and MRR (Test Every 10 Epochs)')
    plt.legend()
    acc_path = os.path.join(dataset_image_path, f'acc_mrr_plot_{args.dim}_{args.type}.png')
    plt.savefig(acc_path)
    plt.show()

    print(f"Accuracy and MRR plot saved to {acc_path}")

if __name__ == '__main__':
    get_mapper(dataset_path=dataset_path)

    update_config(config_path, key_list=['Model', 'seed'], value=args.seed)
    update_config(config_path, key_list=['Embedding', 'base_dim'], value=args.dim)
    update_config(config_path, key_list=['Encoder', 'encoder_type'], value=args.encoder)
    update_config(config_path, key_list=['Model', 'batch_size'], value=args.batch)
    update_config(config_path, key_list=['Model', 'epoch'], value=args.epoch)
    config = get_config(config_path, easy=True)

    dataset = MyDataset(config=config, dataset_path=dataset_path, device=device, load_mode='train')
    batch_size = config.Model.batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True,
                            collate_fn=lambda batch: custom_collate(batch, device, config))
    model = MyModel(config)
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total training samples: {len(dataloader) * batch_size} | Total trainable parameters: {total_params}")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.Adam_optimizer.initial_lr,
                                         weight_decay=config.Adam_optimizer.weight_decay)

    print(f"Dataset: {args.dataset} | Device: {device} | Model: {config.Encoder.encoder_type}")
    print(f"dim: {config.Embedding.base_dim}")

    if test_only:
        save_dir = f'./saved_models/{save_path}'
        top_k_acc,mrr = run_test(dataset_path=dataset_path, model_path=save_dir, model=model, device=device, epoch=49, test_only=test_only)
        exit()

    best_val_loss = float("inf")
    start_time = time.time()
    num_epochs = config.Model.epoch
    warmup_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(dataloader) * 1,
        num_training_steps=len(dataloader) * num_epochs,
    )
    epoch_losses = []
    epoch_acc1 = []
    epoch_acc10 = []
    epoch_mrr = []
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "report.txt"), "w") as report_file:
        print('Train batches:', len(dataloader))
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            average_loss = train_epoch(model, dataloader, optimizer, loss_fn, warmup_scheduler)
            epoch_losses.append(average_loss)

            epoch_str = f"================= Epoch [{epoch + 1}/{num_epochs}]| {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} | LR: {warmup_scheduler.get_last_lr()}=================\n"

            if average_loss <= best_val_loss:
                epoch_str += f"Best Loss: {best_val_loss:.6f} ---> {average_loss:.6f} | Time Token: {time.time() - epoch_start_time:.2f}s"
                best_val_loss = average_loss
            else:
                epoch_str += f"Best Loss: {best_val_loss:.6f} | Epoch Loss: {average_loss:.6f} | Time Token: {time.time() - epoch_start_time:.2f}s"

            report_file.write(epoch_str + '\n\n')
            report_file.flush()
            print(epoch_str)
            if (epoch+1) % 10 == 0:
                top_k_acc,mrr=run_test(dataset_path=dataset_path, model_path=save_dir, model=model, device=device, epoch=epoch, test_only=test_only,type=type)
                epoch_acc1.append(top_k_acc[0])
                epoch_acc10.append(top_k_acc[3])
                epoch_mrr.append(mrr)

    end_time = time.time()
    total_time = end_time - start_time

    with open(os.path.join(save_dir, "report.txt"), "a") as report_file:
        report_file.write(f"Total Running Time: {total_time:.2f} seconds\n")
    plot_metrics(epoch_losses, epoch_acc1, epoch_acc10, epoch_mrr, args)
    print(f"\nModel done.\n")
