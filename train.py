import os
import argparse
import torch
import time
import pickle
import numpy as np
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tools import shuffle
from tools import calculate
from models import model
from tqdm import tqdm

torch.backends.cudnn.benchmark = False

# Base Config
def get_configs():
    parser = argparse.ArgumentParser(description="SPMVF config")
    parser.add_argument("--val_step", type=int, default=200)
    parser.add_argument("--print_interval", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--gpu_id", type=str, default=0)
    parser.add_argument("--log_date", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--static_threshold", type=float, default=0.2)
    parser.add_argument("--sample_num", type=int, default=7)
    parser.add_argument("--filter_ratio", type=float, default=0.6)
    parser.add_argument("--dataset", type=str, default="ped2")
    args = parser.parse_args()

    args.device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
    if args.dataset in ['shanghaitech', 'avenue']:
        args.filter_ratio = 0.8
    elif args.dataset == 'ped2':
        args.filter_ratio = 0.6
    return args


def train(args):
    if not args.log_date:
        running_date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    else:
        running_date = args.log_date
    print("The running_data : {}".format(running_date))
    for k,v in vars(args).items():
        print("-------------{} : {}".format(k, v))

    #Load
    data_dir = f"./datasets/{args.dataset}/training"
    detect_pkl = f'detect/{args.dataset}_train_detect_result.pkl'

    vad_dataset = Shuffle(data_dir, dataset=args.dataset, detect_dir=detect_pkl,fliter_ratio=args.filter_ratio, frame_num=7,puzzle_num=args.sample_num,static_threshold=args.static_threshold)

    vad_dataloader = DataLoader(vad_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    net = model.WideBranchNet_R(time_length=7, num_classes=args.sample_num)

    net.cuda(args.device)
    net = net.train()

    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(params=net.parameters(), lr=1e-4)

    log_dir = './log/{}/'.format(running_date)
    writer = SummaryWriter(log_dir)

    t0 = time.time()
    global_step = 0

    max_acc = -1
    timestamp_in_max = None

    for epoch in range(args.epochs):
        for it, data in enumerate(vad_dataloader):
            video, obj, spat_labels, t_flag = data['video'], data['obj'], data['label'], data["trans_label"], data["temporal"]
            #n_temp = t_flag.sum().item()

            obj = obj.cuda(args.device, non_blocking=True)
            spat_labels = spat_labels[t_flag].long().view(-1).cuda(args.device)

            spat_logits = net(obj)
            spat_logits = spat_logits[t_flag].view(-1, args.sample_num)

            spat_loss = criterion(spat_logits, spat_labels)

            loss = spat_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('Train/Loss', loss.item(), global_step=global_step)

            if (global_step + 1) % args.print_interval == 0:
                print("[{}:{}/{}]\tloss: {:.6f} \ttime: {:.6f}". \
                      format(epoch, it + 1, len(vad_dataloader), loss.item(),
                             time.time() - t0))
                t0 = time.time()

            global_step += 1

            if global_step % args.val_step == 0 and epoch >= 2:
                smoothed_auc, smoothed_auc_avg, temp_timestamp = val(args, net)
                writer.add_scalar('Test/smoothed_auc', smoothed_auc, global_step=global_step)
                writer.add_scalar('Test/smoothed_auc_avg', smoothed_auc_avg, global_step=global_step)

                if smoothed_auc > max_acc:
                    max_acc = smoothed_auc
                    timestamp_in_max = temp_timestamp
                    save = './checkpoint/{}_{}.pth'.format('best', running_date)
                    torch.save(net.state_dict(), save)

                print('cur max: ' + str(max_acc) + ' in ' + timestamp_in_max)
                net = net.train()


def val(args, net=None):
    if not args.log_date:
        running_date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    else:
        running_date = args.log_date
    print("The running_date : {}".format(running_date))

    # Load Data
    data_dir = f"./datasets/{args.dataset}/testing"
    detect_pkl = f'detect/{args.dataset}_test_detect_result.pkl'

    testing_dataset = Shuffle(data_dir, dataset=args.dataset, detect_dir=detect_pkl, fliter_ratio=args.filter_ratio,
                              frame_num=7, puzzle_num=args.sample_num)
    testing_data_loader = DataLoader(testing_dataset, batch_size=256, shuffle=False, num_workers=0, drop_last=False)

    net.eval()

    video_output = {}
    for data in tqdm(testing_data_loader):
        videos = data["video"]
        frames = data["frame"].tolist()
        obj = data["obj"].cuda(args.device)

        with torch.no_grad():
            spat_logits = net(obj)
            spat_logits = spat_logits.view(-1, args.sample_num, args.sample_num)

        spat_probs = F.softmax(spat_logits, -1)
        diag = torch.diagonal(spat_probs, offset=0, dim1=-2, dim2=-1)
        scores = diag.min(-1)[0].cpu().numpy()

        for video_, frame_, score_ in zip(videos, frames, scores):
            if video_ not in video_output:
                video_output[video_] = {}
            if frame_ not in video_output[video_]:
                video_output[video_][frame_] = []
            video_output[video_][frame_].append([score_])

    micro_auc, macro_auc = save_and_evaluate(video_output, running_date, dataset=args.dataset)
    return micro_auc, macro_auc, running_date


def save_and_evaluate(video_output, running_date, dataset='shanghaitech'):
    pickle_path = './log/video_output_ori_{}.pkl'.format(running_date)
    with open(pickle_path, 'wb') as write:
        pickle.dump(video_output, write, pickle.HIGHEST_PROTOCOL)
    if dataset == 'shanghaitech':
        video_output_spatial = remake_video_output(video_output, dataset=dataset)
    else:
        video_output_spatial = remake_video_3d_output(video_output, dataset=dataset)
    evaluate_auc(video_output_spatial, dataset=dataset)
    smoothed_res, smoothed_auc_list = evaluate_auc(video_output_spatial, dataset=dataset)
    return smoothed_res.auc, np.mean(smoothed_auc_list)

if __name__ == '__main__':
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    args = get_configs()
    train(args)
