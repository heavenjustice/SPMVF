import os
import argparse
import torch
import time
import pickle
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tools import shuffle
from tools import calculate
from models import model
from tqdm import tqdm

torch.backends.cudnn.benchmark = False

def get_configs():
    parser = argparse.ArgumentParser(description="SPMVF config")
    parser.add_argument("--gpu_id", type=str, default=0)
    parser.add_argument("--log_date", type=str, default=None)
    parser.add_argument("--sample_num", type=int, default=7)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="ped2")
    args = parser.parse_args()

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

    # Score
    for i, scores in enumerate(video_output_spatial):
        frame_scores = scores
        with open("video_%d.txt" % i, "w") as f:
            for frame_id, score in enumerate(frame_scores):
                f.write("%d %.4f\n" % (frame_id, score))

    smoothed_res, smoothed_auc_list = evaluate_auc(video_output_spatial, dataset=dataset)
    return smoothed_res.auc, np.mean(smoothed_auc_list)

if __name__ == '__main__':
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    args = get_configs()
    state = torch.load(args.checkpoint)
    print('load ' + args.checkpoint)
    net = model.WideBranchNet_R(time_length=7, num_classes=args.sample_num)
    net.load_state_dict(state, strict=True)
    net.cuda()
    smoothed_auc, smoothed_auc_avg, _ = val(args, net)