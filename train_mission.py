import argparse
import json
import numpy as np
import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from solver import solver_RNN, HetNet_solver

from Environment import Mission

import os
from tqdm import tqdm


parser = argparse.ArgumentParser()

parser.add_argument("--model_type", type=str, default="ptrnet")
parser.add_argument("--coverage_num", type=int, default=4)
parser.add_argument("--visiting_num", type=int, default=4)
parser.add_argument("--pick_place_num", type=int, default=4)

parser.add_argument("--num_epochs", type=int, default=200)

parser.add_argument("--num_tr_dataset", type=int, default=1280)
parser.add_argument("--num_te_dataset", type=int, default=1280)

parser.add_argument("--embedding_size", type=int, default=128)
parser.add_argument("--hidden_size", type=int, default=128)

parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--grad_clip", type=float, default=1.5)
parser.add_argument("--beta", type=float, default=0.9)

parser.add_argument("--use_cuda", type=bool, default=True)

parser.add_argument("--log_interval", type=int, default=1)

args = parser.parse_args()



if __name__ =="__main__":
    if args.use_cuda:
        use_pin_memory = True
    else:
        use_pin_memory = False

    torch.manual_seed(100)
    torch.cuda.manual_seed(100)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    np.random.seed(100)
    random.seed(100)

    print("GENERATING TRAINING DATASET")
    train_dataset = Mission.MissionDataset(num_visit=args.visiting_num, num_coverage=args.coverage_num, num_pick_place=args.pick_place_num,
                                           num_samples=args.num_tr_dataset,
                                           random_seed=100,
                                           overlap=True)
    print("GENERATING EVALUATION DATASET")
    eval_dataset = Mission.MissionDataset(num_visit=args.visiting_num, num_coverage=args.coverage_num, num_pick_place=args.pick_place_num,
                                          num_samples=args.num_te_dataset,
                                          random_seed=200,
                                          overlap=True)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=use_pin_memory)

    eval_loader = DataLoader(eval_dataset,
                             batch_size=args.num_te_dataset,
                             shuffle=False)

    if args.model_type == "ptrnet":
        tensorboard_path = 'runs/'+"PtrNet_C%d_V%d_D%d" %(args.coverage_num, args.visiting_num, args.pick_place_num)
        print("RNN model is used")
        model = solver_RNN(
            args.embedding_size,
            args.hidden_size,
            args.visiting_num+args.coverage_num+args.pick_place_num+1,
            2, 10)
    # AM network
    elif args.model_type.startswith("hetnet"):
        tensorboard_path = 'runs/'+"HetNet_C%d_V%d_D%d" %(args.coverage_num, args.visiting_num, args.pick_place_num)
        print("HetNet is used")
        model = HetNet_solver(
            args.embedding_size,
            args.hidden_size,
            # args.visiting_num+args.coverage_num+args.pick_place_num+1,
            2, 10)
    else:
        raise

    writer = SummaryWriter(tensorboard_path)

    if args.use_cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1.0 * 1e-4)

    # Train loop
    moving_avg = torch.zeros(args.num_tr_dataset)
    if args.use_cuda:
        moving_avg = moving_avg.cuda()

    #generating first baseline
    print("GENERATING FIRST BASELINE")
    for (indices, sample_batch) in tqdm(train_data_loader):
        if args.use_cuda:
            sample_batch = sample_batch.cuda()
        rewards, _, _ = model(sample_batch)
        moving_avg[indices] = rewards

    torch.cuda.empty_cache()

    #Training
    print("TRAINING START")
    model.train()
    for epoch in tqdm(range(args.num_epochs)):
        for batch_idx, (indices, sample_batch) in enumerate(train_data_loader):
            if args.use_cuda:
                sample_batch = sample_batch.cuda()
            rewards, log_probs, action = model(sample_batch)
            moving_avg[indices] = moving_avg[indices] * args.beta + rewards * (1.0 - args.beta)
            advantage = rewards - moving_avg[indices]
            log_probs = torch.sum(log_probs, dim=-1)
            log_probs[log_probs < -100] = -100
            loss = (advantage * log_probs).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()


        model.eval()
        ret = []
        for i, batch in eval_loader:
            if args.use_cuda:
                batch = batch.cuda()
            R, _, _ = model(batch)

        mean_R = R.cpu().mean().item()
        print("AVG R", mean_R)

        writer.add_scalar('Reward', mean_R, epoch)


        if epoch % args.log_interval == 0 or epoch == (args.num_epochs-1):
            print("SAVE MODEL")
            if args.model_type=='hetnet':
                dir_name="./Models/HetNet"
                file_name="HetNet_C%d_V%d_D%d" %(args.coverage_num, args.visiting_num, args.pick_place_num)
                param_path = dir_name+"/"+file_name+".param"
                config_path = dir_name+"/"+file_name+".config"

                os.makedirs(dir_name, exist_ok=True)

                with open(config_path, 'w') as f:
                    json.dump(args.__dict__, f, indent=2)

            elif args.model_type=="ptrnet":
                dir_name="./Models/PtrNet"
                file_name="PtrNet_C%d_V%d_D%d" %(args.coverage_num, args.visiting_num, args.pick_place_num)
                param_path = dir_name+"/"+file_name+".param"
                config_path = dir_name+"/"+file_name+".config"

                os.makedirs(dir_name, exist_ok=True)

                with open(config_path, 'w') as f:
                    json.dump(args.__dict__, f, indent=2)

            else:
                print("define appropraite model type")
                raise
            torch.save(model.state_dict(), param_path)

        model.train()

    writer.close()
