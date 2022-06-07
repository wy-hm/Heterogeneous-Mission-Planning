import argparse
import json

import torch
from rl_mission_planning import HetNet_tSNE

from solver import HetNet_Partial_tSNE_solver, HetNet_tSNE_solver, solver_RNN, HetNet_solver, HetNet_Partial_solver

def load_model_eval(param_path, config_path=None, tSNE=False, ablation=False, partial_type=None):
    if config_path is None:
        config_path = param_path[:-6]+'.config'

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open(config_path, 'r') as f:
        args.__dict__ = json.load(f)

    print("LOADING MODEL================")
    print(args)

    if args.model_type == "ptrnet":
            print("Pointer network is used")
            model = solver_RNN(
                args.embedding_size,
                args.hidden_size,
                args.visiting_num+args.coverage_num+args.pick_place_num+1,
                2, 10)
    # AM network
    elif args.model_type.startswith("hetnet"):
        print("HetNet is used")

        if ablation:
            if partial_type is None:
                raise Exception("Please check partial_type")

            if partial_type == 'notype' or partial_type=='nogeo':
                input_size = 5
            elif partial_type == 'nothing':
                input_size = 2
            else:
                raise Exception("Please check partial_type")

            if tSNE:
                print("t-SNE analyzing mode(partial)")
                model = HetNet_Partial_tSNE_solver(partial_type,
                                                   input_size,
                                                   args.embedding_size,
                                                   args.hidden_size,
                                                   2,
                                                   10)

            else:
                print("PARTIAL MODE")
                model = HetNet_Partial_solver(partial_type,
                                              input_size,
                                              args.embedding_size,
                                              args.hidden_size,
                                              2,
                                              10)
        else:
            if tSNE:
                print("t-SNE analyzing mode")
                model = HetNet_tSNE_solver(args.embedding_size,args.hidden_size,2,10)
            else:
                print("NORMAL MODE")
                model = HetNet_solver(
                    args.embedding_size,
                    args.hidden_size,
                    # args.visiting_num+args.coverage_num+args.pick_place_num+1,
                    2, 10)
    else:
        raise

    model.load_state_dict(torch.load(param_path))
    model.eval()

    print("FINSHED LOADING===========")

    return model
