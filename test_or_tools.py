import argparse
from numpy import cos, real

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from torch.utils.data import DataLoader

from Environment import Mission
from Environment.utils_heuristic import ORTOOLS_TYPE1, ORTOOLS_TYPE2
from Environment.utils import Checking_cost, Dae_Bup_Gwan

parser = argparse.ArgumentParser()

parser.add_argument("--model_type", type=str, default="att")
parser.add_argument("--coverage_num", type=int, default=6)
parser.add_argument("--visiting_num", type=int, default=6)
parser.add_argument("--pick_place_num", type=int, default=6)
parser.add_argument("--num_te_dataset", type=int, default=1)
args = parser.parse_args()


def test_single_sample():
  print("GENERATING Single Sample")
  test_dataset = Mission.MissionDataset(num_visit=args.visiting_num, num_coverage=args.coverage_num, num_pick_place=args.pick_place_num,
                                          num_samples=args.num_te_dataset,
                                          random_seed=20,
                                          overlap=True)

  heuristic_data_loader = DataLoader(
                                      test_dataset,
                                      batch_size = 1,
                                      shuffle=True)

  DISTANCE = 6.0
  SCALE = 1000
  VEHICLE_NUM = args.visiting_num+args.coverage_num+args.pick_place_num


  print("CALCULATING ORTOOLS")
  for idx, pointset in heuristic_data_loader:
    # print(pointset)
    type1_solution, type1_cost = ORTOOLS_TYPE1(pointset[0], dist_constraint=DISTANCE*SCALE, scale=SCALE)
    real_cost1, _ = Dae_Bup_Gwan(pointset[0], type1_solution)

    type2_solution, type2_cost = ORTOOLS_TYPE2(pointset[0], DISTANCE*SCALE, VEHICLE_NUM, SCALE)
    real_cost2, _ = Dae_Bup_Gwan(pointset[0], type2_solution)


  print("===TYPE1===")
  print(type1_solution)
  print(real_cost1)
  print("===TYPE2===")
  print(type2_solution)
  print(real_cost2)


if __name__ =="__main__":
  test_single_sample()

