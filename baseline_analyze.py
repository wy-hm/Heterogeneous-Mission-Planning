import numpy as np
import multiprocessing
import os
import time
import pickle
from statistics import stdev
import random

from torch.utils.data import DataLoader

from Environment import Mission
from Environment.utils_heuristic import GREEDY, ORTOOLS_TYPE1, ORTOOLS_TYPE2


class ORTOOL_Analyzer():
  def __init__(self, cpu_num:int = None, solver_type:list = None) -> None:

    assert isinstance(solver_type, list), "Solver type should be set with list"
    assert len(solver_type) > 0, "Empty solver type"

    if cpu_num is None:
      self.cpu_num = max(os.cpu_count() - 2, 1)
    else:
      self.cpu_num = cpu_num

    self.solver_type = solver_type
    self.file_list = []

  def log_performance(self, test_case:list = None,
                            sample_num:int = None,
                            task_type:str='Balanced',
                            seed:int = 777) -> None:

    print("===LOGGING===")

    np.random.seed(seed)
    random.seed(seed)

    data = self.compute_performance(test_case, sample_num, task_type, seed)

    for solver in self.solver_type:
      contents = {}
      contents['solver_type'] = solver
      contents['task_type']   = task_type
      contents['sample_num']  = sample_num
      contents['data']        = data[solver]

      file = task_type+"_"+solver+"_"+str(sample_num)+"_"+str(seed)
      file = 'Data/'+file+'.pkl'
      self.file_list.append(file)

      with open(file, 'wb') as f:
        pickle.dump(contents, f)


  def plot_results(self) -> None:
    pass

  def compute_performance(self, test_case:list = None,
                                sample_num:int = None,
                                task_type:str='Balanced',
                                seed:int = 777) -> dict:
    """
      [Return]
        {type_n: 6 X len(test_case)
                 |Visiting
                 |Coverage
                 |Deliver
                 |Cost
                 |Time
                 |Time_std
    """

    print("===COMPUTING===")

    assert isinstance(test_case, list), "test case should be set"
    assert sample_num is not None, "sample num should be set"

    if task_type=='Balanced':
      print("+++BALANCED TYPE+++")

      results = {k:np.zeros((6, len(test_case))) for k in self.solver_type}
      DISTCONST = [6000.0] * sample_num
      SCALE = [1000.0] * sample_num
      TIME_VERBOSE = [True] * sample_num

      for idx, case in enumerate(test_case):

        print("---Generating {}x3 Missions---".format(case))
        mission_dataset = Mission.MissionDataset(num_visit=case, num_coverage=case, num_pick_place=case,
                                                num_samples=sample_num,
                                                random_seed=seed)

        data_loader = DataLoader(mission_dataset,
                                 batch_size=sample_num)

        if 'type1' in self.solver_type:
          print("TYPE1 solving...")
          pool = multiprocessing.Pool(processes=self.cpu_num)

          start = time.time()
          for i, mission in data_loader:
            args = zip(mission, DISTCONST, SCALE, TIME_VERBOSE)
            result = list(pool.starmap(ORTOOLS_TYPE1, args))
            _, cost, times = zip(*result)
          end = time.time()
          pool.close()
          pool.join()
          print("TYPE1 Done...")

          time_elapsed = (end-start)
          mean_cost = sum(cost)/len(cost)

          results['type1'][:3,idx] = case
          results['type1'][3,idx]  = mean_cost
          results['type1'][4,idx]  = time_elapsed
          results['type1'][5,idx]  = stdev(times)

        # MULTIPLE arguments
        if 'type2' in self.solver_type:
          print("TYPE2 solving...")

          VEH_NUM = [case]*sample_num
          pool = multiprocessing.Pool(processes=self.cpu_num)
          start = time.time()
          for i, mission in data_loader:
            args = zip(mission, DISTCONST, VEH_NUM ,SCALE, TIME_VERBOSE)
            result = list(pool.starmap(ORTOOLS_TYPE2, args))
            _, cost, times = zip(*result)
          end = time.time()
          pool.close()
          pool.join()
          print("TYPE2 Done...")

          time_elapsed = (end-start)
          mean_cost = sum(cost)/len(cost)

          results['type2'][:3,idx] = case
          results['type2'][3,idx]  = mean_cost
          results['type2'][4,idx]  = time_elapsed
          results['type2'][5,idx]  = stdev(times)

        if 'greedy' in self.solver_type:
          print("GREEDY solving...")

          VEH_NUM = [case]*sample_num
          pool = multiprocessing.Pool(processes=self.cpu_num)
          start = time.time()
          for i, mission in data_loader:
            args = zip(mission, DISTCONST, SCALE, TIME_VERBOSE)
            result = list(pool.starmap(GREEDY, args))
            _, cost, times = zip(*result)
          end = time.time()
          pool.close()
          pool.join()
          print("GREEDY Done...")

          time_elapsed = (end-start)
          mean_cost = sum(cost)/len(cost)

          results['greedy'][:3,idx] = case
          results['greedy'][3,idx]  = mean_cost
          results['greedy'][4,idx]  = time_elapsed
          results['greedy'][5,idx]  = stdev(times)

        del mission_dataset

      return results
    else:
      raise ValueError("task_type is not in the choice, please check")


if __name__ =="__main__":




  or_analyzer = ORTOOL_Analyzer(1, ['type1','type2','greedy'])
  or_analyzer.log_performance([1,2,3,4,5,6,7,8,9,10], 10000)

  print("-------DATA LOADING----------")
  for path in or_analyzer.file_list:
    with open(path, 'rb') as f:
      data = pickle.load(f)
      print(data)


