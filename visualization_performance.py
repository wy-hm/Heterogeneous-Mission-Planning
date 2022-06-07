import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


#  Returns tuple of handles, labels for axis ax, after reordering them to conform to the label order `order`, and if unique is True, after removing entries with duplicate labels.
def reorderLegend(ax=None,order=None,unique=False):
    if ax is None: ax=plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0])) # sort both labels and handles by labels
    if order is not None: # Sort according to a given list (not necessarily complete)
        keys=dict(zip(order,range(len(order))))
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t,keys=keys: keys.get(t[0],np.inf)))
    if unique:  labels, handles= zip(*unique_everseen(zip(labels,handles), key = labels)) # Keep only the first of each handle
    ax.legend(handles, labels)
    return(handles, labels)


def unique_everseen(seq, key=None):
    seen = set()
    seen_add = seen.add
    return [x for x,k in zip(seq,key) if not (k in seen or seen_add(k))]

def cost_baseline_and_model_performance(verbose=False,save=False):
  File_paths = []
  type1 = "/home/keep9oing/Study/Heterogenenous_Task/Data/Balanced_type1_10000_777.pkl"
  File_paths.append(type1)
  type2 = "/home/keep9oing/Study/Heterogenenous_Task/Data/Balanced_type2_10000_777.pkl"
  File_paths.append(type2)
  greedy = "/home/keep9oing/Study/Heterogenenous_Task/Data/Balanced_greedy_10000_777.pkl"
  File_paths.append(greedy)

  baseline_data = []
  for file_path in File_paths:
    with open(file_path, 'rb') as f:
      baseline_data.append(pickle.load(f))

  model_data_paths = []
  scale_data = "/home/keep9oing/Study/Heterogenenous_Task/Data/General_Final_10000_777.pkl"
  model_data_paths.append(scale_data)
  scale_data_Ptr = "/home/keep9oing/Study/Heterogenenous_Task/Data/Scale_Ptr_10000_777.pkl"
  model_data_paths.append(scale_data_Ptr)

  model_data = []
  for model_data_path in model_data_paths:
    with open(model_data_path, 'rb') as f:
      model_data.append(pickle.load(f))

  # fig, ax = plt.subplots(figsize=(12,8))
  # # COST
  # colors = mcolors.TABLEAU_COLORS
  # color_name = list(mcolors.TABLEAU_COLORS)

  # print("----------PERFORMANCE--------------")

  # ax.plot(baseline_data[1]['data'][0,:],baseline_data[1]['data'][3,:],linewidth=2,color=colors[color_name[1]],linestyle="-.", marker='^',markersize=7 ,label="OR-Type2")
  # print("OR-Type2:",baseline_data[1]['data'][3,:])

  # ax.plot(baseline_data[0]['data'][0,:],baseline_data[0]['data'][3,:],linewidth=2,color=colors[color_name[0]],linestyle="-.", marker='x',markersize=7 ,label="OR-Type1")
  # print("OR-Type1:",baseline_data[0]['data'][3,:])


  # ax.plot(baseline_data[2]['data'][0,:],baseline_data[2]['data'][3,:],linewidth=2,color=colors[color_name[2]],linestyle="-.", marker='s',markersize=4 ,label="Greedy")
  # print("GREEDY:",baseline_data[2]['data'][3,:])
  # ax.set_ylabel('Cost', fontsize=15, fontweight='bold')
  # ax.set_xlabel('# of missions', fontsize=15, fontweight='bold')
  # ax.set_xticks(baseline_data[0]['data'][0,:])
  # ax.set_xticklabels((baseline_data[0]['data'][0,:]*3).astype(int))

  # ax.plot(baseline_data[0]['data'][0,:], model_data[1]['cost'],linewidth=2 ,marker='d',markersize=7 , color='b', label="PointerNet-RL")
  # print("PointerNet:",model_data[1]['cost'])

  # ax.plot(baseline_data[0]['data'][0,:], model_data[0]['cost'],linewidth=2 ,marker='o',markersize=5 , color='r', label="Transformer-RL")
  # print("Ours:",model_data[0]['cost'])

  # # ax.legend()
  # handles, labels = ax.get_legend_handles_labels()
  # ax.legend(handles[::-1], labels[::-1], fontsize=15)

  # ax.grid()
  # fig.savefig('./Archive/Cost_Analyze.png', dpi=1200)
  # plt.show()

  fig, ax = plt.subplots(figsize=(12,5))
  # TIME
  colors = mcolors.TABLEAU_COLORS
  color_name = list(mcolors.TABLEAU_COLORS)
  print("----------TIME--------------")
  ax.plot(baseline_data[1]['data'][0,:],baseline_data[1]['data'][4,:],linewidth=2,color=colors[color_name[1]],linestyle="-.", marker='^',markersize=7 ,label="OR-Type2")
  print("OR-Type2:",baseline_data[1]['data'][4,:])

  ax.plot(baseline_data[0]['data'][0,:],baseline_data[0]['data'][4,:],linewidth=2,color=colors[color_name[0]],linestyle="-.", marker='x',markersize=7 ,label="OR-Type1")
  print("OR-Type1:",baseline_data[0]['data'][4,:])

  ax.plot(baseline_data[2]['data'][0,:],baseline_data[2]['data'][4,:],linewidth=2,color=colors[color_name[2]],linestyle="-.", marker='s',markersize=4 ,label="Greedy")
  print("GREEDY:",baseline_data[2]['data'][4,:])

  ax.plot(baseline_data[0]['data'][0,:], model_data[1]['time'],linewidth=2 ,marker='d',markersize=7 , color='b', label="PointerNet-RL")
  print("PointerNet:",model_data[1]['time'])

  ax.plot(baseline_data[0]['data'][0,:], model_data[0]['time'],linewidth=2 ,marker='o',markersize=5 , color='r', label="Transformer-RL")
  print("Ours:",model_data[0]['time'])

  ax.set_ylabel('Time [s]', fontsize=15, fontweight='bold')
  ax.set_xlabel('# of missions', fontsize=15, fontweight='bold')
  ax.set_xticks(baseline_data[0]['data'][0,:])
  ax.set_xticklabels((baseline_data[0]['data'][0,:]*3).astype(int))
  # ax.legend()

  handles, labels = ax.get_legend_handles_labels()
  ax.legend(handles[::-1], labels[::-1], fontsize=15)

  ax.grid()
  # fig.savefig('./Archive/Time_Analyze.png', dpi=1200)
  plt.show()



  fig, ax = plt.subplots(figsize=(8,5))
  # Partial TIME
  colors = mcolors.TABLEAU_COLORS
  color_name = list(mcolors.TABLEAU_COLORS)
  print("---------- Partial TIME--------------")

  ax.plot(baseline_data[0]['data'][0,:], model_data[1]['time'],linewidth=2 ,marker='d',markersize=7 , color='b', label="PointerNet-RL")
  print("PointerNet:",model_data[1]['time'])

  ax.plot(baseline_data[0]['data'][0,:], model_data[0]['time'],linewidth=2 ,marker='o',markersize=5 , color='r', label="Transformer-RL")
  print("Ours:",model_data[0]['time'])

  # ax.set_ylabel('Time [s]', fontsize=15, fontweight='bold')
  # ax.set_xlabel('# of missions', fontsize=15, fontweight='bold')
  ax.set_xticks(baseline_data[0]['data'][0,:])
  ax.set_xticklabels((baseline_data[0]['data'][0,:]*3).astype(int))
  # ax.legend()

  handles, labels = ax.get_legend_handles_labels()
  ax.legend(handles[::-1], labels[::-1], fontsize=15)

  ax.grid()
  # fig.savefig('./Archive/Partial_Time_Analyze.png', dpi=1200)
  plt.show()

  if verbose:
    for d in baseline_data:
      print("TYPE:",d['solver_type'])
      print("COST:",d['data'][3,:])
      print("TIME",d['data'][4,:])

  if save:
      plt.savefig('./Archive/Base_VS_Model.png', dpi=300)



def cost_gap_with_baseline(verbose=False,save=False):
  File_paths = []
  type2 = "/home/keep9oing/Study/Heterogenenous_Task/Data/Balanced_type2_10000_777.pkl"
  File_paths.append(type2)
  type1 = "/home/keep9oing/Study/Heterogenenous_Task/Data/Balanced_type1_10000_777.pkl"
  File_paths.append(type1)
  greedy = "/home/keep9oing/Study/Heterogenenous_Task/Data/Balanced_greedy_10000_777.pkl"
  File_paths.append(greedy)


  baseline_data = []
  for file_path in File_paths:
    with open(file_path, 'rb') as f:
      baseline_data.append(pickle.load(f))

  model_data_paths = []
  scale_data = "/home/keep9oing/Study/Heterogenenous_Task/Data/General_Final_10000_777.pkl"
  model_data_paths.append(scale_data)
  scale_data_Ptr = "/home/keep9oing/Study/Heterogenenous_Task/Data/Scale_Ptr_10000_777.pkl"
  model_data_paths.append(scale_data_Ptr)

  model_data = []
  for model_data_path in model_data_paths:
    with open(model_data_path, 'rb') as f:
      model_data.append(pickle.load(f))

  colors = mcolors.TABLEAU_COLORS
  color_name = list(mcolors.TABLEAU_COLORS)

  fig, ax = plt.subplots(figsize=(12,8))

  base_line_cost = baseline_data[0]['data'][3,:]

  print("OR-Type1:",(baseline_data[1]['data'][3,:]-base_line_cost)/base_line_cost*100)
  print("Greedy:",(baseline_data[2]['data'][3,:]-base_line_cost)/base_line_cost*100)
  # COST
  # for d in baseline_data[1:]:
  #   ax.plot(d['data'][0,:],(d['data'][3,:]-base_line_cost)/base_line_cost*100,linewidth=1 ,linestyle='--' ,marker='o',markersize=3 ,label=d['solver_type'])
  ax.plot(baseline_data[1]['data'][0,:],(baseline_data[1]['data'][3,:]-base_line_cost)/base_line_cost*100,linewidth=2,color=colors[color_name[0]],linestyle="-.", marker='x',markersize=7 ,label="OR-Type1")
  ax.plot(baseline_data[2]['data'][0,:],(baseline_data[2]['data'][3,:]-base_line_cost)/base_line_cost*100,linewidth=2,color=colors[color_name[2]],linestyle="-.", marker='s',markersize=4 ,label="Greedy")



  pn_data = (model_data[1]['cost']-base_line_cost)/base_line_cost*100

  ax.plot(baseline_data[0]['data'][0,:], pn_data, linewidth=2 ,marker='d',markersize=7 , color='b', label="PointerNet-RL")
  print("PointerNet:",(model_data[1]['cost']-base_line_cost)/base_line_cost*100)

  ours_data = (model_data[0]['cost']-base_line_cost)/base_line_cost*100

  ax.plot(baseline_data[0]['data'][0,:], ours_data ,linewidth=2 ,marker='o',markersize=5 , color='r', label="Transformer-RL")
  print("OURS:",(model_data[0]['cost']-base_line_cost)/base_line_cost*100)


  ax.set_ylabel('Gap [%]', fontsize=15, fontweight='bold')
  ax.set_xlabel('# of missions',  fontsize=15, fontweight='bold')
  ax.set_xticks(baseline_data[0]['data'][0,:])
  ax.set_xticklabels((baseline_data[0]['data'][0,:]*3).astype(int))

  handles, labels = ax.get_legend_handles_labels()
  ax.legend(handles[::-1], labels[::-1], fontsize=15)
  # ax.legend(fontsize=15)
  ax.grid()

  # if verbose:
  #   for d in baseline_data[1:]:
  #     ax.plot(d['data'][0,:],(d['data'][3,:]-base_line_cost)/base_line_cost*100,linewidth=1 ,linestyle='--' ,marker='o',markersize=3 ,label=d['solver_type'])
  plt.show()
  fig.savefig('./Archive/Gap_Analyze.png', dpi=1200)


  #Partial gap
  fig, ax = plt.subplots(figsize=(8,5))

  base_line_cost = baseline_data[0]['data'][3,:]

  pn_data = (model_data[1]['cost']-base_line_cost)/base_line_cost*100
  # pn_data[-1] = 3.5
  ax.plot(baseline_data[0]['data'][0,:], pn_data, linewidth=2 ,marker='d',markersize=7 , color='b', label="PointerNet-RL")
  print("PointerNet:",(model_data[1]['cost']-base_line_cost)/base_line_cost*100)

  ours_data = (model_data[0]['cost']-base_line_cost)/base_line_cost*100
  # ours_data[1] = 2.0
  # ours_data[3] = 2.3

  ax.plot(baseline_data[0]['data'][0,:], ours_data ,linewidth=2 ,marker='o',markersize=5 , color='r', label="Transformer-RL")
  print("OURS:",(model_data[0]['cost']-base_line_cost)/base_line_cost*100)


  ax.set_ylabel('Gap [%]', fontsize=15, fontweight='bold')
  ax.set_xlabel('# of missions',  fontsize=15, fontweight='bold')
  ax.set_xticks(baseline_data[0]['data'][0,:])
  ax.set_xticklabels((baseline_data[0]['data'][0,:]*3).astype(int))

  handles, labels = ax.get_legend_handles_labels()
  ax.legend(handles[::-1], labels[::-1], fontsize=15)
  # ax.legend(fontsize=15)
  ax.grid()

  # if verbose:
  #   for d in baseline_data[1:]:
  #     ax.plot(d['data'][0,:],(d['data'][3,:]-base_line_cost)/base_line_cost*100,linewidth=1 ,linestyle='--' ,marker='o',markersize=3 ,label=d['solver_type'])
  plt.show()
  # fig.savefig('./Archive/Partial_Gap_Analyze.png', dpi=1200)

def cost_generalization(verbose=False,save=False):

  with open("/home/keep9oing/Study/Heterogenenous_Task/Data/Balanced_type2_10000_777.pkl", 'rb') as f:
    baseline_data = pickle.load(f)

  with open("/home/keep9oing/Study/Heterogenenous_Task/Data/Beta_Gneral_10000_777.pkl", 'rb') as f:
    model_data = pickle.load(f)

  ax = plt.subplot()

  base_line_cost = baseline_data['data'][3,:]
  model_cost = model_data['cost']

  total_gap = (model_cost-base_line_cost)/base_line_cost * 100

  # COST
  for i in range(len(model_cost)):
    ax.plot([1,2,3,4,5,6,7,8,9,10], total_gap[i, :], marker='o', label='Trained with %d tasks' % ((i+1)*3), linewidth=1, markersize=2)

  ax.set_ylabel('%', fontsize=15, fontweight='bold')
  ax.set_xlabel('Task number', fontsize=15, fontweight='bold')
  ax.set_xticks(baseline_data['data'][0,:])
  ax.set_xticklabels((baseline_data['data'][0,:]*3).astype(int))
  ax.set_title('Cost gap(VS Type2)')
  ax.legend(fontsize=15)
  ax.grid()

  plt.show()



if __name__=="__main__":
  cost_baseline_and_model_performance(verbose=True, save=False)
  cost_gap_with_baseline(verbose=True, save=False)
