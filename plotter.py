import matplotlib.pyplot as plt

import numpy as np
import math

def contact_point(point, center, radius):
  assert isinstance(point, tuple), "point must be tuple"
  assert isinstance(center, tuple), "center must be tuple"

  point = np.array(point)
  center = np.array(center)

  d = np.linalg.norm(point-center)

  if d <= radius:
    vector = (center-point)/d
    heading = - (radius-d) * vector
    contact_point = point + heading

  else:
    h = abs(center[1]-point[1])
    l = (d**2 - radius**2)**0.5
    theta = math.asin(h/d)
    beta = math.acos(l/d)
    alpha = theta-beta

    dx = l * math.cos(alpha)
    dy = l * math.sin(alpha)

    x_sign = 1 if center[0] > point[0] else -1
    y_sign = 1 if center[1] > point[1] else -1

    contact_point = point + np.array((x_sign * dx, y_sign * dy))

  return contact_point


###########################################################
# Plot
###########################################################

def plot_result(mission, solution, args):

  assert isinstance(mission, np.ndarray), "The mission must be numpy array"
  assert isinstance(solution, np.ndarray), "The solution must be numpy array"

  Depot = mission[:1, :]
  Area = mission[1:args.coverage_num+1, :]
  Visit = mission[args.coverage_num+1:, :]
  solution = solution


  prev = Depot[0]

  fit, ax = plt.subplots(1,2)
  ax[0].set_title("%d visiting, %d coverage, %d pick_place" % (args.visiting_num, args.coverage_num, args.pick_place_num))
  ax[1].set_title("Solution")

  ax[0].scatter(prev[:1], prev[1:2],marker='s', c='k', s=30, label='Depot')
  ax[1].scatter(prev[:1], prev[1:2],marker='s', c='k', s=30, label='Depot')

  theta = np.radians(np.linspace(0,360*5,1000))
  for i in solution[1:]:
    task = mission[i]

    if task[-2] == 1:
      point = task[:2]

      ax[0].scatter(point[:1], point[1:2],marker='s', color='b', s=10, label='Visiting')
      ax[1].scatter(point[:1], point[1:2],marker='s', color='b', s=10, label='Visiting')
      ax[1].plot([prev[0], point[0]], [prev[1], point[1]], 'r-', linewidth=0.5)

      prev = point
    elif task[-1] == 1:
      x, y, r = task[:3]

      ax[0].add_patch(plt.Circle((x, y), r, fill=False))
      ax[1].add_patch(plt.Circle((x, y), r, fill=False))

      spiral_r = theta / 31 * r
      spiral_x = spiral_r*np.cos(theta)+x
      spiral_y = spiral_r*np.sin(theta)+y
      ax[1].plot(spiral_x, spiral_y,'r-', linewidth=0.5)

      contact = contact_point((prev[0], prev[1]),(x,y),r)
      ax[1].plot([prev[0], contact[0]], [prev[1], contact[1]], 'r-', linewidth=0.5)

      prev = np.array([x, y])

    elif task[-3] == 1:
      pick_point = task[:2]
      place_point = task[3:5]
      points = np.concatenate((pick_point[None,:], place_point[None,:]), axis=0)

      ax[1].plot([prev[0], pick_point[0]], [prev[1], pick_point[1]], 'r-', linewidth=0.5)

      ax[0].scatter(points[:,0], points[:,1], marker='D', color='m', s=20)
      ax[1].scatter(points[:,0], points[:,1], marker='D', color='m', s=20)

      ax[0].arrow(pick_point[0], pick_point[1], 0.8*(place_point[0]-pick_point[0]), 0.8*(place_point[1]-pick_point[1]), width=0.002, color='c', head_width=0.012)
      ax[1].arrow(pick_point[0], pick_point[1], 0.8*(place_point[0]-pick_point[0]), 0.8*(place_point[1]-pick_point[1]), width=0.002, color='c', head_width=0.012)

      prev = place_point

  ax[1].plot([prev[0],0],[prev[1],0],'k--',linewidth=0.5, label='Last path')

  ax[0].set_xlim((-0.05, 1.05))
  ax[0].set_ylim((-0.05, 1.05))
  ax[0].set_aspect('equal')

  ax[1].set_xlim((-0.05, 1.05))
  ax[1].set_ylim((-0.05, 1.05))
  ax[1].set_aspect('equal')

  plt.show()
