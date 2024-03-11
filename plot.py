import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

import torch
import torch.nn as nn
import airfrans as af


def plot2d(dataset_list, dataset_name, net, device):

  X_test_list = []
  # y_test_list = []

  # for i in range(len(dataset_list)):
  for i in range(1):
      simulation_name_splitted = dataset_name[-1].split("_")
      params = [float(j) for j in simulation_name_splitted[-3:]]

      naca = af.naca_generator.naca_generator(params, 100)
      naca = naca[:-1,:].flatten()
      invel=dataset_list[i][0, 2:4]
      naca = np.concatenate((naca, invel))

      #Velocity 7 and 8, Pressure is 9, Turbulent Kinematic Viscosity is 10
      target = 7

      y_position = dataset_list[-1][:, 0:2]
      y_target = dataset_list[-1][:, target]

      for j in range(y_position.shape[0]):
          X_test_list.append(np.concatenate((naca, y_position[j].squeeze())))

      #y_test_list.append(y_target)

  X_test_list = np.array(X_test_list)
  #y_test_list = np.array(y_test_list)
  print(X_test_list.shape)
  X_test = torch.Tensor(X_test_list).to(device)
  net.eval()
  y_pred = net(X_test).cpu().detach().numpy()

  x1_position_bulk = dataset_list[-1][:,0]
  x2_position_bulk = dataset_list[-1][:,1]

  fig, ax = plt.subplots(1, 2, figsize = (36, 12))
  sc0 = ax[0].scatter(x1_position_bulk, x2_position_bulk, c = y_pred[:, 0], s = 0.75)
  ax[0].title.set_text('Predicted Pressure')

  sc1 = ax[1].scatter(x1_position_bulk, x2_position_bulk, c = dataset_list[-1][:,9], s = 0.75)
  ax[1].title.set_text('True Pressure')
  fig.colorbar(sc1)
  plt.show()

  fig, ax = plt.subplots(1, 2, figsize = (36, 12))
  sc2 = ax[0].scatter(x1_position_bulk, x2_position_bulk, c = y_pred[:, 1], s = 0.75)
  ax[0].title.set_text('Predicted Velocity along x')

  sc3 = ax[1].scatter(x1_position_bulk, x2_position_bulk, c = dataset_list[-1][:,7], s = 0.75)
  ax[1].title.set_text('True Velocity along x')
  fig.colorbar(sc3)
  plt.show()

  fig, ax = plt.subplots(1, 2, figsize = (36, 12))
  sc4 = ax[0].scatter(x1_position_bulk, x2_position_bulk, c = y_pred[:, 2], s = 0.75)
  ax[0].title.set_text('Predicted Velocity along y')

  sc5 = ax[1].scatter(x1_position_bulk, x2_position_bulk, c = dataset_list[-1][:,8], s = 0.75)
  ax[1].title.set_text('True Velocity along y')
  fig.colorbar(sc5)
  plt.show()

  fig, ax = plt.subplots(1, 2, figsize = (36, 12))
  sc6 = ax[0].scatter(x1_position_bulk, x2_position_bulk, c = y_pred[:, 3], s = 0.75)
  ax[0].title.set_text('Predicted Kinematic turbulent viscosity')

  sc7 = ax[1].scatter(x1_position_bulk, x2_position_bulk, c = dataset_list[-1][:,10], s = 0.75)
  ax[1].title.set_text('True Kinematic turbulent viscosity')
  fig.colorbar(sc7)
  plt.show()