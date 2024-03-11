import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

import airfrans as af


def preprocess_data_airfrans(root_dir="/content/drive/MyDrive/AirfRANS/Dataset", task="scarce", train=True, download_data=False):
  X_train_list = []
  y_train_list = []
  X_test_list = []
  y_test_list = []
  boundary_line_ratio = 0.1

  if(download_data==True):
    directory_name='Dataset'
    if not os.path.isdir(directory_name):
        af.dataset.download(root = root_dir, file_name = directory_name, unzip = True, OpenFOAM = False)

  try:
    dataset_list, dataset_name = af.dataset.load(root = root_dir,  task = task, train = train)
  except Exception as err:
    print(f"Unexpected {err=}, {type(err)=}")
    raise

  # for i in range(len(dataset_list)):
  for i in tqdm(range(100)):
    simulation_name_splitted = dataset_name[i].split("_")
    params = [float(j) for j in simulation_name_splitted[-3:]]

    naca = af.naca_generator.naca_generator(params, 100)
    naca = naca[:-1,:].flatten()
    invel=dataset_list[i][0, 2:4]
    naca = np.concatenate((naca, invel))

    minimum_x = np.min(dataset_list[i][:,0])
    maximum_x = np.max(dataset_list[i][:,0])
    boundary_line = boundary_line_ratio*maximum_x + (1-boundary_line_ratio)*minimum_x

    for j in range(200):
      dataset_positions = dataset_list[i][:,0:2]

      y_idx = np.random.choice(np.where(dataset_positions[:, 0]<boundary_line)[0], size=1)

      y_position = dataset_list[i][y_idx, 0:2]
      y_pressure = dataset_list[i][y_idx, 9]
      y_velocityx = dataset_list[i][y_idx, 7]
      y_velocityy = dataset_list[i][y_idx, 8]
      y_viscosity = dataset_list[i][y_idx, 10]

      X_train_list.append(np.concatenate((naca, y_position.squeeze())))

      y_train_list.append(np.array([y_pressure, y_velocityx, y_velocityy, y_viscosity]))

    for j in range(800):
      dataset_positions = dataset_list[i][:,0:2]

      y_idx = np.random.choice(np.where(dataset_positions[:, 0]>=boundary_line)[0], size=1)

      y_position = dataset_list[i][y_idx, 0:2]
      y_pressure = dataset_list[i][y_idx, 9]
      y_velocityx = dataset_list[i][y_idx, 7]
      y_velocityy = dataset_list[i][y_idx, 8]
      y_viscosity = dataset_list[i][y_idx, 10]

      X_train_list.append(np.concatenate((naca, y_position.squeeze())))

      y_train_list.append(np.array([y_pressure, y_velocityx, y_velocityy, y_viscosity]))

  X_train_list = np.array(X_train_list)
  y_train_list = np.array(y_train_list).squeeze()

  print(f"Shape of X_train_list : {X_train_list.shape} ; y_train_list : {y_train_list.shape}")

  return X_train_list,y_train_list,X_test_list,y_test_list