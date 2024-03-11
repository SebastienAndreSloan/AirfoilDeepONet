import numpy as np
import tqdm

import torch
import torch.nn as nn

from preprocess_data import dataset_creation
from models import DeepONet

if __name__=="__main__":

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  m=402
  dim_x=2

  iterations = 10000
  batch_size = None
  print_every = iterations/1000
  learning_rate = 2e-4

  X_train_list,y_train_list,X_test_list,y_test_list = dataset_creation.preprocess_data()

  X_train = torch.Tensor(X_train_list).to(device)
  y_train = torch.Tensor(y_train_list).to(device)
  X_test = torch.Tensor(X_test_list).to(device)
  y_test = torch.Tensor(y_test_list).to(device)

  net = DeepONet(branch_dim=m, trunk_dim=dim_x , branch_depth=6, trunk_depth=6, width=100, k=4, device=device).to(device)

  loss_fn = torch.nn.MSELoss().to(device)
  optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

  print('Training...', flush=True)
  loss_history = []

  for i in tqdm(range(iterations)):
      if batch_size is not None:
          mask = np.random.choice(X_train.size(0), batch_size, replace=False)
          loss = loss_fn(net(X_train[mask]), y_train[mask])
      else:
          loss = net.loss(X_train, y_train)
      loss_history.append([i, loss.item()])
      if i % print_every == 0 or i == iterations:
          loss_history.append([i, loss.item()])
          tqdm.write('{:<9}Train loss: {:<25}'.format(i, loss.item()))
      # if i % print_every == 0 or i == iterations:
          # loss_test = net.loss(X_test, y_test)
          # loss_history.append([i, loss.item(), loss_test.item()])
          # print('{:<9}Train loss: {:<25}Test loss: {:<25}'.format(i, loss.item(), loss_test.item()), flush=True)
          if torch.any(torch.isnan(loss)):
              encounter_nan = True
              print('Encountering nan, stop training', flush=True)
      if i < iterations:
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

  torch.save(net.state_dict(), f'/content/drive/MyDrive/AirfRANS/saved_models/model_branch_{m}_trunk_{dim_x}_iter_{iterations}_lr_{learning_rate}')
  np.save(f'/content/drive/MyDrive/AirfRANS/saved_models/loss_history_branch_{m}_trunk_{dim_x}_iter_{iterations}_lr_{learning_rate}', np.array(loss_history))

  print('Done!', flush=True)