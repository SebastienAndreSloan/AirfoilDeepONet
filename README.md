# AirfoilDeepONet

A DeepONet designed to model 2-D incompressible Reynolds-Averaged Navier Stokes flow across airfoils

## 0. Prerequisites


## 1. Preprocess the dataset and training the PI-DON

```
python3 ./train.py \
--root_dir <directory where the AirfRANS Dataset is stored> \
--save_dir <directory where the models and losses will be stored>

argument details :
-- root_dir: directory where the AirfRANS Dataset is stored 
-- save_dir: directory where the models and losses will be stored

example command:
python3 ./train.py \
--root_dir /content/drive/MyDrive/AirfRANS/Dataset \
--save_dir /content/drive/MyDrive/AirfRANS/saved_models
```

## 2. Plotting

## 3. TO DO :
- [ ] Enhance . . . 