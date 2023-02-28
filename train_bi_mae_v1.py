from models import mae_vit_base_patch16
from utils import dataset_division

import os
import glob
import copy
import time
import torch
import numpy as np
import nibabel as nib
import torch.nn as nn
import torch.nn.functional as F

# ==================== model select ====================

model_list = [
    ["MR_PET_mae", [7],],
]

print("Model index: ", end="")
current_model_idx = int(input()) - 1
print(model_list[current_model_idx])
time.sleep(1)

# ==================== basic ====================

train_dict = {}
train_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
train_dict["project_name"] = model_list[current_model_idx][0]
train_dict["gpu_ids"] = model_list[current_model_idx][1]

train_dict["dropout"] = 0.
train_dict["loss_term"] = "SmoothL1Loss"
train_dict["optimizer"] = "AdamW"

train_dict["save_folder"] = "./project_dir/"+train_dict["project_name"]+"/"
train_dict["seed"] = 426
train_dict["input_size"] = [256, 256]
train_dict["epochs"] = 200
train_dict["batch"] = 16
train_dict["PET_norm_factor"] = 4000
train_dict["target_model"] = "./pre_train/mae_pretrain_vit_base.pth"

train_dict["model_term"] = "two-branch mae"
train_dict["continue_training_epoch"] = 0
train_dict["flip"] = False

train_dict["folder_MR"] = "./data/MR/"
train_dict["folder_PET"] = "./data/PET/"
train_dict["val_ratio"] = 0.3
train_dict["test_ratio"] = 0.2
train_dict["train_ratio"] = 1 - train_dict["val_ratio"] - train_dict["test_ratio"]

train_dict["opt_lr"] = 1e-3 # default
train_dict["opt_betas"] = (0.9, 0.999) # default
train_dict["opt_eps"] = 1e-8 # default
train_dict["opt_weight_decay"] = 0.01 # default
train_dict["amsgrad"] = False # default

for path in [train_dict["save_folder"], train_dict["save_folder"]+"npy/", train_dict["save_folder"]+"loss/"]:
    if not os.path.exists(path):
        os.mkdir(path)

np.save(train_dict["save_folder"]+"setting.npy", train_dict)

# ==================== GPU ====================

np.random.seed(train_dict["seed"])
gpu_list = ','.join(str(x) for x in train_dict["gpu_ids"])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== model ====================

model = mae_vit_base_patch16()
pre_train_dir = train_dict["target_model"]
ckpt = torch.load(pre_train_dir, map_location='cpu')["model"]
model.load_state_dict(ckpt, strict=False)

model.train()
model = model.to(device)

# ==================== optimizer ====================

optim = torch.optim.AdamW(
    model.parameters(),
    lr = train_dict["opt_lr"],
    betas = train_dict["opt_betas"],
    eps = train_dict["opt_eps"],
    weight_decay = train_dict["opt_weight_decay"],
    amsgrad = train_dict["amsgrad"]
    )

# ==================== data division ====================

MR_list = glob.glob(train_dict["folder_MR"]+"*.nii.gz")
PET_list = glob.glob(train_dict["folder_PET"]+"*.nii.gz")

MR_train_list, MR_val_list, MR_test_list = dataset_division(
    MR_list, 
    train_dict["train_ratio"], 
    train_dict["val_ratio"], 
    train_dict["test_ratio"],
)

PET_train_list, PET_val_list, PET_test_list = dataset_division(
    PET_list, 
    train_dict["train_ratio"], 
    train_dict["val_ratio"], 
    train_dict["test_ratio"],
)
data_division_dict = {
    "MR_train_list": MR_train_list,
    "MR_val_list": MR_val_list,
    "MR_test_list": MR_test_list,
    "PET_train_list": PET_train_list,
    "PET_val_list": PET_val_list,
    "PET_test_list": PET_test_list,
}
np.save(train_dict["save_folder"]+"data_division.npy", data_division_dict)

# ==================== training ====================

best_val_loss = 1e3
best_epoch = 0

package_train = [[MR_train_list, PET_train_list], True, False, "train"]
package_val = [[MR_test_list, PET_test_list], False, True, "val"]
# package_test = [test_list, False, False, "test"]

for idx_epoch_new in range(train_dict["epochs"]):
    idx_epoch = idx_epoch_new + train_dict["continue_training_epoch"]
    print("~~~~~~Epoch[{:03d}]~~~~~~".format(idx_epoch+1))

    for package in [package_train, package_val]:

        MR_list, PET_list = package[0]
        file_list = MR_list + PET_list
        np.random.shuffle(file_list)
        isTrain = package[1]
        isVal = package[2]
        iter_tag = package[3]

        if isTrain:
            model.train()
        else:
            model.eval()
        
        case_loss = np.zeros((len(file_list), 1))

        # N, C, D, H, W

        for cnt_file, file_path in enumerate(file_list):
            
            x_path = file_path
            curr_modality = None
            if "MR" in file_path:
                curr_modality = "MR"
            if "PET" in file_path:
                curr_modality = "PET"
            file_name = os.path.basename(file_path)
            print(iter_tag + " ===> Epoch[{:03d}]-[{:03d}]/[{:03d}]: --->".format(
                idx_epoch+1, cnt_file+1, len(file_list)), x_path, "<---", end="")
            x_file = nib.load(x_path)
            x_data = x_file.get_fdata()

            batch_x = np.zeros((train_dict["batch"], 1, train_dict["input_size"][0], train_dict["input_size"][1]))

            for idx_batch in range(train_dict["batch"]):
                
                d0_offset = np.random.randint(x_data.shape[0] - train_dict["input_size"][0])
                d1_offset = np.random.randint(x_data.shape[1] - train_dict["input_size"][1])

                x_slice = x_data[d0_offset:d0_offset+train_dict["input_size"][0],
                                 d1_offset:d1_offset+train_dict["input_size"][1],
                                 ]
                
                if curr_modality == "PET":
                    x_slice = x_slice / train_dict["PET_norm_factor"]

                batch_x[idx_batch, 0, :, :, :] = x_slice
                batch_y = copy.deepcopy(batch_x)

            batch_x = torch.from_numpy(batch_x).float().to(device)
            batch_y = torch.from_numpy(batch_y).float().to(device)
            
            if isTrain:

                optim.zero_grad()
                loss, pred, mask = model(batch_x, curr_modality)
                loss.backward()
                optim.step()
                case_loss[cnt_file, 0] = loss.item()
                print("Loss: ", np.mean(case_loss[cnt_file, :]))

            if isVal:

                with torch.no_grad():
                    loss, pred, mask = model(batch_x, curr_modality)

                case_loss[cnt_file, 0] = loss.item()
                print("Loss: ", np.mean(case_loss[cnt_file, :]))

        epoch_loss = np.mean(case_loss)
        print(iter_tag + " ===>===> Epoch[{:03d}]: ".format(idx_epoch+1), end='')
        print("Loss: ", epoch_loss)
        np.save(train_dict["save_folder"]+"loss/epoch_loss_"+iter_tag+"_{:03d}.npy".format(idx_epoch+1), case_loss)

        if isVal:
            # np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_xf.npy", batch_xf.cpu().detach().numpy())
            # np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_fmap.npy", batch_fmap.cpu().detach().numpy())
            torch.save(model, train_dict["save_folder"]+"model_curr.pth".format(idx_epoch + 1))
            
            if epoch_loss < best_val_loss:
                # save the best model
                torch.save(model, train_dict["save_folder"]+"model_best_{:03d}.pth".format(idx_epoch + 1))
                torch.save(optim, train_dict["save_folder"]+"optim_{:03d}.pth".format(idx_epoch + 1))
                print("Checkpoint saved at Epoch {:03d}".format(idx_epoch + 1))
                best_val_loss = epoch_loss

        # del batch_x, batch_y
        # gc.collect()
        # torch.cuda.empty_cache()
