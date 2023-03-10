from models import mae_vit_base_patch16
from models import mae_vit_large_patch16
from models import mae_vit_huge_patch14
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
    ["MR_PET_mae", [3]],
    ["MRCT_brain_NACCT_wb_mae", [5]],
]

print("Model index: ", end="")
current_model_idx = int(input()) - 1
print(model_list[current_model_idx])
time.sleep(1)

# ==================== basic ====================

train_dict = np.load("./project_dir/"+model_list[current_model_idx][0]+"/"+"setting.npy", allow_pickle=True).item()

# train_dict = {}
# train_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
# train_dict["project_name"] = model_list[current_model_idx][0]
# train_dict["gpu_ids"] = model_list[current_model_idx][1]

# train_dict["dropout"] = 0.
# train_dict["loss_term"] = "SmoothL1Loss"
# train_dict["optimizer"] = "AdamW"

# train_dict["save_folder"] = "./project_dir/"+train_dict["project_name"]+"/"
# train_dict["seed"] = 426
# train_dict["input_size"] = [256, 256]
# train_dict["epochs"] = 200
# train_dict["batch"] = 64
# train_dict["PET_norm_factor"] = 4000
# train_dict["target_model"] = "./pre_train/mae_pretrain_vit_large.pth"
# train_dict["modality_club"] = ["MR_brain_norm", "CT_brain_norm", "NAC_wb_norm", "CT_wb_norm"]

# train_dict["model_term"] = "two-branch mae"
# train_dict["continue_training_epoch"] = 0
# train_dict["flip"] = False

# train_dict["folder_club"] = list("./data/"+modalities+"/" for modalities in train_dict["modality_club"])

# train_dict["val_ratio"] = 0.3
# train_dict["test_ratio"] = 0.2

# train_dict["opt_lr"] = 1e-3 # default
# train_dict["opt_betas"] = (0.9, 0.999) # default
# train_dict["opt_eps"] = 1e-8 # default
# train_dict["opt_weight_decay"] = 0.01 # default
# train_dict["amsgrad"] = False # default

for path in [train_dict["save_folder"]+"pred_monai"]:
    if not os.path.exists(path):
        os.mkdir(path)

# np.save(train_dict["save_folder"]+"setting.npy", train_dict)

# ==================== GPU ====================

np.random.seed(train_dict["seed"])
gpu_list = ','.join(str(x) for x in train_dict["gpu_ids"])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== model ====================

ckpt_list = sorted(glob.glob(train_dict["save_folder"]+"model_best_*.pth"))
if len(ckpt_list) > 0:
    model = torch.load(ckpt_list[-1], map_location='cpu')["model"]
# model = mae_vit_large_patch16(modality_club=train_dict["modality_club"])
# model.load_state_dict(ckpt, strict=False)

model.eval()
model = model.to(device)

# ==================== data division ====================

data_division_dict = np.load(train_dict["save_folder"]+"data_division.npy", allow_pickle=True).item()
train_club = data_division_dict["train_club"]
val_club = data_division_dict["val_club"]
test_club = data_division_dict["test_club"]

# ==================== training ====================

package_test = [test_club, False, False, "test"]

file_list = test_club
iter_tag = "test"

case_loss = np.zeros((len(file_list), 1))

# N, C, D, H, W

for cnt_file, file_path in enumerate(file_list):

    x_path = file_path
    curr_modality = None
    if "MR_brain_norm" in file_path:
        curr_modality = "MR_brain_norm"
    if "CT_brain_norm" in file_path:
        curr_modality = "CT_brain_norm"
    if "NAC_wb_norm" in file_path:
        curr_modality = "NAC_wb_norm"
    if "CT_wb_norm" in file_path:
        curr_modality = "CT_wb_norm"

    file_name = os.path.basename(file_path)
    print(iter_tag + " ===> [{:03d}]/[{:03d}]: --->".format(cnt_file+1, len(file_list)), x_path, "<---", end="")
    x_file = nib.load(x_path)
    x_data = x_file.get_fdata()

    # if curr_modality == "PET":
    #     x_data = x_data / train_dict["PET_norm_factor"]
    x_data = np.resize(x_data, (train_dict["input_size"][0], train_dict["input_size"][1], x_data.shape[2]))

    batch_x = np.zeros((train_dict["batch"], 3, train_dict["input_size"][0], train_dict["input_size"][1]))
    recon_x = np.zeros((x_data.shape))
    len_z = x_data.shape[2]
    for idx_z in range(len_z):
        if idx_z == 0:
            batch_x[:, 0, :, :] = x_data[:, :, idx_z]
            batch_x[:, 1, :, :] = x_data[:, :, idx_z]
            batch_x[:, 2, :, :] = x_data[:, :, idx_z+1]
        elif idx_z == len_z-1:
            batch_x[:, 0, :, :] = x_data[:, :, idx_z-1]
            batch_x[:, 1, :, :] = x_data[:, :, idx_z]
            batch_x[:, 2, :, :] = x_data[:, :, idx_z]
        else:
            batch_x[:, 0, :, :] = x_data[:, :, idx_z-1]
            batch_x[:, 1, :, :] = x_data[:, :, idx_z]
            batch_x[:, 2, :, :] = x_data[:, :, idx_z+1]

        batch_x = torch.from_numpy(batch_x).float().to(device)
        with torch.no_grad():
            loss, pred, mask = model(batch_x, curr_modality)
        recon_x[:, :, idx_z] = pred[0, 1, :, :].cpu().numpy()
        
    recon_x = np.resize(recon_x, x_data.shape)
    recon_file = nib.Nifti1Image(recon_x, x_file.affine, x_file.header)
    recon_name = train_dict["save_folder"]+"pred_monai/"+file_name.replace(curr_modality, curr_modality+"_recon")
    nib.save(recon_file, recon_name)
    print(" --->", recon_name, "<---")

