# from models import mae_vit_base_patch16
from models import mae_vit_large_patch16_cont
# from models import mae_vit_huge_patch14
from utils import dataset_division

import os
import gc
import glob
# import copy
import time
import torch
import numpy as np
import nibabel as nib
import torch.nn as nn
import torch.nn.functional as F

# ==================== model select ====================

model_list = [
    ["MRCT_brain_NACCT_wb_mae_base_MRCT_L1", [5], "l1", "base", "MRCT"],
    ["MRCT_brain_NACCT_wb_mae_cont_MRCT_L1", [5], "l1", "cont", "MRCT"],
    ["MRCT_brain_NACCT_wb_mae_base_NACCT_L1", [5], "l1", "base", "NACCT"],
    ["MRCT_brain_NACCT_wb_mae_cont_NACCT_L1", [5], "l1", "cont", "NACCT"],
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
train_dict["loss_term"] = model_list[current_model_idx][2]
train_dict["model_term"] = model_list[current_model_idx][3]
train_dict["modality_term"] = model_list[current_model_idx][4]


train_dict["save_folder"] = "./project_dir/"+train_dict["project_name"]+"/"
train_dict["seed"] = 426
train_dict["input_size"] = [256, 256]
train_dict["epochs"] = 200
train_dict["batch"] = 16

train_dict["cont_folder"] = "./project_dir/MRCT_brain_NACCT_wb_mae_L1/"
train_dict["pretrained_model"] = "./project_dir/MRCT_brain_NACCT_wb_mae_L1/model_best_195.pth"
train_dict["baseline_ckpt"] = "./pre_train/mae_pretrain_vit_large.pth"
if train_dict["modality_term"] == "MRCT":
    train_dict["modality_club"] = ["MR_brain_norm", "CT_brain_norm"]
elif train_dict["modality_term"] == "NACCT":
    train_dict["modality_club"] = ["NAC_wb_norm", "CT_wb_norm"]

# train_dict["modality_club"] = ["MR_brain_norm", "CT_brain_norm", "NAC_wb_norm", "CT_wb_norm"]

# train_dict["model_term"] = "two-branch mae"
train_dict["continue_training_epoch"] = 200
train_dict["flip"] = False

train_dict["folder_club"] = list("./data/"+modalities+"/" for modalities in train_dict["modality_club"])

train_dict["val_ratio"] = 0.3
train_dict["test_ratio"] = 0.2

train_dict["optimizer"] = "AdamW"
train_dict["opt_lr"] = 1e-4 # default
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

model = mae_vit_large_patch16_cont(modality_club=[train_dict["modality_club"][0]], loss_type=train_dict["loss_term"])
if train_dict["model_term"] == "cont":
    pre_train_dir = train_dict["pretrained_model"]
    ckpt = torch.load(pre_train_dir, map_location='cpu')
    ckpt_state_dict = ckpt.state_dict()
    ckpy_state_dict_key = ckpt_state_dict.keys()
    for ckpt_key in ckpy_state_dict_key:
        for modalities in train_dict["modality_club"]:
            if modalities in ckpt_key:
                ckpt_state_dict.pop(ckpt_key, None)

    model.load_state_dict(ckpt_state_dict, strict=False)
elif train_dict["model_term"] == "base":
    ckpt = torch.load(train_dict["baseline_ckpt"], map_location='cpu')["model"]
    num_ckpt_patches = (224 // 16) * (224 // 16) + 1
    num_model_patches = (train_dict["input_size"][0] // 16) * (train_dict["input_size"][1] // 16) + 1
    if num_ckpt_patches != num_model_patches:
        ckpt["pos_embed"] = F.interpolate(ckpt["pos_embed"].unsqueeze(0), size=(num_model_patches, 1024), mode="bilinear", align_corners=False).squeeze(0)
    model.load_state_dict(ckpt, strict=False)

# model_state_dict_keys = list(model.state_dict().keys())
# for key in list(ckpt.keys()):
#     if key in model_state_dict_keys:
#         if ckpt[key].shape != model.state_dict()[key].shape:
#             print(key, ckpt[key].shape, model.state_dict()[key].shape)
#             # pos_embed torch.Size([1, 197, 1024]) torch.Size([1, 257, 1024])

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

data_division_dict = np.load(train_dict["cont_folder"]+"data_division.npy", allow_pickle=True).item()
train_club = data_division_dict["train_club"]
val_club = data_division_dict["val_club"]
test_club = data_division_dict["test_club"]

# ==================== training ====================

best_val_loss = 1e3
best_epoch = 0

package_train = [ train_club, True, False, "train"]
package_val = [ val_club, False, True, "val"]
# package_test = [test_list, False, False, "test"]

for idx_epoch_new in range(train_dict["epochs"]):
    idx_epoch = idx_epoch_new + train_dict["continue_training_epoch"]
    print("~~~~~~Epoch[{:03d}]~~~~~~".format(idx_epoch+1))

    for package in [package_train, package_val]:

        file_list = package[0]
        np.random.shuffle(file_list)
        isTrain = package[1]
        isVal = package[2]
        iter_tag = package[3]

        if isTrain:
            model.train()
        else:
            model.eval()
        
        case_loss = dict()
        case_loss[train_dict["modality_club"][0]] = []

        curr_modality_file_list = []
        for file_path in file_list:
            if train_dict["modality_club"][0] in file_path:
                curr_modality_file_list.append(file_path)


        # N, C, D, H, W

        for cnt_file, file_path in enumerate(curr_modality_file_list):
            
            x_path = file_path
            curr_modality = train_dict["modality_club"][0]

            file_name = os.path.basename(file_path)
            print(iter_tag + " ===> Epoch[{:03d}]-[{:03d}]/[{:03d}]: --->".format(
                idx_epoch+1, cnt_file+1, len(file_list)), x_path, "<---", end="")
            x_file = nib.load(x_path)
            y_file = nib.load(x_path.replace(curr_modality, train_dict["modality_club"][1]))
            x_data = x_file.get_fdata()
            y_data = y_file.get_fdata()
            x_data = np.resize(x_data, (train_dict["input_size"][0], train_dict["input_size"][1], x_data.shape[2]))
            y_data = np.resize(y_data, (train_dict["input_size"][0], train_dict["input_size"][1], y_data.shape[2]))
            batch_x = np.zeros((train_dict["batch"], 3, train_dict["input_size"][0], train_dict["input_size"][1]))
            batch_y = np.zeros((train_dict["batch"], 3, train_dict["input_size"][0], train_dict["input_size"][1]))
            batch_prob = np.mean(x_data, axis=(0, 1))
            batch_prob = batch_prob / np.sum(batch_prob)

            for idx_batch in range(train_dict["batch"]):
                
                # z_offset = np.random.randint(x_data.shape[2]-2)
                z_offset = np.random.choice(x_data.shape[2], p=batch_prob)
                if z_offset == 0:
                    batch_x[idx_batch, 0, :, :] = x_data[:, :, z_offset]
                    batch_x[idx_batch, 1, :, :] = x_data[:, :, z_offset]
                    batch_x[idx_batch, 2, :, :] = x_data[:, :, z_offset+1]
                    batch_y[idx_batch, 0, :, :] = y_data[:, :, z_offset]
                    batch_y[idx_batch, 1, :, :] = y_data[:, :, z_offset]
                    batch_y[idx_batch, 2, :, :] = y_data[:, :, z_offset+1]
                elif z_offset == x_data.shape[2]-1:
                    batch_x[idx_batch, 0, :, :] = x_data[:, :, z_offset-1]
                    batch_x[idx_batch, 1, :, :] = x_data[:, :, z_offset]
                    batch_x[idx_batch, 2, :, :] = x_data[:, :, z_offset]
                    batch_y[idx_batch, 0, :, :] = y_data[:, :, z_offset-1]
                    batch_y[idx_batch, 1, :, :] = y_data[:, :, z_offset]
                    batch_y[idx_batch, 2, :, :] = y_data[:, :, z_offset]
                else:
                    batch_x[idx_batch, 0, :, :] = x_data[:, :, z_offset-1]
                    batch_x[idx_batch, 1, :, :] = x_data[:, :, z_offset]
                    batch_x[idx_batch, 2, :, :] = x_data[:, :, z_offset+1]
                    batch_y[idx_batch, 0, :, :] = y_data[:, :, z_offset-1]
                    batch_y[idx_batch, 1, :, :] = y_data[:, :, z_offset]
                    batch_y[idx_batch, 2, :, :] = y_data[:, :, z_offset+1]
            batch_x = torch.from_numpy(batch_x).float().to(device)
            batch_y = torch.from_numpy(batch_y).float().to(device)
            
            if isTrain:

                optim.zero_grad()
                loss, pred, mask = model(
                    imgs=batch_x, 
                    trgs=batch_y,
                    modality=curr_modality,
                    mask_ratio=0.,
                )
                loss.backward()
                optim.step()
                case_loss[curr_modality].append(loss.item())
                print("Loss: ", loss.item())

            if isVal:

                with torch.no_grad():
                    loss, pred, mask = model(
                        imgs=batch_x, 
                        trgs=batch_y,
                        modality=curr_modality,
                        mask_ratio=0.,
                    )
                case_loss[curr_modality].append(loss.item())
                print("Loss: ", loss.item())

        epoch_loss = []
        print(iter_tag + " ===>===> Epoch[{:03d}]: ".format(idx_epoch+1), end='')
        for curr_modality in case_loss.keys():
            curr_loss = np.mean(case_loss[curr_modality])
            epoch_loss.append(curr_loss)
            print(curr_modality, curr_loss, end=' ')
        print()
        np.save(train_dict["save_folder"]+"loss/epoch_loss_"+iter_tag+"_{:03d}.npy".format(idx_epoch+1), case_loss)
        epoch_loss = np.mean(epoch_loss)

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
        
        del batch_x, batch_y
        gc.collect()
        torch.cuda.empty_cache()
