import nibabel as nib
import numpy as np
import os
import glob

NAC_list = sorted(glob.glob("./data/Iman/PET_256_MAX_MIN_2_Not_Normalized/*.nii"))
CT_list = sorted(glob.glob("./data/Iman/CT_256_MAX_MIN_2_Not_Normalized/*.nii"))

for folder_path in ["./data/NAC_wb_norm/", "./data/CT_wb_norm/"]:
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

for file_path in NAC_list:
    file_name = os.path.basename(file_path)
    print(file_name, end=" -> ")
    NAC_file = nib.load(file_path)
    NAC_data = NAC_file.get_fdata()
    print(NAC_data.shape)
    NAC_data = np.resize(NAC_data, (256, 256, NAC_data.shape[2]))
    NAC_data = NAC_data / 4000
    new_NAC_file = nib.Nifti1Image(NAC_data, NAC_file.affine, NAC_file.header)
    new_save_name = "./data/NAC_wb_norm/" + file_name + ".gz"
    nib.save(new_NAC_file, new_save_name)

for file_path in CT_list:
    file_name = os.path.basename(file_path)
    print(file_name, end=" -> ")
    CT_file = nib.load(file_path)
    CT_data = CT_file.get_fdata()
    print(CT_data.shape)
    CT_data = np.resize(CT_data, (256, 256, CT_data.shape[2]))
    CT_data = (CT_data + 1000) / 4000
    new_CT_file = nib.Nifti1Image(CT_data, CT_file.affine, CT_file.header)
    new_save_name = "./data/CT_wb_norm/" + file_name + ".gz"
    nib.save(new_CT_file, new_save_name)

    

