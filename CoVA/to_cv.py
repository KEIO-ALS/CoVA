from data_utils import Dataset_CEU, DataParallel
from utils import exists

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys

def to_cv(dataset_dir, output_dir, trained_ceu_model):
    # バッチサイズを搭載されているGPUの数に変更
    batch_size = 2
    dataset = Dataset_CEU(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    CEU = torch.load(trained_ceu_model)
    CEU = DataParallel(CEU).cuda()


    for step, (tokens, filenames) in enumerate(dataloader):
        file_ids = list(map(lambda filename: filename.split("/")[-1].split(".")[0], filenames))
        file_ids_str = ""
        for file_id in file_ids:
            file_ids_str += file_id + " & "
        file_ids_str = file_ids_str[:-3]
        print(f"Start transform {file_ids_str} tokens into CVs.")
        pre_cls_vecs = [None for _ in range(batch_size)]
        for segment_index in tqdm(range(7500)):
            start_pos, end_pos = segment_index*1000, (segment_index+1)*1000
            segment = tokens[:, start_pos:end_pos]

            with torch.no_grad():
                cls_vecs = CEU(segment, None, True)

            for idx, cls_vec in enumerate(cls_vecs):
                if exists(pre_cls_vecs[idx]):
                    pre_cls_vecs[idx] = torch.cat((pre_cls_vecs[idx], cls_vec), dim=0)
                else:
                    pre_cls_vecs[idx] = cls_vec
        for idx, vec in enumerate(pre_cls_vecs):
            torch.save(vec, os.path.join(output_dir ,f"{file_ids[idx]}.cv.pt"))
        print(f"Completed to transform {file_ids_str} tokens into CVs.")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Argument is missing. (e.g. python3 tokenize.py <dataset directory> <output directory> <trained_ceu_model>)")
        sys.exit(1)
    dataset_dir = os.path.join(sys.argv[1])
    output_dir = os.path.join(sys.argv[2])
    trained_ceu_model = os.path.join(sys.argv[3])
    to_cv(dataset_dir, output_dir, trained_ceu_model)