from .utils import *

import os
import gzip
from tqdm import tqdm
import pandas as pd
import torch

class Tokenizer:
    def __init__(
            self,
            max_length:int=int(7.5e6),
    ) -> None:
        self.max_length = max_length

    def __call__(
            self,
            input_dir:str,
            output_dir:str,
            *args,
            **kwargs,
    ) -> None:
        
        for file_name in get_file_names(input_dir):
            file_path = os.path.join(input_dir, file_name)
            file_id, chr_id, _, _ = file_name.split(".")
            _output_file = lambda filename: os.path.join(output_dir, f"{file_id}.{chr_id}."+filename)
            print(f"Start the process of converting the {chr_id} region of {file_id}.")

            with gzip.open(file_path, 'rt') as file:
                lines = file.readlines()

            record=[]
            for line in tqdm(lines):
                if line[0]!="#":
                    for info in get_info_from_line(line):
                        if info is not None:
                            record.append(info)
            for _ in range(self.max_length-len(record)):
                record.append([0, 0, "NNNN", 0])

            record_df = pd.DataFrame(record, columns=["chr", "positions", "token_str", "tokens"])
            # print(record_df.head)
            record_df.to_csv(_output_file("records.csv"), index=False)
            print("Record creation is completed.")

            tokens = torch.tensor(record_df["tokens"].values)
            torch.save(tokens, _output_file("tokens.pt"))
            print("tokens creation is completed.")

            positions_combined = list(zip(record_df["chr"].values, record_df["positions"].values))
            positions_tensor = torch.tensor(positions_combined)
            torch.save(positions_tensor, _output_file("positions.pt"))
            print("positions creation is completed.")