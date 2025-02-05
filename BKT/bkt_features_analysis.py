"""

Lesson_Id: 
1. Modeling_Two-Step_Expressions_2019-2020_v2
2. 

"""

import os
import sys
import numpy as np
from bkt_model import Model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from helper import reshape_for_pyBKT

Lesson_Id="Modeling_Two-Step_Expressions_2019-2020_v2"
sparsity_level=10

file_name=Lesson_Id+f"_dense_sparsity_{sparsity_level}.npy" 

file_path=os.path.join(os.getcwd(),"dataset","sparse_dataset_sim","MATHia", file_name) 

print("file path is ", file_path)

data=np.load(file_path)

df_data = reshape_for_pyBKT(data)

print(df_data) 

model = Model(
    seed=42, 
    num_fits=1
)

model.fit(data=df_data, forgets=True)

df_parameters=model.params()

bkt_output_path=os.path.join(os.getcwd(),"dataset","bkt_ranking_dataset")
os.makedirs(bkt_output_path, exist_ok=True)

output_csv_path = os.path.join(bkt_output_path, "all_features_records.csv")

df_parameters.to_csv(output_csv_path)

