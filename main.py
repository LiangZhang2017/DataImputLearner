import argparse
import sys
from config import Model_Config
from helper import reshape_for_pyBKT
# from pyBKT.models import Model 
from BKT.bkt_model import Model
import sklearn.metrics as sk
import os
import pandas as pd

def data_input_config():
    
    '''
    Focus on the Mathia dataset
    Lesson_Id: 
    1) Using_Scale_Factor_2019-2020
    2) Modeling_Two-Step_Expressions_2019-2020_v2
    3) Analyzing_Models_of_Two-Step_Linear_Relationships_2019-2020
    
    KC model is used to explore the similiar learner data based on the foure features. 
    1. BKT github: https://github.com/CAHLR/pyBKT
    2. Prior Per Student (KtPPS.py): https://github.com/CAHLR/pyBKT-examples
    '''
    
    parser=argparse.ArgumentParser(description='Arguments for Parameters Setting')
    parser.add_argument("--Course",nargs=1,type=str,default=['MATHia'])
    parser.add_argument("--sparse_data_path",nargs=2,type=str,default=['/dataset/sparse_dataset_sim','/MATHia'])
    parser.add_argument("--dense_data_path",nargs=2,type=str,default=['/dataset/dense_dataset','/MATHia'])
    parser.add_argument("--Lesson_Id",nargs=1,type=str,default=['Modeling_Two-Step_Expressions_2019-2020_v2'])
    parser.add_argument("--kc_model",nargs=1,type=str,default=['IBKT'])
    parser.add_argument("--Imputation_model",nargs=1,type=str,default=['GAIN'])
        
    args=parser.parse_args()
    
    return args

if __name__ == '__main__':
    print("main")
    print(sys.executable)
    
    args=data_input_config()
    
    print("args are ", args)
    
    model_obj=Model_Config(args=args)
    sparsity_tensors_all=model_obj.extract_sparsity_tensors_all()
    print("sparsity_data_all is ", sparsity_tensors_all[90].shape)
    
    dense_tensor=model_obj.extract_dense_tensor() 
    
    print("########### Start of BKT Similarity Exploration ############")
    print("dense_tensor is ", dense_tensor.shape)
    # Convert to a DataFrame
    df_pyBKT = reshape_for_pyBKT(dense_tensor)
    
    # df_pyBKT['combined_resource'] = df_pyBKT['skill_name'].astype(str) + "_" + df_pyBKT['user_id'].astype(str)
    
    model = Model(
        seed=42, 
        num_fits=100
    )

    model.fit(data=df_pyBKT, forgets=True)
    
    training_auc=model.evaluate(data = df_pyBKT, metric = 'auc')
    training_rmse=model.evaluate(data = df_pyBKT)
    
    print("training_auc is ", training_auc)
    print("training_rmse is ", training_rmse)
    
    print("model.params()", model.params())
    
    df_parameters=model.params()
    
    bkt_output_path=os.path.join(os.getcwd(),"BKT","outputs")
    os.makedirs(bkt_output_path, exist_ok=True)
    
    output_csv_path = os.path.join(bkt_output_path, "output.csv")
    df_parameters.to_csv(output_csv_path)
    
    print("########### End of  BKT Similarity Exploration ############")

    # Try to get both the dense tensor and sparse tensor
    sparse_tensor=sparsity_tensors_all[10]
    
    # model_obj.model_running(sparse_tensor, dense_tensor)
    