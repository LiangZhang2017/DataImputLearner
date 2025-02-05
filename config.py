import os
import pandas as pd
import numpy as np
from GAIN.gain_model import GAIN

class Model_Config:
    def __init__(self,args):
        self.args=args
        
        print("args")
        
    
    def model_factory(self, model_name, sparse_tensor, dense_tensor, set_parameters):
        model_constructors = {
            "Standard_TC": lambda: Standard_TC(sparse_tensor, dense_tensor, set_parameters),
            "Standard_CPD": lambda: CPDecomposition(sparse_tensor, dense_tensor, set_parameters),
            "BPTF": lambda: BPTF(sparse_tensor, dense_tensor, set_parameters),
            "GAN": lambda: New_GAN(sparse_tensor, dense_tensor, set_parameters),  # Assuming New_GAN is the desired class
            "CGAN": lambda: CGAN(sparse_tensor, dense_tensor, set_parameters),
            "GAIN": lambda: GAIN(sparse_tensor, dense_tensor, set_parameters),
            "AmbientGAN": lambda: AmbientGAN(sparse_tensor, dense_tensor, set_parameters),
            "Embedding_GAIN": lambda: Embedding_GAIN(sparse_tensor, dense_tensor, set_parameters),
            "InfoGAN":lambda: InfoGAN(sparse_tensor, dense_tensor, set_parameters)
        }
        constructor = model_constructors.get(model_name)
        return constructor() if constructor else None
    
    def extract_sparsity_tensors_all(self):
        
        print("generate parameters")
        
        course=self.args.Course[0]
        
        print(os.getcwd())
        print(self.args.sparse_data_path[0])
        print(self.args.sparse_data_path[1])
        
        file_path = os.path.join(
            os.getcwd(),
            self.args.sparse_data_path[0].lstrip("/"),
            self.args.sparse_data_path[1].lstrip("/")
        )
                
        print("file_path is ", file_path)
        
        Sparsity_levels=[10,20,30,40,50,60,70,80,90]
        
        sparsity_data_all = {}
        
        for sp_level in Sparsity_levels:
            
            file_name=self.args.Lesson_Id[0]+"_dense_sparsity_"+str(sp_level)+".npy"
            
            print("file_name is ", file_name)
            
            full_file_path = os.path.join(file_path, file_name)
            
            # Read the .npy file (assuming it exists)
            try:
                data_saprsity_tensor = np.load(full_file_path)
                print(f"Sparsity level {sp_level}: Loaded array shape =", data_saprsity_tensor.shape)
                sparsity_data_all[sp_level] = data_saprsity_tensor

            except FileNotFoundError:
                print(f"File not found: {full_file_path}")
            except Exception as e:
                print(f"Error loading file {full_file_path}: {e}")
                
        return sparsity_data_all
    
    def extract_dense_tensor(self):
        
        print("generate parameters")
        
        course=self.args.Course[0]
        
        print(os.getcwd())
        print(self.args.dense_data_path[0])
        print(self.args.dense_data_path[1])
        
        file_path = os.path.join(
            os.getcwd(),
            self.args.dense_data_path[0].lstrip("/"),
            self.args.dense_data_path[1].lstrip("/")
        )
  
        print("file_path is ", file_path)

        file_name=self.args.Lesson_Id[0]+"_dense.npy"
        
        print("file_name is ", file_name)
        
        full_file_path = os.path.join(file_path, file_name)
        
        # Read the .npy file (assuming it exists)
        try:
            dense_data_tensor = np.load(full_file_path)
            print(f"dense_data_tensor is loaded successfully", dense_data_tensor.shape)

        except FileNotFoundError:
            print(f"File not found: {full_file_path}")
        except Exception as e:
            print(f"Error loading file {full_file_path}: {e}")
        
        return dense_data_tensor

    def model_running(self, sparse_tensor, dense_tensor):
        
        model_name=self.args.Imputation_model[0]
        
        print("model name is ", model_name)
        
        set_parameters="GAIN parameters"
        model = self.model_factory(model_name, sparse_tensor, dense_tensor, set_parameters)
        
        max_iter=5
        
        model.RunModel() 