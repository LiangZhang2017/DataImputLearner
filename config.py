import os
import pandas as pd

class Model_Config:
    def __init__(self,args):
        self.args=args
        
        print("args")
    
    
    def generate_paradic(self):
        
        print("generate parameters")
        
        course=self.args.Course[0]
        
        print(os.getcwd())
        print(self.args.data_path[0])
        print(self.args.data_path[1])
        
        file_path = os.path.join(
            os.getcwd(),
            self.args.data_path[0].lstrip("/"),
            self.args.data_path[1].lstrip("/"),
            "MathiaAllData.txt"
        )
                
        print("file_path is ", file_path)
        
        data = pd.read_csv(file_path, sep='\t')  # Adjust `sep` as needed (e.g., ',' for CSV)
        
        # Display basic information about the data
        print("Data successfully loaded!")
        print("Shape of the DataFrame:", data.shape)
        print("First few rows of the data:")
        print(data.columns)
        
        # print(data['Outcome'])
        # print(data['Attempt At Step'])
        
        # print(data['Anon Student Id'])
        print(data['Problem Name'].unique())
        print(data['Level (Workspace Id)'])
        
        unique_students = data['Anon Student Id'].nunique()
        
        print("unique_students are ",unique_students)
        
        
        
        