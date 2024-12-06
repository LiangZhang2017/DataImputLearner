
import argparse


def data_input_config():
    
    parser=argparse.ArgumentParser(description='Arguments for Parameters Setting')
    parser.add_argument("--Course",nargs=1,type=str,default=['CSAL'])
    parser.add_argument("--data_path",nargs=2,type=str,default=['/dataset','/CSAL'])
    parser.add_argument("--Lesson_Id",nargs=1,type=str,default=['lesson21'])
    parser.add_argument("--Imput_model",nargs=1,type=str,default=['GAIN'])
        
    args=parser.parse_args()
    
    return args

if __name__ == '__main__':
    
    print("main")
    
    args=data_input_config()
    
    print("args are ", args)
    
    