import time
import argparse
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
from Tools.utils import *
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from Compare.PyOD import compare
from binary.Bprototype import BPROT

# conda install cudatoolkit conda install -c dglteam/label/cu116 dgl   conda install dgl-cuda11.6-0.9.1-py39_0.tar.bz2

if __name__ == '__main__':
    st = time.time()
    #tips  0(compare_methods)  2(PHAD) 
    compare_alg = 2
    # 0: SDAD
    # 2: PHAD

    c_alg = 'SDAD'

    for dataname in dataname_list:
        seed_torch(seed)
        parser = argparse.ArgumentParser()
        args = parser.parse_args()

        args.device = device
        args.num_embedding = 16
        args.width = 7
        args.k_nebor = 20
        args.w_list = 't'
        args.mad = 't'
        args.alpha = 0.7
        args.t = 1000

        train_x, train_y, test_x, test_y = getdataNN(dataname, 0.2)

        if compare_alg == 0:
            name, maxauc, maxpr, timetaken = compare(c_alg, dataname, train_x, train_y, test_x, test_y, seed, args.k_nebor)
        elif compare_alg == 2:
            maxauc, maxpr, timetaken = BPROT(dataname, device, train_x, train_y, test_x, test_y, args)
      

        output_file = "t.xlsx"
        if os.path.exists(output_file):
            df = pd.read_excel(output_file)
        else:
            print("fdasfffff")
        result_str = '{:.4f}, {:.4f}'.format(np.mean(maxauc), np.mean(maxpr))
        df = pd.concat([df, pd.DataFrame([result_str])], ignore_index=True)
        print(result_str)
        df.to_excel(output_file, index=False)

    time_taken = time.time() - st
    print("TimeTaken:",time_taken)

