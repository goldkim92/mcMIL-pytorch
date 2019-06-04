import os
import argparse
from tqdm import tqdm

from solver import MIL_trainer


# ===========================================================
# settings
# ===========================================================
parser = argparse.ArgumentParser(description='')

parser.add_argument('--gpu_number', type=str, default='0')
parser.add_argument('--epochs',     type=int, default=100)
parser.add_argument('--lr',         type=float, default=1e-4)
parser.add_argument('--patch_size', type=int, default=128)
parser.add_argument('--mc',         type=int, default=64)
parser.add_argument('--phase',       type=str, default='train', help='"train" or "test" or "continue_train"') 

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_number


# ===========================================================
# main
# ===========================================================
if __name__ == "__main__":
    
    net = MIL_trainer(args)
    
    if args.phase in ['train', 'continue_train']:
        print('\n===== Train Monte Carlo AMIL =====')
        
        # Validate the initialized model
        net.valid(0)
        
        # Train model
        for epoch in range(1, args.epochs+1):
            print(f'\nEpoch: {epoch}')
            net.train(epoch)    
            net.valid(epoch)


#         # Evaluate the final model
#         print('\n\n Evaluate normal {}-MIL'.format(args.method))
#         net.test()
#     
#     elif args.phase == 'test':
#         print('\n\n Evaluate normal {}-MIL'.format(args.method))
#         net.test()
        
    else:
        raise Exception("phase should be in ['train', 'test', 'continue_train']")
        
    
    
