import os
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import util
import dataloader 
import model


class MIL_trainer(object):
    def __init__(self, args):
        super(MIL_trainer, self).__init__()
        self.epochs = args.epochs
        self.lr = args.lr
        self.patch_size = args.patch_size
        self.mc = args.mc
        self.phase = args.phase
        
        self.lowest_loss = np.inf
        
        # directory for saving model 
        self.parent_dir = os.path.join('runs')
        self.ckpt_dir = os.path.join(self.parent_dir, 'ckpt')
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load data & buil model
        self.load_dataset()
        self.build_model()

        
    def load_dataset(self):
        self.train_loader, self.valid_loader = dataloader.ICIAR_loader()

        
    def build_model(self):
        if self.phase == 'continue_train':
            self.model = util.load_model(self.parent_dir, self.valid_fold)
        else:
            self.model = model.Attention()
                
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=self.lr, 
                                    betas=(0.9, 0.999), 
                                    weight_decay=5e-4)
        
         
    def train(self, epoch):
        self.model.train()
        for (bag, bag_label) in tqdm(self.train_loader):
            instances = util.get_instances_mesh(bag, self.patch_size, self.mc)
            instances, bag_label = instances.to(self.device), bag_label.to(self.device) 

            self.optimizer.zero_grad()
            output, _ = self.model(instances)
            loss = self.model.criterion(output, bag_label)
            loss.backward()
            self.optimizer.step()

        print(f'===> Train Loss: {loss.cpu().detach():.5f}')

    
    def valid(self, epoch):
        self.model.eval()
        valid_loss, correct = 0., 0.

        for (bag, bag_label) in self.valid_loader:
            instances = util.get_instances_mesh(bag, self.patch_size, self.mc)
            instances, bag_label = instances.to(self.device), bag_label.to(self.device)

            output, _ = self.model(instances)
            valid_loss += self.model.criterion(output, bag_label).cpu().detach()
            pred = output.max(dim=1)[1]
            correct += pred.eq(bag_label).float().cpu().item()

        valid_loss /= len(self.valid_loader)
        valid_accr = correct / len(self.valid_loader)

        # saving model criteria : lowest valid_loss
        self.save_model(valid_loss, valid_accr, epoch)

        print('===> Valid Loss: {:.5f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            valid_loss, correct, len(self.valid_loader), 100. * valid_accr)) 
            
    
#     def test(self):
#         # load model
#         model = util.load_model(self.parent_dir, self.valid_fold)
#         
#         self.model.eval()
#         test_loss, correct = 0., 0.
# 
#         for (data, label) in self.test_loader:
#             data, label = data.squeeze(dim=0), label.squeeze(dim=0)
#             data, bag_label = data.type(torch.float32).to(self.device), \
#                               label.type(torch.float32).to(self.device)
# 
#             output, attention = model(data)
#             test_loss += self.model.criterion(output, bag_label, attention).detach()
#             pred = output.ge(0.).float()
#             correct += pred.eq(bag_label).float().cpu().item()
#             print(pred.cpu(), bag_label.cpu())
# 
#         test_loss /= len(self.test_loader)
# 
#         print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
#             test_loss, correct, len(self.test_loader), 100. * correct / len(self.test_loader)))
        

    def save_model(self, valid_loss, valid_accr, epoch):
        if valid_loss <= self.lowest_loss:
            print('===> Saving the model....')
            state = {
                'model': self.model.state_dict(),
                'accuracy': valid_accr, 
                'epoch': epoch
            }
            torch.save(state, os.path.join(self.ckpt_dir, 'ckpt.pth'))
            self.lowest_loss = valid_loss
        
        
        
        
