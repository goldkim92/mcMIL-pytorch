import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            
#             nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            
#             nn.MaxPool2d(2),
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(64 * 8 * 8, self.L),
            nn.ReLU(),
#             nn.Dropout(0.5),
            
            nn.Linear(self.L, self.L),
            nn.ReLU(),
#             nn.Dropout(0.5),
            
        )
        
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 4),
        )

        # weight initialization
        for k,m in self._modules.items():
            for mm in m:
                if any([isinstance(mm, module) for module in [nn.Conv2d, nn.Linear]]):
                    nn.init.xavier_normal_(mm.weight)
                    
    
    def forward(self, x):
        H = self.feature_extractor_part1(x) 
        H = H.view(-1, 64 * 8 * 8)
        H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        
        M = torch.mm(A, H)  # KxL

        score = self.classifier(M)
        return score, A
    
    
    def criterion(self, output, bag_label, attention=None):
        ''' binary cross entropy loss '''
        return F.cross_entropy(output, bag_label)
    
    
