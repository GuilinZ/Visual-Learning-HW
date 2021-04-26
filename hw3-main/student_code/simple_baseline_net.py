import torch.nn as nn
from external.googlenet.googlenet import GoogLeNet
import torch
class SimpleBaselineNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering (Zhou et al, 2017) paper.
    """
    def __init__(self, question_encoding_len, answer_encoding_len):
        super().__init__()
	    ############ 2.2 TODO
        self.backbone = GoogLeNet(aux_logits=False)
        # self.word_embedding = nn.Embedding(question_encoding_len, 1024)
        self.word_embedding = nn.Linear(question_encoding_len, 1024)
        self.fc = nn.Linear(2048, answer_encoding_len)
	    ############

    def forward(self, image, question_encoding):
	    ############ 2.2 TODO
        img_features = self.backbone(image) #Bx1000
        bow_encoding = torch.clamp(torch.sum(question_encoding, dim=1), min=0, max=1).type(torch.FloatTensor).cuda() #Bxquestion_encoding_len
        word_features = self.word_embedding(bow_encoding) #Bx1000

        new_features = torch.cat((img_features, word_features), dim=1) #Bx2000
        out = self.fc(new_features) #Bx5217
        return out
	    ############
        # raise NotImplementedError()
