import torch.nn as nn
import torch

class CoattentionNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Hierarchical Question-Image Co-Attention
    for Visual Question Answering (Lu et al, 2017) paper.
    """
    def __init__(self, question_encoding_len, answer_encoding_len, emb_len, question_seq_len):
        super().__init__()
        ############ 3.3 TODO
        self.get_word_embedding = nn.Sequential(
            nn.Linear(question_encoding_len, emb_len),
            nn.Tanh(),
            nn.Dropout(0.5)
        )

        self.phrase_unigram = nn.Conv1d(emb_len, emb_len, kernel_size=1, stride=1, padding=0)
        self.phrase_biagram = nn.Conv1d(emb_len, emb_len, kernel_size=2, stride=1, padding=0)
        self.phrase_triagram = nn.Conv1d(emb_len, emb_len, kernel_size=3, stride=1, padding=1)
        self.phrase_activation = nn.Sequential(nn.Tanh(),
                                               nn.Dropout(0.5))

        self.get_question_embedding = nn.LSTM(input_size=emb_len, hidden_size=emb_len, batch_first=True)

        self.get_img_embedding = nn.Sequential(nn.Linear(512, emb_len),
                                               nn.Tanh(),
                                               nn.Dropout(0.5))
        self.get_question_attention = AttentionNet(emb_len, hidden_len=512)
        self.get_img_attention = AttentionNet(emb_len, hidden_len=512)
        self.final_linear_1 = nn.Sequential(
            nn.Linear(emb_len, emb_len),
            nn.Tanh(),
            nn.Dropout(0.5)
        )
        self.final_linear_2 = nn.Sequential(
            nn.Linear(emb_len * 2, emb_len),
            nn.Tanh(),
            nn.Dropout(0.5)
        )
        self.final_linear_3 = nn.Sequential(
            nn.Linear(emb_len * 2, emb_len * 2),
            nn.Tanh(),
            nn.Dropout(0.5)
        )
        self.output_layer = nn.Linear(emb_len * 2, answer_encoding_len)
        ############
    def get_coattention(self, q, v):
        s = self.get_question_attention(q, None)
        v = self.get_img_attention(v, s)
        q = self.get_question_attention(q, v)
        return v.squeeze(), q.squeeze()
    def forward(self, image, question_encoding):
        ############ 3.3 TODO
        ##image
        img_encoding = image.view(-1, 512, 196)
        img_encoding_trans = img_encoding.permute(0, 2, 1) #B x 196 x 512
        img_features = self.get_img_embedding(img_encoding_trans)

        ##word
        word_features = self.get_word_embedding(question_encoding) #B x 26 x emb_len
        word_features_trans = word_features.permute(0,2,1) #B x emb_len x 26
        v_word, q_word = self.get_coattention(word_features, img_features)
        ##phrase
        phrase_fea_unigram = self.phrase_unigram(word_features_trans)
        phrase_fea_biagram = self.phrase_biagram(nn.functional.pad(word_features_trans, (0, 1)))
        phrase_fea_triagram = self.phrase_triagram(word_features_trans) #B x emb_len x 26

        phrase_features_trans = torch.max(torch.max(phrase_fea_unigram, phrase_fea_biagram), phrase_fea_triagram)
        phrase_features = phrase_features_trans.permute(0, 2, 1) #B x 26 x emb_len
        phrase_features = self.phrase_activation(phrase_features)#B x 26 x emb_len
        v_phrase, q_phrase = self.get_coattention(phrase_features, img_features)
        ##question
        question_features, _ = self.get_question_embedding(phrase_features) #B x 26 x emb_len
        v_question, q_question = self.get_coattention(question_features, img_features)

        ##Attention
        ##Concate and MLP
        h_word = self.final_linear_1(q_word + v_word)
        h_phrase = self.final_linear_2(torch.cat(((q_phrase + v_phrase),h_word), dim = 1))
        h_question = self.final_linear_3(torch.cat(((q_question + v_question),h_phrase), dim=1))

        output = self.output_layer(h_question)

        return output







        ############ 
        # raise NotImplementedError()

class AttentionNet(nn.Module):
    def __init__(self, emb_len, hidden_len):
        super().__init__()
        self.process_x = nn.Linear(emb_len, hidden_len)
        self.process_g = nn.Linear(emb_len, hidden_len)
        self.activation = nn.Sequential(nn.Tanh(),
                                        nn.Dropout(0.5))
        self.get_att_weight = nn.Sequential(nn.Linear(hidden_len, 1),
                                            nn.Softmax(dim=1))
    def forward(self, X, g=None):
        if g is None:
            H = self.process_x(X)
        else:
            H = self.process_x(X) + self.process_g(g)
        H = self.activation(H)
        attention_weight = self.get_att_weight(H)
        attented_feature = torch.sum(attention_weight * X, dim=1, keepdim=True)
        return attented_feature # B x 1 x emb_len