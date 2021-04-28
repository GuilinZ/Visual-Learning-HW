from student_code.coattention_net import CoattentionNet
from student_code.experiment_runner_base import ExperimentRunnerBase
from student_code.vqa_dataset import VqaDataset
import torchvision.transforms as transforms
import torchvision.models as models
import torch
import torch.nn as nn
import os

class CoattentionNetExperimentRunner(ExperimentRunnerBase):
    """
    Sets up the Co-Attention model for training. This class is specifically responsible for creating the model and optimizing it.
    """
    def __init__(self, train_image_dir, train_question_path, train_annotation_path,
                 test_image_dir, test_question_path,test_annotation_path, batch_size, num_epochs,
                 num_data_loader_workers, cache_location, lr, log_validation, writer):

        ############ 3.1 TODO: set up transform and image encoder
        transform = transforms.Compose([transforms.Resize((448, 448)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
        image_encoder = models.resnet18(pretrained=True)
        image_encoder = nn.Sequential(*(list(image_encoder.children())[:-2]))
        # image_encoder = image_encoder.cuda()
        image_encoder.eval()
        for p in image_encoder.parameters():
            p.requires_grad = False
        #reference https://stackoverflow.com/questions/52548174/how-to-remove-the-last-fc-layer-from-a-resnet-model-in-pytorch
        ############ 

        question_word_list_length = 5746 + 1
        answer_list_length = 5216 + 1
        emb_len = 256
        question_seq_len = 26
        train_dataset = VqaDataset(image_dir=train_image_dir,
                                   question_json_file_path=train_question_path,
                                   annotation_json_file_path=train_annotation_path,
                                   image_filename_pattern="COCO_train2014_{}.jpg",
                                   transform=transform,
                                   cache_location=os.path.join(cache_location, "tmp_train"),
                                   ############ 3.1 TODO: fill in the arguments
                                   question_word_to_id_map=None,
                                   answer_to_id_map=None,
                                   ############
                                   pre_encoder=image_encoder)
        val_dataset = VqaDataset(image_dir=test_image_dir,
                                 question_json_file_path=test_question_path,
                                 annotation_json_file_path=test_annotation_path,
                                 image_filename_pattern="COCO_val2014_{}.jpg",
                                 transform=transform,
                                 cache_location=os.path.join(cache_location, "tmp_val"),
                                 ############ 3.1 TODO: fill in the arguments
                                 question_word_to_id_map=train_dataset.question_word_to_id_map,
                                 answer_to_id_map=train_dataset.answer_to_id_map,
                                 ############
                                 pre_encoder=image_encoder)

        self._model = CoattentionNet(question_word_list_length, answer_list_length, emb_len, question_seq_len)
        self.model_path = '/home/ubuntu/Visual-Learning-HW/hw3-main/attention_models/'
        super().__init__(train_dataset, val_dataset, self._model, batch_size, num_epochs,
                         num_data_loader_workers=num_data_loader_workers, log_validation=log_validation, writer=writer)

        ############ 3.4 TODO: set up optimizer
        self.optimizer = torch.optim.RMSprop(self._model.parameters(), lr=4e-4, alpha=0.99, eps=1e-8)
        ############ 

    def _optimize(self, predicted_answers, true_answer_ids):
        ############ 3.4 TODO: implement the optimization step
        self.optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss()
        loss = criterion(predicted_answers, true_answer_ids)
        if self._model.training:
            loss.backward()
            self.optimizer.step()
        ############ 
        return loss
