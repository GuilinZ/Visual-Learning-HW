from torch.utils.data import DataLoader
import torch
import random
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
class ExperimentRunnerBase(object):
    """
    This base class contains the simple train and validation loops for your VQA experiments.
    Anything specific to a particular experiment (Simple or Coattention) should go in the corresponding subclass.
    """

    def __init__(self, train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers=10, log_validation=False, writer=None):
        self._model = model
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._log_freq = 10  # Steps
        self._test_freq = 100  # Steps
        self._writer = writer
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset


        self._train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_data_loader_workers)

        # If you want to, you can shuffle the validation dataset and only use a subset of it to speed up debugging
        self._val_dataset_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_data_loader_workers)

        # Use the GPU if it's available.
        self._cuda = torch.cuda.is_available()

        if self._cuda:
            self._model = self._model.cuda()
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self._log_validation = log_validation

    def _optimize(self, predicted_answers, true_answers):
        """
        This gets implemented in the subclasses. Don't implement this here.
        """
        raise NotImplementedError()

    def validate(self,current_step, epoch):
        ############ 2.8 TODO
        # Should return your validation accuracy
        num_of_batches = 20
        val_loss_total = 0
        val_acc_total = 0
        chosen_batches = [random.randint(0,100) for _ in range(20)]
        for batch_id, batch_data in enumerate(self._val_dataset_loader):
            if batch_id > 100:
                break
            if batch_id not in chosen_batches:
                continue

            img = batch_data['img'].to(self.device)
            q_id = batch_data['question_id'][0].item()
            questions = batch_data['question'].to(self.device)
            predicted_answer = self._model(img, questions)  # Bx5217 TODO

            answers = batch_data['answer']  # Bx10x5217
            ground_truth_answer = self.majority_vote(answers).to(self.device)  # Bx5217 TODO
            loss = self._optimize(predicted_answer, ground_truth_answer).cpu().item()
            val_loss_total += loss

            _, f_prediction = torch.max(predicted_answer, dim=1)
            # _, f_answer = torch.max(ground_truth_answer, dim=1)
            acc = (f_prediction == ground_truth_answer).float().cpu().mean().item()
            val_acc_total += acc
        ############
        val_loss = val_loss_total / num_of_batches
        val_acc = val_acc_total / num_of_batches
        print('validation loss: ', val_loss)
        if self._log_validation:
            ############ 2.9 TODO
            # you probably want to plot something here
            self._writer.add_scalar('Validation Loss', val_loss, current_step)

            #====================================
            #Validation Image Visualization
            #Remove comments for visualizing baseline validation images
            # log_img = img.cpu()[0]
            # invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
            #                                                     std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            #                                transforms.Normalize(mean=[-0.485, -0.456, -0.406],
            #                                                     std=[1., 1., 1.]),
            #                                # transforms.ToPILImage()
            #                                ])
            # log_img = invTrans(log_img)
            #====================================
            # plt.imshow(log_img)
            # plt.show()
            q_map = self._train_dataset.id_to_question_word_map
            a_map = self._train_dataset.id_to_answer_map

            log_question = self._val_dataset._vqa.qqa[q_id]['question']

            # _, questions = torch.max(questions.cpu()[0], dim=1)
            # log_question = [q_map[idx.item()] if idx.item() in q_map else '' for idx in questions]
            # log_question = ' '.join(log_question)

            f_prediction = f_prediction[0]
            log_prediction = (a_map[f_prediction.item()] if f_prediction.item() in a_map else 'unknown')

            gt = ground_truth_answer[0]
            log_gt = (a_map[gt.item()] if gt.item() in a_map else 'unknown')
            self._writer.add_text('Step%d Ground Truth'%current_step, log_gt)
            self._writer.add_text('Step%d Prediction'%current_step, log_prediction)
            self._writer.add_text('Step%d Question'%current_step, log_question)
            # self._writer.add_image('Step%d Image'%current_step, log_img)
        return val_acc
            ############
        # raise NotImplementedError()

    def majority_vote(self, answers):
        answers = torch.sum(answers, dim=1)
        # voted_ans = torch.zeros((answers.shape))
        pos = torch.argmax(answers, dim=1)
        return pos
        # voted_ans[list(range(self._batch_size)),pos] = 1
        # return voted_ans #B x 5217
    def train(self):

        for epoch in range(self._num_epochs):
            num_batches = len(self._train_dataset_loader)
            train_acc_total = 0
            train_loss_total = 0
            for batch_id, batch_data in enumerate(self._train_dataset_loader):
                # if batch_id > 10:
                #     break
                self._model.train()  # Set the model to train mode
                current_step = epoch * num_batches + batch_id

                ############ 2.6 TODO
                # Run the model and get the ground truth answers that you'll pass to your optimizer
                # This logic should be generic; not specific to either the Simple Baseline or CoAttention.
                img = batch_data['img'].to(self.device)
                questions = batch_data['question'].to(self.device)
                predicted_answer = self._model(img, questions) # Bx5217 TODO

                answers = batch_data['answer'] #Bx10x5217
                ground_truth_answer = self.majority_vote(answers).to(self.device)# Bx5217 TODO

                ############

                # Optimize the model according to the predictions
                loss = self._optimize(predicted_answer, ground_truth_answer).item()
                _, f_prediction = torch.max(predicted_answer, dim=1)
                # _, f_answer = torch.max(ground_truth_answer, dim=1)
                acc = (f_prediction == ground_truth_answer).float().cpu().mean().item()
                train_acc_total += acc
                train_loss_total += loss
                acc = train_acc_total / (batch_id + 1)
                loss = train_loss_total / (batch_id + 1)
                if current_step % self._log_freq == 0:
                    print("Epoch: {}, Batch {}/{} has loss {} and acc {}".format(epoch, batch_id, num_batches, loss, acc))
                    ############ 2.9 TODO
                    # you probably want to plot something here
                    self._writer.add_scalar('Train Loss', loss, current_step)
                    self._writer.add_scalar('Train Acc', acc, current_step)
                    ############

                if current_step % self._test_freq == 0:
                    self._model.eval()
                    val_accuracy = self.validate(current_step, epoch)
                    print("Epoch: {} has val accuracy {}".format(epoch, val_accuracy))
                    ############ 2.9 TODO
                    # you probably want to plot something here
                    self._writer.add_scalar('Validation Accuracy', val_accuracy, current_step)
                    ############
            torch.save(self._model.state_dict(), self.model_path + 'epoch_%d.pth'%(epoch))
            print('epoch %d model saved'%epoch, 'to ', self.model_path)