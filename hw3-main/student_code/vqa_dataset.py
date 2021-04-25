from torch.utils.data import Dataset
from external.vqa.vqa import VQA
import re
import os
from PIL import Image
import numpy as np
import torch
import torchvision

class VqaDataset(Dataset):
    """
    Load the VQA dataset using the VQA python API. We provide the necessary subset in the External folder, but you may
    want to reference the full repo (https://github.com/GT-Vision-Lab/VQA) for usage examples.
    """

    def __init__(self, image_dir, question_json_file_path, annotation_json_file_path, image_filename_pattern,
                 transform=None, question_word_to_id_map=None, answer_to_id_map=None, question_word_list_length=5746, answer_list_length=5216,
                 pre_encoder=None, cache_location=None):
        """
        Args:
            image_dir (string): Path to the directory with COCO images
            question_json_file_path (string): Path to the json file containing the question data
            annotation_json_file_path (string): Path to the json file containing the annotations mapping images, questions, and
                answers together
            image_filename_pattern (string): The pattern the filenames of images in this dataset use (eg "COCO_train2014_{}.jpg")
        """
        self._vqa = VQA(annotation_file=annotation_json_file_path, question_file=question_json_file_path)
        self._image_dir = image_dir
        self._image_filename_pattern = image_filename_pattern
        self._transform = transform
        self._max_question_length = 26

        # Publicly accessible dataset parameters
        self.question_id_list = self._vqa.getQuesIds()
        self.question_word_list_length = question_word_list_length + 1
        self.unknown_question_word_index = question_word_list_length
        self.answer_list_length = answer_list_length + 1
        self.unknown_answer_index = answer_list_length
        self._pre_encoder = pre_encoder
        self._cache_location = cache_location
        if self._cache_location is not None:
            try:
                os.makedirs(self._cache_location)
            except OSError:
                pass

        # Create the question map if necessary
        if question_word_to_id_map is None:
            ############ 1.6 TODO
            question_word_list = []
            for q in self._vqa.questions['questions']:
                question_word_list.append(q['question'])
            question_word_list = self._create_word_list(question_word_list)
            self.question_word_to_id_map = self._create_id_map(question_word_list, question_word_list_length)
            ############
            # raise NotImplementedError()
        else:
            self.question_word_to_id_map = question_word_to_id_map

        # Create the answer map if necessary
        if answer_to_id_map is None:
            ############ 1.7 TODO
            ans_list = []
            for ann in self._vqa.dataset['annotations']:
                for ans in ann['answers']:
                    sentence = ans['answer']
                    sentence = sentence.lower()
                    #treat the answer sentence as a unique word
                    # sentence = ''.join(re.split(r'\W+', sentence))
                    ans_list.append(sentence)

            self.answer_to_id_map = self._create_id_map(ans_list, answer_list_length)

            ############
            # raise NotImplementedError()
        else:
            self.answer_to_id_map = answer_to_id_map


    def _create_word_list(self, sentences):
        """
        Turn a list of sentences into a list of processed words (no punctuation, lowercase, etc)
        Args:
            sentences: a list of str, sentences to be splitted into words
        Return:
            A list of str, words from the split, order remained.
        """

        ############ 1.4 TODO
        out = []
        # punctuations =set(string.punctuation)
        for s in sentences:
            s = s.lower()
            s = re.split(r'\W+', s)
            out.extend([vocab for vocab in s if vocab is not ''])

        return out
        ############
        # raise NotImplementedError()


    def _create_id_map(self, word_list, max_list_length):
        """
        Find the most common str in a list, then create a map from str to id (its rank in the frequency)
        Args:
            word_list: a list of str, where the most frequent elements are picked out
            max_list_length: the number of strs picked
        Return:
            A map (dict) from str to id (rank)
        """

        ############ 1.5 TODO
        str_to_freq = {}
        str_to_rank = {}
        for _str in word_list:
            if _str not in str_to_freq:
                str_to_freq[_str] = 1
            else:
                str_to_freq[_str] += 1
        sorted_list = sorted(str_to_freq.items(), key = lambda x: x[1], reverse = True)
        for i in range(max_list_length):
            str_to_rank[sorted_list[i]] = i

        return str_to_rank
        ############
        # raise NotImplementedError()


    def __len__(self):
        ############ 1.8 TODO
        return len(self._vqa.questions['questions'])

        ############
        # raise NotImplementedError()

    def __getitem__(self, idx):
        """
        Load an item of the dataset
        Args:
            idx: index of the data item
        Return:
            A dict containing multiple torch tensors for image, question and answers.
        """

        ############ 1.9 TODO
        # figure out the idx-th item of dataset from the VQA API
        question_id = self.question_id_list[idx]
        img_id = self._vqa.qa[question_id]['image_id']

        ############

        if self._cache_location is not None and self._pre_encoder is not None:
            ############ 3.2 TODO
            # implement your caching and loading logic here

            ############
            raise NotImplementedError()
        else:
            ############ 1.9 TODO
            # load the image from disk, apply self._transform (if not None)
            img_file = self._image_filename_pattern.format('%012d'%(img_id))
            img = Image.open(os.path.join(self._image_dir, img_file))
            if not self._transform:
                img = self._transform(img)
            else:
                img = torchvision.tranforms.ToTensor()(img)
            ############
            # raise NotImplementedError()

        ############ 1.9 TODO
        # load and encode the question and answers, convert to torch tensors
        #encode question
        q_encoding = np.zeros((self._max_question_length, self.question_word_list_length))
        question = self._vqa.qqa[question_id]['question']
        question_list = self._create_word_list([question])
        for i, word in enumerate(question_list):
            if i >= self._max_question_length:
                break
            if word in self.question_word_to_id_map:
                q_encoding[i][self.question_word_to_id_map[word]] = 1
            else:
                q_encoding[i][-1] = 1

        #encode answer
        a_encoding = np.zeros((10, self.answer_list_length))
        answers = self._vqa.qa[question_id]['answers']
        answer_list = []
        for ans in answers:
            sentence = ans['answer']
            sentence = sentence.lower()
            # treat the answer sentence as a unique word
            # sentence = ''.join(re.split(r'\W+', sentence))
            answer_list.append(sentence)

        for i, word in enumerate(answer_list):
            if word in self.answer_to_id_map:
                a_encoding[i][self.answer_to_id_map[word]] = 1
            else:
                a_encoding[i][-1] = 1

        res = {}
        res['img'] = img
        res['question'] = torch.tensor(q_encoding)
        res['answer'] = torch.tensor(a_encoding)

        return res
        ############
        # raise NotImplementedError()
