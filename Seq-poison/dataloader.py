# -*- coding:utf-8 -*-

import os
import random
import math
from cv2 import randn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

class GenDataset(Dataset):
    """ Toy data iter to load digits"""
    def __init__(self, max_seq_length, user_seq, mode="train"):
        super(GenDataset, self).__init__()
        self.max_seq_length = max_seq_length
        self.user_seq = self.padding(user_seq)
        
        if mode == 'train':
            self.user_seq = self.user_seq[:int(0.9*len(self.user_seq))] 
        else:
            self.user_seq = self.user_seq[int(0.9*len(self.user_seq)):] 

    def padding(self, user_seq):
        user_seqs = []
        for s in user_seq:
            pad_len = self.max_seq_length - len(s)
            s = s + [0] * pad_len
            user_seqs.append(s)
        return user_seqs
    
    def __len__(self):
        return len(self.user_seq)
    
    def __getitem__(self, index):
        item = self.user_seq[index]
        label = torch.LongTensor(np.array(item,dtype="int64"))
        data = [0] + item[:-1]
        data = torch.LongTensor(np.array(data,dtype="int64"))
        return data, label
        
        

class DisDataset(Dataset):
    """ Toy data iter to load digits"""
    def __init__(self, real_data, fake_data, max_seq_length):
        super(DisDataset, self).__init__()
        self.max_seq_length = max_seq_length
        self.data = self.padding(real_data) + fake_data
        self.labels = [1 for _ in range(len(real_data))] +\
                        [0 for _ in range(len(fake_data))]
        self.pairs = list(zip(self.data, self.labels))
        
    def padding(self, user_seq):
        user_seqs = []
        for s in user_seq:
            pad_len = self.max_seq_length - len(s)
            s = s + [0] * pad_len
            user_seqs.append(s)
        return user_seqs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        data = torch.LongTensor(np.array(pair[0],dtype="int64"))
        label = torch.LongTensor([pair[1]])
        return data, label


class ClaDataset(Dataset):
    def __init__(self, max_seq_length, real_seq, fake_seq, mask_id):
        self.real_seq = self.unpadding(real_seq)
        self.fake_seq = self.unpadding(fake_seq)
        self.masked_segment_sequence = []
        self.anti_masked_segment_sequence = []
        self.data_pairs = [] #数据对
        self.max_len = max_seq_length
        self.mask_id = mask_id

        self.mask_sequence()
        self.get_pos_neg_pairs()

    def unpadding(self, fake_seq):
        #tensor to list
        fake_seq = fake_seq.cpu().data.numpy().tolist()
        fake_seqs = []
        for fake_data in fake_seq:
            seq = []
            for i, f in enumerate(fake_data):
                if f != 0:
                    seq.append(f)
                else:
                    # 如果是连续出现两个0，则认为可以进行裁剪
                    if i == (len(fake_data) - 1) or fake_data[i+1] == 0:
                        break
                    else:
                        continue
            fake_seqs.append(seq)
 
        return fake_seqs
    
    def mask_sequence(self):
        '''对user_seq进行遮罩操作并做padding处理'''
        for fake_data in self.fake_seq:
            masked_segment_sequence = []
            anti_masked_segment_sequence = []
            # Masked Item Prediction
            if len(fake_data) < 2:
                masked_segment_sequence = fake_data
                anti_masked_segment_sequence = [self.mask_id] * len(fake_data)
            else:
                real_sample = self.real_seq[random.randint(0, len(self.real_seq)-1)]
                min_len = len(fake_data) if len(fake_data)<len(real_sample) else len(real_sample)
                
                sample_length = random.randint(1, min_len // 2)
                start_id = random.randint(0, min_len - sample_length)
                masked_segment_sequence =  [self.mask_id] * len(real_sample[:start_id]) + fake_data[start_id:start_id+sample_length] +\
                                                    [self.mask_id] * len(real_sample[start_id + sample_length:])
                
                anti_masked_segment_sequence = real_sample[:start_id] + [self.mask_id] * sample_length + \
                                                    real_sample[start_id + sample_length:]


            # padding sequence
            pad_len_masked = self.max_len - len(masked_segment_sequence)
            pad_len_anti_masked = self.max_len - len(anti_masked_segment_sequence)
            masked_segment_sequence = masked_segment_sequence + [0] * pad_len_masked
            anti_masked_segment_sequence = anti_masked_segment_sequence + [0] * pad_len_anti_masked


            masked_segment_sequence = masked_segment_sequence[:self.max_len]
            anti_masked_segment_sequence = anti_masked_segment_sequence[:self.max_len]
            
            self.masked_segment_sequence.append(masked_segment_sequence)
            self.anti_masked_segment_sequence.append(anti_masked_segment_sequence)

    def get_pos_neg_pairs(self):
        '''根据mask后的数据进行两两随机重组,并打标'''
        for masked, anti_masked in zip(self.masked_segment_sequence,self.anti_masked_segment_sequence):
            self.data_pairs.append([masked,anti_masked,0])


    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, index):
        '''根据重组结果返回数据'''
        data = torch.cat((torch.tensor(self.data_pairs[index][0],dtype=torch.long),torch.tensor(self.data_pairs[index][1],dtype=torch.long)),dim=0)
        target = torch.tensor(self.data_pairs[index][2],dtype=torch.long)
        return data, target
