from classify import Classify
from torch.utils.data import Dataset, DataLoader
import random
import torch

max_seq_len = 50

d_emb_dim = 64
d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]

d_dropout = 0.75
d_num_class = 2


class PretrainDataset(Dataset):
    def __init__(self, max_seq_length, user_seq, long_sequence, mask_p, mask_id, mode):
        self.user_seq = user_seq
        self.masked_segment_sequence = []
        self.anti_masked_segment_sequence = []
        self.data_pairs = [] # data pair
        self.long_sequence = long_sequence
        self.max_len = max_seq_length
        self.mask_id = mask_id
        self.mask_p = mask_p
        self.mask_sequence()
        self.get_pos_neg_pairs()

        if mode == 'train':
            self.data_pairs = self.data_pairs[:int(0.9*len(self.data_pairs)/2)] + \
                self.data_pairs[int(len(self.data_pairs)/2):int(len(self.data_pairs)/2) + int(0.9*len(self.data_pairs)/2)]
        else:
            self.data_pairs = self.data_pairs[int(0.9*len(self.data_pairs)/2):int(len(self.data_pairs)/2)] + \
                self.data_pairs[int(len(self.data_pairs)/2) + int(0.9*len(self.data_pairs)/2):]


    def mask_sequence(self):
        """
        mask user_seq and do padding
        """
        for seq in self.user_seq:
            masked_segment_sequence = []
            anti_masked_segment_sequence = []

            # Masked Item Prediction
            # for item in s:
            #     prob = random.random()
            #     if prob < self.mask_p:
            #         masked_segment_sequence.append(self.mask_id)
            #         anti_masked_segment_sequence.append(item)
            #     else:
            #         masked_segment_sequence.append(item)
            #         anti_masked_segment_sequence.append(self.mask_id)
            # Segment Prediction
            if len(seq) < 2:
                masked_segment_sequence = seq
                anti_masked_segment_sequence = [self.mask_id] * len(seq)
            else:
                sample_length = random.randint(1, len(seq) // 2)
                start_id = random.randint(0, len(seq) - sample_length)
                masked_segment_sequence = seq[:start_id] + [self.mask_id] * sample_length + seq[start_id + sample_length:]
                anti_masked_segment_sequence = [self.mask_id] * len(seq[:start_id]) + seq[start_id:start_id+sample_length] + \
                                                    [self.mask_id] * len(seq[start_id + sample_length:])

            # padding sequence
            pad_len = self.max_len - len(seq)
            masked_segment_sequence = masked_segment_sequence + [0] * pad_len
            anti_masked_segment_sequence = anti_masked_segment_sequence + [0] * pad_len

            masked_segment_sequence = masked_segment_sequence[:self.max_len]
            anti_masked_segment_sequence = anti_masked_segment_sequence[:self.max_len]
            
            self.masked_segment_sequence.append(masked_segment_sequence)
            self.anti_masked_segment_sequence.append(anti_masked_segment_sequence)

    def get_pos_neg_pairs(self):
        """
        Randomly reorganize the masked data pairs and label them
        """
        for masked, anti_masked in zip(self.masked_segment_sequence,self.anti_masked_segment_sequence):
            self.data_pairs.append([masked,anti_masked,1])
        
        for masked in self.masked_segment_sequence:
            index = self.masked_segment_sequence.index(masked)
            item = random.randint(0, len(self.masked_segment_sequence)-1)
            while item == index:
                item = random.randint(0, len(self.masked_segment_sequence)-1)
            self.data_pairs.append([masked,self.anti_masked_segment_sequence[item],0])

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, index):
        """
        return data pairs
        """
        data = torch.cat((torch.tensor(self.data_pairs[index][0],dtype=torch.long),torch.tensor(self.data_pairs[index][1],dtype=torch.long)),dim=0)
        target = torch.tensor(self.data_pairs[index][2],dtype=torch.long)
        return data, target

def get_user_seqs_long(data_file):
    lines = open(data_file).readlines()
    user_seq = []
    long_sequence = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        long_sequence.extend(items) # negative samples
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    return user_seq, max_item, long_sequence

def main():
    device = torch.device("cuda:0")
    #dataset
    dataset = "Beauty"
    dataset_path = "./dataset/"+ dataset + ".txt"
    user_seq, max_item, long_sequence = get_user_seqs_long(dataset_path)
    mask_id  = max_item + 1
    mask_p = 0.2
    model = Classify(d_num_class,max_item+2,d_emb_dim,d_filter_sizes,d_num_filters,d_dropout)
    model.to(device)
    criterion = torch.nn.NLLLoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    pretrain_dataset = PretrainDataset(max_seq_len, user_seq, long_sequence, mask_p, mask_id, mode="train")
    eval_dataset = PretrainDataset(max_seq_len, user_seq, long_sequence, mask_p, mask_id, mode="eval")
    pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=64, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=64, shuffle=False)
    """
    start training bi-classifier
    """
    print("Training bi-classifier...'")
    epochs = 20
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0

        eval_loss = 0
        eval_acc = 0
        for mask_sequecne, target in pretrain_dataloader:
            mask_sequecne = mask_sequecne.to(device)
            target = target.to(device)
            output = model(mask_sequecne)
            loss = criterion(output,target)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _, pred = output.max(1)
            num_correct = (pred == target).sum().item()
            acc = num_correct / mask_sequecne.size(0)
            train_acc += acc

        for mask_sequecne, target in eval_dataloader:
            mask_sequecne = mask_sequecne.to(device)
            target = target.to(device)
            output = model(mask_sequecne)
            loss = criterion(output,target)
            eval_loss += loss.item()
            
            _, pred = output.max(1)
            num_correct = (pred == target).sum().item()
            acc = num_correct / mask_sequecne.size(0)
            eval_acc += acc
        print(f"epoch:{epoch}, train_loss:{train_loss/len(pretrain_dataloader)},train_acc:{train_acc/len(pretrain_dataloader)},\
            eval_loss:{eval_loss/len(eval_dataloader)},eval_acc:{eval_acc/len(eval_dataloader)}")
    
    torch.save(model.state_dict(), f"{dataset}_bi_classify.pt")


if __name__ == "__main__":
    main()
