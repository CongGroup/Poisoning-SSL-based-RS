import torch
import generator
import os

CUDA = True

MAX_SEQ_LEN = 50
START_LETTER = 0
BATCH_SIZE = 32
MLE_TRAIN_EPOCHS = 20
ADV_TRAIN_EPOCHS = 100
POS_NEG_SAMPLES = 10000
a, b, c = 0.2, 0.6, 0.2
GEN_EMBEDDING_DIM = 64
GEN_HIDDEN_DIM = 64
DIS_EMBEDDING_DIM = 64
DIS_HIDDEN_DIM = 64

def unpadding(fake_seq, start_id):
    #tensor to list
    fake_seqs = []
    id = start_id
    for fake_data in fake_seq:
        seq = []
        seq.append(id)
        for i, f in enumerate(fake_data):
            if f != 0:
                seq.append(f)
            else:
                # 如果是连续出现两个0，则认为可以进行裁剪
                if i == (len(fake_data) - 1) or fake_data[i+1] == 0:
                    break
                else:
                    continue
        if len(seq) == 1:
            continue
        fake_seqs.append(seq)
        id += 1
    return fake_seqs

def get_user_seqs_long(data_file):
    lines = open(data_file).readlines()
    user_seq = []
    long_sequence = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        long_sequence.extend(items) # 后面的都是采的负例
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    return user_seq, max_item, long_sequence

def generate_data():

    dataset = "Toys_and_Games"
    dataset_path = os.path.join("./dataset", dataset) + ".txt"
    percentage = 0.03
    user_seq, max_item, _ = get_user_seqs_long(dataset_path)
    data_num = len(user_seq)
    start_id = len(user_seq) + 1
    
    gen = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, max_item+1, MAX_SEQ_LEN, gpu=CUDA).cuda()
    
    i = 78
    gen.load_state_dict(torch.load(f"./output/{dataset}/attack-generator-epochs-{i}.pt"))
    datas = gen.sample(int(data_num * percentage)).cpu().data.numpy().tolist()
    datas = unpadding(datas,start_id=start_id)
    with open(f"./output/{dataset}/generate_epoch{i}_data_{str(percentage)}.txt", 'w') as fout:
        for sample in datas:
            string = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % string)


if __name__ == "__main__":
    generate_data()