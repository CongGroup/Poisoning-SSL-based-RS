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

torch.cuda.set_device(0)

def unpadding(fake_seq, start_id, target_item):
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
                # clip when there are two consecutive "0"
                if i == (len(fake_data) - 1) or fake_data[i+1] == 0:
                    break
                else:
                    continue
        if len(seq) == 1 or len(seq) == 2 or target_item not in seq:
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
        long_sequence.extend(items) # negative samples
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    return user_seq, max_item, long_sequence

def generate_data():

    dataset = "Yelp"
    dataset_path = os.path.join("./dataset", dataset) + ".txt"
    percentage = 0.01
    user_seq, max_item, _ = get_user_seqs_long(dataset_path)
    data_num = len(user_seq)
    start_id = len(user_seq) + 1
    
    gen = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, max_item+1, MAX_SEQ_LEN, gpu=CUDA).cuda()
    
    i = 79  #epoch
    
    gen.load_state_dict(torch.load(f"./output/{dataset}/output/attack-generator-epochs-{i}.pt"))
    datas = []

    while True:
        data = gen.sample(10).cpu().data.numpy().tolist()
        data = unpadding(data,start_id=start_id, target_item=5556) #target item
        datas.extend(data)
        start_id = start_id + len(data)
        if len(datas) > int(data_num * percentage):
            break

    with open(f"./output/{dataset}/generate_epoch{i}_data_{str(percentage)}.txt", 'w') as fout:
        for sample in datas:
            string = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % string)


if __name__ == "__main__":
    generate_data()
