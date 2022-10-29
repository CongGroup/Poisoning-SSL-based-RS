from __future__ import print_function
from math import ceil
from matplotlib import use
import numpy as np
import sys
import pdb
import random
import os
import torch
import torch.optim as optim
import torch.nn as nn

import generator
import discriminator
import helpers
import matplotlib.pyplot as plt

from classify import Classify
#from log import Logger
from dataloader import GenDataset, DisDataset, ClaDataset
from torch.utils.data import Dataset, DataLoader

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

d_emb_dim = 64
d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
d_dropout = 0.75
d_num_class = 2

def padding(user_seq, max_seq_length):
    user_seqs = []
    for s in user_seq:
        pad_len = max_seq_length - len(s)
        s = s + [0] * pad_len
        s = s[:max_seq_length]
        user_seqs.append(s)
    return user_seqs

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
        fake_seqs.append(seq)
        id += 1
    return fake_seqs

def read_file(data_file):
    with open(data_file, 'r') as f:
        lines = f.readlines()
    lis = []
    for line in lines:
        l = line.strip().split(' ')
        l = [int(s) for s in l]
        lis.append(l)
    return lis

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

def train_generator_MLE(gen, gen_opt, real_data_samples, epochs):
    """
    Max Likelihood Pretraining for the generator
    """
    train_len = len(real_data_samples)
    
    for epoch in range(epochs):
        print('epoch %d : ' % (epoch + 1), end='')
        sys.stdout.flush()
        total_loss = 0

        for i in range(0, train_len, BATCH_SIZE):
            inp, target = helpers.prepare_generator_batch(real_data_samples[i:i + BATCH_SIZE], start_letter=START_LETTER,
                                                          gpu=CUDA)
            gen_opt.zero_grad()
            loss = gen.batchNLLLoss(inp, target)
            loss.backward()
            gen_opt.step()

            total_loss += loss.data.item()

        # each loss in a batch is loss per sample
        total_loss = total_loss / ceil(train_len / float(BATCH_SIZE)) / MAX_SEQ_LEN
        
        print(' average_train_NLL = %.4f' % (total_loss))

def eval_generator(gen, eval_samples):
    oracle_loss = 0
    eval_len = len(eval_samples)
    for i in range(0, eval_len, BATCH_SIZE):
        inp, target = helpers.prepare_generator_batch(eval_samples[i:i + BATCH_SIZE], start_letter=START_LETTER,
                                                        gpu=CUDA)
        loss = gen.batchNLLLoss(inp, target)

        oracle_loss += loss.data.item()

        if (i / BATCH_SIZE) % ceil(
                        ceil(eval_len / float(BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
            print('.', end='')
            sys.stdout.flush()

        # each loss in a batch is loss per sample
    oracle_loss = oracle_loss / ceil(eval_len / float(BATCH_SIZE)) / MAX_SEQ_LEN
    return oracle_loss

def train_generator_PG(gen, gen_opt, bi_classify, train_samples, dis, num_batches, attack_item, mask_id):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for num_batches batches.
    """
    
    for batch in range(num_batches):
        s = gen.sample(BATCH_SIZE*2)        # 64 works best
        inp, target = helpers.prepare_generator_batch(s, start_letter=START_LETTER, gpu=CUDA)
        rewards = dis.batchClassify(target)

        loss_a = gen.batchLoss_A(inp, target, attack_item=attack_item)
        loss_b = 0
        dataset = ClaDataset(MAX_SEQ_LEN, train_samples, s, mask_id=mask_id)
        dataloader = DataLoader(dataset, batch_size=64)
        for data, label in dataloader:
            loss_b = gen.batchLoss_B(inp, target, data, label, bi_classify)

        gen_opt.zero_grad()
        loss_c = gen.batchPGLoss(inp, target, rewards)
        loss = c*loss_c + b*loss_b + a*loss_a

        loss.backward()
        gen_opt.step()

        print("training PG total_loss = %.4f, loss_a = %.4f, loss_b = %.4f, loss_c = %.4f" % (loss,loss_a,loss_b,loss_c))
        return loss, loss_a, loss_b, loss_c

    # sample from generator and compute oracle NLL
    # train_loss = eval_generator(gen, train_samples)
    # eval_loss = eval_generator(gen, eval_samples)
    # print(' train_sample_NLL = %.4f, eval_sample_NLL = %.4f' % (train_loss,eval_loss))


def train_discriminator(discriminator, dis_opt, real_data_samples, generator, d_steps, epochs):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """
    # generating a small validation set before training (using oracle and generator)
    for d_step in range(d_steps):
        s = helpers.batchwise_sample(generator, POS_NEG_SAMPLES, BATCH_SIZE)
        dis_inp, dis_target = helpers.prepare_discriminator_data(real_data_samples, s, gpu=CUDA)
        for epoch in range(epochs):
            print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')
            sys.stdout.flush()
            total_loss = 0
            total_acc = 0

            for i in range(0, len(s)+len(real_data_samples), BATCH_SIZE):
                inp, target = dis_inp[i:i + BATCH_SIZE], dis_target[i:i + BATCH_SIZE]
                dis_opt.zero_grad()
                out = discriminator.batchClassify(inp)
                loss_fn = nn.BCELoss()
                loss = loss_fn(out, target)
                loss.backward()
                dis_opt.step()

                total_loss += loss.data.item()
                total_acc += torch.sum((out>0.5)==(target>0.5)).data.item()

                if (i / BATCH_SIZE) % ceil(ceil((len(s)+len(real_data_samples)) / float(
                        BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                    print('.', end='')
                    sys.stdout.flush()

            total_loss /= ceil((len(s)+len(real_data_samples)) / float(BATCH_SIZE))
            total_acc /= float(len(s)+len(real_data_samples))

            print(' average_loss = %.4f, train_acc = %.4f' % (total_loss, total_acc))

def get_attack_item(dataset):
    if dataset == "Beauty":
        return 8887
    elif dataset == "Toys_and_Games":
        return 6662
    elif dataset == "Sports_and_Outdoors":
        return 7775
    elif dataset == "Yelp":
        return 5556
    
def main():
    #oracle = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
    #oracle.load_state_dict(torch.load(oracle_state_dict_path))
    dataset = "Yelp"
    attack_item = get_attack_item(dataset)
    dataset_path = os.path.join("./dataset", dataset) + ".txt"

    user_seq, max_item, long_sequence = get_user_seqs_long(dataset_path)
    mask_id = max_item + 1
    user_seq = padding(user_seq, MAX_SEQ_LEN)
    train_samples = torch.Tensor(user_seq).type(torch.LongTensor)
    
    # a new oracle can be generated by passing oracle_init=True in the generator constructor
    # samples for the new oracle can be generated using helpers.batchwise_sample()

    gen = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, max_item+1, MAX_SEQ_LEN, gpu=CUDA)
    dis = discriminator.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, max_item+1, MAX_SEQ_LEN, gpu=CUDA)

    bi_classify = Classify(d_num_class, max_item+2, d_emb_dim, d_filter_sizes, d_num_filters, d_dropout)
    bi_classify.load_state_dict(torch.load(dataset + "_bi_classify.pt"))

    if CUDA:
        #oracle = oracle.cuda()
        gen = gen.cuda()
        dis = dis.cuda()
        train_samples = train_samples.cuda()

    # GENERATOR MLE TRAINING
    print('Starting Generator MLE Training...')
    gen_optimizer = optim.Adam(gen.parameters())
    train_generator_MLE(gen, gen_optimizer, train_samples, MLE_TRAIN_EPOCHS)

    # torch.save(gen.state_dict(), pretrained_gen_path)
    # gen.load_state_dict(torch.load(pretrained_gen_path))

    # PRETRAIN DISCRIMINATOR
    print('\nStarting Discriminator Training...')
    dis_optimizer = optim.Adagrad(dis.parameters())
    train_discriminator(dis, dis_optimizer, train_samples, gen, 20, 3)

    # torch.save(dis.state_dict(), pretrained_dis_path)
    # dis.load_state_dict(torch.load(pretrained_dis_path))

    # ADVERSARIAL TRAINING
    print('\nStarting Adversarial Training...')

    loss_A = []
    loss_B = []
    loss_C = []
    total_loss = []
    
    output_path = os.path.join("./output", dataset)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for epoch in range(ADV_TRAIN_EPOCHS):
        print('\n--------\nEPOCH %d\n--------' % (epoch+1))
        # TRAIN GENERATOR
        print('\nAdversarial Training Generator : ', end='')
        sys.stdout.flush()
        loss, loss_a, loss_b, loss_c = train_generator_PG(gen, gen_optimizer, bi_classify, train_samples, dis, 1, attack_item=attack_item, mask_id=mask_id)
        total_loss.append(loss.cpu().detach().numpy())
        loss_A.append(loss_a.cpu().detach().numpy())
        loss_B.append(loss_b.cpu().detach().numpy())
        loss_C.append(loss_c.cpu().detach().numpy())

        if (epoch+1) % 10 == 0:
            torch.save(gen.cpu().state_dict(), os.path.join(output_path, f'attack-generator-epochs-{epoch+1}.pt'))
            gen.cuda()
            datas = gen.sample(int(len(user_seq) * 0.01)).cpu().data.numpy().tolist()
            datas = unpadding(datas, len(user_seq)+1)
            with open(os.path.join(output_path, f"generate_epoch{epoch+1}_data.txt"), 'w') as fout:
                for sample in datas:
                    string = ' '.join([str(s) for s in sample])
                    fout.write('%s\n' % string)

        if epoch in range(70,80):
            torch.save(gen.cpu().state_dict(), os.path.join(output_path, f'attack-generator-epochs-{epoch+1}.pt'))
            gen.cuda()
            datas = gen.sample(int(len(user_seq) * 0.01)).cpu().data.numpy().tolist()
            datas = unpadding(datas, len(user_seq)+1)
            with open(os.path.join(output_path, f"generate_epoch{epoch+1}_data.txt"), 'w') as fout:
                for sample in datas:
                    string = ' '.join([str(s) for s in sample])
                    fout.write('%s\n' % string)

        # TRAIN DISCRIMINATOR
        print('\nAdversarial Training Discriminator : ')
        train_discriminator(dis, dis_optimizer, train_samples, gen, 3, 2)


    plt.title(dataset + '_training loss')
    plt.plot(np.arange(len(total_loss)), total_loss, label="total loss")

    plt.plot(np.arange(len(loss_A)), loss_A, label="loss A")

    plt.plot(np.arange(len(loss_B)), loss_B, label="loss B")

    plt.plot(np.arange(len(loss_C)), loss_C, label="loss C")
    plt.legend(loc='upper right')

    plt.savefig(dataset + "_loss.png")

def main_():
    #oracle = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
    #oracle.load_state_dict(torch.load(oracle_state_dict_path))
    dataset = "Beauty"
    attack_item = get_attack_item(dataset)
    dataset_path = os.path.join("./dataset", dataset) + ".txt"

    user_seq, max_item, long_sequence = get_user_seqs_long(dataset_path)
    mask_id = max_item + 1
    user_seq = padding(user_seq, MAX_SEQ_LEN)
    train_samples = torch.Tensor(user_seq).type(torch.LongTensor)
    
    # a new oracle can be generated by passing oracle_init=True in the generator constructor
    # samples for the new oracle can be generated using helpers.batchwise_sample()

    gen = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, max_item+1, MAX_SEQ_LEN, gpu=CUDA)
    gen.load_state_dict(torch.load(f"./output/{dataset}/attack-generator-epochs-60.pt"))
    dis = discriminator.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, max_item+1, MAX_SEQ_LEN, gpu=CUDA)

    bi_classify = Classify(d_num_class, max_item+2, d_emb_dim, d_filter_sizes, d_num_filters, d_dropout)
    bi_classify.load_state_dict(torch.load(dataset + "_bi_classify.pt"))

    if CUDA:
        #oracle = oracle.cuda()
        gen = gen.cuda()
        dis = dis.cuda()
        train_samples = train_samples.cuda()

    # GENERATOR MLE TRAINING
    gen_optimizer = optim.Adam(gen.parameters())


    # torch.save(gen.state_dict(), pretrained_gen_path)
    # gen.load_state_dict(torch.load(pretrained_gen_path))

    # PRETRAIN DISCRIMINATOR
    print('\nStarting Discriminator Training...')
    dis_optimizer = optim.Adagrad(dis.parameters())
    train_discriminator(dis, dis_optimizer, train_samples, gen, 20, 3)


    # torch.save(dis.state_dict(), pretrained_dis_path)
    # dis.load_state_dict(torch.load(pretrained_dis_path))

    # ADVERSARIAL TRAINING
    print('\nStarting Adversarial Training...')

    loss_A = []
    loss_B = []
    loss_C = []
    total_loss = []
    
    output_path = os.path.join("./output", dataset)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for epoch in range(60,70):
        print('\n--------\nEPOCH %d\n--------' % (epoch+1))
        # TRAIN GENERATOR
        print('\nAdversarial Training Generator : ', end='')
        sys.stdout.flush()
        loss, loss_a, loss_b, loss_c = train_generator_PG(gen, gen_optimizer, bi_classify, train_samples, dis, 1, attack_item=attack_item, mask_id=mask_id)
        total_loss.append(loss.cpu().detach().numpy())
        loss_A.append(loss_a.cpu().detach().numpy())
        loss_B.append(loss_b.cpu().detach().numpy())
        loss_C.append(loss_c.cpu().detach().numpy())

        
        torch.save(gen.cpu().state_dict(), os.path.join(output_path, f'attack-generator-epochs-{epoch+1}.pt'))
        gen.cuda()
        datas = gen.sample(int(len(user_seq) * 0.01)).cpu().data.numpy().tolist()
        datas = unpadding(datas, len(user_seq)+1)
        with open(os.path.join(output_path, f"generate_epoch{epoch+1}_data.txt"), 'w') as fout:
            for sample in datas:
                string = ' '.join([str(s) for s in sample])
                fout.write('%s\n' % string)

        # TRAIN DISCRIMINATOR
        print('\nAdversarial Training Discriminator : ')
        train_discriminator(dis, dis_optimizer, train_samples, gen, 3, 2)



# MAIN
if __name__ == '__main__':
    main_()