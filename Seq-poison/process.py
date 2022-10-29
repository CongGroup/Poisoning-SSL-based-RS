

from importlib.resources import read_binary


MAX_SEQ_LEN = 204


def read_file(data_file):
    with open(data_file, 'r') as f:
        lines = f.readlines()
    lis = []
    for line in lines:
        l = line.strip().split(' ')
        l = [int(s) for s in l]
        l += [0] * (MAX_SEQ_LEN - len(l))

        lis.append(l)
    with open("_beauty.txt","w") as f:
        for l in lis:
            string = ' '.join([str(s) for s in l])
            f.write('%s\n' % string)

def get_train_eval_dataset(data_file):
    with open(data_file,'r') as f:
        lines = f.readlines()
    lis = []
    for line in lines:
        l = line.strip().split(' ')
        l = [int(s) for s in l]
        lis.append(l)

    train_list = lis[0:20000]
    eval_list = lis[20000:]
    
    with open("beauty_train.txt", "w") as f:
        for l in train_list:
            string = ' '.join([str(s) for s in l])
            f.write('%s\n' % string)

    with open("beauty_eval.txt", "w") as f:
        for l in eval_list:
            string = ' '.join([str(s) for s in l])
            f.write('%s\n' % string)

if __name__ == "__main__":
    #read_file("./beauty.txt")
    get_train_eval_dataset("./beauty_padding.txt")