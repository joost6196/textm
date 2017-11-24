import glob
import os
import random


# Code the TA's used to generate the train/test split, it is not necessary for you to run this unless you want to have
# a different test split then 0.25. You can download the data here from the following link:
# https://www.cs.drexel.edu/~mb553/drexel-amt-corpus.tar.gz
def create_train_test_split(test_split=0.25):
    dirs = glob.glob('Drexel-AMT-Corpus/*')
    dirs = [os.path.basename(x) for x in dirs if os.path.basename(x)]

    os.makedirs('data/')
    os.makedirs('data/test/')
    os.makedirs('data/train/')
    for dir in dirs:
        os.makedirs('data/test/%s/' % dir)
        os.makedirs('data/train/%s/' % dir)

        files = glob.glob('Drexel-AMT-Corpus/%s/*_[0-9].*' % dir)
        random.shuffle(files)
        N_test = int(test_split * float(len(files)))

        train_files = files[:-N_test]
        for train_file in train_files:
            os.rename(train_file, 'data/train/%s/%s' % (dir, os.path.basename(train_file)))

        test_files = files[-N_test:]
        for test_file in test_files:
            os.rename(test_file, 'data/test/%s/%s' % (dir, os.path.basename(test_file)))


if __name__ == '__main__':
    create_train_test_split()
