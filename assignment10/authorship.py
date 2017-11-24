from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize, sent_tokenize
import nltk
import sklearn.datasets
import numpy as np
import sklearn.metrics
import sklearn.model_selection
import string
from collections import Counter
from sklearn.svm import SVC
import os
os.chdir("/home/pieter/projects/textm/assignment10")

# Download the 'stopwords' and 'punkt' from the Natural Language Toolkit, you can comment the next lines if already present.
#nltk.download('stopwords')
#nltk.download('punkt')
stop_words = set(stopwords.words('english'))
functionwords = [str(x) for x in open("functionwords.txt", "r").readlines()]

# Load the dataset into memory from the filesystem
def load_data(dir_name):
    return sklearn.datasets.load_files('data/%s' % dir_name, encoding='utf-8')

def load_train_data():
    return load_data('train')

def load_test_data():
    return load_data('test')

# Extract features from a given text
def extract_features(text):
    features = []
    #first sentence features
    sentences = nltk.sent_tokenize(text)
    features.append(len(sentences)) #length of sentences
    features.append(sum(len(sent) for sent in sentences)/len(sentences)) #mean length
    std = np.std([len(x) for x in sentences])
    features.append(std) #standard deviation
    #features.append(std**2)
    #character features
    features.append(len(text)) #text length

    for letter in list(string.ascii_lowercase):
        features.append(len([x for x in text if x == letter]))

    for letter in list(string.ascii_uppercase):
        features.append(len([x for x in text if x == letter]))
    #bag of word features
    tokens = wordpunct_tokenize(text)
    bag_of_words = [x for x in tokens]
    features.append(len([x for x in bag_of_words if len(x) < 2])) #punctation or short words
    features.append(len([x for x in bag_of_words if x.isdigit()]))
    features.append(len([x for x in bag_of_words if x.isupper()]))
    features.append(len([x for x in bag_of_words if x.lower()]))
    features.append(float(sum(map(len, bag_of_words))) / len(bag_of_words)) #average word length
    features.append(len(bag_of_words)) #number of words
    features.append(len([x for x in bag_of_words if x.lower() not in stop_words])) #
    #function words
    for word in functionwords:
        features.append(len([x for x in bag_of_words if x in word]))
    #bigram trigram  collocation counts
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    finder = nltk.BigramCollocationFinder.from_words(tokens)
    features.append(len(finder.score_ngrams(bigram_measures.raw_freq)))

    finder3 = nltk.TrigramCollocationFinder.from_words(tokens)
    features.append(len(finder3.score_ngrams(trigram_measures.raw_freq)))
    features.append(len(set(tokens)))
    features.append(len(set(tokens))/float(len(tokens)))

    #syntactical features
    postags = nltk.pos_tag(nltk.word_tokenize(text))
    counts = Counter(tag for word, tag in postags)
    features.append(counts['NN'])
    features.append(counts['CD'])
    features.append(counts['FW'])
    features.append(counts['NNP'])
    features.append(counts['NNS'])
    features.append(counts['VBD'])
    features.append(counts['EX'])
    features.append(counts['JJ'])
    features.append(counts['PRP'])
    features.append(counts['SYM'])
    """
    features.append(len([x for (x, y) in postags if y is 'NN']))
    features.append(len([x for (x, y) in postags if y is 'DT']))
    features.append(len([x for (x, y) in postags if y is 'JJ']))
    features.append(len([x for (x, y) in postags if y is 'CD']))
    features.append(len([x for (x, y) in postags if y is 'FW']))
    """

    return features





# Classify using the features
def classify(train_features, train_labels, test_features):
    # TODO: (Optional) If you would like to test different how classifiers would perform different, you can alter
    # TODO: the classifier here.
    clf = SVC(kernel='linear')
    clf.fit(train_features, train_labels)
    return clf.predict(test_features)


# Evaluate predictions (y_pred) given the ground truth (y_true)
def evaluate(y_true, y_pred):
    # TODO: What is being evaluated here and what does it say about the performance? Include or change the evaluation
    # TODO: if necessary.
    recall = sklearn.metrics.recall_score(y_true, y_pred, average='macro')
    print("Recall: %f" % recall)

    precision = sklearn.metrics.precision_score(y_true, y_pred, average='macro')
    print("Precision: %f" % precision)

    f1_score = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
    print("F1-score: %f" % f1_score)

    return recall, precision, f1_score


# The main program
def main():
    train_data = load_train_data()

    # Extract the features
    features = list(map(extract_features, train_data.data))

    # Classify and evaluate
    skf = sklearn.model_selection.StratifiedKFold(n_splits=10)
    scores = []
    for fold_id, (train_indexes, validation_indexes) in\
        enumerate(skf.split(train_data.filenames, train_data.target)):
        # Print the fold number
        print("Fold %d" % (fold_id + 1))
        # Collect the data for this train/validation split
        train_features = [features[x] for x in train_indexes]
        train_labels = [train_data.target[x] for x in train_indexes]
        validation_features = [features[x] for x in validation_indexes]
        validation_labels = [train_data.target[x] for x in validation_indexes]
        # Classify and add the scores to be able to average later
        y_pred = classify(train_features, train_labels, validation_features)
        scores.append(evaluate(validation_labels, y_pred))
        # Print a newline
        print("")

    # Print the averaged score
    recall = sum([x[0] for x in scores]) / len(scores)
    print("Averaged total recall", recall)
    precision = sum([x[1] for x in scores]) / len(scores)
    print("Averaged total precision", precision)
    f_score = sum([x[2] for x in scores]) / len(scores)
    print("Averaged total f-score", f_score)
    print("")

    # TODO: Once you are done crafting your features and tuning your model, also test on the test set and report your
    # TODO: findings. How does the score differ from the validation score? And why do you think this is?
    # test_data = load_test_data()
    # test_features = list(map(extract_features, test_data.data))
    #
    # y_pred = classify(features, train_data.target, test_features)
    # evaluate(test_data.target, y_pred)



if __name__ == '__main__':
    main()
