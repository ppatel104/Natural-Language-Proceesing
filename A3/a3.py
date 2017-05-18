# coding: utf-8
"""CS585: Assignment 3

In this assignment, you will build a named-entity classifier
using LogisticRegression.

We'll use the labeled data from the CoNLL 2003 Shared Task:
http://www.cnts.ua.ac.be/conll2003/ner/

This is downloaded by the download_data method below.

The main goals of this assignment are to have you:
1- Implement different feature sets for the classifier.
2- Compute evaluation metrics for the classifier on the test set.
3- Enumerate over various settings of the features to determine
   which features result in the highest accuracy.

See Log.txt for the expected output of running the main method.
(Subject to minor variants based on computing environment.)
"""

### DO NOT ADD TO THESE IMPORTS. ####
from collections import Counter
from itertools import product
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import urllib.request


#####################################


def download_data():
    """ Download labeled data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/bqitsnhk911ndqs/train.txt?dl=1'
    urllib.request.urlretrieve(url, 'train.txt')
    url = 'https://www.dropbox.com/s/s4gdb9fjex2afxs/test.txt?dl=1'
    urllib.request.urlretrieve(url, 'test.txt')


def read_data(filename):
    """
    Read the data file into a list of lists of tuples.

    Each sentence is a list of tuples.
    Each tuple contains four entries:
    - the token
    - the part of speech
    - the phrase chunking tag
    - the named entity tag

    For example, the first two entries in the
    returned result for 'train.txt' are:

    > train_data = read_data('train.txt')
    > train_data[:2]
    [[('EU', 'NNP', 'I-NP', 'I-ORG'),
      ('rejects', 'VBZ', 'I-VP', 'O'),
      ('German', 'JJ', 'I-NP', 'I-MISC'),
      ('call', 'NN', 'I-NP', 'O'),
      ('to', 'TO', 'I-VP', 'O'),
      ('boycott', 'VB', 'I-VP', 'O'),
      ('British', 'JJ', 'I-NP', 'I-MISC'),
      ('lamb', 'NN', 'I-NP', 'O'),
      ('.', '.', 'O', 'O')],
     [('Peter', 'NNP', 'I-NP', 'I-PER'), ('Blackburn', 'NNP', 'I-NP', 'I-PER')]]
    """
    ###TODO
    train = open(filename)
    data = train.read()
    data = data.replace("-DOCSTART- -X- -X- O\n\n", "")
    data = data.split('\n')
    sentences = []
    sentence = []
    for d in range(0, len(data)):
        if (data[d] != ''):
            sentence.append(tuple(data[d].split(' ')))
        else:
            sentences.append(sentence)
            sentence = []

    return sentences


def make_feature_dicts(data,
                       token=True,
                       caps=True,
                       pos=True,
                       chunk=True,
                       context=True):
    """
    Create feature dictionaries, one per token. Each entry in the dict consists of a key (a string)
    and a value of 1.
    Also returns a numpy array of NER tags (strings), one per token.

    See a3_test.

    The parameter flags determine which features to compute.
    Params:
    data.......the data returned by read_data
    token......If True, create a feature with key 'tok=X', where X is the *lower case* string for this token.
    caps.......If True, create a feature 'is_caps' that is 1 if this token begins with a capital letter.
               If the token does not begin with a capital letter, do not add the feature.
    pos........If True, add a feature 'pos=X', where X is the part of speech tag for this token.
    chunk......If True, add a feature 'chunk=X', where X is the chunk tag for this token
    context....If True, add features that combine all the features for the previous and subsequent token.
               E.g., if the prior token has features 'is_caps' and 'tok=a', then the features for the
               current token will be augmented with 'prev_is_caps' and 'prev_tok=a'.
               Similarly, if the subsequent token has features 'is_caps', then the features for the
               current token will also include 'next_is_caps'.
    Returns:
    - A list of dicts, one per token, containing the features for that token.
    - A numpy array, one per token, containing the NER tag for that token.
    """
    ###TODO
    features = []
    lables = []
    temp_dict = {}
    temp_dict1 = {}
    temp_features = []

    for sentence in data:
        temp_list = []
        temp_list1 = []
        for t, p, c, e_tag in sentence:
            if (token == True):
                temp_dict['tok' + '=' + t.lower()] = 1
                temp_dict1['tok' + '=' + t.lower()] = 1
            if (caps == True and t[0].isupper()):
                temp_dict['is_caps'] = 1
                temp_dict1['is_caps'] = 1
            if (pos == True):
                temp_dict['pos' + '=' + p] = 1
                temp_dict1['pos' + '=' + p] = 1
            if (chunk == True):
                temp_dict['chunk' + '=' + c] = 1
                temp_dict1['chunk' + '=' + c] = 1
            temp_list.append(temp_dict)
            temp_list1.append(temp_dict1)
            temp_dict = {}
            temp_dict1 = {}
            lables.append(e_tag)
        features.append(temp_list)
        temp_features.append(temp_list1)

    if (context == True):
        for i in range(0, len(temp_features)):
            for j in range(0, len(temp_features[i])):
                if (j == 0 and len(temp_features[i]) > 1):
                    for key in temp_features[i][j + 1]:
                        features[i][j]['next_' + key] = features[i][j + 1][key]
                elif (j == len(temp_features[i]) - 1 and len(temp_features[i]) > 1):
                    for key in temp_features[i][j - 1]:
                        features[i][j]['prev_' + key] = features[i][j - 1][key]
                elif (j > 0 and j != (len(temp_features[i]) - 1) and len(temp_features[i]) > 1):
                    for key in temp_features[i][j - 1]:
                        features[i][j]['prev_' + key] = features[i][j - 1][key]
                    for key in temp_features[i][j + 1]:
                        features[i][j]['next_' + key] = features[i][j + 1][key]

    final_features = []
    for sublist in features:
        for item in sublist:
            final_features.append(item)

    lables = np.array(lables)

    return final_features, lables


def confusion(true_labels, pred_labels):
    """
    Create a confusion matrix, where cell (i,j)
    is the number of tokens with true label i and predicted label j.

    Params:
      true_labels....numpy array of true NER labels, one per token
      pred_labels....numpy array of predicted NER labels, one per token
    Returns:
    A Pandas DataFrame (http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html)
    See Log.txt for an example.
    """
    ###TODO
    c_matrix = []
    true_labels = true_labels.tolist()
    pred_labels = pred_labels.tolist()
    tp_labels = list(zip(true_labels, pred_labels))
    c = Counter()
    c.update(tp_labels)
    u_tlabels = sorted(np.unique(true_labels))
    u_plabels = sorted(np.unique(pred_labels))
    for i in u_tlabels:
        rows = []
        for j in u_plabels:
            if ((i, j) in c):
                rows.append(c[i, j])
            else:
                rows.append(0)
        c_matrix.append(rows)

    confusion_matrix = pd.DataFrame(c_matrix, index=u_tlabels, columns=u_plabels)

    return confusion_matrix


def evaluate(confusion_matrix):
    """
    Compute precision, recall, f1 for each NER label.
    The table should be sorted in ascending order of label name.
    If the denominator needed for any computation is 0,
    use 0 as the result.  (E.g., replace NaNs with 0s).

    NOTE: you should implement this on your own, not using
          any external libraries (other than Pandas for creating
          the output.)
    Params:
      confusion_matrix...output of confusion function above.
    Returns:
      A Pandas DataFrame. See Log.txt for an example.
    """
    ###TODO
    confusion_matrix.fillna(0, inplace=True)
    precision = []
    recall = []
    f_score = []
    for i in range(0, len(confusion_matrix.index)):
        if (confusion_matrix.sum(axis=0)[i] == 0):
            precision.append(0)
        else:
            precision.append(
                confusion_matrix[confusion_matrix.index[i]][confusion_matrix.columns[i]] / confusion_matrix.sum(axis=0)[
                    i])

    for i in range(0, len(confusion_matrix.index)):
        if (confusion_matrix.sum(axis=1)[i] == 0):
            recall.append(0)
        else:
            recall.append(
                confusion_matrix[confusion_matrix.index[i]][confusion_matrix.columns[i]] / confusion_matrix.sum(axis=1)[
                    i])

    for i in range(0, len(precision)):
        if (precision[i] == 0 or recall[i] == 0):
            f_score.append(0)
        else:
            f_score.append(2 * precision[i] * recall[i] / (precision[i] + recall[i]))
    e_matrix = [precision, recall, f_score]
    evaluation_matrix = pd.DataFrame(e_matrix, index=['Precision', 'Recall', 'F_score'], columns=confusion_matrix.index)
    evaluation_matrix.reindex_axis(sorted(evaluation_matrix.columns), axis=1)
    return evaluation_matrix


def average_f1s(evaluation_matrix):
    """
    Returns:
    The average F1 score for all NER tags,
    EXCLUDING the O tag.
    """
    ###TODO
    sum = evaluation_matrix.sum(axis=1)[len(evaluation_matrix.index) - 1] - evaluation_matrix['O']['F_score']
    avg_fscore = sum / (len(evaluation_matrix.columns) - 1)
    return avg_fscore


def evaluate_combinations(train_data, test_data):
    """
    Run 16 different settings of the classifier,
    corresponding to the 16 different assignments to the
    parameters to make_feature_dicts:
    caps, pos, chunk, context
    That is, for one setting, we'll use
    token=True, caps=False, pos=False, chunk=False, context=False
    and for the next setting we'll use
    token=True, caps=False, pos=False, chunk=False, context=True

    For each setting, create the feature vectors for the training
    and testing set, fit a LogisticRegression classifier, and compute
    the average f1 (using the above functions).

    Returns:
    A Pandas DataFrame containing the F1 score for each setting,
    along with the total number of parameters in the resulting
    classifier. This should be sorted in descending order of F1.
    (See Log.txt).

    Note1: You may find itertools.product helpful for iterating over
    combinations.

    Note2: You may find it helpful to read the main method to see
    how to run the full analysis pipeline.
    """
    ###TODO
    l = [[True], [True, False], [True, False], [True, False], [True, False]]
    settings = list(product(*l))
    matrix = []
    for s in settings:
        rows = []
        dicts, labels = make_feature_dicts(train_data, token=s[0], caps=s[1], pos=s[2], chunk=s[3], context=s[4])
        vec = DictVectorizer()
        X = vec.fit_transform(dicts)
        clf = LogisticRegression()
        clf.fit(X, labels)
        test_d, test_l = make_feature_dicts(test_data, token=s[0], caps=s[1], pos=s[2], chunk=s[3], context=s[4])
        X_test = vec.transform(test_d)
        preds = clf.predict(X_test)
        confusion_m = confusion(test_l, preds)
        evaluation_m = evaluate(confusion_m)
        f1_score = average_f1s(evaluation_m)
        rows.append(f1_score)
        rows.append((clf.coef_).size)
        rows.append(s[1])
        rows.append(s[2])
        rows.append(s[3])
        rows.append(s[4])
        matrix.append(rows)

    d_frame = pd.DataFrame(matrix, index=list(range(0, 16)),
                           columns=['f1', 'n_params', 'caps', 'pos', 'chunk', 'context'])
    d_frame = d_frame.sort_values('f1', ascending=False)

    return d_frame


if __name__ == '__main__':
    """
    This method is done for you.
    See Log.txt for expected output.
    """
    download_data()
    train_data = read_data('train.txt')
    dicts, labels = make_feature_dicts(train_data,
                                       token=True,
                                       caps=True,
                                       pos=True,
                                       chunk=True,
                                       context=True)
    vec = DictVectorizer()
    X = vec.fit_transform(dicts)
    print('training data shape: %s\n' % str(X.shape))
    clf = LogisticRegression()
    clf.fit(X, labels)

    test_data = read_data('test.txt')
    test_dicts, test_labels = make_feature_dicts(test_data,
                                                 token=True,
                                                 caps=True,
                                                 pos=True,
                                                 chunk=True,
                                                 context=True)
    X_test = vec.transform(test_dicts)
    print('testing data shape: %s\n' % str(X_test.shape))

    preds = clf.predict(X_test)

    confusion_matrix = confusion(test_labels, preds)
    print('confusion matrix:\n%s\n' % str(confusion_matrix))

    evaluation_matrix = evaluate(confusion_matrix)
    print('evaluation matrix:\n%s\n' % str(evaluation_matrix))

    print('average f1s: %f\n' % average_f1s(evaluation_matrix))

    combo_results = evaluate_combinations(train_data, test_data)
    print('combination results:\n%s' % str(combo_results))
