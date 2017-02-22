# coding: utf-8
"""CS585: Assignment 2

In this assignment, you will complete an implementation of
a Hidden Markov Model and use it to fit a part-of-speech tagger.
"""

from collections import Counter, defaultdict
import math
import numpy as np
import os.path
import urllib.request


class HMM:
    def __init__(self, smoothing=0):
        """
		Construct an HMM model with add-k smoothing.
		Params:
		  smoothing...the add-k smoothing value

		This is DONE.
		"""
        self.smoothing = smoothing

    def fit_transition_probas(self, tags):
        """
        Estimate the HMM state transition probabilities from the provided data.

        Creates a new instance variable called `transition_probas` that is a
        dict from a string ('state') to a dict from string to float. E.g.
        {'N': {'N': .1, 'V': .7, 'D': 2},
         'V': {'N': .3, 'V': .5, 'D': 2},
         ...
        }
        See test_hmm_fit_transition.

        Params:
          tags...a list of lists of strings representing the tags for one sentence.
        Returns:
            None
        """

        ###TODO
        def iter_bigrams(l=[]):
            return (l[i:i + 2] for i in range(len(l) - 1))

        self.transition_probas = defaultdict(lambda: Counter())
        unique_words = set(x for l in tags for x in l)

        for sent in tags:
            for bigram in iter_bigrams(sent):
                self.transition_probas[bigram[0]].update([bigram[-1]])

        self.states = list(unique_words)

        for bigram, word_counts in self.transition_probas.items():
            total = sum(word_counts.values())
            for i in self.transition_probas.values():
                for j in unique_words:
                    if (j not in i):
                        i.update({j: 0})
            self.transition_probas[bigram] = {
                word: (count + self.smoothing) / (total + (len(unique_words) * self.smoothing)) for word, count in
                word_counts.items()}


    def fit_emission_probas(self, sentences, tags):
        """
		Estimate the HMM emission probabilities from the provided data.
		Creates a new instance variable called `emission_probas` that is a
		dict from a string ('state') to a dict from string to float. E.g.
		{'N': {'dog': .1, 'cat': .7, 'mouse': 2},
		 'V': {'run': .3, 'go': .5, 'jump': 2},
		 ...
		}
		Params:
		  sentences...a list of lists of strings, representing the tokens in each sentence.
		  tags........a list of lists of strings, representing the tags for one sentence.
		Returns:
			None

		See test_hmm_fit_emission.
		"""
        ###TODO
        composite = []
        for tag, sentence in zip(tags, sentences):
            for t, s in zip(tag, sentence):
                composite.append([t, s])

        def iter_bigrams(l=[]):
            return (l[i:i + 2] for i in range(len(l) - 1))

        self.emission_probas = defaultdict(lambda: Counter())
        for compo in composite:
            for bigram in iter_bigrams(compo):
                self.emission_probas[bigram[0]].update([bigram[-1]])
        values = self.emission_probas.values()
        unique_words = set(x for l in values for x in l)
        for bigram, word_counts in self.emission_probas.items():
            total = sum(word_counts.values())
            for i in self.emission_probas.values():
                for j in unique_words:
                    if (j not in i):
                        i.update({j: 0})
            self.emission_probas[bigram] = {
                word: (count + self.smoothing) / (total + (len(unique_words) * self.smoothing)) for word, count in
                word_counts.items()}

    def fit_start_probas(self, tags):
        """
		Estimate the HMM start probabilities form the provided data.

		Creates a new instance variable called `start_probas` that is a
		dict from string (state) to float indicating the probability of that
		state starting a sentence. E.g.:
		{
			'N': .4,
			'D': .5,
			'V': .1
		}

		Params:
		  tags...a list of lists of strings representing the tags for one sentence.
		Returns:
			None

		See test_hmm_fit_start
		"""
        ###TODO
        start_states = []
        for tag in tags:
            start_states.append(tag[0])
        unique_words = set(x for l in tags for x in l)
        d = Counter()
        d.update(tuple(start_states))
        total_tokens = sum(d.values())
        self.start_probas = d
        for i in unique_words:
            if i not in self.start_probas.keys():
                self.start_probas.update({i: 0})
        self.start_probas = {token: (value + self.smoothing) / (total_tokens + len(unique_words) * self.smoothing) for
                             token, value in d.items()}

    def fit(self, sentences, tags):
        """
		Fit the parameters of this HMM from the provided data.

		Params:
		  sentences...a list of lists of strings, representing the tokens in each sentence.
		  tags........a list of lists of strings, representing the tags for one sentence.
		Returns:
			None

		DONE. This just calls the three fit_ methods above.
		"""
        self.fit_transition_probas(tags)
        self.fit_emission_probas(sentences, tags)
        self.fit_start_probas(tags)

    def viterbi(self, sentence):
        """
		Perform Viterbi search to identify the most probable set of hidden states for
		the provided input sentence.

		Params:
		  sentence...a lists of strings, representing the tokens in a single sentence.

		Returns:
		  path....a list of strings indicating the most probable path of POS tags for
		  		  this sentence.
		  proba...a float indicating the probability of this path.
		"""
        ###TODO

        states = self.states
        viterbi = np.zeros(shape=(len(states),len(sentence)))
        backp = np.zeros(shape=(len(states),len(sentence)))
        for i in range(0,len(states)):
            viterbi[i][0] = self.start_probas[states[i]] * self.emission_probas[states[i]][sentence[0]]
            backp[i][0] = 0

        for i in range(1,len(sentence)):
            for j in range(0,len(states)):
                v_max = []
                b_max = []
                for k in range(0,len(states)):
                    v_max.append(viterbi[k][i-1] * self.transition_probas[states[k]][states[j]]*self.emission_probas[states[j]][sentence[i]])
                    b_max.append(viterbi[k][i-1] * self.transition_probas[states[k]][states[j]])
                viterbi[j][i] = max(v_max)
                backp[j][i] = b_max.index(max(b_max))

        index_path = []
        for l in range(len(sentence)-1,-1,-1):
            if(l==len(sentence)-1):
                last_column = list(viterbi[:,len(sentence)-1])
                final_prob = max(last_column)
                last_index = int(last_column.index(max(last_column)))
                value = int(backp[:, l][last_index])
                index_path.append(last_index)
            else:
                index_path.append(value)
                value = int(backp[:,l][value])
        index_path.reverse()
        final_path = []
        for i in range(0,len(index_path)):
            final_path.append(states[index_path[i]])


        return final_path,final_prob







def read_labeled_data(filename):

    """
        Read in the training data, consisting of sentences and their POS tags.

        Each line has the format:
        <token> <tag>

        New sentences are indicated by a newline. E.g. two sentences may look like this:
        <token1> <tag1>
        <token2> <tag2>

        <token1> <tag1>
        <token2> <tag2>
        ...

        See data.txt for example data.

        Params:
          filename...a string storing the path to the labeled data file.
        Returns:
          sentences...a list of lists of strings, representing the tokens in each sentence.
          tags........a lists of lists of strings, representing the POS tags for each sentence.
    """

    ###TODO
    file = open(filename)
    Text = file.read()
    text = Text.strip('\n').split('\n')
    sentences = []
    tags = []
    sentence = []
    tag = []
    for text1 in text:
        l = text1.split(' ')
        if (len(l) == 2):
            sentence.append(l[0])
            tag.append(l[1])
        else:
            sentences.append(sentence)
            tags.append(tag)
            sentence = []
            tag = []
    return sentences,tags


def download_data():

    """ Download labeled data.
        DONE ALREADY.
    """

    url = 'https://www.dropbox.com/s/ty7cclxiob3ajog/data.txt?dl=1'
    urllib.request.urlretrieve(url, 'data.txt')

if __name__ == '__main__':
    """
        Read the labeled data, fit an HMM, and predict the POS tags for the sentence
        'Look at what happened'

        DONE - please do not modify this method.

        The expected output is below. (Note that the probability may differ slightly due
        to different computing environments.)

        $ python3 a2.py
        model has 33 states
        ['$', "''", ',', ':', 'CC', 'CD', 'DT', 'EX', 'IN', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'TO', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WRB', '``']
        predicted parts of speech for the sentence ['Look', 'at', 'what', 'happened']
        (['VB', 'IN', 'WP', 'VBD'], 2.752640843234699e-10)
    """
    fname = 'data.txt'
    if not os.path.isfile(fname):
        download_data()
    sentences, tags = read_labeled_data(fname)

    model = HMM(.001)
    model.fit(sentences, tags)
    print('model has %d states' % len(model.states))
    print(model.states)
    sentence = ['Look', 'at', 'what', 'happened']
    print('predicted parts of speech for the sentence %s' % str(sentence))
    print(model.viterbi(sentence))
