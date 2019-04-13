from collections import Counter, defaultdict
import numpy as np
from viterbi import viterbi
import random

class hmm_tagger:
    def __init__(self, tags):
        self._tags = tags

    def train(self, data):
        self.word_to_index = dict()
        curr_index_word = 0

        count_word_tags = defaultdict(Counter)
        count_tag_pairs = defaultdict(Counter)

        self.Pi = np.zeros(len(self._tags))

        for sentence in data:
            self.Pi[self._tags.index(sentence[0][1])] += 1
            for i in xrange(len(sentence)):
                word, tag = sentence[i]

                # Map the word to an index, if it isn't already mapped
                if word not in self.word_to_index:
                    self.word_to_index[word] = curr_index_word
                    curr_index_word += 1

                count_word_tags[word][tag] += 1
                # Skip the last tag, because there's no following tag to check
                if i < len(sentence) - 1:
                    count_tag_pairs[tag][sentence[i+1][1]] += 1
        self.Pi /= float(len(data))

        self.A = np.zeros((len(self._tags), len(self._tags)))
        for first_tag in self._tags:
            # Count the amount of appearences of this first tag as the first tag in
            # a tag pair
            first_tag_count = sum(count_tag_pairs[first_tag].itervalues())
            for second_tag in self._tags:
                # The probability for the pair is the amount of appearences of the second tag
                # right after the first tag, divided by the total amount of appearences of the first tag
                # as the first tag in a pair
                pair_prob = count_tag_pairs[first_tag][second_tag] / float(first_tag_count)
                first_tag_index = self._tags.index(first_tag)
                second_tag_index = self._tags.index(second_tag)
                self.A[first_tag_index][second_tag_index] = pair_prob

        self.B = np.zeros((len(self._tags), len(count_word_tags)))
        for word in count_word_tags:
            word_count = sum(count_word_tags[word].itervalues())
            for tag in count_word_tags[word]:
                word_tag_prob = count_word_tags[word][tag] / float(word_count)
                self.B[self._tags.index(tag)][self.word_to_index[word]] = word_tag_prob

    def evaluate(self, data):
        word_success = 0
        word_count = 0
        sentence_correct = True
        sentence_success = 0
        for sentence in data:
            prediction = self._do_viterbi(sentence)
            prediction = map(lambda tag_index: None if tag_index is None else self._tags[int(tag_index)],
                             prediction)
            for i in xrange(len(sentence)):
                word_count += 1
                if prediction[i] is None:
                    prediction[i] = random.choice(self._tags)
                if sentence[i][1] == prediction[i]:
                    word_success += 1
                else:
                    sentence_correct = False
            if sentence_correct:
                sentence_success += 1
            sentence_correct = True

        word_accuracy = word_success / float (word_count)
        sentence_accuracy = sentence_success / float (len(data))

        return word_accuracy, sentence_accuracy


    def _do_viterbi(self, sentence):
        word_list = []
        viterbi_results = np.array([])
        for word, tag in sentence:
            if word in self.word_to_index:
                word_list.append(self.word_to_index[word])
            else:
                if len(word_list) != 0:
                    viterbi_results = np.concatenate((viterbi_results, viterbi(word_list,self.A,self.B,self.Pi)))
                viterbi_results = np.append(viterbi_results,None)
                word_list = []
        if len(word_list) != 0:
            viterbi_results = np.concatenate((viterbi_results, viterbi(word_list,self.A,self.B,self.Pi)))

        return viterbi_results

def main():
    from nltk.corpus import treebank
    train_data = treebank.tagged_sents()[:3000]
    test_data = treebank.tagged_sents()[3000:]
    hmm = hmm_tagger()
    print 'start train'
    hmm.train(train_data)
    print 'start test'
    word_accuracy, sentence_accuracy = hmm.evaluate(test_data)
    print "Word accuracy = {0}% | Sentence accuracy = {1}%".format(word_accuracy * 100, sentence_accuracy * 100)

if __name__ == '__main__':
    main()

