from collections import Counter, defaultdict
from viterbi import viterbi

class hmm_tagger:
    # This list consists of all the tags that exist in treebank.tagged_sents()
    _TAGS = ['PRP$', 'VBG', 'VBD', '``', 'VBN', 'POS', "''", 'VBP', 'WDT', 'JJ', 'WP', 'VBZ', 'DT', '#',
         'RP', '$', 'NN', 'FW', ',', '.', 'TO', 'PRP', 'RB', '-LRB-', ':', 'NNS', 'NNP', 'VB', 'WRB',
         'CC', 'LS', 'PDT', 'RBS', 'RBR', 'CD', '-NONE-', 'EX', 'IN', 'WP$', 'MD', 'NNPS', '-RRB-', 'JJS',
         'JJR', 'SYM', 'UH']

    def train(self, data):
        word_to_index = dict()
        curr_index_word = 0
        tag_to_index = dict()
        curr_index_tag = 0

        count_word_tags = defaultdict(Counter)
        count_tag_pairs = defaultdict(Counter)

        for sentence in data:
            for i in xrange(len(sentence)):
                word, tag = sentence[i]

                # Map the word to an index, if it isn't already mapped
                if word not in word_to_index:
                    word_to_index[word] = curr_index_word
                    curr_index_word += 1
                if tag not in tag_to_index:
                    tag_to_index[word] = curr_index_tag
                    curr_index_tag += 1

                count_word_tags[word][tags] += 1
                # Skip the last tag, because there's no following tag to check
                if i < len(sentence) - 1:
                    count_tag_pairs[tag][sentence[i+1][1]] += 1

        self.A = np.zero(len(self._TAGS), len(self._TAGS))
        for first_tag in self._TAGS:
            # Count the amount of appearences of this first tag as the first tag in
            # a tag pair
            first_tag_count = sum(count_tag_pairs[first_tag].itervalues())
            for second_tag in self._TAGS:
                # The probability for the pair is the amount of appearences of the second tag
                # right after the first tag, divided by the total amount of appearences of the first tag
                # as the first tag in a pair
                pair_prob = count_tag_pairs[first_tag][second_tag] / first_tag_count
                first_tag_index = self._TAGS.index(first_tag)
                second_tag_index = self._TAGS.index(second_tag)
                self.A[first_tag_index][second_tag_index] = pair_prob

        self.B = np.zero(len(self._TAGS), len(count_word_tags))
        for word in count_word_tags:
            word_count = sum(count_word_tags[word].itervalues())
            for tag in count_word_tags[word]:
                word_tag_prob = count_word_tags[word][tag] / word_count
                self.B[self._TAGS.index(tag)][word_to_index[word]] = word_tag_prob
        

    def evaluate(self, data):
        for sentence in data:
            for word, tag in sentence:

