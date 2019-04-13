#nltk.download('treebank')
#nltk.download('tagsets')

from collections import Counter, defaultdict
import random

class simple_tagger:
    def __init__(self, tags):
        self._tags = tags


    def train(self, data):
        """
        Trains the tagger using the given data.
        """
        counts = defaultdict(Counter)
        # For each word in the data, count how many times it's tagged with each tag
        for sentence in data:
            for word, tag in sentence:
                counts[word][tag] += 1

        self._model = dict()
        # Map each word to its most frequent tag
        for word in counts:
            self._model[word] = counts[word].most_common(1)[0][0]

    def evaluate(self, data):
        """
        Evaluates the tagger on the given data.
        :return: (word level accuracy, sentence level accuracy)
        :type: tuple
        """
        words_count = 0
        success_count_words = 0
        success_count_sentences = 0
        for sentence in data:
            sentence_success = True
            for word, tag in sentence:
                words_count += 1

                prediction = self._model.get(word, None)
                # If the examined word doesn't exist in the model, choose a random tag
                if prediction is None:
                    prediction = random.choice(self._tags)

                if tag == prediction:
                    # Count this word as successful
                    success_count_words += 1
                else:
                    # Mark this sentence as failed
                    sentence_success = False
            # Count this as a successful sentence only if all words in this sentence were correct in the model
            if sentence_success:
                success_count_sentences += 1

        return (float(success_count_words) / words_count), (float(success_count_sentences) / len(data))


if __name__ == "__main__":
    from nltk.corpus import treebank
    train_data = treebank.tagged_sents()[:3000]
    test_data = treebank.tagged_sents()[3000:]
    simple = simple_tagger()
    simple.train(train_data)
    word_acc, sent_acc = simple.evaluate(test_data)
    print "Word accuracy = {0}% | Sentence accuracy = {1}%".format(word_acc * 100, sent_acc * 100)
