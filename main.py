from nltk.corpus import treebank
from nltk.tag import tnt
import simple_tagger
import hmm_tagger
import random

# This list consists of all the tags that exist in treebank.tagged_sents()
TAGS = ['PRP$', 'VBG', 'VBD', '``', 'VBN', 'POS', "''", 'VBP', 'WDT', 'JJ', 'WP', 'VBZ', 'DT', '#',
         'RP', '$', 'NN', 'FW', ',', '.', 'TO', 'PRP', 'RB', '-LRB-', ':', 'NNS', 'NNP', 'VB', 'WRB',
         'CC', 'LS', 'PDT', 'RBS', 'RBR', 'CD', '-NONE-', 'EX', 'IN', 'WP$', 'MD', 'NNPS', '-RRB-', 'JJS',
         'JJR', 'SYM', 'UH']

def main():
    train_data = treebank.tagged_sents()[:3000]
    test_data = treebank.tagged_sents()[3000:]
    simple = simple_tagger.simple_tagger(TAGS)
    simple.train(train_data)
    word_acc_simple, sent_acc_simple = simple.evaluate(test_data)
    print "simple_tagger:"
    print "Word accuracy = {0}% | Sentence accuracy = {1}%".format(word_acc_simple * 100, sent_acc_simple * 100)

    hmm = hmm_tagger.hmm_tagger(TAGS)
    hmm.train(train_data)
    word_acc_hmm, sent_acc_hmm = hmm.evaluate(test_data)
    print "hmm_tagger:"
    print "Word accuracy = {0}% | Sentence accuracy = {1}%".format(word_acc_hmm * 100, sent_acc_hmm * 100)

    tnt_pos_tagger = tnt.TnT()
    tnt_pos_tagger.train(train_data)
    word_acc_memm, sent_acc_memm = evaluate_memm(test_data, tnt_pos_tagger)
    print "memm_tagger:"
    print "Word accuracy = {0}% | Sentence accuracy = {1}%".format(word_acc_memm * 100, sent_acc_memm * 100)

def evaluate_memm(data, tagger):
    word_count = 0
    word_success = 0
    sentence_correct = True
    sentence_success = 0
    for sentence in data:
        data_only_words = (map(lambda word_tag: word_tag[0],sentence))
        prediction = tagger.tag(data_only_words)
        for i in xrange(len(sentence)):
            word_count += 1
            predicted_tag = random.choice(TAGS) if prediction[i][1] == 'Unk' else prediction[i][1]
            if sentence[i][1] == predicted_tag:
                word_success += 1
            else:
                sentence_correct = False
        if sentence_correct:
            sentence_success += 1
        sentence_correct = True

    word_accuracy = word_success / float (word_count)
    sentence_accuracy = sentence_success / float (len(data))

    return word_accuracy, sentence_accuracy

if __name__ == '__main__':
    main()
