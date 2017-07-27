import dynet as dy
from collections import Counter
import random
import os
import numpy as np
import io
import logging
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d-%m-%Y %H:%M:%S', level=logging.INFO)


UNK_TOKEN = "_UNK_"
START_TOKEN = "_START_"


class POSTagger:
    """
    A POS-tagger implemented in Dynet, based on https://github.com/clab/dynet/tree/master/examples/python
    """

    def __init__(self, use_mlp=True, train_path="train.conll",
                 dev_path="dev.conll", test_path="test.conll",
                 log_frequency=1000,
                 n_epochs=5, learning_rate=0.001,
                 max_sent_length=None):
        """
        Initialize the POS tagger.
        :param use_mlp: classify using an MLP or not (linear)
        :param train_path: path to training data (CONLL format)
        :param dev_path: path to dev data (CONLL format)
        :param test_path: path to test data (CONLL format)
        """
        self.log_frequency = log_frequency
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.use_mlp = use_mlp
        self.max_sent_length = max_sent_length

        # load data
        self.train_data, self.dev_data, self.test_data = POSTagger.load_data(
            train_path, dev_path, test_path, max_sent_length=max_sent_length)

        # create vocabularies
        self.word_vocab, self.tag_vocab = self.create_vocabularies()

        self.unk = self.word_vocab.w2i[UNK_TOKEN]
        self.n_words = self.word_vocab.size()
        self.n_tags = self.tag_vocab.size()

        self.model, self.params, self.builders = self.build_model()

    @staticmethod
    def load_data(train_path=None, dev_path=None, test_path=None, max_sent_length=None):
        """
        Load all POS data.
        We store all data in memory so that we can shuffle easily each epoch.
        :return: train, test, dev
        """
        train_data = list(read_conll_pos(train_path, max_sent_length=max_sent_length))
        dev_data = list(read_conll_pos(dev_path))
        test_data = list(read_conll_pos(test_path))
        return train_data, dev_data, test_data

    def create_vocabularies(self):
        """
        Create vocabularies from the data.
        :return:
        """
        words = []
        tags = []
        counter = Counter()
        for sentence in self.train_data:
            for word, tag in sentence:
                words.append(word)
                tags.append(tag)
                counter[word] += 1
        words.append(UNK_TOKEN)

        # replace frequency 1 words with unknown word token
        words = [w if counter[w] > 1 else UNK_TOKEN for w in words]

        tags.append(START_TOKEN)

        word_vocab = Vocab.from_corpus([words])
        tag_vocab = Vocab.from_corpus([tags])

        return word_vocab, tag_vocab

    def build_model(self):
        """
        This builds our POS-tagger model.
        :return:
        """
        model = dy.Model()

        params = {}
        params["E"] = model.add_lookup_parameters((self.n_words, 128))
        params["p_t1"] = model.add_lookup_parameters((self.n_tags, 30))

        if self.use_mlp:
            params["H"] = model.add_parameters((32, 50*2))
            params["O"] = model.add_parameters((self.n_tags, 32))
        else:
            params["O"] = model.add_parameters((self.n_tags, 50*2))

        builders = [
            dy.LSTMBuilder(1, 128, 50, model),
            dy.LSTMBuilder(1, 128, 50, model)]

        return model, params, builders

    def tag_sent(self, sent, builders):
        """
        Tags a single sentence.
        :param sent:
        :param builders:
        :return:
        """
        dy.renew_cg()
        f_init, b_init = [b.initial_state() for b in builders]
        wembs = [self.params["E"][self.word_vocab.w2i.get(w, self.unk)] for w, t in sent]

        fw = [x.output() for x in f_init.add_inputs(wembs)]
        bw = [x.output() for x in b_init.add_inputs(reversed(wembs))]

        if self.use_mlp:
            H = dy.parameter(self.params["H"])
            O = dy.parameter(self.params["O"])
        else:
            O = dy.parameter(self.params["O"])
        tags = []
        for f, b, (w, t) in zip(fw, reversed(bw), sent):
            if self.use_mlp:
                r_t = O * (dy.tanh(H * dy.concatenate([f, b])))
            else:
                r_t = O * dy.concatenate([f, b])
            out = dy.softmax(r_t)
            chosen = np.argmax(out.npvalue())
            tags.append(self.tag_vocab.i2w[chosen])
        return tags

    def build_tagging_graph(self, words, tags, builders):
        """
        Builds the graph for a single sentence.
        :param words:
        :param tags:
        :param builders:
        :return:
        """
        dy.renew_cg()
        f_init, b_init = [b.initial_state() for b in builders]

        wembs = [self.params["E"][w] for w in words]
        wembs = [dy.noise(we, 0.1) for we in wembs]

        fw = [x.output() for x in f_init.add_inputs(wembs)]
        bw = [x.output() for x in b_init.add_inputs(reversed(wembs))]

        if self.use_mlp:
            H = dy.parameter(self.params["H"])
            O = dy.parameter(self.params["O"])
        else:
            O = dy.parameter(self.params["O"])
        errs = []
        for f, b, t in zip(fw, reversed(bw), tags):
            f_b = dy.concatenate([f, b])
            if self.use_mlp:
                r_t = O * (dy.tanh(H * f_b))
            else:
                r_t = O * f_b
            err = dy.pickneglogsoftmax(r_t, t)
            errs.append(err)
        return dy.esum(errs)

    def evaluate(self, eval_data):
        """
        Evaluate the model on the given data set.
        :param eval_data:
        :return:
        """
        good = bad = 0.0
        for sent in eval_data:
            tags = self.tag_sent(sent, self.builders)
            golds = [t for w, t in sent]
            for go, gu in zip(golds, tags):
                if go == gu:
                    good += 1
                else:
                    bad += 1
        accuracy = good / (good+bad)
        return accuracy

    def train(self):
        """
        Training loop.
        :return:
        """
        trainer = dy.AdamTrainer(self.model, alpha=self.learning_rate)
        tagged = 0
        loss = 0

        for EPOCH in range(self.n_epochs):
            random.shuffle(self.train_data)
            for i, s in enumerate(self.train_data, 1):

                # print loss
                if i % self.log_frequency == 0:
                    # trainer.status()
                    accuracy = self.evaluate(self.dev_data)
                    logging.info("Epoch {} Iter {} Loss: {:1.6f} Accuracy: {:1.4f}".format(
                        EPOCH, i, loss / tagged, accuracy))
                    loss = 0
                    tagged = 0

                # get loss for this training example
                words = [self.word_vocab.w2i.get(word, self.unk) for word, _ in s]
                tags = [self.tag_vocab.w2i[tag] for _, tag in s]
                sum_errs = self.build_tagging_graph(words, tags, self.builders)
                loss += sum_errs.scalar_value()
                tagged += len(tags)

                # update parameters
                sum_errs.backward()
                trainer.update()


class Vocab:
    def __init__(self, w2i):
        self.w2i = dict(w2i)
        self.i2w = {i: w for w, i in w2i.items()}

    @classmethod
    def from_corpus(cls, corpus):
        w2i = {}
        for sent in corpus:
            for word in sent:
                w2i.setdefault(word, len(w2i))
        return Vocab(w2i)

    def size(self):
        return len(self.w2i.keys())


def read_conll_pos(fname, max_sent_length=None):
    """
    Read words and POS-tags from a CONLL file.
    :param fname:
    :param max_sent_length:
    :return:
    """
    sent = []
    for line in io.open(fname, mode="r", encoding="utf-8"):
        line = line.strip().split()
        if not line:
            if sent:
                if max_sent_length:
                    if len(sent) <= max_sent_length:
                        yield sent
                else:
                    yield sent
            sent = []
        else:
            word = line[1]
            tag = line[4]
            sent.append((word, tag))


def main():

    # set up our data paths
    data_dir = "/home/joost/git/lxmls-toolkit/data"  # EDIT
    train_path = os.path.join(data_dir, "train-02-21.conll")
    dev_path = os.path.join(data_dir, "dev-22.conll")
    test_path = os.path.join(data_dir, "test-23.conll")

    # create a POS tagger object
    pt = POSTagger(train_path=train_path, dev_path=dev_path, test_path=test_path, max_sent_length=15, n_epochs=5)

    # let's train it!
    pt.train()

    test_accuracy = pt.evaluate(pt.test_data)
    logging.info("Test accuracy: {}".format(test_accuracy))


if __name__ == '__main__':
    main()
