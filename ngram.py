from collections import defaultdict, Counter
import math
import random

import nltk
nltk.download('punkt')


class NGram(object):
    fnames = [
        # 'anna-karenina.txt', 
        # 'alisa.txt', 
        # 'prestuplenie_i_nakazanie.txt', 
        # 'mertvye-dushi.txt', 
        'idiot.txt'
    ]
    EPS = 10e-15
    UNK_TOKEN = '<UNK>'
    
    def __init__(self, log=False):
        super().__init__()
        self.log = log
        if self.log:
            print('initializing model')
        self.corpus = self._get_corpus()
        self.corpus = self._mark_corpus_unk(self.corpus)
        self.vocab = set(self.corpus)

    @staticmethod    
    def _read_file(fname):
        text = ''
        with open(fname, 'r', encoding='utf8') as fin:
            text = fin.read()
        return text

    @staticmethod
    def _tokenize_text(text):
        tokens = []
        sentences = nltk.sent_tokenize(text, language='russian')
        for sent in sentences:
            tokens.extend(nltk.word_tokenize(sent))
        return tokens

    @classmethod
    def _tokenize_text_from_file(cls, fname):
        text = cls._read_file(fname)
        return cls._tokenize_text(text)
    
    def _get_corpus(self):
        tokens = []
        for f in self.fnames:
            if self.log:
                print(f'reading "{f}"')
            tokens.extend(self._tokenize_text_from_file(f))
        if self.log:
            print(f'read {len(tokens)} tokens')
        return tokens

    def _mark_corpus_unk(self, tokens, thresh=2):
        if self.log:
            print(f'marking {self.UNK_TOKEN} with threshold: {thresh}')
        _tokens = [t for t in tokens]
        fdist = nltk.FreqDist(_tokens)
        unk_tokens = [w for w in _tokens if fdist[w] < thresh]
        unks = set(unk_tokens)
        if self.log:
            print(f'found {len(unks)} {self.UNK_TOKEN} tokens')
        count = 1
        percent = -1
        new_percent = 0
        for i, w in enumerate(_tokens):
            if w in unks:
                new_percent = int(100 * count / len(unk_tokens))
                if new_percent != percent:
                    if self.log:
                        print(f'\rmarked {new_percent}% tokens', end='')
                    percent = new_percent
                count += 1
                _tokens[i] = self.UNK_TOKEN
        if self.log:
            print('')
        return _tokens

    def _train_without_smoothing(self):
        for w1 in self.model:
            history_count = float(sum(self.model[w1].values()))
            self.model[w1]['HISTORY_COUNT'] = history_count
            for w2 in self.model[w1]:
                self.model[w1][w2] /= history_count
    
    def _train_add_k_smoothing(self, k=0.05):
        for w1 in self.model:
            history_count = float(sum(self.model[w1].values()))
            frac = history_count + k*len(self.vocab)
            self.model[w1]['HISTORY_COUNT'] = history_count
            for w2 in self.model[w1]:
                self.model[w1][w2] = (self.model[w1][w2] + k) / frac
    
    def _get_unigrams_model(self):
        return dict(Counter(self.corpus))

    def _get_bigrams_model(self):
        model = defaultdict(lambda: defaultdict(lambda: 0))
        bigrams = nltk.bigrams(self.corpus)
        for w1, w2 in bigrams:
            model[w1][w2] += 1
        return model
    
    def _get_trigrams_model(self):
        model = defaultdict(lambda: defaultdict(lambda: 0))
        trigrams = nltk.trigrams(self.corpus)
        for w1, w2, w3 in trigrams:
            model[(w1, w2)][w3] += 1
        return model

    def _train_good_turing(self):
        bigrams = nltk.bigrams(self.corpus)
        Ncs = defaultdict(lambda: 0)
        self.bigrams_cnt = Counter(bigrams)
        for c in self.bigrams_cnt:
            self.model[c[0]][c[1]] = self.bigrams_cnt[c]
        Ncs[0] = len(self.vocab)**2 - len(self.bigrams_cnt)
        for c in self.bigrams_cnt:
            Ncs[self.bigrams_cnt[c]] += 1
        max_c = self.bigrams_cnt.most_common(1)[0][1]
        Ncs.update({max_c + 1: 0})
        N = sum([i*Ncs[i] for i in Ncs])
        self.discounts = defaultdict(lambda: 0)
        zero_divs = []
        for i in range(max_c):
            if Ncs[i]:
                self.discounts[i] = (i + 1) * Ncs[i + 1] / (Ncs[i] * N)
    
    def _train_interpolation(self):
        self.model = defaultdict(lambda: defaultdict(lambda: 0))
        l2 = 0.9
        l1 = 1 - l2
        unigrams = self._get_unigrams_model()
        bigrams = self._get_bigrams_model()

        for w1 in unigrams:
            history_count = float(sum(bigrams[w1].values()))
            for w2 in unigrams:
                self.model[w1][w2] = l2 * bigrams[w1][w2] / history_count + l1 * unigrams[w2] / len(unigrams)

    def train(self, n=2, smoothing=None, k=0.00016):
        """
        :n: n value for n-gram
        :smoothing: type of smoothing to apply (None, 'add-one', 'add-k',)
        """
        _bigrams_only = ('good-turing', 'interpolation')
        _available = (None, 'add-one', 'add-k') + _bigrams_only
        self.model = defaultdict(lambda: defaultdict(lambda: 0))
        self.n = n
        self.smoothing = smoothing if smoothing in _available else None
        if self.log:
            if self.smoothing:
                print(f'training model with "{self.smoothing}" smoothing')
            else:
                print('training model without smoothing')
        if smoothing == 'add-one':
            self.k = 1
        else:
            self.k = k if 0 < k < 1 else 0.05
        if smoothing in _bigrams_only:
            if smoothing == 'good-turing':
                self._train_good_turing()
            elif smoothing == 'interpolation':
                self._train_interpolation()
            return
        elif n == 2:
            self.model = self._get_bigrams_model()
        elif n == 3:
            self.model = self._get_trigrams_model()
        if smoothing is None:
            self._train_without_smoothing()
        elif smoothing in ['add-k', 'add-one']:
            self._train_add_k_smoothing(k=self.k)
    
    def _count_bigram_prob(self, tokens):
        prob = 1
        bigrams = nltk.bigrams(tokens)
        if self.smoothing is None:
            for w1, w2 in bigrams:
                prob *= self.model[w1][w2]
        elif self.smoothing in ['add-k', 'add-one']:
            for w1, w2 in bigrams:
                prob *= (self.model[w1][w2] + self.k) / \
                    (self.model[w1]['HISTORY_COUNT'] + self.k*len(self.vocab))
        return prob
    
    def _count_trigram_prob(self, tokens):
        prob = 1
        trigrams = nltk.trigrams(tokens)
        if self.smoothing is None:
            for w1, w2, w3 in trigrams:
                prob *= self.model[(w1, w2)][w3]
        elif self.smoothing in ['add-k', 'add-one']:
            for w1, w2, w3 in trigrams:
                prob *= (self.model[(w1, w2)][w3] + self.k) / \
                    (self.model[(w1, w2)]['HISTORY_COUNT'] + self.k*len(self.vocab))
        return prob
    
    def _count_good_turing_prob(self, tokens):
        prob = 1
        bigrams = nltk.bigrams(tokens)
        for w1, w2 in bigrams:
            if (w1, w2) in self.bigrams_cnt:
                prob *= self.discounts[self.bigrams_cnt[(w1, w2)]]
            else:
                prob *= self.discounts[0]
        return prob
    
    def _count_interpolation_prob(self, tokens):
        prob = 1
        bigrams = nltk.bigrams(tokens)
        for w1, w2 in bigrams:
            prob *= self.model[w1][w2]
        return prob

    def _mark_unk(self, tokens):
        _tokens = [t for t in tokens]
        for i, t in enumerate(_tokens):
            if t not in self.corpus:
                _tokens[i] = self.UNK_TOKEN
        return _tokens

    def _preprocess_phrase(self, phrase):
        tokens = self._tokenize_text(phrase)
        return self._mark_unk(tokens)

    def probability(self, phrase):
        tokens = self._preprocess_phrase(phrase)
        print(f'tokens: {tokens}')
        prob = 1
        if self.smoothing == 'good-turing':
            prob = self._count_good_turing_prob(tokens)
        elif self.smoothing == 'interpolation':
            prob = self._count_interpolation_prob(tokens)
        elif self.n == 2:
            prob = self._count_bigram_prob(tokens)
        elif self.n == 3:
            prob = self._count_trigram_prob(tokens)
        return prob

    def perplexity(self, phrase):
        tokens = self._preprocess_phrase(phrase)
        prob = self.probability(phrase)
        if abs(prob) < self.EPS:
            return math.inf
        return prob**(1/float(len(tokens)))

    def count_prob_and_perp(self, test_sents):
        probs = []
        perps = []
        for s in test_sents:
            probs.append(self.probability(s))
            perps.append(self.perplexity(s))
        avg_prob = sum(probs) / len(probs)
        max_prob = max(probs)
        min_prob = min(probs)
        avg_perp = sum(perps) / len(perps)
        max_perp = max(perps)
        min_perp = min(perps)
        return avg_prob, max_prob, min_prob, avg_perp, max_perp, min_perp

    def print_prob_and_perp(self, test_sents):
        avg_prob, max_prob, min_prob, avg_perp, max_perp, min_perp = \
            self.count_prob_and_perp(test_sents)
        print(f'avg prob: {avg_prob:.15f}')
        print(f'max prob: {max_prob:.15f}')
        print(f'min prob: {min_prob:.15f}')
        print(f'avg perp: {avg_perp:.15f}')
        print(f'max perp: {max_perp:.15f}')
        print(f'min perp: {min_perp:.15f}')
    
    def test(self, smoothing=None):
        return
