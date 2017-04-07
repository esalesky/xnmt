from __future__ import division, generators

import sys
import math

class Logger:
    '''
    A template class to generate report for training.
    '''

    REPORT_TEMPLATE = 'Epoch %.4f: {}_ppl=%.4f (loss/word=%.4f, words=%d)'

    def __init__(self, eval_every, total_train_sent):

        self.dev_every = eval_every
        self.train_every = 5000
        self.total_train_sent = total_train_sent

        self.sent_num = 0

        self.train_loss = self.dev_loss = 0.0
        self.train_words = self.dev_words = 0
        self.since_train = self.since_dev = 0

        self.best_dev = sys.float_info.max

    def fractional_epoch(self):
      return self.sent_num / self.total_train_sent

    def update_train_loss(self, src, tgt, loss):
        batch_sent_num = self.count_sent_num(src)
        self.sent_num += batch_sent_num
        self.since_train += batch_sent_num
        self.since_dev += batch_sent_num
        self.train_words += self.count_tgt_words(tgt)
        self.train_loss += loss

    def needs_report_train(self):
        return self.since_train >= self.train_every

    def report_train(self, do_clear=True):
        print(Logger.REPORT_TEMPLATE.format('train') % (
            self.fractional_epoch(), math.exp(self.train_loss / self.train_words),
            self.train_loss / self.train_words, self.train_words))
        self.since_train %= self.train_every
        if do_clear: self.clear_train()

    def clear_train(self):
        self.train_words = 0
        self.train_loss = 0.0

    def update_dev_loss(self, tgt, loss):
        self.dev_loss += loss
        self.dev_words += self.count_tgt_words(tgt)

    def needs_report_dev(self):
        return self.since_dev >= self.dev_every

    def report_dev(self, do_clear=False):
        print(Logger.REPORT_TEMPLATE.format('test') % (
            self.fractional_epoch(), math.exp(self.dev_loss / self.dev_words),
            self.dev_loss / self.dev_words, self.dev_words))
        self.since_dev %= self.dev_every
        if do_clear: clear_dev()

    def clear_dev(self):
        self.best_dev = min(self.dev_loss, self.best_dev)
        self.dev_words = 0
        self.dev_loss = 0.0

    def needs_write_model(self):
        return self.dev_loss < self.best_dev

    def count_tgt_words(self, tgt_words):
        raise NotImplementedError('count_tgt_words must be implemented in Logger subclasses')

    def count_sent_num(self, obj):
        raise NotImplementedError('count_tgt_words must be implemented in Logger subclasses')

class BatchLogger(Logger):

    def count_tgt_words(self, tgt_words):
        return sum(len(x) for x in tgt_words)

    def count_sent_num(self, obj):
        return len(obj)


class NonBatchLogger(Logger):

    def count_tgt_words(self, tgt_words):
        return len(tgt_words)

    def count_sent_num(self, obj):
        return 1
