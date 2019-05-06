import os
from general_utils import get_logger
from my_util_4_15 import load_vocab,get_processing_word,get_processing_tag
class Config():
    def __init__(self, load=True):
        if not os.path.exists(self.dir_output):  # 不重要
            os.makedirs(self.dir_output)

            # create instance of logger
        self.logger = get_logger(self.path_log)
        if load:
            self.load()

    def load(self):
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)#Returns:d: dict[word] = index
        self.vocab_tags  = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)

        self.nwords     = len(self.vocab_words)
        self.nchars     = len(self.vocab_chars)
        self.ntags      = len(self.vocab_tags)
        self.processing_word = get_processing_word(self.vocab_words, self.vocab_chars, lowercase=True, chars=True)
        self.processing_tag = get_processing_tag(self.vocab_tags)
        # 2. get processing functions that map str -> id
        """self.processing_word = get_processing_word(self.vocab_words,
                self.vocab_chars, lowercase=True, chars=self.use_chars)
        self.processing_tag  = get_processing_word(self.vocab_tags,
                lowercase=False, allow_unk=False)
        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed)
                if self.use_pretrained else None)
"""
    # general config
    dir_output = "results/test/"
    dir_model  = dir_output + "model.weights/"
    path_log   = dir_output + "log.txt"
    # embeddings
    dim_word = 300
    dim_char = 100
    # glove files
    # filename_glove = "data/glove.6B/glove.6B.{}d.txt".format(dim_word)
    filename_glove="glove.6B.50d"
    #filename_glove = "data/vocab.txt"
    # trimmed embeddings (created from glove_filename with build_data.py)
    filename_trimmed = "data/vocab_4_28.trimmed.npz"
    use_pretrained = True#是否加载预处理词向量
    # dataset
    filename_dev = "data/CoNLL-2003/eng.mytesta.txt"
    #filename_dev = "data/CoNLL-2003/eng.try.dev"
    filename_test = "data/CoNLL-2003/eng.mytestb.txt"
    filename_train = "data/CoNLL-2003/eng.mytrain.txt"
    #filename_train = "data/CoNLL-2003/eng.try.train"
    max_iter = None # if not None, max number of examples in Dataset
    # vocab (created from dataset with build_data.py)
    filename_words = "data/words.txt"#这时候还没生成
    filename_tags = "data/tags.txt"
    filename_chars = "data/chars.txt"
    #max_iter = None # if not None, max number of examples in Dataset
    # training
    train_embeddings = False
    nepochs          = 15#迭代总轮次
    dropout          = 0.5
    batch_size       = 20
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = -1 # if negative, no clipping

    nepoch_no_imprv  = 3
    # model hyperparameters
    hidden_size_char = 100 # lstm on chars
    hidden_size_lstm = 300 # lstm on word embeddings
#   use_crf = True # if crf, training is 1.7x slower on CPU
    #use_chars = True # if char embedding, training is 3.5x slower on CPU