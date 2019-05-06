import numpy as np
import os
import tensorflow as tf
from general_utils import Progbar
from base_model import BaseModel
from my_config_4_15 import Config
from my_util_4_15 import pad_sequences1, pad_sequences2, load_vocab, get_trimmed_glove_vectors,get_chunks,try_minibatches
class NERModel(BaseModel):
    def __init__(self, config):
        super(NERModel, self).__init__(config)
        self.init_configer()
    def init_configer(self):
        #self.vocab_tags = self.config.vocab_tags
        #filename_tags="data/tags.txt"
        #self.vocab_tags =vocab_tags = load_vocab(filename_tags)
        self.idx_to_tag = {idx: tag for tag, idx in self.config.vocab_tags.items()}
        self.embeddings = get_trimmed_glove_vectors(self.config.filename_trimmed)
    def build(self):
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_loss_op()
        self.add_train_op(self.config.lr_method, self.lr, self.loss, self.config.clip)#继承了base_model中的类
        self.initialize_session()
        # Generic functions that add training op and initialize session
    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
        # 加载char_embedding则还需使用下边这两个
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None], name="char_ids")
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None], name="word_lengths")
    def add_word_embeddings_op(self):
        with tf.variable_scope("words"):  # words是个命名空间，使用的时候只需要调用
            _word_embeddings = tf.Variable(self.embeddings, name="_word_embeddings", dtype=tf.float32, trainable=False)
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.word_ids,name="word_embeddings")  # word_ids是个啥？
            # Tensor("words/word_embeddings:0", shape=(?, ?, 300), dtype=float32)  # (batch_size,max_length,og sentence,dim)
        with tf.variable_scope("chars"):
            _char_embeddings = tf.get_variable(name="_char_embeddings",dtype=tf.float32, shape=[self.config.nchars, self.config.dim_char])
            char_embeddings = tf.nn.embedding_lookup(_char_embeddings, self.char_ids, name="char_embeddings")
            # _char_embeddings shape=(61或84,100)  self.char_ids shape=(batch_size,max length of sentence,max length of word)
            # put the time dimension on axis=1 for dynamic_rnn
            s = tf.shape(char_embeddings)  # char_embeddings在哪定义的
            char_embeddings = tf.reshape(char_embeddings,shape=[s[0] * s[1], s[-2], self.config.dim_char])  # 从(?,?,?,100)变成了（？，？，100）
            word_lengths = tf.reshape(self.word_lengths, shape=[s[0] * s[1]])#数据集中出现的词汇的总数量个，内容是每个单词的长度

            # bi lstm on chars
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,state_is_tuple=True)
            #hidden_size=num_unit输出维度.max_time=step,num_unit输出维度
            #不设输出维度应该是输出维度不变
            # 100和char的维度一致，也就是要训练char_embedding
            # state_is_tuple=True输入和输出的states为接受状态和返回状态是(c_state,m_state)元组
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char, state_is_tuple=True)#_nume_units=100,_outut_size=100,c,h=100
            _output = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, char_embeddings,sequence_length=word_lengths, dtype=tf.float32)
            #sequence_length=word_lengths确定了lstme单层神经元的个数
            #输入X是一个 char_embeddings=[batch_size，step，input_size] = [?，?，100]
            # 应用在文本中时，max_time可以为句子的长度（一般以最长的句子为准，短句需要做padding），depth为输入句子词向量的维度。
            # read and concat output
            #shape=[?,?,100]
            #output最后一层每个step的输出，结构是[batch_size,step,n_neurons=默认]
            _, ((_, output_fw), (_, output_bw)) = _output  # （outputs, output_states) = _output
            # output_states为(output_state_fw, output_state_bw)，包含了前向和后向每层最后一个的隐藏状态h和输出结果c的组成的,类型都是LSTMStateTuple
            # 每一个维度是[batch_size,num_units]
            output = tf.concat([output_fw, output_bw], axis=-1)
            # output_states是一个元组的元组，我个人的处理方法是用c_fw,h_fw = output_state_fw和c_bw,h_bw = output_state_bw
            # shape = (batch size, max sentence length, char hidden size)这一句写的啥又不知道了
            output = tf.reshape(output, shape=[s[0], s[1], 2 * self.config.hidden_size_char])  # shape(?,?,200)#回归原来的形状
            word_embeddings = tf.concat([word_embeddings, output], axis=-1)
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)  # (?,?,500)
#疑问：数据输入lstm后经历了怎样的矩阵变化，h是怎么得到的
    def add_logits_op(self):
        """
        对于批处理的每个句子中的每个单词，它对应于一个向量分数，尺寸等于标签的数量。
        每个单词对应的标签得分情况
        word-->[scores of tags]
        """
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            #_num_units=_output_size=300
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.word_embeddings,
                                                                        sequence_length=self.sequence_lengths,
                                                                        dtype=tf.float32)
            #对于文本分类来说，需要最后一个time_step的输出，而中文实体抽取则需要最终的outputs，即所有time_step的输出。
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)
        #两个图
        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,shape=[2 * self.config.hidden_size_lstm, self.config.ntags])  # 从隐藏层到分类tags
            b = tf.get_variable("b", shape=[self.config.ntags], dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]#此图接上图”bi-lstm“图
            output = tf.reshape(output, [-1, 2 * self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])

    def add_loss_op(self):
        log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(self.logits, self.labels,self.sequence_lengths)
        #在测试的时候语句是这样写的
        # viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
        #log_likelihood: 标量,log-likelihood
        #transition_params: 形状为[num_tags, num_tags] 的转移矩阵
        # inputs: 一个形状为[batch_size, max_seq_len, num_tags] 的tensor,
        # 一般使用BILSTM处理之后输出转换为他要求的形状作为CRF层的输入.
        # self.logits是预测出来的标签，labels是正确标签，
        self.trans_params = trans_params  # need to evaluate it for decoding
        # log_likeiood:包含给定序列标签索引的对数似然的标量
        # trans_params一个[num_tags,num_tags]转换矩阵，即转移矩阵
        self.loss = tf.reduce_mean(-log_likelihood)
        tf.summary.scalar("loss", self.loss)

    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        char_ids, word_ids = zip(*words)  # 查字典，将字典中的词做成list 含ids     没懂word_ids,char_ids是什么# word_ids=pad_tok
        word_ids, sequence_lengths = pad_sequences1(word_ids, 0)  # 填充sequences
        ##word_ids把整理好的seq加入新的sequence_padded中
        ##sequence_lengths记录有效长度
        #这时候已经不行了
        char_ids, word_lengths = pad_sequences2(char_ids, pad_tok=0, nlevels=2)
        # build feed dictionary
        feed = {self.word_ids: word_ids,  # word_ids=sequence_padded
                self.sequence_lengths: sequence_lengths,  #
                self.char_ids: char_ids, self.word_lengths: word_lengths,}
        if labels is not None:
            labels, _ = pad_sequences1(labels, 0)
            feed[self.labels] = labels
        if lr is not None:
            feed[self.lr] = lr
        if dropout is not None:
            feed[self.dropout] = dropout
        return feed, sequence_lengths  # 为啥还要返回sequence_length

    def run_evaluate(self, dev):
        """Evaluates performance on test set
        Args:
            test: dataset that yields tuple of (sentences, tags)
        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...
        """
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in try_minibatches(dev, self.config.batch_size):
            #batch_size=9时，训练两轮
            labels_pred, sequence_lengths = self.predict_batch(words)
            #label_pred
            # return的是viterbi_sequences, sequence_lengths
            #预测：lab_pred对于local softmax直接选择每个time step最高的值就可以:
            for lab, lab_pred, length in zip(labels, labels_pred, sequence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                accs += [a == b for (a, b) in zip(lab, lab_pred)]
                lab_chunks = set(get_chunks(lab, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred, self.config.vocab_tags))
                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        #输出结果前三轮p\r\f1都是0
        acc = np.mean(accs)
        return {"acc": 100 * acc, "f1": 100 * f1}

   # def predict(self, words_raw):
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        pred_ids, _ = self.predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]
        return preds
    def predict_batch(self, words):
        """
    Args:
        words: list of sentences
    Returns:
        labels_pred: list of labels for each sentence
        sequence_length
        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)
        # get tag scores and transition params of CRF
        viterbi_sequences = []
        logits, trans_params = self.sess.run([self.logits, self.trans_params], feed_dict=fd)
        #logits训练过程只蕴含了bilstm  proj的信息。没有crf
        #trans_params是crf后的转移矩阵、参数
        # iterate over the sentences because no batching in vitervi_decode
        #遍历句子，因为vitervi_decode中没有批处理
        for logit, sequence_length in zip(logits, sequence_lengths):
            logit = logit[:sequence_length]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
            #原本的训练模型是这样的，然后再run
            #log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(self.logits, self.labels,self.sequence_lengths)
            ##解码，得到预测的序列，以及预测序列的得分
            #对于CRF，传递一下训练时候得到的转移矩阵T，用viterbi的方法搜索到最优解即可
            # 通俗一点,作用就是返回最好的标签序列.这个函数只能够在测试时使用,在tensorflow外部解码
            # score: 一个形状为[seq_len, num_tags] matrix of unary potentials.
            # transition_params: 形状为[num_tags, num_tags] 的转移矩阵
            viterbi_sequences += [viterbi_seq]
        return viterbi_sequences, sequence_lengths

    def run_epoch(self, train, dev, epoch):
        """Performs one complete pass over the train set and evaluate on dev
    Args:
        train: dataset that yields tuple of sentences, tags
        dev: dataset
        epoch: (int) index of the current epoch
    Returns:
        f1: (python float), 选择模型的分数越高越好
        """
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)  # 初始化general_utils类
        # iterate over dataset
        for i, (words, labels) in enumerate(try_minibatches(train, batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.config.lr, self.config.dropout)
            # return的是feed, sequence_lengths
            _, train_loss, summary = self.sess.run([self.train_op, self.loss, self.merged], feed_dict=fd)
            #loss中蕴藏了bilstm  proj  crf三个图
            #self.train_op=add_train_op 定义的optimizer.minimize(loss)
            #self.loss=add_loss_op定义的tf.reduce_mean(-log_likelihood)
            #self.merged=add_summary在base_model中
            #self.sess.run（output,feed_dict）
            prog.update(i + 1, [("train loss", train_loss)])
            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch * nbatches + i)
        metrics = self.run_evaluate(dev)  # dev: dataset、test
        msg = " - ".join(["{} {:04.2f}".format(k, v) for k, v in metrics.items()])
        self.logger.info(msg)
        return metrics["f1"]