import numpy as np
UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"
def get_trimmed_glove_vectors(filename):#取出修剪的glove矢量
    with np.load(filename) as data:
        return data["embeddings"]#只返回embedding
def pad_sequences1(sequences, pad_tok, nlevels=1):
    # 传过来的pad_tok=0
    # sequences=word_ids:(277,9,277,272,69,43,277,317,277)
    if nlevels == 1:
        max_length = max(map(lambda x: len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)
    return sequence_padded, sequence_length
##把整理好的seq加入新的sequence_padded中
##记录有效长度
def pad_sequences2(sequences, pad_tok, nlevels=2):
    max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
    # map第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表
    # 对每个sequences中的seq求长度的最大值
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        # all words are same length now
        sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
        sequence_padded += [sp]
        sequence_length += [sl]
    max_length_sentence = max(map(lambda x: len(x), sequences))
    sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_word, max_length_sentence)
    sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)
    return sequence_padded, sequence_length

def _pad_sequences(sequences, pad_tok, max_length):
    # sequences:#列表或元组生成器
    # pad_tok:#要填充的pad
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)  # 大于长度则截取最大长度后边的#双保险，在上一步调用已经找出最大值了
        sequence_padded += [seq_]  # 把整理好的seq加入新的sequence_padded中
        sequence_length += [min(len(seq), max_length)]  # 记录有效长度
    return sequence_padded, sequence_length

def load_vocab(filename):
    d = dict()
    with open(filename) as f:
        for idx, word in enumerate(f):  # f中只有word 没有tag
            word = word.strip()
            d[word] = idx
    return d

def get_processing_word(vocab_words=None, vocab_chars=None,lowercase=False, chars=False, allow_unk=True):
    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]#char_ids中存放的是idx
        if lowercase:#小写字母
            word = word.lower()
        if word.isdigit():#是数字
            word = NUM
        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                if allow_unk:
                    word = vocab_words[UNK]
                else:
                    raise Exception("Unknow key is not allowed. Check that "\
                                    "your vocab (tags?) is correct")

        # 3. return tuple char ids, word id
        if vocab_chars is not None and chars == True:
            return char_ids, word
        else:
            return word
    return f
def get_processing_tag(vocab_tags=None):
    UNK = "$UNK$"
    NUM = "$NUM$"
    def f(tag):
        if tag.isdigit():  # 是数字
            tag = NUM
        if tag in vocab_tags:
            tag=vocab_tags[tag]  # word= (d[word] = idx)
            return tag
        else:
            tag= vocab_tags[UNK]
            raise Exception("Unknow key is not allowed. Check that your tags? is correct")
            return tag
    return f

def try_data_iter(data):
    words_in_sequence= []
    tags_in_sequece=[]
    data=zip(data[0],data[1])
    for aa in data:
        if (aa[0]!= ' '):
            words_in_sequence .append(aa[0])
            tags_in_sequece.append(aa[1])
        else:
            if (len(words_in_sequence) != 0):
                yield words_in_sequence,tags_in_sequece
                words_in_sequence=[]
                tags_in_sequece=[]
def try_minibatches(data, minibatch_size):
    x_batch, y_batch = [], []
    dat=try_data_iter(data)
    # for (x,y) in(try_get_data_arr(data))
    i=0
    for x,y in dat:
        #print(x)
        #print(y)#len的错误还没有出现
        if len(x_batch) == minibatch_size:
            i = i + 1
            yield x_batch, y_batch
            x_batch, y_batch = [], []
        if type(x[0]) == tuple:  # 是元组
            x = zip(*x)
        x_batch += [x]
        y_batch += [y]
    if len(x_batch) != 0:
        yield x_batch, y_batch

def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position#给定一系列标签、组实体及其位置
    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4
    Returns:
        list of (chunk_type, chunk_start, chunk_end)
    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]
    """
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    #Ex:idx_to_tag={0:0,B-LOC:3,B-PER:4,I-PER:5}
    #将seq中的序列判断idx_to_tag中对应哪一项，然后将per/loc这种类别标出seq中开始标号和结束标号，
    #如果是0则将标出的值添加到chunks中
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass
    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        #结束时如果类别界限有未完成，则结束标号为类别结束符，并添加到chunks中
        chunks.append(chunk)
    return chunks
def get_chunk_type(tok, idx_to_tag):#获取块类型，处理了标签
    """不知道是做什么的
    Args:
        tok: id of token, ex 4
       idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"
    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type#就是找到4对应的tag然后给拆开

