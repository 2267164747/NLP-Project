import numpy as np
UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"
filename_words = "data/words.txt"
filename_tags = "data/tags.txt"
filename_chars = "data/chars.txt"
#processing_word = get_processing_word(vocab_words,vocab_chars, lowercase=True)
filename_dev = "data/CoNLL-2003/eng.mytesta.txt"
filename_test = "data/CoNLL-2003/eng.mytestb.txt"
filename_train = "data/CoNLL-2003/eng.mytrain.txt"
filename_glove="data/vocab.txt"
filename_trimmed= "data/vocab_4_27.trimmed.npz"
dim_word=300
def load_word_char_tag_embedding(filename):
    d = dict()
    with open(filename) as f:
        for idx, word in enumerate(f):  # f中只有word 没有tag
            word = word.strip()
            d[word] = idx
    return d

def  CoNLLDataset_make(path):
    max_iter = None
    niter = 0
    temp=0
    with open(path, 'r') as f:
        words, tags = [], []
        for line in f:
            line = line.strip()
            if (len(line) != 0 and line.startswith("-DOCSTART-") == False):
                ls = line.split(' ')
                word, tag = ls[0], ls[1]
                #word =processing_word(word)
                #tag = processing_tag(tag)
                words += [word]
                tags += [tag]
                niter = 0
                temp=1
            else:
                if (line.startswith("-DOCSTART-") == True):
                    niter = 0
                    temp=0
                if (niter == 1):
                    break
                if(len(line) == 0 and temp==1):
                    words += [' ']
                    tags += [' ']
                    niter = niter + 1
                if(len(line) == 0 and temp==0):
                    niter = niter + 1
    return words, tags
def  CoNLLDataset_find(processing_word,processing_tag,path):
    max_iter = None
    niter = 0
    temp=0
    with open(path, 'r') as f:
        words, tags = [], []
        for line in f:
            line = line.strip()
            if (len(line) != 0 and line.startswith("-DOCSTART-") == False):
                ls = line.split(' ')
                word, tag = ls[0], ls[1]
                word =processing_word(word)
                tag = processing_tag(tag)
                words += [word]
                tags += [tag]
                niter = 0
                temp=1
            else:
                if (line.startswith("-DOCSTART-") == True):
                    niter = 0
                    temp=0
                if (niter == 1):
                    break
                if(len(line) == 0 and temp==1):
                    words += [' ']
                    tags += [' ']
                    niter = niter + 1
                if(len(line) == 0 and temp==0):
                    niter = niter + 1
    return words, tags
def get_vocab_words_and_tags(datasets):
    print("Building vocab...")
    vocab_words = set()
    vocab_tags = set()
    for dataset in datasets:
        dataset = zip(dataset[0], dataset[1])
        for word, tag in dataset:
            vocab_words.add(word)#居然给拆了
            vocab_tags.add(tag)
    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_words, vocab_tags#其实没什么用
def get_glove_vocab(filename_glove):
    print("Building vocab...glove...")
    vocab = set()
    with open(filename_glove) as f:  # 关注fliename中写的是什么
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print("- done. {} tokens".format(len(vocab)))
    return vocab
def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
    print("Building export_trimmed")
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename) as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)#np.asarray修改初始数据，则自己也修改，有形状的数据
    np.savez_compressed(trimmed_filename, embeddings=embeddings)#保存为二进制压缩文件
    #set a& set b求交集
    #把train_word_vocab_embeding中对应word加载embeding
    #打开glove逐个提取每个单词word
    #在共有vocab中找到word,获取其在vocab中的index,
    #在新建的embedding中同样index中放入该单词的embedding
    #此后glove应该用不到了，用压缩文件做字典
def write_vocab(vocab, filename):
    print("Writing vocab...")
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    f.close()
    print("- done. {} tokens".format(len(vocab)))
def get_char_vocab(dataset):
    vocab_char = set()
    dataset=zip(dataset[0],dataset[1])
    for words, _ in dataset:
        for char in words:
            vocab_char.update(char)
    return vocab_char
def main():
    # Generators
    #vocab_words = load_word_char_tag_embedding(filename_words)
    #vocab_tags = load_word_char_tag_embedding(filename_tags)
    #vocab_chars = load_word_char_tag_embedding(filename_chars)
    dev = CoNLLDataset_make(filename_dev)  # 返回words[],tags[],processing_word好像没有用
    #print(len(dev[1]))
    #print(dev[0][len(dev[0])-2])
    #print(dev[1][len(dev[1]) - 2])
    test = CoNLLDataset_make(filename_test)
    #print(len(test[1]))
    train = CoNLLDataset_make(filename_train)
    #print(len(train[1]))
    #数据集获取没有问题
    # Build Word and Tag vocab
    vocab_words, vocab_tags = get_vocab_words_and_tags([train, dev, test])
    vocab_glove = get_glove_vocab(filename_glove)  # 返回glove词向量中单词
    # len(vocab_glove)=71290
    vocab = vocab_words & vocab_glove  # 两个set与完还是set
    # len(vocab)=14595
    vocab.add(UNK)
    vocab.add(NUM)
    # Save vocab
    write_vocab(vocab,filename_words)
    # 这时写进去的是14595个单词,同时出现在glove和数据集中的单词
    write_vocab(vocab_tags,filename_tags)
    vocab = load_word_char_tag_embedding(filename_words)
    export_trimmed_glove_vectors(vocab, filename_glove, filename_trimmed,dim_word)  # config.dim_word
    # Build and save char vocab
    train = CoNLLDataset_make(filename_train)#只有训练数据需要用char????
    vocab_chars = get_char_vocab(train)#得到char set()字典
    write_vocab(vocab_chars,filename_chars)#写下来
if __name__ == "__main__":
    main()
#只是写了三个文件，压缩文件中没有char_embedding，只是对word_embedding做了处理
