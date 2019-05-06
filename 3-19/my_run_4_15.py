
from my_model_4_15 import NERModel
from my_util_4_15 import load_vocab
from my_config_4_15 import Config
from my_data_util_4_11 import CoNLLDataset_find
filename_tags="data/tags.txt"
def main():
    config=Config()
    model = NERModel(config)
    model.build()
    dev = CoNLLDataset_find(config.processing_word,config.processing_tag,config.filename_dev)
    train = CoNLLDataset_find(config.processing_word,config.processing_tag,config.filename_train)
    # train model
    model.train(train, dev)

if __name__ == "__main__":
    main()