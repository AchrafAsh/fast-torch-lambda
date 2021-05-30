import io
import torch
import pickle

from collections import Counter

from torch.nn.utils.rnn import pad_sequence  # padding of every batch
from torch.utils.data import DataLoader

from torchtext.data.utils import get_tokenizer
from torchtext.utils import download_from_url, extract_archive
from torchtext.vocab import Vocab

def save_vocab():
    train_filepahts, _, _ = get_filepaths()
    fr_tokenizer = get_tokenizer('spacy', language='fr')
    en_tokenizer = get_tokenizer('spacy', language='en')

    fr_vocab = build_vocab(train_filepahts[0], fr_tokenizer)
    en_vocab = build_vocab(train_filepahts[1], en_tokenizer)

    print("Pickling vocabs...")
    with open("./fr_vocab.pkl", "wb") as f:
        pickle.dump(fr_vocab, f)
    with open("./en_vocab.pkl", "wb") as f:
        pickle.dump(en_vocab, f)
    
    print("Vocabs pickled 🎉")
    return


def build_vocab(filepath, tokenizer):
    counter = Counter()
    with io.open(filepath, encoding="utf8") as f:
        for string_ in f:
            counter.update(tokenizer(string_))
    return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])


def data_process(filepaths, fr_vocab, en_vocab, fr_tokenizer, en_tokenizer):
    raw_fr_iter = iter(io.open(filepaths[0], encoding="utf8"))
    raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
    data = []
    for (raw_fr, raw_en) in zip(raw_fr_iter, raw_en_iter):
        fr_tensor_ = torch.tensor([fr_vocab[token] for token in fr_tokenizer(raw_fr)],
                                dtype=torch.long)
        en_tensor_ = torch.tensor([en_vocab[token] for token in en_tokenizer(raw_en)],
                                dtype=torch.long)
        data.append((fr_tensor_, en_tensor_))
    return data

def generate_batch(opts):
    def f(data_batch):
        fr_batch, en_batch = [], []
        for (fr_item, en_item) in data_batch:
            fr_batch.append(torch.cat([torch.tensor([opts['BOS_IDX']]), fr_item, torch.tensor([opts['EOS_IDX']])], dim=0))
            en_batch.append(torch.cat([torch.tensor([opts['BOS_IDX']]), en_item, torch.tensor([opts['EOS_IDX']])], dim=0))
        fr_batch = pad_sequence(fr_batch, padding_value=opts['PAD_IDX'])
        en_batch = pad_sequence(en_batch, padding_value=opts['PAD_IDX'])
        return fr_batch, en_batch
    return f

def get_filepaths():
    url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
    train_urls = ('train.fr.gz', 'train.en.gz')
    val_urls = ('val.fr.gz', 'val.en.gz')
    test_urls = ('test_2016_flickr.fr.gz', 'test_2016_flickr.en.gz')

    train_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in train_urls]
    val_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in val_urls]
    test_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in test_urls]

    return train_filepaths, val_filepaths, test_filepaths


def get_iterators(batch_size:int = 12):
    fr_tokenizer = get_tokenizer('spacy', language='fr')
    en_tokenizer = get_tokenizer('spacy', language='en')

    train_filepaths, val_filepaths, test_filepaths = get_filepaths()

    fr_vocab = build_vocab(train_filepaths[0], fr_tokenizer)
    en_vocab = build_vocab(train_filepaths[1], en_tokenizer)

    train_data = data_process(train_filepaths, fr_vocab, en_vocab,
                              fr_tokenizer, en_tokenizer)
    val_data = data_process(val_filepaths, fr_vocab, en_vocab,
                            fr_tokenizer, en_tokenizer)
    test_data = data_process(test_filepaths, fr_vocab, en_vocab,
                             fr_tokenizer, en_tokenizer)

    opts = {}
    opts['PAD_IDX'] = fr_vocab['<pad>']
    opts['BOS_IDX'] = fr_vocab['<bos>']
    opts['EOS_IDX'] = fr_vocab['<eos>']
    opts['TGT_PAD_IDX'] = en_vocab['<pad>']
    opts['BATCH_SIZE'] = batch_size
    opts['src_vocab_size'] = len(fr_vocab)
    opts['tgt_vocab_size'] = len(en_vocab)

    train_iter = DataLoader(train_data, batch_size=opts['BATCH_SIZE'],
                            shuffle=True, collate_fn=generate_batch(opts))
    valid_iter = DataLoader(val_data, batch_size=opts['BATCH_SIZE'],
                            shuffle=True, collate_fn=generate_batch(opts))
    test_iter = DataLoader(test_data, batch_size=opts['BATCH_SIZE'],
                        shuffle=True, collate_fn=generate_batch(opts))

    return train_iter, valid_iter, test_iter, opts

if __name__ == "__main__":
    save_vocab()