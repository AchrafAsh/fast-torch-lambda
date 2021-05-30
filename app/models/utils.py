import argparse
import time
import pickle

import torch
from torchtext.data.utils import get_tokenizer
from models.baseline import Seq2Seq, Encoder, Decoder, Attention
from models.data import get_filepaths, build_vocab


def build_model(src_vocab_size, tgt_vocab_size, device):
    MODEL_PATH = "../app/models/seq2seq.pt"
    
    INPUT_DIM = src_vocab_size
    OUTPUT_DIM = tgt_vocab_size
    ENC_EMB_DIM = 12
    DEC_EMB_DIM = 12
    ENC_HID_DIM = 24
    DEC_HID_DIM = 24
    ATTN_DIM = 2
    # 1,202,392
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
    model = Seq2Seq(enc, dec, device).to(device)

    print("trying to load model...")
    model.load_state_dict(torch.load(MODEL_PATH), map_location=device)
    print("model loaded 🎉")
    
    return model

def translate(sentence: str) -> str:
    """Takes a sentence in French and returns it's translation in English.
    """
    start_time = time.time()
    MAX_LEN = 25
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fr_tokenizer = get_tokenizer('spacy', language='fr')
    en_tokenizer = get_tokenizer('spacy', language='en')

    try:
        with open("./en_vocab.pkl", "rb") as f:
            en_vocab = pickle.load(f)
        with open("./fr_vocab.pkl", "wb") as f:
            fr_vocab = pickle.load(f)
    except:
        train_filepahts, _, _ = get_filepaths()
        fr_vocab = build_vocab(train_filepahts[0], fr_tokenizer)
        en_vocab = build_vocab(train_filepahts[1], en_tokenizer)

    tokens = fr_tokenizer(sentence)
    x = torch.tensor([fr_vocab.stoi[word] for word in tokens])
    src = x.view(-1, 1).to(device)
    tgt = torch.tensor([en_vocab['<sos>']] + [0] * MAX_LEN).view(-1, 1).to(device)
    
    # Load the model    
    model = build_model(len(fr_vocab), len(en_vocab), device)

    output = model(src, tgt, 0)

    print(f"output shape: {output.shape} | vocab size: {len(en_vocab)}")
    
    translation = []
    for i in range(output.size(0)):
        idx = torch.argmax(output[i][0])
        word = en_vocab.itos[idx]
        if word == "<unk>": continue
        if word == "<eos>": break
        translation.append(word)

    end_time = time.time()
    print(f"⌛ Time to process: {end_time - start_time}")
    return " ".join(translation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence", "-s", type=str, help="French sentence to translate", required=True)
    args = parser.parse_args()
    
    print(f"Francais: {args.sentence}\nEng: {' '.join(translate(args.sentence))}")