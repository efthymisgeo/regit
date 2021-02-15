import os
import sys
import torch
import numpy as np
import torch
import argparse
import pickle
import torch.nn as nn
import torch.optim as optim
from transformers import PreTrainedModel, PreTrainedTokenizer, \
    PretrainedConfig
from transformers import RobertaForSequenceClassification, \
    RobertaTokenizerFast, RobertaConfig
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
from captum.attr import LayerConductance
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))
from utils.berta_utils import *
from utils.model_utils import EarlyStopping

DATA_PATH = "data/aclImdb"
TEST_SPLIT = 0.01
# define model, tokenizer, config
MODEL_CLASSES = {
    'roberta': (RobertaForSequenceClassification,
                RobertaTokenizerFast,
                RobertaConfig),
                }

EPOCHS = 10
BS = 10
ETA = 5e-5


def train_loop(model, data_loader, optim, device):
    model.train()
    train_loss = 0
    for batch in tqdm(data_loader):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        import pdb; pdb.set_trace()
        loss = outputs[0]
        loss.backward()
        optim.step()
        train_loss += loss.item()
    return train_loss / len(data_loader)

def test_loop(model, data_loader, device):
    model.eval()
    test_loss, correct = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            test_loss += loss.item()
    test_loss /= len(data_loader)
    return test_loss

def attribute_loop(model, attributor, data_loader, device="cpu", ref_token_id=129837):
    model.eval()
    model.zero_grad()
    for batch in tqdm(data_loader):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        # input_ids.type(torch.LongTensor)
        # attention_mask.type(torch.LongTensor)
        # labels.type(torch.LongTensor)
        input_ids, attention_mask, labels = \
            input_ids.to(device).long(), attention_mask.to(device).long(), labels.to(device).long()
        # attention_mask = torch.tensor(attention_mask).to(device).long()
        # labels = torch.tensor(labels).to(device).long()
        # # input_ids, ref_ids, _, sep_idx = \
        #     construct_input_ref_pair(input_text, ref_token_id)
        # SOS: works only for single batch
        ref_ids = attention_mask * input_ids
        sep_idx = sum(sum(attention_mask)) - 1
        ref_ids[0, 1:(sep_idx)] = ref_token_id
        ref_ids.to(device).long()
        seq_len = input_ids.size(1)
        token_type_ids = \
            torch.tensor([[0 if i <= sep_idx else 1 for i in range(seq_len)]]).to(device).long()
        # import pdb; pdb.set_trace()
        attr, delta = \
            attributor.attribute(inputs=input_ids,
                               baselines=ref_ids,
                            #    target=labels,
                               additional_forward_args=(labels, token_type_ids, attention_mask),
                               return_convergence_delta=True,
                               n_steps=5)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        test_loss += loss.item()




def get_baseline_inputs(tokenizer):
    # A token used for generating token reference
    ref_token_id = tokenizer.pad_token_id
    # A token used as a separator between question and text and it is 
    # also added to the end of the text. 
    sep_token_id = tokenizer.sep_token_id
    # A token used for prepending to the concatenated question-text word sequence
    cls_token_id = tokenizer.cls_token_id 

def construct_input_ref_token_type_pair(input_ids, sep_ind=0):
    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]])
    ref_token_type_ids = torch.zeros_like(token_type_ids)# * -1
    return token_type_ids, ref_token_type_ids

# def old_construct_input_ref_pair(text, ref_token_id, sep_token_id, cls_token_id):
#     # question_ids = tokenizer.encode(question, add_special_tokens=False)
#     text_ids = tokenizer.encode(text, add_special_tokens=False)
#     # construct input token ids
#     input_ids = [cls_token_id] + text_ids + [sep_token_id]

#     # construct reference token ids 
#     ref_input_ids = [cls_token_id] + [ref_token_id] * len(text_ids) + [sep_token_id]

#     return torch.tensor([input_ids], device=device),
#            torch.tensor([ref_input_ids], device=device),
#            len(question_ids)


def construct_input_ref_pair(text, ref_token_id):
    """
    Construct a reference baseline of equal length with a given input text
    Args:
        text (str): an input string 
        ref_token_id (int): an int which will be used as the referecne baseline
    """
    text_dict = \
        tokenizer.encode_plus(text,
                              add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                              max_length=512,
                              padding='max_length',
                              return_attention_mask=True,
                              return_tensors='pt')
    input_ids = text_dict["input_ids"]
    attention_mask = text_dict["attention_mask"]
    ref_ids = attention_mask * input_ids
    sep_idx = sum(sum(attention_mask)) - 1
    ref_ids[0, 1:(sep_idx)] = ref_token_id
    return input_ids, ref_ids, attention_mask, sep_idx


def construct_input_ref_token_type_pair(input_ids, sep_ind=0):
    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]])
    ref_token_type_ids = torch.zeros_like(token_type_ids)# * -1
    return token_type_ids, ref_token_type_ids


def get_model_conductance(model_importance, data_loader):
    """
    Function which calculates the overal conductance of the roberta model on a given data split
    Args:
        model_importance (dict): dict of dicts
    """
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, required=True, help='device name, e.g cuda:2')
    parser.add_argument('--use_y', action="store_true",
                        help='Argument which handles the usage of Y (default: False)')
    # parser.add_argument('--out', type=str, default='out.mp4')

    in_args = parser.parse_args()
    use_cond = in_args.use_y
    dev = in_args.device
    # load data
    train_texts, train_labels = \
        read_imdb_split(os.path.join(DATA_PATH, "train"))
    test_texts, test_labels = \
        read_imdb_split(os.path.join(DATA_PATH, "test"))

    # model type definition
    model_type = 'roberta'
    # 'roberta-large', 'roberta-large-mnli', 
    # 'distilroberta-base', 
    # 'roberta-base-openai-detector', 'roberta-large-openai-detector'
    pretrained_model_name = 'roberta-base' 
    model_class, tokenizer_class, config_class = MODEL_CLASSES[model_type]
    tokenizer = tokenizer_class.from_pretrained(pretrained_model_name)
    model = model_class.from_pretrained(pretrained_model_name)
    device = torch.device(dev) if torch.cuda.is_available() else torch.device('cpu')
    print(f"Available device is {device} \n")
    model.to(device)
    print(model)

    # load data
    train_dataset, val_dataset, test_dataset = \
        get_train_dev_test_data(train_texts, train_labels,
                                test_texts, test_labels,
                                tokenizer, dev_size=TEST_SPLIT)

    # torch dataloader
    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BS, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=False)

    if use_cond:

        # def custom_forward(inputs, labels, token_type_ids=None, attention_mask=None):
        #     outputs = \
        #         model(input_ids=input_ids,
        #               token_type_ids=None,
        #               attention_mask=attention_mask,
        #               labels=labels)
        #     import pdb; pdb.set_trace()
        #     preds = outputs[1]
        #     return torch.softmax(preds, dim = 1)[0]


        def custom_forward(input_ids, labels, attention_mask=None):
            outputs = \
                model(input_ids=input_ids,
                      attention_mask=attention_mask,
                      labels=labels)
            import pdb; pdb.set_trace()
            preds = outputs[1]
            return torch.softmax(preds, dim = 1)[0]


        def custom_forward_func(inputs,
                                labels,
                                token_type_ids=None,
                                attention_mask=None):
            print("entered custom func")
            outputs = \
                model(input_ids=inputs,
                      token_type_ids=token_type_ids,
                      attention_mask=attention_mask,
                      labels=labels)
            print("finished forward custom pass")
            # import pdb; pdb.set_trace()
            preds = outputs[0]
            return torch.softmax(preds, dim = 1)[0]

        # texts = [
        #     "Hello world",
        #     "hello darkness my old friend, i came to talk with you again",
        #     "            ",
        #     "a a a a a a a a a a a a a a a a a"
        # ]
        # for text in texts:
        #     encoded_dict = tokenizer.encode_plus(
        #                     text,
        #                     add_special_tokens=True, # Add '[CLS]' and '[SEP]'
        #                     max_length=512,
        #                     pad_to_max_length=True,
        #                     return_attention_mask=True,
        #                     return_tensors='pt')
        #     input_ids = encoded_dict["input_ids"]
        #     attention_mask = encoded_dict["attention_mask"]
        #     import pdb; pdb.set_trace()

        # reference ids
        ref_text = "                                "
        input_text = "Hello darkness my old friend i came to talk with you again"
        # ref_dict = tokenizer.encode_plus(
        #                 ref_text,
        #                 add_special_tokens=True, # Add '[CLS]' and '[SEP]'
        #                 max_length=512,
        #                 pad_to_max_length=True,
        #                 return_attention_mask=True,
        #                 return_tensors='pt')
        # input_dict = tokenizer.encode_plus(
        #                 input_text,
        #                 add_special_tokens=True, # Add '[CLS]' and '[SEP]'
        #                 max_length=512,
        #                 pad_to_max_length=True,
        #                 return_attention_mask=True,
        #                 return_tensors='pt')
        # input_ids = input_dict["input_ids"]
        # inp_attention_mask = input_dict["attention_mask"]
        # ref_ids = ref_dict["input_ids"]
        # ref_attention_mask = ref_dict["attention_mask"]
        ref_token_id = 1234
        labels = torch.tensor([1, 1])
        input_ids, ref_ids, attention_mask, sep_idx = construct_input_ref_pair(input_text, ref_token_id)
        token_type_ids, ref_token_type_ids = construct_input_ref_token_type_pair(input_ids, sep_idx)
        

        # test attribution
        attr_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        y_embeds = LayerConductance(custom_forward_func,
                                    model.roberta.encoder.layer[0].intermediate.dense)
        attribute_loop(model, y_embeds, attr_loader, device)

        # attr, delta = y_embeds.attribute(
        #     inputs=input_ids,
        #     baselines=ref_ids,
        #     # additional_forward_args=(labels, token_type_ids, attention_mask),
        #     additional_forward_args=(attention_mask, labels),
        #     return_convergence_delta=True,
        #     n_steps=5,
        # )
        import pdb; pdb.set_trace()







    ###############################################################################################
    ##### TRAINING PROCEDURE
    ###############################################################################################
    # load model and optimizer
    # model.to(device)
    optim = AdamW(model.parameters(), lr=ETA)

    # early stopping
    earlystop = EarlyStopping(patience=2,
                              verbose=False,
                              save_model=True,
                              ckpt_path="checkpoints/berta/roberta-imdb")

    # train loop
    for epoch in range(EPOCHS):
        print(f"Finetuning epoch {epoch}/{EPOCHS}")
        train_loss = train_loop(model, train_loader, optim, device)
        val_loss = test_loop(model, val_loader, device)
        print(f"At epoch {epoch} train loss: {train_loss} and dev loss {val_loss}")
        earlystop(val_loss, model)
        if earlystop.early_stop:
            print(f"Early Stopping training at epoch {epoch}")
            break
    # load model with best validation loss
    model.load_state_dict(torch.load("checkpoints/berta/roberta-imdb.pt", map_location="cpu"))
    model.to(device)
    test_loss = test_loop(model, test_loader, device)
    print(f"The test loss of the model with the best validation loss is {test_loss}")
    
