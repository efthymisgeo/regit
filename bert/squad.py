#!/usr/bin/env python
# coding: utf-8

# # Interpreting BERT Models (Part 1)

# In this notebook we demonstrate how to interpret Bert models using  `Captum` library. In this particular case study we focus on a fine-tuned Question Answering model on SQUAD dataset using transformers library from Hugging Face: https://huggingface.co/transformers/
# 
# We show how to use interpretation hooks to examine and better understand embeddings, sub-embeddings, bert, and attention layers. 
# 
# Note: Before running this tutorial, please install `seaborn`, `pandas` and `matplotlib`, `transformers`(from hugging face) python packages.

# In[1]:


import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from transformers import BertTokenizer, BertForQuestionAnswering, BertConfig, BertForSequenceClassification

from captum.attr import visualization as viz
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer


# In[2]:


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


# The first step is to fine-tune BERT model on SQUAD dataset. This can be easiy accomplished by following the steps described in hugging face's official web site: https://github.com/huggingface/transformers#run_squadpy-fine-tuning-on-squad-for-question-answering 
# 
# Note that the fine-tuning is done on a `bert-base-uncased` pre-trained model.

# After we pretrain the model, we can load the tokenizer and pre-trained BERT model using the commands described below. 

# In[3]:


# replace <PATH-TO-SAVED-MODEL> with the real path of the saved model
model_path = 'bert-base-uncased'

# load model
# model = BertForQuestionAnswering.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()
model.zero_grad()

# load tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)


# A helper function to perform forward pass of the model and make predictions.

# In[4]:


def predict(inputs, token_type_ids=None, position_ids=None, attention_mask=None):
    return model(inputs, token_type_ids=token_type_ids,
                 position_ids=position_ids, attention_mask=attention_mask, )


# Defining a custom forward function that will allow us to access the start and end postitions of our prediction using `position` input argument.

# In[5]:


def squad_pos_forward_func(inputs, token_type_ids=None, position_ids=None, attention_mask=None, position=0):
    pred = predict(inputs,
                   token_type_ids=token_type_ids,
                   position_ids=position_ids,
                   attention_mask=attention_mask)
    pred = pred[position]
    return pred.max(1).values


# Let's compute attributions with respect to the `BertEmbeddings` layer.
# 
# To do so, we need to define baselines / references, numericalize both the baselines and the inputs. We will define helper functions to achieve that.
# 
# The cell below defines numericalized special tokens that will be later used for constructing inputs and corresponding baselines/references.

# In[6]:


ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
sep_token_id = tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
cls_token_id = tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence


# Below we define a set of helper function for constructing references / baselines for word tokens, token types and position ids. We also provide separate helper functions that allow to construct the sub-embeddings and corresponding baselines / references for all sub-embeddings of `BertEmbeddings` layer.

# In[7]:


def construct_input_ref_pair(question, text, ref_token_id, sep_token_id, cls_token_id):
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    text_ids = tokenizer.encode(text, add_special_tokens=False)

    # construct input token ids
    input_ids = [cls_token_id] + question_ids + [sep_token_id] + text_ids + [sep_token_id]

    # construct reference token ids 
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(question_ids) + [sep_token_id] +         [ref_token_id] * len(text_ids) + [sep_token_id]

    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device), len(question_ids)

def construct_input_ref_token_type_pair(input_ids, sep_ind=0):
    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=device)
    ref_token_type_ids = torch.zeros_like(token_type_ids, device=device)# * -1
    return token_type_ids, ref_token_type_ids

def construct_input_ref_pos_id_pair(input_ids):
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    # we could potentially also use random permutation with `torch.randperm(seq_length, device=device)`
    ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=device)

    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
    return position_ids, ref_position_ids
    
def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)

def construct_bert_sub_embedding(input_ids, ref_input_ids,
                                   token_type_ids, ref_token_type_ids,
                                   position_ids, ref_position_ids):
    input_embeddings = interpretable_embedding1.indices_to_embeddings(input_ids)
    ref_input_embeddings = interpretable_embedding1.indices_to_embeddings(ref_input_ids)

    input_embeddings_token_type = interpretable_embedding2.indices_to_embeddings(token_type_ids)
    ref_input_embeddings_token_type = interpretable_embedding2.indices_to_embeddings(ref_token_type_ids)

    input_embeddings_position_ids = interpretable_embedding3.indices_to_embeddings(position_ids)
    ref_input_embeddings_position_ids = interpretable_embedding3.indices_to_embeddings(ref_position_ids)
    
    return (input_embeddings, ref_input_embeddings),            (input_embeddings_token_type, ref_input_embeddings_token_type),            (input_embeddings_position_ids, ref_input_embeddings_position_ids)
    
def construct_whole_bert_embeddings(input_ids, ref_input_ids,                                     token_type_ids=None, ref_token_type_ids=None,                                     position_ids=None, ref_position_ids=None):
    input_embeddings = interpretable_embedding.indices_to_embeddings(input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
    ref_input_embeddings = interpretable_embedding.indices_to_embeddings(ref_input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
    
    return input_embeddings, ref_input_embeddings


# Let's define the `question - text` pair that we'd like to use as an input for our Bert model and interpret what the model was forcusing on when predicting an answer to the question from given input text 

# In[8]:


question, text = "What is important to us?", "It is important to us to include, empower and support humans of all kinds."


# Let's numericalize the question, the input text and generate corresponding baselines / references for all three sub-embeddings (word, token type and position embeddings) types using our helper functions defined above.

# In[9]:


input_ids, ref_input_ids, sep_id = construct_input_ref_pair(question, text, ref_token_id, sep_token_id, cls_token_id)
token_type_ids, ref_token_type_ids = construct_input_ref_token_type_pair(input_ids, sep_id)
position_ids, ref_position_ids = construct_input_ref_pos_id_pair(input_ids)
attention_mask = construct_attention_mask(input_ids)

indices = input_ids[0].detach().tolist()
all_tokens = tokenizer.convert_ids_to_tokens(indices)


# Also, let's define the ground truth for prediction's start and end positions.

# In[10]:


ground_truth = 'to include, empower and support humans of all kinds'

ground_truth_tokens = tokenizer.encode(ground_truth, add_special_tokens=False)
ground_truth_end_ind = indices.index(ground_truth_tokens[-1])
ground_truth_start_ind = ground_truth_end_ind - len(ground_truth_tokens) + 1


# Now let's make predictions using input, token type, position id and a default attention mask.

# In[11]:


# start_scores, end_scores = predict(input_ids,
#                                   token_type_ids=token_type_ids,                                    
#                                   position_ids=position_ids,
#                                 attention_mask=attention_mask)


print(f'Question: {question}')
# print(f'Predicted Answer: ', ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))


# There are two different ways of computing the attributions for `BertEmbeddings` layer. One option is to use `LayerIntegratedGradients` and compute the attributions with respect to that layer. The second option is to pre-compute the embeddings and wrap the actual embeddings with `InterpretableEmbeddingBase`. The pre-computation of embeddings for the second option is necessary because integrated gradients scales the inputs and that won't be meaningful on the level of word / token indices.
# 
# Since using `LayerIntegratedGradients` is simpler, let's use it here.

# In[12]:



encoded_dict = tokenizer.encode_plus(
                          text,           
                          add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                          max_length = 512,         
                          padding = True,
                          return_attention_mask = True,   
                          return_tensors = 'pt',   
                     )

input_ids = encoded_dict['input_ids'].to(device)
attention_mask = encoded_dict['attention_mask'].to(device)

def custom_forward(inputs, labels=None, token_type_ids=None, attention_mask=None):
    outputs = model(input_ids=inputs,
                    labels=labels,
                    token_type_ids=None,
                    attention_mask=attention_mask)
    preds = outputs[0]
    import pdb; pdb.set_trace()
    return torch.softmax(preds, dim = 1)[0]

lig = LayerIntegratedGradients(custom_forward, model.bert.embeddings)
labels = torch.tensor([[1]]).to(device)
import pdb; pdb.set_trace()
attributions, delta = lig.attribute(inputs=input_ids,
                                    additional_forward_args=(token_type_ids, attention_mask), # revise this
                                    return_convergence_delta=True,
                                    n_steps=5)

import pdb;pdb.set_trace()



lig = LayerIntegratedGradients(squad_pos_forward_func, model.bert.embeddings)

attributions_start, delta_start = lig.attribute(inputs=input_ids,
                                  baselines=ref_input_ids,
                                  additional_forward_args=(token_type_ids, position_ids, attention_mask, 0),
                                  return_convergence_delta=True)
attributions_end, delta_end = lig.attribute(inputs=input_ids, baselines=ref_input_ids,
                                additional_forward_args=(token_type_ids, position_ids, attention_mask, 1),
                                return_convergence_delta=True)


import pdb;pdb.set_trace()
# A helper function to summarize attributions for each word token in the sequence.

# In[13]:


def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


# In[14]:


attributions_start_sum = summarize_attributions(attributions_start)
attributions_end_sum = summarize_attributions(attributions_end)

