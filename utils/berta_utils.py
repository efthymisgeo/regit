import torch
from pathlib import Path
from captum.attr import LayerConductance
from sklearn.model_selection import train_test_split

def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir is "neg" else 1)

    return texts, labels


def train_val_split(split, split_labels, val_size=0.2):
  'split a given dataset into two namely train and dev'
  train_split, val_split, train_labels, val_labels = train_test_split(split,
                                                                      split_labels,
                                                                      test_size=val_size)
  return train_split, val_split, train_labels, val_labels

def tokenize(tokenizer, split, truncation=True, padding=True):
    return tokenizer(split, truncation=truncation, padding=padding)


class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def get_imdb_data(encodings, labels):
    return IMDbDataset(encodings, labels)

def get_train_dev_test_data(train, train_labels, test, test_labels, tokenizer, dev_size=0.2):
    # get train dev split
    train_split, val_split, train_labels, val_labels = \
            train_val_split(train, train_labels, val_size=dev_size)
    # tokenize
    train_encodings = tokenize(tokenizer, train_split)
    val_encodings = tokenize(tokenizer, val_split)
    test_encodings = tokenize(tokenizer, test)

    # get torch datasets
    train_dateset = IMDbDataset(train_encodings, train_labels)
    val_dataset = IMDbDataset(val_encodings, val_labels)
    test_dataset = IMDbDataset(test_encodings, test_labels)

    return train_dateset, val_dataset, test_dataset

def get_attn(model, roberta_layer, use_concat=True):
  '''
  Get the attention module from roberta
  Args:
    model (huggingface): the model's forward pass
    roberta_layer (model.roberta.encoder.layer): the self attention layer
    use_concat (bool): take into account the linear transform which aggregates
      attention heads
  Output:
    importance_dict (dict): a dict with all the importance attributors
  '''
  attn = roberta_layer.attention.self
  importance_dict = {}
  importance_dict['q'] = LayerConductance(model, attn.query)
  importance_dict['k'] = LayerConductance(model, attn.key)
  importance_dict['v'] = LayerConductance(model, attn.value)
  if use_concat:
    importance_dict['concat'] = LayerConductance(model, roberta_layer.output.dense)
  return importance_dict

def get_linear(model, linear_layer):
  importance_dict = {}
  importance_dict['dense'] = LayerConductance(model, linear_layer)
  return importance_dict

def parse_model(model, use_attn=True, use_intermed=True, use_out=False):
  '''
  Function used to parse the RobertaModel and returns a model importance dict.
  Args:
    model (roberta model)
    use_attn (bool): flag which handles the usage of attention heads 
      in importance calculation
    use_intermed (bool): flag which handles the usage of intermed projection 
      in importance calculation
    use_out (bool): flag which handles the usage of output projection 
      in importance calculation
  Output:
    model_importance (dict): a dict with all the relevant importances per
      encoder block
  '''
  model_importance = {}
  for id, layers in enumerate(model.roberta.encoder.layer):
    # intialize dict for encoder block -id-
    model_importance[str(id)] = {}
    print(f"Parsing Encoder Block {id}")
    if use_attn:
      model_importance[str(id)]["attn"] = get_attn(model, layers)
    if use_intermed:
      model_importance[str(id)]["intermed"] = \
        get_linear(model, layers.intermediate.dense)
    if use_out:
      model_importance[str(id)]["out"] = \
        get_linear(model, layers.output.dense)
    # import pdb; pdb.set_trace()

  return model_importance

