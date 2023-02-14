import numpy as np

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from transformers import BertModel, BertConfig
from transformers import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup


class IntentModel(nn.Module):
  def __init__(self, args, tokenizer, target_size):
    super().__init__()
    self.tokenizer = tokenizer
    self.model_setup(args)
    self.target_size = target_size

    # task1: add necessary class variables as you wish.
    
    # task2: initilize the dropout and classify layers
    self.dropout = nn.Dropout(p=args.drop_rate)
    self.classify = Classifier(args, target_size)
    
  def model_setup(self, args):
    print(f"Setting up {args.model} model")

    # task1: get a pretrained model of 'bert-base-uncased'
    self.encoder = BertModel.from_pretrained("bert-base-uncased")
    
    self.encoder.resize_token_embeddings(len(self.tokenizer))  # transformer_check

  def forward(self, inputs, targets, plot=False):
    """
    task1: 
        feeding the input to the encoder, 
    task2: 
        take the last_hidden_state's <CLS> token as output of the
        encoder, feed it to a drop_out layer with the preset dropout rate in the argparse argument, 
    task3:
        feed the output of the dropout layer to the Classifier which is provided for you.
    """
    encoder_out = self.encoder(**inputs) # TODO: **inputs --> inputs["input_ids"]
    sequence_last_hidden_state, pooled = encoder_out["last_hidden_state"], encoder_out["pooler_output"]

    clr_hidden_state = sequence_last_hidden_state[:, 0, :] #size: batch_size * embedded_dimension
    
    if plot:
      return clr_hidden_state
    
    dropped_out_clr_hidden_state = self.dropout(clr_hidden_state)
    return self.classify(dropped_out_clr_hidden_state)
  
class Classifier(nn.Module):
  def __init__(self, args, target_size):
    super().__init__()
    input_dim = args.embed_dim
    self.top = nn.Linear(input_dim, args.hidden_dim)
    self.relu = nn.ReLU()
    self.bottom = nn.Linear(args.hidden_dim, target_size)

  def forward(self, hidden):
    middle = self.relu(self.top(hidden))
    logit = self.bottom(middle)
    return logit

# same as baseline model
class CustomModel(IntentModel):
  def __init__(self, args, tokenizer, target_size):
    super().__init__(args, tokenizer, target_size)
    
    # task1: use initialization for setting different strategies/techniques to better fine-tune the BERT model

class SupConModel(IntentModel):
  def __init__(self, args, tokenizer, target_size, feat_dim=768):
    super().__init__(args, tokenizer, target_size)

    # task1: initialize a linear head layer
    self.head = nn.Linear(args.embed_dim, feat_dim)
    
    
  def forward(self, inputs, targets, plot=False):

    """
    task1: 
        feeding the input to the encoder, 
    task2: 
        take the last_hidden_state's <CLS> token as output of the
        encoder, feed it to a drop_out layer with the preset dropout rate in the argparse argument, 
    task3:
        feed the normalized output of the dropout layer to the linear head layer; return the embedding
    """
    encoder_out = self.encoder(**inputs)
    sequence_last_hidden_state, _ = encoder_out["last_hidden_state"], encoder_out["pooler_output"]
    clr_hidden_state = sequence_last_hidden_state[:, 0, :] #size: batch_size * embedded_dimension
    
    if plot:
      return clr_hidden_state
    
    dropped_out_clr_hidden_state = self.dropout(clr_hidden_state)
    out = F.normalize(self.head(dropped_out_clr_hidden_state), dim=1)
    
    return out