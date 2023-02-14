import time
import numpy as np
import random
import torch

from tqdm import tqdm as progress_bar

from utils import set_seed, setup_gpus, check_directories
from dataloader import get_dataloader, check_cache, prepare_features, process_data, prepare_inputs
from load import load_data, load_tokenizer
from arguments import params
from model import IntentModel, SupConModel, CustomModel
from torch import nn
from transformers import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

import umap
import umap.plot


device='cuda'


def baseline_train(args, model, datasets, tokenizer):
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()
    # task1: setup train dataloader
    train_dataloader = get_dataloader(args, datasets["train"], "train")

    # task2: setup model's optimizer_scheduler if you have
    model.optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
    #model.scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size=args.n_epochs / 3)
    model.scheduler = torch.optim.lr_scheduler.ExponentialLR(model.optimizer, 0.9)

    # task3: write a training loop
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()

        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch, model) 
            logits = model(inputs, labels)
            loss = criterion(logits, labels)
            loss.backward()

            model.optimizer.step()  # backprop to update the weights
            #model.scheduler.step()  # Update learning rate schedule #Why step inside batch?
            model.zero_grad()
            losses += loss.item()
    
        model.scheduler.step()
        run_eval(args, model, datasets, tokenizer, split='validation')
        losses /= len(train_dataloader)
        print('epoch', epoch_count + 1, '| losses:', losses)
    
    if args.plot != "False":
        visualize_embeddings(args, model, datasets, caller="baseline")
  
def custom_train(args, model, datasets, tokenizer):
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()
    # task1: setup train dataloader
    train_dataloader = get_dataloader(args, datasets["train"], "train")

    # task2: setup model's optimizer_scheduler if you have
    total_step = args.n_epochs * len(train_dataloader)
    model.optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
    #model.scheduler = torch.optim.lr_scheduler.ExponentialLR(model.optimizer, 0.9)
    model.scheduler = get_linear_schedule_with_warmup(                
                optimizer = model.optimizer,
                num_warmup_steps = total_step / 15,
                num_training_steps = total_step)

    # task3: write a training loop
    step_cnt = 0
    step_per_validation = total_step / 50
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()

        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch, model) 
            logits = model(inputs, labels)
            loss = criterion(logits, labels) 
            loss.backward()

            model.optimizer.step()  # backprop to update the weights
            model.scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            losses += loss.item()
            step_cnt += 1

            # For frequent validation
            # if (step_cnt == step_per_validation):
            # 	run_eval(args, model, datasets, tokenizer, split='validation')
            # 	step_cnt = 0
    
        run_eval(args, model, datasets, tokenizer, split='validation')
        #model.scheduler.step()
        losses /= len(train_dataloader)
        print('epoch', epoch_count + 1, '| losses:', losses)
  

def run_eval(args, model, datasets, tokenizer, split='validation'):
    model.eval()
    dataloader = get_dataloader(args, datasets[split], split)
    criterion = nn.CrossEntropyLoss()

    acc = 0
    losses = 0
    for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
        inputs, labels = prepare_inputs(batch, model)
        logits = model(inputs, labels)
        eval_loss = criterion(logits, labels)
        eval_loss.backward()
        losses += eval_loss.item()
        
        tem = (logits.argmax(1) == labels).float().sum()
        acc += tem.item()
  
    losses /= len(dataloader)
    print(f'{split} acc:', acc/len(datasets[split]), f'|dataset split {split} size:', len(datasets[split]))
    print(f'{split} losses:', losses)

def supcon_train(args, model, datasets, tokenizer):
    from loss import SupConLoss
    criterion = SupConLoss(temperature=args.temperature)
    train_dataloader = get_dataloader(args, datasets["train"], "train")
    caller = ""
    # set optimizer (and scheduler?)
    model.optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
    total_step = args.n_epochs * len(train_dataloader)
    model.scheduler = get_linear_schedule_with_warmup(                
                optimizer = model.optimizer,
                num_warmup_steps = total_step / 15,
                num_training_steps = total_step)
    
    for epoch_count in range(args.n_epochs):
        losses = 0.0
        model.train()
        
        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch, model)
            
            # compute loss
            output1 = model(inputs, labels)    # forward passes
            output2 = model(inputs, labels)
            logits = torch.cat([output1.unsqueeze(1), output2.unsqueeze(1)], dim=1)
            if args.loss == "SupCon" or args.loss == "supcon":
                loss = criterion(logits, labels)    # supcon loss
                caller = "supcon"
            elif args.loss == "SimCLR" or args.loss == "simclr":
                loss = criterion(logits)        # simclr loss
                caller = "simclr"
            else:
                raise ValueError("contrastive method not supported")
            
            # update losses    
            losses += loss.item()
            
            # SGD
            model.optimizer.zero_grad()     # updating weights
            loss.backward()
            model.optimizer.step()          # calculating gradients
            model.scheduler.step() 
            
        # run_eval(args, model, datasets, tokenizer, split='validation')    
        losses /= len(train_dataloader)
        
        print('epoch', epoch_count + 1, '| losses:', losses)
        
    if args.plot != "False":
        visualize_embeddings(args, model, datasets, caller)
    

# helper function to draw the embeddings of BERT encoder of the CLS token
def visualize_embeddings(args, model, datasets, caller):
    dataloader = get_dataloader(args, datasets['test'], 'test')
    
    # lists we wanna feed into umap
    embeddings_list = np.array([[]])
    label_list = np.array([[]])
    
    for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
        inputs, labels = prepare_inputs(batch, model)
        
        # forward pass to get the embeddings
        embeddings = model(inputs, labels, plot=True)
        
        # get a list of labels and embeddings of first 10 classes
        embeddings = embeddings.cpu().detach().numpy()
        labels_to_plot = labels.cpu().detach().numpy()
        idx = []
        for i in range(len(labels)):
            if labels[i] < 10:
                idx.append(i)
        np.array(idx)
        
        embeddings = embeddings[idx]
        labels_to_plot = labels_to_plot[idx]
        
        # append the current batch's list to the total list
        if embeddings_list.shape[0] == 1:
            embeddings_list = embeddings
            label_list = labels_to_plot
        else:
            embeddings_list = np.concatenate((embeddings_list, embeddings), axis=0)
            label_list = np.concatenate((label_list, labels_to_plot), axis=0)
    
    print("------ visualizing embeddings ------")
    
    mapper = umap.UMAP().fit(embeddings_list)
    image = umap.plot.points(mapper, label_list)
    figure = image.get_figure()
    
    # save and name the figure according to this function's caller
    random.seed(time.process_time())
    name = caller + str(random.randint(0,1000))
    figure.savefig(name)
    print("------ done, check "+ name + ".png")

if __name__ == "__main__":
    args = params()
    args = setup_gpus(args)
    args = check_directories(args)
    set_seed(args)

    cache_results, already_exist = check_cache(args)
    tokenizer = load_tokenizer(args)

    if already_exist:
        features = cache_results
    else:
        data = load_data()
        features = prepare_features(args, data, tokenizer, cache_results)
    datasets = process_data(args, features, tokenizer)
    for k,v in datasets.items():
        print(k, len(v))
 
    if args.task == 'baseline':
        model = IntentModel(args, tokenizer, target_size=60).to(device)
        run_eval(args, model, datasets, tokenizer, split='validation')
        run_eval(args, model, datasets, tokenizer, split='test')
        baseline_train(args, model, datasets, tokenizer)
        run_eval(args, model, datasets, tokenizer, split='test')
    elif args.task == 'custom': # you can have multiple custom task for different techniques
        model = CustomModel(args, tokenizer, target_size=60).to(device)
        run_eval(args, model, datasets, tokenizer, split='validation')
        run_eval(args, model, datasets, tokenizer, split='test')
        custom_train(args, model, datasets, tokenizer)
        run_eval(args, model, datasets, tokenizer, split='test')
    elif args.task == 'supcon':
        model = SupConModel(args, tokenizer, target_size=60).to(device)
        supcon_train(args, model, datasets, tokenizer)
   

