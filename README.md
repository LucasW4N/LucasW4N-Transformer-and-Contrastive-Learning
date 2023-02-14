This respository is refered and modified from this repository of the [paper](https://arxiv.org/abs/2109.03079).

# PA4: Transformers and Contrastive Learning
In this repo we explore and use a pre-trained BERT model to classify user intent. We will first fine-tune BERT as a baseline model, and then apply various training techniques obtained from a blog to build a custom model, and finally, train models with contrastive losses (SupCon and SimCLR).

## Contributors
- Jared Zhang
- Lucas Wan


## Command Line Flags
- `--task TASK`           baseline is fine-tuning bert for classification; tune is advanced techiques fine-tune bert; constast is contrastive learning method
- `--temperature TEMPERATURE`
                        temperature parameter for contrastive loss
- `--reinit_n_layers REINIT_N_LAYERS`
                        number of layers that are reinitialized. Count from last to first.
- `--input-dir INPUT_DIR`
                        The input training data file (a text file).
- `--output-dir OUTPUT_DIR`
                        Output directory where the model predictions and checkpoints are written.
- `--model MODEL`         The model architecture to be trained or fine-tuned.
- `--seed SEED`
- `--dataset {amazon}`    dataset
- `--ignore-cache`        Whether to ignore cache and create a new input data
- `--debug`               Whether to run in debug mode which is exponentially faster
- `--do-train `           Whether to run training.
- `--do-eval`             Whether to run eval on the dev set.
- `--batch-size BATCH_SIZE`
                        Batch size per GPU/CPU for training and evaluation.
- `--learning-rate LEARNING_RATE`
                        Model learning rate starting point.
- `--hidden-dim HIDDEN_DIM`
                        Model hidden dimension.
- `--drop-rate DROP_RATE`
                        Dropout rate for model training
- `--embed-dim EMBED_DIM`
                        The embedding dimension of pretrained LM.
- `--adam-epsilon ADAM_EPSILON`
                        Epsilon for Adam optimizer.
- `--n-epochs N_EPOCHS`   Total number of training epochs to perform.
- `--max-len MAX_LEN `    maximum sequence length to look back
- `--loss LOSS `          choose to use SupCon loss or SimCLR loss, put SupCon or SinCLR
- `--plot PLOT `          put True to let Umap visualize embedding, False to not plot

## How to Run
1. **Install the required packages**
    - Run `pip install -r requirements.txt`

2. **Run main.py with command line argument**
    - For instance: 
  ```python main.py --task supcon --loss simclr --plot true --temperature 0.1 --drop-rate 0.3```
## Files
- `main.py`: Main driver class
- `load.py`: Helper class to download dataset
- `loss.py`: Contains SupCon loss SimCLR loss functions
- `model.py`: Contains three models: Baseline, Custom, and SupCon
- `util.py`: Contains helper functions