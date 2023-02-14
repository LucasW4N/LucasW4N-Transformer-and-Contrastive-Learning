mkdir assets
python main.py --do-train
#python main.py --n-epochs 8 --do-train --task custom --reinit_n_layers 3
#python main.py --n-epochs 14 --do-train --task supcon --batch-size 64
# python main.py --task supcon --loss supcon --plot true --temperature 0.1 --drop-rate 0.5
# python main.py --task supcon --loss simclr --plot true --temperature 0.3 --drop-rate 0.3