#python3 train.py --momentum=0.9 --lr=100 --batch_size=128 --resume=3240000 --finetune=True --max_epoch=200|tee finetune.log
python3 train.py --momentum=0.9 --lr=2 --batch_size=64 --resume=4040000 --finetune=True --max_epoch=200|tee finetune.log
