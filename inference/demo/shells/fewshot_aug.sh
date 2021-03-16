source /home/wdy/anaconda/bin/activate cc
cd /home/wdy/cc/darknet/
python aug_inference.py --weights $1 --input $2 --output $3
