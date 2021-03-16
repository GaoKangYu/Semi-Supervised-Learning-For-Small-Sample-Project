source /home/wdy/anaconda/bin/activate cc
cd /home/wdy/cc/darknet/
python dota_inference.py --weights $1 --input $2 --output $3
