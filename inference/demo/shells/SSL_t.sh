source /home/wdy/anaconda/bin/activate SSL
cd /home/wdy/SSL/darknet
python inference_SSL-T.py --weights $1 --input $2 --output $3
