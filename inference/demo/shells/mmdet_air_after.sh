source /home/wdy/anaconda/bin/activate mmdet
cd /home/wdy/tai/tianzhi/AerialDetection
python dect_cls_final_gaoair.py --cls_model_pth $1 --input $2 --output $3
