import os
import time
import cv2
import shutil
import random


def random_select(input_img_path, output_img_path):
    file_path = os.listdir(input_img_path)
    img_files = []
    for img_file in file_path:
        if img_file.split('.')[1] == 'jpg':
            img_files.append(img_file)
    sample = random.sample(img_files, 150)
    for name in sample:
        shutil.move(input_img_path + '/' + name, output_img_path + '/' + name)


def move_img(reference_txt, img_fold, output_fold):
    print('start save img')
    f_reference = open(reference_txt, 'r')
    reference_list = f_reference.read().split('\n')
    j = 0
    for i in reference_list:
        img_file_path = os.path.join(img_fold, i.split('.')[0] + '.jpg')
        txt_file_name = str(i).strip().split('.')[0] + '.txt'
        txt_file_path = os.path.join(img_fold, txt_file_name)
        shutil.move(img_file_path, output_fold)
        shutil.move(txt_file_path, output_fold)
        j = j + 1

def move_img_with_label(label_fold, img_fold, output_fold):
    print('start move img')
    label_list = os.listdir(label_fold)
    for i in label_list:
        img_file_path = os.path.join(img_fold, i.split('.')[0] + '.txt')
        shutil.move(img_file_path, output_fold)



def move_txt(txt_fold, source_img_fold):
    print('start save txt')
    img_file_path = os.listdir(source_img_fold)
    for img_name in img_file_path:
        print(img_name.split('.')[0])
        shutil.move(txt_fold + img_name.split('.')[0] + '.txt', source_img_fold)



if __name__ == "__main__":
    # 实现读取txt内容另存同名图片
    start_time = time.time()
    label_fold = 'F:/8-1_RE/experiment/faster-rcnn/label'
    reference_txt = 'F:/7_PLA/sever_process/new_val-local.txt'
    img_fold = 'F:/8-1_RE/experiment/faster-rcnn/RDD2020_data/annotations/xmls'
    output_fold = 'F:/8-1_RE/experiment/faster-rcnn/xml'
    #move_img_with_label(label_fold, img_fold, output_fold)
    #move_img(reference_txt, img_fold, output_fold)
    #在指定目录下随机筛选100张图片并转移其对应的标注文件
    input_img_path = 'F:/7_pla/cry/2/data/1-out'
    output_img_path = 'F:/7_pla/cry/2/data/append'
    random_select(input_img_path, output_img_path)
    move_img_with_label(output_img_path, input_img_path, output_img_path)
    txt_fold = 'F:/7_PLA/report/Done/Teacher_model_test/test_2/2_train/add_label/'
    source_img_fold = 'F:/7_PLA/report/Done/Teacher_model_test/test_2/2_train/900'
    #move_txt(txt_fold, source_img_fold)
    end_time = time.time()
    cost_time = round(end_time - start_time, 4)
    print("全部数据筛选完毕，用时" + str(cost_time) + "秒，完成图片的筛选")