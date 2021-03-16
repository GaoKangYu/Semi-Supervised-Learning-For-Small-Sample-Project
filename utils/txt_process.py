# -*- coding:utf-8 -*-
import os
import time

#对比两个txt文档，筛选出不同的内容并写入另一个txt
def BianLi(source_file_path, reference_file_path, output_file_path):
    print("开始数据筛查，请稍等...")
    start_time = time.time()
    f_reference = open(reference_file_path, "r")
    f_source = open(source_file_path, "r")
    f_out = open(output_file_path, "a")
    source_list = f_source.read().split('\n')
    reference_list = f_reference.read().split('\n')
    for i in source_list:
        line = str(i).strip()
        if line in reference_list:
            continue
        else:
            f_out.write(line + '\n')
    end_time = time.time()
    print("全部数据筛选完毕，用时%.2f秒" % (end_time - start_time))



#给txt文档每一行的开头添加指定内容
def add_path(input_txt, output_txt):
    print("开始添加路径，请稍等...")
    start_time = time.time()
    f_source = open(input_txt, "r")
    f_out = open(output_txt, "a")
    source_list = f_source.read().split('\n')
    for i in source_list:
        line = str(i).strip()
        line = '/home/guest/gky/darknet/newData/cnm-val-append/' + line
        f_out.write(line + '\n')
    end_time = time.time()
    print("添加完毕，用时%.2f秒" % (end_time - start_time))


#获取指定文件夹下所有图片名并按一行一个的格式写入TXT文档，最后给每行开头添加指定的额外内容，实现train_list.txt制作
def to_txt(input_path, output_txt_path):
    print("开始制作train_list，请稍等...")
    img_list = os.listdir(input_path)
    f_output_txt = open(output_txt_path, 'a')
    for img_name in img_list:
        if os.path.splitext(img_name)[1] == ".jpg":
            print(img_name)
            f_output_txt.write(img_name)
            f_output_txt.write('\n')


if __name__ == "__main__":
    input_path = 'F:/7_pla/cry/cnm/cnm-val-append'
    output_txt_path = 'F:/7_pla/cry/1.txt'
    output_train_txt_path = 'F:/7_pla/cry/cnm/cnm-val-1.txt'
    to_txt(input_path, output_txt_path)
    add_path(output_txt_path, output_train_txt_path)
    os.remove(output_txt_path)
