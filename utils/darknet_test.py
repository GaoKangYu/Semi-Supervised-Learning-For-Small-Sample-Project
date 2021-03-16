#!python3
#pylint: disable=R, W0401, W0614, W0703
from ctypes import *
import random
import os
import cv2
import numpy as np
import math
import time
import shutil



def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

hasGPU = True
if os.name == "nt":
    cwd = os.path.dirname(__file__)
    os.environ['PATH'] = cwd + ';' + os.environ['PATH']
    winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")
    winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_no_gpu.dll")
    envKeys = list()
    for k, v in os.environ.items():
        envKeys.append(k)
    try:
        try:
            tmp = os.environ["FORCE_CPU"].lower()
            if tmp in ["1", "true", "yes", "on"]:
                raise ValueError("ForceCPU")
            else:
                print("Flag value '"+tmp+"' not forcing CPU mode")
        except KeyError:
            # We never set the flag
            if 'CUDA_VISIBLE_DEVICES' in envKeys:
                if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:
                    raise ValueError("ForceCPU")
            try:
                global DARKNET_FORCE_CPU
                if DARKNET_FORCE_CPU:
                    raise ValueError("ForceCPU")
            except NameError:
                pass
            # print(os.environ.keys())
            # print("FORCE_CPU flag undefined, proceeding with GPU")
        if not os.path.exists(winGPUdll):
            raise ValueError("NoDLL")
        lib = CDLL(winGPUdll, RTLD_GLOBAL)
    except (KeyError, ValueError):
        hasGPU = False
        if os.path.exists(winNoGPUdll):
            lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
            print("Notice: CPU-only mode")
        else:
            # Try the other way, in case no_gpu was
            # compile but not renamed
            lib = CDLL(winGPUdll, RTLD_GLOBAL)
            print("Environment variables indicated a CPU run, but we didn't find `"+winNoGPUdll+"`. Trying a GPU run anyway.")
else:
    lib = CDLL("./libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE,c_char_p]

def network_width(net):
    return lib.network_width(net)

def network_height(net):
    return lib.network_height(net)

predict = lib.network_predict_ptr
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

init_cpu = lib.init_cpu

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict_ptr
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

predict_image_letterbox = lib.network_predict_image_letterbox
predict_image_letterbox.argtypes = [c_void_p, IMAGE]
predict_image_letterbox.restype = POINTER(c_float)

def array_to_image(arr):
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w,h,c,data)
    return im, arr

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        if altNames is None:
            nameTag = meta.names[i]
        else:
            nameTag = altNames[i]
        res.append((nameTag, out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45, debug= False):
    """
    Performs the meat of the detection
    """
    #pylint: disable= C0321
    im = load_image(image, 0, 0)
    if debug: print("Loaded image")
    ret = detect_image(net, meta, im, thresh, hier_thresh, nms, debug)
    free_image(im)
    if debug: print("freed image")
    return ret

def detect_image(net, meta, im, thresh=.5, hier_thresh=.5, nms=.45, debug= False):
    num = c_int(0)
    if debug: print("Assigned num")
    pnum = pointer(num)
    if debug: print("Assigned pnum")
    predict_image(net, im)
    letter_box = 0
    #predict_image_letterbox(net, im)
    #letter_box = 1
    if debug: print("did prediction")
    #dets = get_network_boxes(net, custom_image_bgr.shape[1], custom_image_bgr.shape[0], thresh, hier_thresh, None, 0, pnum, letter_box) # OpenCV
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, letter_box)
    if debug: print("Got dets")
    num = pnum[0]
    if debug: print("got zeroth index of pnum")
    if nms:
        do_nms_sort(dets, num, meta.classes, nms)
    if debug: print("did sort")
    res = []
    if debug: print("about to range")
    for j in range(num):
        if debug: print("Ranging on "+str(j)+" of "+str(num))
        if debug: print("Classes: "+str(meta), meta.classes, meta.names)
        for i in range(meta.classes):
            if debug: print("Class-ranging on "+str(i)+" of "+str(meta.classes)+"= "+str(dets[j].prob[i]))
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                if altNames is None:
                    nameTag = meta.names[i]
                else:
                    nameTag = altNames[i]
                if debug:
                    print("Got bbox", b)
                    print(nameTag)
                    print(dets[j].prob[i])
                    print((b.x, b.y, b.w, b.h))
                res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h), i))
    if debug: print("did range")
    res = sorted(res, key=lambda x: -x[1])
    if debug: print("did sort")
    free_detections(dets, num)
    if debug: print("freed detections")
    return res


netMain = None
metaMain = None
altNames = None

def write_label(imagePath, file_name, line):
    txt_file = file_name + '.txt'
    txt_file_path = os.path.join(imagePath.split('\\')[0], txt_file)
    line = str(line)
    with open(txt_file_path, "a") as f:
        f.write(line)
        f.write('\n')


def performDetect(imagePath, thresh = 0.3, configPath = "F:/7_pla/[!]inference/cfg/SSL.cfg", weightPath = "F:/7_pla/[!]inference/weight/SSL-S.weights", metaPath= "F:/7_pla/[!]inference/cfg/SSL.data", initOnly= False):
    # Import the global variables. This lets us instance Darknet once, then just call performDetect() again without instancing again
    global metaMain, netMain, altNames #pylint: disable=W0603
    if netMain is None:
        netMain = load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = load_meta(metaPath.encode("ascii"))
    if altNames is None:
        # In Python 3, the metafile default access craps out on Windows (but not Linux)
        # Read the names file and create a list to feed to detect
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    # Do the detection
    img = cv2.imread(imagePath)
    detections = detect(netMain, metaMain, imagePath.encode("ascii"), thresh)
    img_file_name = imagePath.split('\\')[1].split('.')[0]
    confidence = 0
    count_ship1 = 0
    count_ship2 = 0
    count_j6 = 0
    count_j8 = 0
    count_z9 = 0
    count_chf = 0
    button = (0, img.shape[0]-20)
    for detection in detections:
        #模型预测bbox，[x中心点绝对值，y中心点绝对值，宽绝对值，高绝对值]
        confidence = detection[1]
        print('置信度：', confidence)
        bounds = detection[2]
        #变为相对值，保留前6位（labelimg生成的标签是6位）
        x_center = round(bounds[0] / img.shape[1], 6)
        y_center = round(bounds[1] / img.shape[0], 6)
        width = round(bounds[2] / img.shape[1], 6)
        height = round(bounds[3] / img.shape[0], 6)
        name = detection[0]
        if name == 'ship1':
            count_ship1 = count_ship1 + 1
        elif name == 'ship2':
            count_ship2 = count_ship2 + 1
        elif name == 'J6':
            count_j6 = count_j6 + 1
        elif name == 'J8':
            count_j8 = count_j8 + 1
        elif name == 'Z9':
            count_z9 = count_z9 + 1
        elif name == 'CHF':
            count_chf = count_chf + 1
        yExtent = int(bounds[3])
        xEntent = int(bounds[2])
        xCoord = int(bounds[0] - bounds[2] / 2)
        yCoord = int(bounds[1] - bounds[3] / 2)
        boundingBox = [
            [xCoord, yCoord],
            [xCoord, yCoord + yExtent],
            [xCoord + xEntent, yCoord + yExtent],
            [xCoord + xEntent, yCoord]
        ]
        #print(boundingBox[0], boundingBox[2])
        p1 = (boundingBox[0][0], boundingBox[0][1])
        p2 = (boundingBox[2][0], boundingBox[2][1])
        cv2.rectangle(img, p1, p2, (255, 0, 0), 2)
        re_text_line = str(name) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height)
        write_label(imagePath, img_file_name, re_text_line)
        txt_file = img_file_name + '.txt'
        txt_file_path = os.path.join(imagePath.split('\\')[0], txt_file)
        if confidence < 0.95:
            print('存在低置信度目标，该标注文件已被删除')
            os.remove(txt_file_path)
        cv2.putText(img, str(name), p1, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                          (0, 0, 255), 2)
    return img


def dota2yolo(label_path, label_file_name):
    labeled_img_path = 'F:/7_PLA/dataset/crop_output_val/images/' + label_file_name.split('.')[0] + '.png'
    print('当前正在处理：', label_file_name.split('.')[0] + '.png')
    labled_img = cv2.imread(labeled_img_path)
    output_label_fold = 'F:/7_PLA/dataset/crop_output_val/val_label/' + label_file_name
    with open(label_path, "r") as f:  # 打开文件
        for lines in f.readlines():
            label = lines.split(' ')
            if label[9] != '2\n' and label[9] != '2':
                # dota:x1, y1, x2, y2, x3, y3, x4, y4, category, difficult
                x_1 = float(label[0])
                y_1 = float(label[1])
                x_2 = float(label[2])
                y_2 = float(label[3])
                x_3 = float(label[4])
                y_3 = float(label[5])
                x_4 = float(label[6])
                y_4 = float(label[7])
                name = label[8]
                if name == 'plane':
                    name = 0
                elif name == 'ship':
                    name = 1
                elif name == 'helicopter':
                    name = 2
                # yolo:类名，x中心点, y中心点,宽，高
                x_center = round((x_1 + x_2 + x_3 + x_4) / 4 / labled_img.shape[1], 6)
                y_center = round((y_1 + y_2 + y_3 + y_4) / 4 / labled_img.shape[0], 6)
                width = round(math.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2) / labled_img.shape[1], 6)
                height = round(math.sqrt((x_1 - x_4) ** 2 + (y_1 - y_4) ** 2) / labled_img.shape[0], 6)
                label = str(name) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height)
                print(label)
                with open(output_label_fold, "a") as f_out:  # 打开文件
                    f_out.write(label)
                    f_out.write('\n')


if __name__ == "__main__":
    #完成批处理预测、生成伪标签
    start_time = time.time()
    print('start')
    fold_path = "F:/7_pla/[!]inference/test_data"
    output_path = "F:/7_pla/[!]inference/output/"
    file_names = os.listdir(fold_path)
    for file_name in file_names:
        if os.path.splitext(file_name)[1] == ".jpg":
            print('当前正在处理：', file_name)
            img_path = os.path.join(fold_path, file_name)
            img = performDetect(img_path)
            cv2.imwrite(output_path + file_name, img)
            #cv2.imshow(file_name, img)
            #cv2.waitKey(0)
    end_time = time.time()
    print("全部数据检测完毕，用时%.2f秒" % (end_time - start_time))
    print('done')
    '''
    #完成批处理标注格式转换
    print('start change label')
    label_fold = 'F:/7_PLA/dataset/crop_output_val/labelTxt'
    file_names = os.listdir(label_fold)
    for file_name in file_names:
        if os.path.splitext(file_name)[1] == ".txt":
            label_path = os.path.join(label_fold, file_name)
            dota2yolo(label_path, file_name)
    print('done')
    '''
