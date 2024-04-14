"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import io
from PIL import Image
import datetime

import torch
import cv2
import numpy as np
import tensorflow as tf
from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for, Response
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
from subprocess import Popen
import re
import requests
import shutil
import time
from flask import send_file, jsonify
from collections import Counter

app = Flask(__name__)

    
@app.route("/", methods=["POST"])
def predict_img():
    names = [ 'paper', 'paper_pack', 'paper_cup','can','reusable_glass','brown_glass','green_glass','white_glass','etc_glass','pet','plastic','vinyl','paper_dirty','paper_cup_dirty','can_dirty','etc_glass_dirty','pet_dirty_packaging','pet_dirty','plastic_dirty','vinyl_dirty','reusable_glass_packaging','brown_glass_packaging','green_glass_packaging','white_glass_packaging','pet_packaging','white_Styrofoam','color_Styrofoam','Styrofoam_dirty','battery']
    
    if 'file' not in request.files:
        print("nononono")
        return
    
    f = request.files['file']
    basepath = os.path.dirname(__file__)
    filepath = os.path.join(basepath,'uploads.jpg')
    f.save(filepath)
    
    img = Image.open('uploads.jpg')
    img_resized = img.resize((640,640))
    img_resized.save('uploads.jpg')

    process = Popen(["python3", "detect.py", '--source', filepath,"--save-txt", "--weights","best.pt",'--name','result','--conf-thres','0.4'], shell=False)
    process.wait()
    
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    image_path = folder_path+'/'+latest_subfolder+'/'+'uploads.jpg'
    label_path = folder_path+'/'+latest_subfolder+'/labels/uploads.txt'
    
    labels = [] 
    with open(label_path, 'r') as file:
        for line in file:
            label = line.split()[0]
            labels.append(names[int(label)])
    element_count = Counter(labels)
    print(element_count)
    
    # element_count가 쓰레기 종류:개수 적혀있는 Counter 객체
    # image_path가 이미지가 결과 이미지 저장되어있는 경로
    
    return send_file(image_path,mimetype='image/jpeg')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=80, type=int, help="port number")
    args = parser.parse_args()
    model = torch.hub.load('.', 'custom','best.pt', source='local')
    model.eval()
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat

