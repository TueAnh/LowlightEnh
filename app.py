from __future__ import division, print_function
# coding=utf-8
from pathlib import Path
import cv2
import sys
import os
import glob
import torch, gc
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import numpy as np
import model
import torch.optim
import time
import model
from PIL import Image
import time

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
# Load model
DCE_net = model.enhance_net_nopool().cuda()
DCE_net_noCol = model.enhance_net_nopool().cuda()
DCE_net_noExp = model.enhance_net_nopool().cuda()
DCE_net_noSpa = model.enhance_net_nopool().cuda()
DCE_net_noLtv = model.enhance_net_nopool().cuda()
DCE_net.load_state_dict(torch.load('snapshots/Epoch99.pth'))
DCE_net_noCol.load_state_dict(torch.load('snapshots/Epoch99_noCol.pth'))
DCE_net_noExp.load_state_dict(torch.load('snapshots/Epoch199_noLexp.pth'))
DCE_net_noLtv.load_state_dict(torch.load('snapshots/Epoch199_noLtv.pth'))
DCE_net_noSpa.load_state_dict(torch.load('snapshots/Epoch199_noLspa.pth'))
# DCE_net.load_state_dict(torch.load('snapshots/Epoch199.pth'))
print('Model loaded. Check http://127.0.0.1:5000/')


def gamma_correction(img, gamma):
    return np.power(img, gamma)

def gamma_like(img, enhanced):
    x, y = img.mean(), enhanced.mean()
    gamma = np.log(y) / np.log(x)
    return gamma_correction(img, gamma)

def make_grid(nrow, ncol, h, w, hspace, wspace):
    grid = np.ones(
        (nrow * h + hspace * (nrow - 1), ncol * w + (ncol - 1) * wspace, 3),
        dtype=np.float32
    )
    return grid

def putText(im, text, pos, color, size=1, scale=2, font=cv2.FONT_HERSHEY_SIMPLEX):
    im = cv2.putText(im, text, pos, font, size, color, scale)
    return im

def read_image(fp, h, w):
    fp = str(fp)
    img = cv2.imread(fp)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.
    img = cv2.resize(img, (w, h))
    return img

def row_arrange(wspace, images):
    n = len(images)
    h, w, c = images[0].shape
    row = np.ones((h, (n - 1) * wspace + n * w, c))
    curr_w = 0
    for image in images:
        row[:, curr_w:curr_w + w, :] = image
        curr_w += w + wspace
    return row





def lowlight(image_path, model_enh='zerodce'):
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    data_lowlight = (Image.open(image_path))
    data_lowlight = (np.asarray(data_lowlight)/255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2,0,1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)
    method_param = {
        'zerodce': DCE_net,
        'nolspa': DCE_net_noSpa,
        'nolexp': DCE_net_noExp,
        'nolcol': DCE_net_noCol,
        'noltvA': DCE_net_noLtv,
    }
    # DCE_net = model.enhance_net_nopool().cuda()
    # # checkpoint = torch.load('snapshots/Epoch99.pth')
    
    # DCE_net.load_state_dict(torch.load('snapshots/Epoch199.pth'))
    # print("Model's state_dict:")
    # for param_tensor in DCE_net.state_dict():
    #   print(param_tensor, "\t", DCE_net.state_dict()[param_tensor].size())
    start = time.time()
    _,enhanced_image,_ = method_param[model_enh](data_lowlight)
    end_time = (time.time() - start)
    print("time = {0}".format(end_time))
    
    image_path = image_path.replace('uploads','static/result/'+model_enh)
    image_path = image_path.replace('\\','/')
    result_path = image_path
    if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
        os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))
    torchvision.utils.save_image(enhanced_image, result_path)
    return result_path

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        return os.path.relpath(lowlight(file_path),basepath)
    return None


@app.route('/compare', methods=['GET', 'POST'])
def compare():
    
    if request.method == 'GET':
        return render_template('compare.html')
    method1 = request.form['model']
    method2 = request.form['compare']
    print(method2)
    f = request.files['file']
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)

    h, w = 384, 512
    hspace, wspace = 8, 4
    ncol = 2
    nrow = 2
    grid = make_grid(nrow, ncol, h, w, hspace=hspace, wspace=wspace)
    outpath = lowlight(file_path, method1)
    outpath2 = lowlight(file_path, method2)
    ori = read_image(file_path, h, w)
    comp1 = read_image(outpath, h, w)
    comp2 = read_image(outpath2, h, w)
    gamma_fixed = gamma_correction(ori, 0.4)
    gamma_alike = gamma_like(ori, comp1)

    pos = (20, 42)
    color = (1.0, 1.0, 1.0)
    comp1 = putText(comp1, method1, pos, color=color)
    comp2 = putText(comp2, method2, pos, color=color)
    gamma_fixed = putText(gamma_fixed, 'Gamma(0.4)', pos, color=color)
    gamma_alike = putText(gamma_alike, 'Gamma(adaptive)', pos, color=color)
    row = row_arrange(wspace, [comp1, comp2])
    grid[0:h,:,:] = row
    grid[hspace+h:hspace+2*h,:,:] = row_arrange(wspace,[gamma_fixed, gamma_alike])
    # curr_h += h + hspace

    grid_image = Image.fromarray((grid * 255).astype(np.uint8), mode='RGB')
    result_path = os.path.join(
        basepath, 'static/result', 'comparison_'+method1+method2+outpath.split('/')[-1])
    grid_image.save(result_path)
    # # Compare page
    return os.path.relpath(result_path,basepath)


if __name__ == '__main__':
    app.run(debug=True)

