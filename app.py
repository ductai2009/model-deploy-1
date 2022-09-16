
from flask import Flask,render_template, request
# import pickle
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
app = Flask(__name__)
# model = pickle.load(open('./model.pkl','rb'))
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
model = load_model('model1.h5')
# decode = ['airplane',
#  'bicycle',
#  'boat',
#  'motorbus',
#  'motorcycle',
#  'seaplane',
#  'train',
#  'truck']
decode = {'living room': 0,
 'lion': 1,
 'smiling dog': 2,
 'lion with closed eyes.': 3,
 'dog': 4,
 'dog with a collar on its neck': 5,
 'horse': 6,
 'canine.': 7,
 'parrot': 8,
 'lion with an open mouth.': 9,
 'conference room': 10,
 'giraffe': 11,
 'bedroom': 12,
 'boat': 13,
 'bird.': 14,
 'lion with open eyes.': 15,
 'lion with mane on its neck.': 16,
 ' elephant': 17,
 'car': 18,
 'seaplane': 19,
 ' airplane': 20,
 'truck': 21}
def name_to_vec(name:str):
  if name in decode.keys():
    return tf.keras.utils.to_categorical(decode[name] , len(decode))
  else:
    return np.zeros(len(decode))


def cut_img(img):
  num = 9
  range_pixel = int(img.shape[0]/3)
  list_img = []
  for i in range(int(num**0.5)):
    for j in range(int(num**0.5)):
      list_img.append(img[i*range_pixel:i*range_pixel+range_pixel,j*range_pixel:j*range_pixel+range_pixel])
  return list_img

# def process(img):
#   im_new = cv2.resize(img, (128,128))
#   return im_new
# def process(img):
#     img = cv2.resize(img, (128, 128))
#     return img
# def predict1(img):
#     # img = process(img).reshape(1,128,128,3)
#     max = np.argmax(model.predict(img))
    # return decode[max]
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")


@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    imgfile = request.files['imagefile']
    img_path = "./templates/image/" + imgfile.filename
    imgfile.save(img_path)
    img_9 = cv2.imread(img_path)[:,:,::-1]
    print(img_9.shape)
    # img_9 = load_img(img_path)
    
    # img_9 = img_to_array(img_9)
    # img_9 = img_9.reshape(1, img_9.shape[0], img_9.shape[1],img_9.shape[2])
    # full_path = 'img/train/00058_seaplane.png'
    name = img_path.split('_')[-1].split('.')[0]
    # img_9 = cv2.imread(img_path)[:,:,::-1]
    if img_9.shape[0] != 384: img_9 = cv2.resize(img_9, (384,384))
    list_img = cut_img(img_9)
    x1 = np.array(list_img)
    x2 = np.array([name_to_vec(name)]*9)
    # np.argmax(model.predict([x1,x2]))
    # plt.imshow(img_9)
    model.predict([x1,x2])*1
    predict = (model.predict([x1,x2]) >= 0.5).reshape(1,9)*1
    # predict = (model.predict([x1,x2]) >= 0.1)*1
    # list = []
    # for i in predict:
    #     if i == 0:
    #         print("co")
    #         list.append("co")
    #     if i ==1:
    #         print("ko")
    #         list.append("ko")
    # xuat = '%s %s %s %s %s %s %s %s %s', (list[0], list[1], list[2], list[3], list[4], list[5], list[6], list[7], list[8] )
    # xuat = predict(img_9)
    predict = str(predict)
    
    return render_template("index.html", pre = predict)
# def predict_label(img_path):
# 	i = image.load_img(img_path, target_size=(100,100))
# 	i = image.img_to_array(i)/255.0
# 	i = i.reshape(1, 100,100,3)
# 	p = model.predict_classes(i)
# 	return dic[p[0]]
# @app.route("/submit", methods = ['GET', 'POST'])
# def get_output():
# 	if request.method == 'POST':
# 		img = request.files['my_image']

# 		img_path = "static/" + img.filename	
# 		img.save(img_path)

# 		p = predict(img_path)

	# return render_template("index.html", prediction = p, img_path = img_path)
if __name__ == '__main__':
    app.run(port = 1000,debug = True)