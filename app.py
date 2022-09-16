from flask import Flask,render_template, request
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
app = Flask(__name__)
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
model = load_model('model1.h5')
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
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")
@app.route('/predict', methods = ['POST'])
def predict():
    imgfile = request.files['imagefile']
    img_path = "./templates/image/" + imgfile.filename
    imgfile.save(img_path)
    img_9 = cv2.imread(img_path)[:,:,::-1]
    print(img_9.shape)
    name = img_path.split('_')[-1].split('.')[0]
    if img_9.shape[0] != 384: img_9 = cv2.resize(img_9, (384,384))
    list_img = cut_img(img_9)
    x1 = np.array(list_img)
    x2 = np.array([name_to_vec(name)]*9)
    model.predict([x1,x2])*1
    predict = (model.predict([x1,x2]) >= 0.5).reshape(1,9)*1
    predict = str(predict)
    return render_template("index.html", pre = predict)
if __name__ == '__main__':
    app.run( host = '0.0.0.0', port= 8080)