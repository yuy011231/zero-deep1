import pickle
import random
import numpy as np
from mnist import load_mnist
from PIL import Image

def img_save(flat_img: np.ndarray):
    img_shaped=flat_img.reshape(28, 28)
    Image.fromarray(np.uint8(img_shaped)).save("output.png")

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    c=np.max(x)
    exp_x=np.exp(x-c)
    sum_exp_x=np.sum(exp_x)
    return exp_x/sum_exp_x

def get_data():
    (train_img, train_label), (test_img, test_label) = load_mnist(normalize=False)
    return (train_img, train_label), (test_img, test_label)

def init():
    with open("chapter_3/sample_weight.pkl", "rb") as f:
        return pickle.load(f)
    
def predict(weight, x):
    W1, W2, W3=weight["W1"], weight["W2"], weight["W3"]
    b1, b2, b3=weight["b1"], weight["b2"], weight["b3"]
    
    a1=np.dot(x, W1)+b1
    x2=sigmoid(a1)
    a2=np.dot(x2, W2)+b2
    x3=sigmoid(a2)
    a3=np.dot(x3, W3)+b3
    return softmax(a3)

if __name__=="__main__":
    _, test_data = get_data()

    target_index=random.randint(0, 10000)
    target_img=test_data[0][target_index]
    target_label=test_data[1][target_index]

    img_save(target_img)
    result=predict(init(), target_img)
    print(f"predict : {np.argmax(result)}")
    print(f"gt : {target_label}")
