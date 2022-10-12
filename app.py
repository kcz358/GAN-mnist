import torch
from flask import Flask, render_template, request, get_template_attribute
import os
import numpy as np
from base64 import b64encode
import matplotlib.pyplot as plt
from gan_train import Generator

app = Flask(__name__)

@app.route("/", methods = ['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/generate", methods = ['GET', 'POST'])
def generate():
    if request.method == 'POST' or request.method == 'GET':
        epoch_num = request.form.get('model_selector')
        checkpoints = torch.load("checkpoints/epoch_{}".format(epoch_num) , map_location=torch.device('cpu'))
        generator.load_state_dict(checkpoints['generator_state_dict'])
        images = np.zeros((28*10, 28*15))
        for w in range(10):
            for h in range(15):
                images[w*28:(w+1)*28, h*28:(h+1)*28] = generator(torch.randn(100)).reshape(28,28).detach().numpy()
                    
        plt.imsave("static/images/pic.jpeg", images, cmap = 'Greys')
        with open('static/images/pic.jpeg', 'rb') as imagefile:
            img = b64encode(imagefile.read()).decode('utf-8')
        
        generate_img = get_template_attribute("_generator.html", "generate")
        html = generate_img(img)
        return html
port = int(os.getenv("PORT", 5000))
if __name__ == "__main__":
    generator = Generator()
    app.run(debug = True, host='0.0.0.0', port=port)