#|export
import timm
from fastai.vision.all import *
import gradio as gr
#export
learn = load_learner('model.pkl')
#export
categories = learn.dls.vocab

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))
#export
image = gr.Image()
label = gr.Label()
examples = ['basset.jpg']
#export
intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)