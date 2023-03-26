import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from keras.models import model_from_json
import librosa 
import numpy as np

import librosa
import math
import os
import re

import numpy as np


genre_list = [
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
    ]

def load_model(model_path, weights_path):
    "Load the trained LSTM model from directory for genre classification"
    with open(model_path, "r") as model_file:
        trained_model = model_from_json(model_file.read())
    trained_model.load_weights(weights_path)
    trained_model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return trained_model


def get_genre(model, music_path):
    "Predict genre of music using a trained model"
    prediction = model.predict(extract_audio_features(music_path))
    predict_genre = genre_list[np.argmax(prediction)]
    return predict_genre

def extract_audio_features(file):
    "Extract audio features from an audio file for genre classification"
    timeseries_length = 128
    features = np.zeros((1, timeseries_length, 33), dtype=np.float64)

    y, sr = librosa.load(file)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
    spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=512)

    features[0, :, 0:13] = mfcc.T[0:timeseries_length, :]
    features[0, :, 13:14] = spectral_center.T[0:timeseries_length, :]
    features[0, :, 14:26] = chroma.T[0:timeseries_length, :]
    features[0, :, 26:33] = spectral_contrast.T[0:timeseries_length, :]
    return features

# # create the root window
# root = tk.Tk()
# root.title('Tkinter Open File Dialog')
# root.resizable(False, False)
# root.geometry('300x150')




# # open button
# open_button = ttk.Button(
#     root,
#     text='Open a File',
#     command=select_file
# )

# # open button
# predict_button = ttk.Button(
#     root,
#     text='Predict',
#     command=select_file
# )

# open_button.pack(expand=True)
# predict_button.pack()

# # run the application
# root.mainloop()

from tkinter import *


model = load_model("LSTM-old/weights/model.json", "LSTM-old/weights/model_weights.h5")


root = Tk()
root.title('Pytorch model prediction')
root.geometry('400x300')
root.config(bg='#9FD996')

ws = Frame(root,bg='#9FD996')


llb = Label(
    ws,
    text='Choose an Audio file',
    bg='#9FD996'
)

llb.grid(row=0, column=0,columnspan=2,pady=20,padx=20)


global filename

def select_file():
    global filename
    filetypes = (
        ('Audio files', '*.wav'),
        ('All files', '*.*')
    )

    filename = fd.askopenfilename(
        title='Open a file',
        initialdir='/',
        filetypes=filetypes)

    llb.config(text = "Selected file: " +  filename.split("/")[-1])


# open button
open_button = Button(
    ws,
    text='Open a File',
    command=select_file
).grid(row=0, column=2)


# progress = ttk.Progressbar(
#     ws,
#     orient='horizontal',
#     length = 100, 
#     mode = 'determinate'
# )
# progress.grid(row=3, column=0,columnspan=2,pady=20,padx=20)

import time

predict_llb = Label(
        ws,
        text='Result...',
        bg='#9FD996'
    )
predict_llb.grid(row=3, column=0,columnspan=2,pady=20,padx=20)

import threading

# def real_progress_update():
#     progress['value'] = 0
#     for i in range(10):
#         progress['value'] += 10
#         root.update_idletasks()
#         time.sleep(0.2)
        
def predict_and_update():
    predict_llb.config(text = "Predicting...")
    predicted_text = get_genre(model, filename)
    predict_llb.config(text = "Predicted Class: " +  predicted_text)

    
def update_progress():
    # t = threading.Thread(target=real_progress_update)
    # t.start()
    
    threading.Thread(target=predict_and_update).start()
    

    
    # for i in genre_list:
    #     if i in filename:
    #         predicted_text = i
    

    
Button(
    ws,
    text='Predict',
    command=update_progress
).grid(row=3, column=2)




ws.pack(expand=True)

root.mainloop()