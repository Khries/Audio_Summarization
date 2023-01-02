import streamlit as st
from deepspeech import Model
import numpy as np
import os
import wave
import subprocess
from IPython.display import Audio
from PIL import Image
import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px
from pytube import YouTube
import os
import torch
from transformers import pipeline

class Transcript:
    def __init__(self,link):
        self.link=link
        self.long_text=''
        
    def youtube_audio(self):
        '''
        
        Extract video from Youtube
        ->Convert the video to audio
        ->Convert audio to 16kHZ format
        
        '''
        if os.path.exists('yt_audio.mp3'):
            os.remove('yt_audio.mp3')
        if os.path.exists('out.wav'):
            os.remove('out.wav')
        yt=YouTube(self.link)
        video = yt.streams.filter(only_audio=True).first()
        output_file = video.download()
        base, ext = os.path.splitext(output_file)
        new_file =  'yt_audio.mp3'
        os.rename(output_file, new_file)

        subprocess.call('ffmpeg -i "yt_audio.mp3" -acodec pcm_s16le -ac 1 -ar 16000 out.wav -y', shell=True)
        
   
    def listen_audio(self):
        '''
        method to listen to the audio file
        '''
        Audio('out.wav')
    
    def run_transcript(self):
        '''
        Method to use the deepSpeech model to 
        generate transcript of the audio file
        '''
        self.youtube_audio()

        model_file_path='deepspeech-0.9.3-models.pbmm'
        lm_file_path='deepspeech-0.9.3-models.scorer'

        beam_width=100
        lm_alpha=0.93
        lm_beta=1.18
        
        model= Model(model_file_path)
        model.enableExternalScorer(lm_file_path)
        
        model.setScorerAlphaBeta(lm_alpha, lm_beta)
        model.setBeamWidth(beam_width)
        buffer, rate= self.read_wav_file()
        data16=np.frombuffer(buffer, dtype=np.int16)
        long_text=model.stt(data16)
        trans=open('transcript.txt','w')
        trans.write(long_text)
        trans.close()
        print(model.stt(data16))
        return model.stt(data16)


          
    def summarized(self):
        summarizer = pipeline("summarization",
    "pszemraj/long-t5-tglobal-base-16384-book-summary",
        device=0 if torch.cuda.is_available() else -1,
)
        with open('transcript.txt','r') as f:
          lines= f.readlines()
        result = summarizer(lines[0])
        summarized=result[0]["summary_text"]
        print(summarized)

        summary=open('summary.txt','w')
        summary.write(summarized)
        
        
        return summarized

    # def summarized(self):
    #   summarizer=pipeline("summarization","pszemraj/long-t5-tglobal-base-16384-book-summary",
    #     device=0 if torch.cuda.is_available() else -1,)
    #   result=summarizer(self.long_text)
    #   return result[0]["summary_text"]
        
    def read_wav_file(self):
        with wave.open('out.wav','rb') as w:
            rate=w.getframerate()
            frames=w.getnframes()
            buffer=w.readframes(frames)
            print("Rate:", rate)
            print("Frames:", frames)
        return buffer, rate 
      
    
    def sentiment(self):
        classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
        with open('summary.txt','r') as f:
            lines= f.readlines()
        sequence_to_classify = lines[0]
        candidate_labels = ['Positive', 'Neutral', 'Negative']
        result=classifier(sequence_to_classify, candidate_labels)
        return result['scores']


image= Image.open('yt_logo.jpg')
def get_text(path):
  with open(path,'r') as f:
    lines=f.readlines()
  return lines[0]



st.image(image,width=450)

with st.spinner("Please Wait Running Inference.."):
  link=st.text_input(label='Paste YouTube link 🔗')
  if st.button('Compute'):
    transcript=Transcript(link).run_transcript()
    summary=Transcript(link).summarized()


# Listen to audio
st.write('Listen To Audio 🎧')
st.audio('out.wav', format='wav')

with st.sidebar:
  options=st.selectbox('Which Method of text Summarization would you like to use?'
                      ,('High Accuracy Slow Inference','Low Accuracy Fast Inference'))


if st.button('Get Transcript & Summary'):
  transcript=get_text('transcript.txt')
  st.write('''### Transcript''')
  st.write(transcript)
  st.write('''### Summary''')
  summary=get_text('summary.txt')
  st.write(summary)

#encorporate bar chart in sentiment analysis


if st.button('Get Sentiment Score 🤗'):
  labels=['Positive', 'Neutral', 'Negative']
  scores=Transcript(link).sentiment()
  scores=list(scores)
  data={'Sentiments':labels,'Probabiliy':scores}
  data=pd.DataFrame(data)
  fig = plt.figure(figsize = (3, 2))
  fig = px.bar(data, x='Sentiments', y='Probabiliy')
  st.plotly_chart(fig) 




  

