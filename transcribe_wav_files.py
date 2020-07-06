#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip3 install SpeechRecognition')
get_ipython().system('pip3 install spacy')


# In[ ]:


import os
import speech_recognition as sr
import spacy


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


r = sr.Recognizer()
nlp = spacy.load('en_core_web_sm')
data_directory = "/content/drive/My Drive/Trump"


# In[ ]:


with open("prompts.gui", "w", buffering=1) as text_file:
    for file in sorted(os.listdir(data_directory)):
        filename = os.fsdecode(file)
        print("Current Filename: %s" % filename)

        if filename.endswith(".wav"):
            try:
                audio_file = sr.AudioFile(data_directory + "/" + filename)

                # Open audio file with speech_recognition
                with audio_file as source:
                    audio = r.record(source)
                    text = r.recognize_google(audio)

                # Convert speech_recognition audio data to spaCy audio document
                document = nlp(text)

                # Extract sentences from document
                sentences = list(document.sents)

                # Write out parsed sentences with their original filenames
                for sentence in sentences:
                    if sentence and sentence[0]:
                      sentence_str = str(sentence)
                      if sentence_str and sentence_str[0]:
                        processed_sentence = sentence_str[0].upper() + sentence_str[1:]
                    text_file.write("%s\n%s.\n\n" % (filename, processed_sentence))
            
            except:
                print("Failed to extract sentence from %s" % filename)
            continue
        else:
            continue

