# Conversational_GPT
Conversational GPT using Langchain achives a human level capability to understand the conversations in any language converting it into English and utilizes the power of  OPENAI Chat GPT model to analyze and get an understanding of possible potential customers for buiness growth.

### Objective 
The Repo objective was for the companies who are having a database of conversations with multiple customers and want to ChatGpt on there own data , Hence helping them to track down such customers and subsequently targeting for sales and better service .

### Project Pipleine 
![Image text](https://github.com/Samarth-991/Conversational_GPT/blob/main/pipeline.png)

### Word Embeddings - openAI vs Huggingface
The project is build in such a way that it can be used with and witout the OpenAI key to create Vector database . Use either of  OpenAI vector Embeddings or Hugging face vector embeddings to create Vector DB and store into local FAISS vector database . 


### Web UI
The project uses streamlit to create an UI interface .Streamlit turns data scripts into shareable web apps in minutes.

## Audio Processors 
The repo uses 2 audio processors namely Whisper and Hugging face.
- <b>Whisper</b>

OpenAI  new speech recognition model  Whisper. Unlike GPT-3, Whisper is a free and open-source model. 
You can toggle with multiple whipser models from base , small , medium , large. With higher complexity model accuracy increases but becomes more compute intensive and memory hungry .

##### Setting up Whisper
 You can download and install (or update to) the latest release of Whisper with the following command
```
pip install -U openai-whisper
````
Additional dependencies required by whipser API
```
sudo apt update && sudo apt install ffmpeg
pip install setuptools-rust

```
If  you want to install without hassel simply :
```
<Setup virtual or Conda environment>
pip install -r requirements.txt
```
- <b>Hugging Face -Facebook model </b> 
Wav2Vec2 is a speech model that accepts a float array corresponding to the raw waveform of the speech signal.Wav2Vec2 model was trained using connectionist temporal classification (CTC) so the model output has to be decoded using Wav2Vec2CTCTokenizer .It is a type of neural network output and associated scoring function, for training recurrent neural networks (RNNs) such as LSTM networks to tackle sequence problems where the timing is variable.

##### setting up Hugging Face 

Setting up Hugging face is little tough. Its advicebale to use requirements.txt to install.
```
pip install transformers[torch] librosa soundfile nltk  
 	
```
#### Using Facebook models :
Within hugging face there are certain models which require you to create a token. This token is one time and need to add in Audio2Text/huggingface_api.py once . Once added code automatically creates a hugging facehub login file at your root.
```
pip install huggingface-hub
```




