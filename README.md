##  EchoMindLite: Intelligent Voice and Sentiment Analysis for Enhanced Call Center Insights in real-time streaming 

It has inherited some enhanced features for [sensevoice](funaudiollm.github.io/):
- **VAD detection**
- **real-time streaming recognition**
- **speaker verification**.

## Installation TODO

First, clone this repository to your local machine:

```bash
git clone https://github.com/BiancaZYCao/call-center-sentiment-sense.git
cd call-center-sentiment-sense
```

Then, install the required dependencies using the following command: 

```bash
conda create -n echomindlite python=3.10
conda activate echomindlite

conda install -c conda-forge ffmpeg

pip install -r requirements.txt
```
Also install nltk pkgs

## Running

### Backend API Server - Streaming Real-time Recognition WebSocket Server

```bash
python server_wss.py 
```

### Client Page

- `index.html`
- Change `wsUrl` to your own WebSocket server address to test


## Roadmap

- [x]  VAD and STT with SenseVoice - real time with speaker verification
- [x]  Accent Recognition Singaporean English V.S. Native Speaker
- [x]  Text Sentiment
- [ ]  Topic Modelling 
- [ ]  Reply Hint, RAG
- [ ]  Summary Page



## License

This project is sponsored by NCS, Singapore.   
**Note**: This project should not be used in commercial products. 
It is for educational demo purposes only.


## Dependencies
- [https://github.com/FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)
- [https://modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced](https://modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced)
- [https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch)
- More 

## Explanation

1) The following folders are used:
	- "models -lda" folder contain the trained BERTopic models
	- "gpt_store3" folder contains the vector store of the data using RAG
	
2) The following files no need to be run when deployed:
	- rag2.py: used to convert data to vectors and stored in gpt_store3 folder using RAG
	- BERTopicModelTrainer9.py: used to train the topic model using the data in the "data" folder
	
3) stopwords.txt contains custom stopwords used by BERTopicModelTrainer9.py and TopicModel.py during text preprocessing.

4) TextPreprocessing.py contains code to preprocess text.

5) TopicModel.py contains functions to retrieve topics, questions, and responses from RAG vector store.

6) testTopicModel.py contains examples to call the functions in TopicModel.py.


## Instruction

1) Unzip gpt_store3.zip in same folder as all the above files.

2) You only need to use the following functions in TopicModel.py:
	- getResponseForQuestions(input_text)
		Function to get response from RAG based on the question or prompt.
		Parameters:
			input_text - the question or prompt
		Return a text response.
	- getTopics(sentence, num_of_topics=7)
		Function to get the best topics from a list of topics.
		Parameters:
			sentence - the inut sentence, can consist of multiple sentences.
			num_of_topics - the number of topics to retrieve. Maximum is 10. I've set default to 7.
		Return a list of topic words.
	- getTopicsAndQuestions()
		Function to get generated questions for each topic.
		Return a dictionary in the format: 
		{topic: [question1, question2,...]}