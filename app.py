import streamlit as st
import moviepy.editor as mp 
import speech_recognition as sr 
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmltemplates import css, bot_template, user_template
import docx2txt
import pandas as pd
import openpyxl
import faiss
import os


def extract_video_text(video):
    video = mp.VideoFileClip(video) 
    audio_file = video.audio 
    
    audio_file.write_audiofile("video.wav") 
    r = sr.Recognizer() 
    
    with sr.AudioFile("video.wav") as source: 
        data = r.record(source) 
 
    text = r.recognize_google(data)
    return text




def read_excel_to_raw_text(file):
    raw_text = ""
    workbook = openpyxl.load_workbook(file)
    sheet = workbook.active

    for row in sheet.iter_rows(values_only=True):
        raw_text += ' '.join(map(str, row)) + '\n'

    return raw_text


def get_conversation_chain(vectorstore):
   llm=ChatOpenAI()
   #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
   memory=ConversationBufferMemory(memory_key='chat_history',return_messages=True)
   conversaation_chain=ConversationalRetrievalChain.from_llm(
       llm=llm,
       retriever=vectorstore.as_retriever(),
       memory=memory
       )
   return conversaation_chain


def get_vectorstore(text_chunks):
       key=st.secrets["OPENAI_APIKEY"]
       embeddings = OpenAIEmbeddings(openai_api_key=key)
       vectorstore=FAISS.from_texts(texts=text_chunks,embedding=embeddings)
       return vectorstore



def get_raw_chunk(doc_text):
    text_splitter=CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks=text_splitter.split_text(doc_text)
    return chunks



def get_raw_text(docs_files):
    text=''
    for doc in docs_files:
        pdf_reader=PdfReader(doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_web(web_link):
    url =web_link  
    response = requests.get(url)   
    if response.status_code == 200:
          soup = BeautifulSoup(response.content, 'html.parser')
          text = ''.join([p.get_text() for p in soup.find_all('p')])
          return text
    else:
         print("Failed to fetch the web page")

def mock_conversation_handler(input_data):
    # Placeholder implementation, replace with your actual logic
    response = {'chat_history': [{'content': 'Hello!'}, {'content': 'How can I help you?'}]}
    return response

def initialize_conversation():
    # Initialize the conversation handler in session state
    st.session_state.conversation = mock_conversation_handler
        


def handle_userinput(user_questions):
    if 'conversation' not in st.session_state:
        # Initialize conversation handler if not already initialized
       initialize_conversation()
    
    response = st.session_state.conversation({'question': user_questions})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
    

def main():
    st.set_page_config(page_title='smart Chatbot based on your documents',page_icon='https://icons.iconarchive.com/icons/graphicloads/colorful-long-shadow/48/Files-icon.png',layout='wide')
    st.write(css, unsafe_allow_html=True)
    st.header("Chat With Your Documents:book:")
    user_questions=st.text_input("Ask a question about your documents:")
   
    if user_questions:
        handle_userinput(user_questions)
    
    with st.sidebar:
        st.subheader("Your documents".title())
        docs_files = st.file_uploader("Upload your Documents here and click on 'Process'", accept_multiple_files=True,type=None)
        st.subheader("If you have any questions about any website".upper())
        web_link=st.text_input("please upload the link here ! else skip it ")
        if st.button("Process"):
            with st.spinner("Processing ..."):
                
                
                if web_link:
                    raw_text=get_text_web(web_link)
                
                elif docs_files[0].type.startswith('video/'):
                    raw_text = extract_video_text(docs_files)
                
                elif docs_files[0].type=='application/pdf':
                    raw_text=get_raw_text(docs_files)    
                
                elif docs_files[0].type=='text/plain':
                    raw_text=docs_files[0].read()    
                    
                
                elif docs_files[0].type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    raw_text=docx2txt.process(docs_files[0])  
                
                elif docs_files[0].type == "text/csv":
                    df = pd.read_csv(docs_files[0])
                    raw_text = df.to_string(index=False, header=False)  

                elif docs_files[0].type == "application/json":
                    
                    raw_text= ' '.join(f"{key} {value}" for key, value in docs_files[0])
                    
                
                elif docs_files[0].type =="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                    raw_text =read_excel_to_raw_text(docs_files[0])  


                else:
                   st.write(f"Unsupported file type: {docs_files[0].name}")
                   return 

                text_chunks=get_raw_chunk(raw_text)
           
                vectorstore = get_vectorstore(text_chunks)
               
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__=="__main__":
    main()
