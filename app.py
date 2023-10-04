#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('pip', 'install langchain gradio')


# In[ ]:


import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()


# In[ ]:


get_ipython().system('wget https://patents.justia.com/patent/11663394')


# In[3]:


from langchain.document_loaders import BSHTMLLoader

loader = BSHTMLLoader("11663394.html")
data = loader.load()
text = data[0].page_content.replace("\n", " ")
data[0].page_content = data[0].page_content.replace("\n", " ")


# In[4]:


from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 100,
    chunk_overlap  = 20,
    length_function = len,
    add_start_index = True,
)

texts = text_splitter.split_documents(data)
print(texts)


# In[5]:


from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os

openai_key = os.environ["OPENAI_API_KEY"]

vectorstore = Chroma.from_documents(documents=texts, embedding=OpenAIEmbeddings(openai_api_key=openai_key))


# In[6]:


from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

question = "what is this patent about?"
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever())
result = qa_chain({"query": question})


# In[9]:


import gradio as gr
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Define the Q&A model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())

# Define the Gradio interface
def qa_interface(query):
    result = qa_chain({"query": query})
    return result["result"]

inputs = gr.Textbox(label="Question")
# outputs = gr.Textbox(label="Answer")
outputs = "text"
interface = gr.Interface(fn=qa_interface, inputs=inputs, outputs=outputs, title="Q&A Model")

# Launch the interface
interface.launch()

