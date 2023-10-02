#pip install langchain bs4 sentence_transformers chromadb flask

#set the api key
import os
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
key = os.getenv('google_key')


#for passing vector store to llm
from langchain.chains import RetrievalQA
#for llm
# from langchain.llms import HuggingFaceHub
from langchain.llms import VertexAI
#for getting data from web
from langchain.document_loaders import WebBaseLoader
#for splitting text into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
#for embeddings
from langchain.embeddings import HuggingFaceEmbeddings
#for the vector store
from langchain.vectorstores import Chroma
# for the prompt template
from langchain.prompts import PromptTemplate

# text splitter
text_splitter_r = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

# embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"}
)

# # llm model(s)
# llm = HuggingFaceHub(
#     repo_id="google/flan-t5-large",
#     model_kwargs={"temperature": 0}
# )

llm = VertexAI(
    model_name='text-bison@001',
)



### ----------------------------- DOCUMENT LOADING INTO VECTOR STORE --------------------------- ####

# loading text from webpages into variable web_documents. Formatted as a list, with documents, in each document a
# dictionary with metadata
webpage_list = ["https://wiki.albiononline.com/wiki/Account_Creation", "https://wiki.albiononline.com/wiki/Achievements", "https://wiki.albiononline.com/wiki/Premium_Features", "https://wiki.albiononline.com/wiki/Gold", "https://wiki.albiononline.com/wiki/New_Player_FAQ"]

# # setting up so that it does 3 webpages at a time.
# batch_size = 3
# for i in range(0, len(webpage_list), batch_size):
#     current_webpage_batch = webpage_list[i:i+batch_size]

web_documents = []
for url in webpage_list:
    loader = WebBaseLoader(url)
    web_doc = loader.load()
    web_documents.extend(web_doc)

# splitting that doc into chunks and storing that split doc into variable split_doc
split_doc = text_splitter_r.split_documents(web_documents)

# storing embeddings into vectorstore variable called vectordb
vectordb = Chroma.from_documents(
    documents=split_doc,
    embedding=embeddings,
    persist_directory='./persist_directory_data',
)

# clear any persisting data out
del split_doc



### ---------------------- QA SETUP ---------------------------------- ###

template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Reply in full sentences, being sure to rephrase the question in your first sentence. Use three sentences maximum. Keep the answer as concise as possible.
{context}
Question: {question}
Helpful Answer:"""
qa_chain_prompt = PromptTemplate.from_template(template)


#setting up our retrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": qa_chain_prompt}
)


def run_llm(question):
    answer = qa_chain({"query": question})
    return answer


# #let's test everything to make sure its working
# test_question = input("What is your question?: ")
# test_answer = qa_chain({"query": test_question})
# print(test_answer['result'])
