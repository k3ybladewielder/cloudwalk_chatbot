from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import logging
logging.getLogger("langchain.text_splitter").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")
from langchain_community.document_loaders import RecursiveUrlLoader
import yaml

#----------- PARAMS -----------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

with open('./config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
# Acessar o prompt
rebuild_vector_store = config.get('rebuild_vector_store')

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

#----------- PATH VS ----------
VS_BASE = "./vector_store/vs_base"
VS_CENTRAL = "./vector_store/vs_central_ajuda"

#---------- VECTOR DATABASE -------------------
## knowledge base offline
url_list = [
"https://www.infinitepay.io", 
"https://www.infinitepay.io/maquininha", 
"https://www.infinitepay.io/maquininha-celular", 
"https://www.infinitepay.io/tap-to-pay", 
"https://www.infinitepay.io/pdv", 
"https://www.infinitepay.io/receba-na-hora", 
"https://www.infinitepay.io/gestao-de-cobranca",  
"https://www.infinitepay.io/gestao-de-cobranca-2",  
"https://www.infinitepay.io/link-de-pagamento", 
"https://www.infinitepay.io/loja-online", 
"https://www.infinitepay.io/boleto", 
"https://www.infinitepay.io/conta-digital", 
"https://www.infinitepay.io/conta-pj", 
"https://www.infinitepay.io/pix",
"https://www.infinitepay.io/pix-parcelado", 
"https://www.infinitepay.io/emprestimo", 
"https://www.infinitepay.io/cartao", 
"https://www.infinitepay.io/rendimento",
'https://www.infinitepay.io/taxas',
'https://www.cloudwalk.io/',
'https://www.cloudwalk.io/#our-mission',
'https://www.cloudwalk.io/#our-pillars',
'https://www.cloudwalk.io/#our-products',
]

def rebuild_vector_store_func(EMBEDDING_MODEL):
    loader = WebBaseLoader(web_paths=url_list)
    docs = loader.load()    
    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(split_docs, embeddings)
    os.makedirs(VS_BASE, exist_ok=True)
    vector_store.save_local(VS_BASE)
    print(f"vs_base salva em {VS_BASE}")
