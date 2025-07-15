import os
import yaml
import gradio as gr 
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate 
from dotenv import load_dotenv
from vs import rebuild_vector_store_func

# -------- PARAMS --------
os.environ["HF_HUB_USER_AGENT"] = "CloudWalk_Chatbot"

# --- CONFIGURAÇÕES DE MODELOS ---
# LLM_MODEL = 'google/gemma-3-4b-it'
# LLM_MODEL = 'google/gemma-3-1b-it'
LLM_MODEL = 'google/gemma-3-12b-it'
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- CONFIGURAÇÃO DO TOKEN HF ---
load_dotenv('./env')
HF_TOKEN = os.getenv("HF_TOKEN") 

if not HF_TOKEN:
    print("Token HF não encontrado no config.yaml ou arquivo não existe.")
    if "HF_TOKEN" not in os.environ:
        import getpass
        os.environ["HF_TOKEN"] = getpass.getpass("Por favor, digite seu token da API Hugging Face: ")
    HF_TOKEN = os.environ["HF_TOKEN"]

# --- 1. Inicializa o LLM Hugging Face ---
llm = HuggingFaceEndpoint(
    repo_id=LLM_MODEL,
    task="text-generation",
    max_new_tokens=1024,
    do_sample=False,
    repetition_penalty=1.03,
    huggingfacehub_api_token=HF_TOKEN,
)

chat_model = ChatHuggingFace(llm=llm)


# --- vs -----------

with open('./config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
rebuild_vector_store = config.get('rebuild_vector_store')

if rebuild_vector_store:
    rebuild_vector_store_func(EMBEDDING_MODEL)


# --- 2. Inicializa os embeddings ---
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# --- 3. Carrega a vector store FAISS salva localmente ---
vector_store_path = './vector_store/vs_base/'
try:
    faiss_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
except Exception as e:
    print(f"Erro ao carregar a vector store FAISS: {e}")
    print("Verifique se o caminho está correto e se o arquivo não está corrompido.")
    exit()

# --- 4. Retriever a partir da vector store (usando similarity como exemplo) ---
retriever = faiss_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# retriever = faiss_store.as_retriever(
#     search_type="mmr",
#     search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7}
# )

# --- 5. PromptTemplate personalizado ---
custom_prompt_template = """
You are a useful chatbot for customer service.
ALWAYS RESPONDE IN THE SAME LANGUAGE AS THE INPUT.
Use the following pieces of context to answer the user's question.
{context}

Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(custom_prompt_template)

# --- 6. Cria uma cadeia de QA que usa o retriever e o modelo ---
qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# --- 7. Função principal para a interface do Gradio ---
def chat_response(question: str, history: list): # `history` é um parâmetro obrigatório para gr.ChatInterface
    """
    Gera a resposta do chatbot usando o modelo de QA e formata para exibição no Gradio.

    Args:
        question (str): A pergunta do usuário.
        history (list): Histórico de conversas (não usado diretamente aqui, mas necessário para a interface).

    Returns:
        str: A resposta formatada do chatbot, incluindo as fontes.
    """
    print(f"Recebida pergunta: '{question}'") # Para debug no console

    try:
        result = qa_chain.invoke({"query": question})
        answer = result["result"]
        sources = result.get("source_documents", [])

        response_text = f"**Resposta:** {answer}\n\n"

        if sources:
            response_text += "**Saiba mais em:**\n"
            unique_sources = set()
            source_list_for_printing = []

            for doc in sources:
                source_name = doc.metadata.get('source', 'Fonte desconhecida')
                if source_name not in unique_sources:
                    unique_sources.add(source_name)
                    source_list_for_printing.append(source_name)

            for i, source_name in enumerate(source_list_for_printing):
                response_text += f"- {i+1}. '{source_name}'\n"
        else:
            response_text += "Nenhuma fonte específica foi utilizada para esta resposta.\n"
        
        return response_text

    except Exception as e:
        error_message = f"Ocorreu um erro ao processar sua pergunta: {e}. Por favor, tente novamente."
        print(error_message) # Para debug no console
        return error_message

# --- 8. Criação da Interface Gradio ---
if __name__ == "__main__":
    print("Iniciando a interface Gradio...")
    
    # gr.ChatInterface é ideal para interfaces de chatbot
    demo = gr.ChatInterface(
        type="messages",
        fn=chat_response,
        title="CloudWalk Chatbot",
        description="Olá! Estou aqui para responder suas dúvidas.",
        submit_btn="Enviar Pergunta",
        examples=[
            ["What product and services InfinitePay offer ?"],
            ['What are the fees for the card machine?'],
            ['How can I order one card machine?'],
            ["Quais serviços a infinite pay oferece?"],
            ["Quais as taxas da maquininha?"],
            ["Como pedir uma maquininha?"],
        ],
        chatbot=gr.Chatbot(type="messages")
    )

    # Inicia a interface Gradio
    demo.launch(share=True)
