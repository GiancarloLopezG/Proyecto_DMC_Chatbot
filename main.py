# Importaciones necesarias
import openai
import streamlit as st
from typing import List
from dataclasses import dataclass, field
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from streamlit_chat import message
from openai import OpenAI
from pinecone import Pinecone
from PIL import Image

# Configuración de parámetros
@dataclass
class Config:
    """Application configuration parameters."""
    PINECONE_INDEX_NAME: str = st.secrets["pinecone"]["index_name"]
    PINECONE_API_KEY: str = st.secrets["pinecone"]["api_key"]
    OPENAI_API_KEY: str = st.secrets["openai"]["api_key"]
    OPENAI_EMBEDDING_MODEL_NAME : str = st.secrets["openai"]["embedding_model_name"]
    AVAILABLE_MODELS: List[str] = field(default_factory=lambda: ['gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4'])
    DEFAULT_MODEL: str = "gpt-3.5-turbo"
    SYSTEM_PROMPT: str = """
    Eres un experto en parrillas uruguayas y tu deber es responder todas las consultas que los usuarios tengan de forma amigable y concisa.
    Para ello antes de cada pregunta se te brindará contexto el cuál puedes utilizar para responder a las consultas.
    Si no encuentras la respuesta en el contexto o este no tiene sentido, responde 'No tengo información sobre ello actualmente'.
    """

# Instanciación de objetos necesarios
config = Config()
pinecone_client = Pinecone(api_key=config.PINECONE_API_KEY)
index = pinecone_client.Index(name=config.PINECONE_INDEX_NAME)
openai_client = OpenAI(api_key=config.OPENAI_API_KEY)

# Configuración del modelo
llm_model = ChatOpenAI(model_name=config.DEFAULT_MODEL, openai_api_key=config.OPENAI_API_KEY)
system_msg_template = SystemMessagePromptTemplate.from_template(template=config.SYSTEM_PROMPT)
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
prompt_template = ChatPromptTemplate.from_messages([
    system_msg_template,
    MessagesPlaceholder(variable_name="history"),
    human_msg_template
])
conversation = ConversationChain(
    memory=ConversationBufferWindowMemory(k=3, return_messages=True),
    prompt=prompt_template,
    llm=llm_model,
    verbose=True
)

# Funciones auxiliares
def find_match(query: str) -> str:
    """Busca los mejores resultados en Pinecone para el query dado."""
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=query
    )
    query_embedding = response.data[0].embedding
    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )
    return "\n".join([result["metadata"]["text"] for result in results["matches"]])

def query_refiner(conversation: str, query: str) -> str:
    """Refina la consulta basándose en el historial de conversación."""
    response = openai_client.chat.completions.create(
        model=config.DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": "Refina consultas basándote en el contexto del historial de conversación."},
            {"role": "user", "content": f"CONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:"}
        ],
        temperature=0.1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content

def get_conversation_string() -> str:
    """Genera un historial de la conversación en formato string."""
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string

# Inicialización de estados de sesión
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["**¿Qué tal?¿Cómo puedo ayudarte?**\nHablas con el mayor experto en parrillas uruguayas"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

# Configuración de la página
st.set_page_config(
    page_title="Experto parrillas uruguayas 🍖🥩",
    page_icon="https://python.langchain.com/img/favicon.ico"
)

# Sidebar
with st.sidebar:
    st.title("Configuración de la API de OpenAI")
    llm_model_name = st.selectbox(
        'Elige el modelo',
        config.AVAILABLE_MODELS,
        key="model",
        index=config.AVAILABLE_MODELS.index(config.DEFAULT_MODEL)
    )
    image = Image.open('parrilla.jpeg')
    st.image(image, caption='OpenAI, Langchain y Streamlit')

# Interfaz principal
st.subheader("Chatbot con Langchain, ChatGPT, Pinecone y Streamlit")

# Contenedores para el historial y entrada del usuario
response_container = st.container()
text_container = st.container()

with text_container:
    query = st.text_input("Consulta: ", key="input")
    if query:
        with st.spinner("Escribiendo respuesta..."):
            conversation_string = get_conversation_string()
            refined_query = query_refiner(conversation_string, query)
            st.subheader("Consulta refinada:")
            st.write(refined_query)
            context = find_match(refined_query)
            response = conversation.predict(input=f"Context:\n{context}\n\nQuery:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
