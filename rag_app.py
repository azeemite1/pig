import os, base64
from pathlib import Path
import hmac
import tempfile
import pandas as pd
import uuid
import streamlit as st
import plotly.express as px
import numpy as np
import openai

from langchain_community.vectorstores import AstraDB
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import AstraDBChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader,  # Removed PDF, WebBaseLoader
    # Note: We will load TXT and Excel files using pandas
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from langchain.schema import StrOutputParser
from langchain.callbacks.base import BaseCallbackHandler

print("Started")
st.set_page_config(page_title='Pigging Management System', page_icon='🐷')

# Get a unique session id for memory
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4()

# Streaming callback handler for responses
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

###############
### Globals ###
###############

global lang_dict, language, rails_dict, session, embedding, vectorstore, chat_history, memory
global disable_vector_store, strategy, prompt_type, custom_prompt

#################
### Functions ###
#################

def check_password():
    """Returns `True` if the user had a correct password."""
    def login_form():
        with st.form("credentials"):
            st.text_input('Username', key='username')
            st.text_input('Password', type='password', key='password')
            st.form_submit_button('Login', on_click=password_entered)

    def password_entered():
        if st.session_state['username'] in st.secrets['passwords'] and hmac.compare_digest(
            st.session_state['password'], st.secrets.passwords[st.session_state['username']]
        ):
            st.session_state['password_correct'] = True
            st.session_state.user = st.session_state['username']
            del st.session_state['password']
        else:
            st.session_state['password_correct'] = False

    if st.session_state.get('password_correct', False):
        return True

    login_form()
    if "password_correct" in st.session_state:
        st.error('😕 User not known or password incorrect')
    return False

def logout():
    for key in st.session_state.keys():
        del st.session_state[key]
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()

# Function for vectorizing uploaded data into Astra DB
def vectorize_text(uploaded_files):
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            temp_dir = tempfile.TemporaryDirectory()
            file = uploaded_file
            print(f"Processing: {file}")
            temp_filepath = os.path.join(temp_dir.name, file.name)
            with open(temp_filepath, 'wb') as f:
                f.write(file.getvalue())
            # Process TXT files
            if uploaded_file.name.endswith('txt'):
                file_content = [uploaded_file.read().decode()]
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
                texts = text_splitter.create_documents(file_content, [{'source': uploaded_file.name}])
                vectorstore.add_documents(texts)
                st.info(f"{len(texts)} documents added from TXT")
            # Process CSV files
            elif uploaded_file.name.endswith('csv'):
                docs = []
                loader = CSVLoader(temp_filepath)
                docs.extend(loader.load())
                vectorstore.add_documents(docs)
                st.info(f"{len(docs)} documents added from CSV")
            # Process Excel files (.xls, .xlsx)
            elif uploaded_file.name.endswith(('xls', 'xlsx')):
                try:
                    df = pd.read_excel(temp_filepath)
                    # Convert the DataFrame to text (e.g., CSV string) for vectorization
                    text = df.to_csv(index=False)
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
                    texts = text_splitter.create_documents([text], [{'source': uploaded_file.name}])
                    vectorstore.add_documents(texts)
                    st.info(f"{len(texts)} documents added from Excel")
                except Exception as e:
                    st.error(f"Error processing Excel file: {e}")

def get_prompt(type):
    template = ''
    if type == 'Extended results':
        print("Prompt type: Extended results")
        template = f"""You're a helpful AI assistant tasked to answer questions about the pigging data.
You answer in a detailed manner using bulletpoints.
If you don't know the answer, just say 'I do not know the answer'.

Use the following context to answer the question:
{{context}}

Use the following chat history to answer the question:
{{chat_history}}

Question:
{{question}}

Answer in {language}:"""
    if type == 'Short results':
        print("Prompt type: Short results")
        template = f"""You're a helpful AI assistant tasked to answer questions about the pigging data.
You answer very briefly.
If you don't know the answer, just say 'I do not know the answer'.

Use the following context to answer the question:
{{context}}

Use the following chat history to answer the question:
{{chat_history}}

Question:
{{question}}

Answer in {language}:"""
    if type == 'Custom':
        print("Prompt type: Custom")
        template = custom_prompt

    return ChatPromptTemplate.from_messages([("system", template)])

def load_model():
    print("load_model")
    return ChatOpenAI(temperature=0.3, model='gpt-4-1106-preview', streaming=True, verbose=True)

def load_retriever(top_k_vectorstore):
    print(f"load_retriever with top_k_vectorstore='{top_k_vectorstore}'")
    return vectorstore.as_retriever(search_kwargs={"k": top_k_vectorstore})

@st.cache_resource()
def load_memory(top_k_history):
    print(f"load_memory with top-k={top_k_history}")
    return ConversationBufferWindowMemory(
        chat_memory=chat_history,
        return_messages=True,
        k=top_k_history,
        memory_key="chat_history",
        input_key="question",
        output_key='answer',
    )

def generate_queries():
    prompt = f"""You are a helpful assistant that generates multiple search queries based on a single input query about pigging data in language {language}.
Generate multiple search queries related to: {{original_query}}
OUTPUT (4 queries):"""
    return ChatPromptTemplate.from_messages([("system", prompt)]) | model | StrOutputParser() | (lambda x: x.split("\n"))

def reciprocal_rank_fusion(results: list[list], k=60):
    from langchain.load import dumps, loads
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results

##################
### Data Cache ###
##################

@st.cache_data()
def load_localization(locale):
    print("load_localization")
    df = pd.read_csv("./customizations/localization.csv")
    df = df.query(f"locale == '{locale}'")
    lang_dict = {df.key.to_list()[i]: df.value.to_list()[i] for i in range(len(df.key.to_list()))}
    return lang_dict

@st.cache_data()
def load_rails(username):
    print("load_rails")
    df = pd.read_csv("./customizations/rails.csv")
    df = df.query(f"username == '{username}'")
    rails_dict = {df.key.to_list()[i]: df.value.to_list()[i] for i in range(len(df.key.to_list()))}
    return rails_dict

#############
### Login ###
#############

if not check_password():
    st.stop()

username = st.session_state.user
language = st.secrets.languages[username]
lang_dict = load_localization(language)

#######################
### Resources Cache ###
#######################

@st.cache_resource(show_spinner=lang_dict['load_embedding'])
def load_embedding():
    print("load_embedding")
    return OpenAIEmbeddings()

@st.cache_resource(show_spinner=lang_dict['load_vectorstore'])
def load_vectorstore(username):
    print(f"load_vectorstore for {username}")
    return AstraDB(
        embedding=embedding,
        collection_name=f"vector_context_{username}",
        token=st.secrets["ASTRA_TOKEN"],
        api_endpoint=os.environ["ASTRA_ENDPOINT"],
    )

@st.cache_resource(show_spinner=lang_dict['load_message_history'])
def load_chat_history(username):
    print(f"load_chat_history for {username}_{st.session_state.session_id}")
    return AstraDBChatMessageHistory(
        session_id=f"{username}_{st.session_state.session_id}",
        api_endpoint=os.environ["ASTRA_ENDPOINT"],
        token=st.secrets["ASTRA_TOKEN"],
    )

#####################
### Session state ###
#####################

if 'messages' not in st.session_state:
    st.session_state.messages = [AIMessage(content=lang_dict['assistant_welcome'])]

# ----------------------------------------------------
# Cumulative Data Store for Pigging Trends (Unified)
# ----------------------------------------------------
if 'cumulative_data' not in st.session_state:
    st.session_state['cumulative_data'] = pd.DataFrame()

############
### Main ###
############

# Display welcome text and logo
try:
    st.markdown(Path(f"./customizations/welcome/{username}.md").read_text())
except:
    st.markdown(Path("./customizations/welcome/default.md").read_text())

with st.sidebar:
    try:
        st.image(f"./customizations/logo/{username}.svg", use_container_width=True)
        st.text('')
    except:
        try:
            st.image(f"./customizations/logo/{username}.png", use_container_width=True)
            st.text('')
        except:
            st.image('./customizations/logo/default.png', use_container_width=True)
            st.text('')

with st.sidebar:
    st.markdown(f"{lang_dict['logout_caption']} :orange[{username}]")
    logout_button = st.button(lang_dict['logout_button'])
    if logout_button:
        logout()

with st.sidebar:
    st.divider()

# Initialize resources
with st.sidebar:
    rails_dict = load_rails(username)
    embedding = load_embedding()
    vectorstore = load_vectorstore(username)
    chat_history = load_chat_history(username)

with st.sidebar:
    disable_chat_history = st.toggle(lang_dict['disable_chat_history'])
    top_k_history = st.slider(lang_dict['k_chat_history'], 1, 50, 5, disabled=disable_chat_history)
    memory = load_memory(top_k_history if not disable_chat_history else 0)
    delete_history = st.button(lang_dict['delete_chat_history_button'], disabled=disable_chat_history)
    if delete_history:
        with st.spinner(lang_dict['deleting_chat_history']):
            memory.clear()
    disable_vector_store = st.toggle(lang_dict['disable_vector_store'])
    top_k_vectorstore = st.slider(lang_dict['top_k_vector_store'], 1, 50, 5, disabled=disable_vector_store)
    strategy = st.selectbox(lang_dict['rag_strategy'], ('Basic Retrieval', 'Maximal Marginal Relevance', 'Fusion'),
                            help=lang_dict['rag_strategy_help'], disabled=disable_vector_store)
    custom_prompt_text = ''
    custom_prompt_index = 0
    try:
        custom_prompt_text = open(f"./customizations/prompt/{username}.txt").read()
        custom_prompt_index = 2
    except:
        custom_prompt_text = open("./customizations/prompt/default.txt").read()
        custom_prompt_index = 0
    prompt_type = st.selectbox(lang_dict['system_prompt'], ('Short results', 'Extended results', 'Custom'),
                               index=custom_prompt_index)
    custom_prompt = st.text_area(lang_dict['custom_prompt'], custom_prompt_text,
                                 help=lang_dict['custom_prompt_help'], disabled=(prompt_type != 'Custom'))
    print(f"{disable_vector_store}, {top_k_history}, {top_k_vectorstore}, {strategy}, {prompt_type}")

with st.sidebar:
    st.divider()
    # Uploader for new context files (vectorization)
    uploaded_files = st.file_uploader(lang_dict['load_context'], type=['txt', 'csv', 'xls', 'xlsx'], accept_multiple_files=True)
    upload = st.button(lang_dict['load_context_button'])
    if upload and uploaded_files:
        vectorize_text(uploaded_files)

if (username in st.secrets['delete_option'] and st.secrets.delete_option[username] == 'True'):
    with st.sidebar:
        st.caption(lang_dict['delete_context'])
        submitted = st.button(lang_dict['delete_context_button'])
        if submitted:
            with st.spinner(lang_dict['deleting_context']):
                vectorstore.clear()
                memory.clear()
                st.session_state.messages = [AIMessage(content=lang_dict['assistant_welcome'])]

with st.sidebar:
    st.divider()
    st.subheader(lang_dict['rails_1'])
    st.caption(lang_dict['rails_2'])
    for i in rails_dict:
        st.markdown(f"{i}. {rails_dict[i]}")

# ----------------------------------------------------
# Dashboard Section: Cumulative Pigging Trends
# ----------------------------------------------------
st.subheader("📊 Pigging Trends Dashboard")
# Uploader for new dashboard data (CSV, TXT, or Excel)
dashboard_file = st.file_uploader("Upload Pigging Data for Dashboard", type=["csv", "txt", "xls", "xlsx"], key="dashboard_csv")
if dashboard_file:
    # Read file based on extension
    ext = dashboard_file.name.split('.')[-1].lower()
    if ext == "csv":
        new_df = pd.read_csv(dashboard_file)
    elif ext in ["xls", "xlsx"]:
        new_df = pd.read_excel(dashboard_file)
    elif ext == "txt":
        new_df = pd.read_csv(dashboard_file, delimiter="\t")
    else:
        new_df = pd.DataFrame()
    
    # Convert Date columns if present
    if "Date Launched" in new_df.columns:
        new_df["Date Launched"] = pd.to_datetime(new_df["Date Launched"])
    if "Date Recovered" in new_df.columns:
        new_df["Date Recovered"] = pd.to_datetime(new_df["Date Recovered"])
    # Append new data to cumulative DataFrame in session state
    st.session_state['cumulative_data'] = pd.concat([st.session_state['cumulative_data'], new_df], ignore_index=True)
    st.success("New data appended to cumulative trends.")

if not st.session_state['cumulative_data'].empty:
    cumulative_df = st.session_state['cumulative_data']
    # Sidebar filters for cumulative data:
    # Assuming CSV contains a "Line" column for pipeline line selection
    if "Line" in cumulative_df.columns:
        line_filter = st.sidebar.multiselect("Select Pipeline Line", cumulative_df["Line"].unique(), default=cumulative_df["Line"].unique())
        cumulative_df = cumulative_df[cumulative_df["Line"].isin(line_filter)]
    contractor_filter = st.sidebar.multiselect("Select Contractor (Dashboard)", cumulative_df["Contractor"].unique(), default=cumulative_df["Contractor"].unique())
    pig_type_filter = st.sidebar.multiselect("Select Pig Type (Dashboard)", cumulative_df["Pig Type"].unique(), default=cumulative_df["Pig Type"].unique())
    cumulative_df = cumulative_df[
        (cumulative_df["Contractor"].isin(contractor_filter)) &
        (cumulative_df["Pig Type"].isin(pig_type_filter))
    ]
    # Display charts:
    fig1 = px.line(cumulative_df, x="Date Launched", y=["Launch Pressure (bar)", "Recovery Pressure (bar)"], title="Pressure Trend Over Time")
    st.plotly_chart(fig1, use_container_width=True)
    fig2 = px.bar(cumulative_df, x="Date Launched", y="Debris QTY (kg)", color="Debris Type", title="Debris Collected Over Time")
    st.plotly_chart(fig2, use_container_width=True)
    fig3 = px.histogram(cumulative_df, x="Date Launched", title="Pigging Frequency")
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("No cumulative dashboard data available yet. Upload data to build trends over time.")

# ----------------------------------------------------
# Chat Section: Draw chat messages so far
# ----------------------------------------------------
for message in st.session_state.messages:
    st.chat_message(message.type).markdown(message.content)

# ----------------------------------------------------
# AI Agent: Chat Input and RAG processing (Text-only)
# ----------------------------------------------------
question = st.chat_input(lang_dict['assistant_question'])
if question:
    print(f"Got question: {question}")
    st.session_state.messages.append(HumanMessage(content=question))
    with st.chat_message('human'):
        st.markdown(question)
    model = load_model()
    retriever = load_retriever(top_k_vectorstore)
    content = ''
    fusion_queries = []
    relevant_documents = []
    if not disable_vector_store:
        if strategy == 'Basic Retrieval':
            relevant_documents = retriever.get_relevant_documents(query=question, k=top_k_vectorstore)
        elif strategy == 'Maximal Marginal Relevance':
            relevant_documents = vectorstore.max_marginal_relevance_search(query=question, k=top_k_vectorstore)
        elif strategy == 'Fusion':
            generate_queries_fn = generate_queries()
            fusion_queries = generate_queries_fn.invoke({"original_query": question})
            print(f"Fusion queries: {fusion_queries}")
            content += f"\n*{lang_dict['using_fusion_queries']}*\n"
            for fq in fusion_queries:
                content += f"📙 :orange[{fq}]\n"
            with st.chat_message('assistant'):
                st.markdown(content)
            st.session_state.messages.append(AIMessage(content=content))
            chain = generate_queries_fn | retriever.map() | reciprocal_rank_fusion
            relevant_documents = chain.invoke({"original_query": question})
            print(f"Fusion results: {relevant_documents}")
    print("Chat message")
    with st.chat_message('assistant'):
        content = ''
        response_placeholder = st.empty()
        history = memory.load_memory_variables({})
        print(f"Using memory: {history}")
        inputs = RunnableMap({
            'context': lambda x: x['context'],
            'chat_history': lambda x: x['chat_history'],
            'question': lambda x: x['question']
        })
        print(f"Using inputs: {inputs}")
        chain = inputs | get_prompt(prompt_type) | model
        print(f"Using chain: {chain}")
        response = chain.invoke({'question': question, 'chat_history': history, 'context': relevant_documents}, 
                                  config={'callbacks': [StreamHandler(response_placeholder)]})
        print(f"Response: {response}")
        content += response.content
        memory.save_context({'question': question}, {'answer': content})
        if disable_vector_store:
            content += f"\n*{lang_dict['no_context']}*\n"
        else:
            content += f"\n*{lang_dict['sources_used']}*\n"
        sources = []
        for doc in relevant_documents:
            if strategy == 'Fusion':
                doc = doc[0]
            print(f"DOC: {doc}")
            source = doc.metadata['source']
            if source not in sources:
                content += f"📙 :orange[{os.path.basename(os.path.normpath(source))}]\n"
                sources.append(source)
        if disable_chat_history:
            content += f"\n*{lang_dict['no_chat_history']}*\n"
        else:
            content += f"\n*{lang_dict['chat_history_used']}: ({int(len(history['chat_history'])/2)} / {top_k_history})*\n"
        response_placeholder.markdown(content)
        st.session_state.messages.append(AIMessage(content=content))

with st.sidebar:
    st.caption("Azeemite")
