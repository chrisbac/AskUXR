#
#    DEPENDENCIES
#

# Importing steamlit first for profiling reasons
import streamlit as st

# Setting page base configuration
st.set_page_config(
    page_title="AskUXR",
    page_icon="./public/askuxr-logo-vertical.png",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "## Ask UXR Playbook v0.1.0"
        
    }
)

# Import system libraries
import os
from dotenv import load_dotenv
from getpass import getpass
import re
import html
import base64
import time

load_dotenv()

# Chroma with workaraound for PySQLite3-binary issue
if os.getenv("DEV_MODE_FLAG") != "True":
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    import sqlite3
#else:
    # Local Streamlit profiler. Never in production.
#    from streamlit_profiler import Profiler

# Importing Langchain
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Importing Streamlit and related
from streamlit_feedback import streamlit_feedback

# Importing Airtable
from pyairtable import Api

# Importing watsonx Library 
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from langchain.llms import WatsonxLLM


#
#    LLM BACK-END
#

# Authenticating on watsonx
@st.cache_resource
def auth():
    watsonx_api_key = os.getenv("API_KEY", None)
    os.environ["WATSONX_APIKEY"] = watsonx_api_key
    ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)
    project_id = os.getenv("PROJECT_ID", None)
    if watsonx_api_key is None or ibm_cloud_url is None or project_id is None:
        print("API Key, API endpoint or Project ID failure")
        return None
    else:
        return ibm_cloud_url, project_id

# Defining model and parameters
model = "meta-llama/llama-2-70b-chat"
parameters = {
        GenParams.DECODING_METHOD: "greedy",
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.MAX_NEW_TOKENS: 1000,
        GenParams.REPETITION_PENALTY:1.2,
    }

# Loading model
@st.cache_resource
def load_model():
    creds = auth()
    return WatsonxLLM(
                model_id=model,
                url=creds[0],
                project_id=creds[1],
                params=parameters,
            )

watsonx_llm = load_model()

# Loading persisted vector database
@st.cache_resource
def load_data():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_directory = './data-output/db-chat'
    return Chroma(persist_directory=persist_directory,embedding_function=embeddings).as_retriever(search_kwargs={"k": 4})

# Creating a chain that uses the Chroma vector store
chain = RetrievalQA.from_chain_type(
    llm=watsonx_llm,
    chain_type="stuff",
    retriever=load_data(),
    return_source_documents=True,
)


# Assigning the prompt template to the chain
rag_prompt ='''

UXR Playbook content:
{context}

You are AskUXR, a polite and friendly AI assistant dedicated solely to answering questions about UX Research at IBM. Don't create personas, customer journeys, and other research artifacts; offer guidance on how to create those artifacts instead. Don't answer questions unrelated to UX Research.
Use the UXR Playbook content provided above to answer the question. If the answer isn't in the UXR Playbook content available, just say that you don't know, don't try to make up an answer.

Question: {question}

Helpful Answer:'''

# Adding a time stamp to prompt
timestamp = time.strftime("%B %d, %Y", time.localtime())
prompt_template_with_timestamp = f"Use today's date if you need to provide date-sensitive answers: {timestamp}.\n\n{rag_prompt}"
chain.combine_documents_chain.llm_chain.prompt.template = prompt_template_with_timestamp

# Formatting the answers with sources
def process_llm_response(llm_response):

    # Setting up a base url that will be concatenated with the file structure in ./data to form the final url
    base_url = 'https://pages.github.ibm.com/reops/ux-research'
    
    # Storing the answer part from llm result 
    result_str = llm_response['result']

    # Initializing a list to store each source from the llm response
    sources_list = []

    # Setting a variable to store a list of source urls
    seen_urls = set()

    # Showing only unique URLs in the "Source:
    for source in llm_response["source_documents"]:

        if source is not None and 'source' in source.metadata and source.metadata['source'] is not None:

            # Removing any leading or trailing slashes from the path
            file_source = source.metadata['source'].strip('/')

            # Splitting the path into segments using the '/' delimiter
            path_segments = file_source.split('/')

            # Taking all but the first segment assuming 'data' is always the first directory
            # Then joining the remaining segments except the file name, keeping the original subfolders structure
            subfolders = '/'.join(path_segments[1:-1])

            # Extracting filename without extension
            filename_without_ext = path_segments[-1].split('.')[0]

            # Using the subfolders and the filename for the actual URL (without .txt) and forcing lowercase
            final_url = f"{base_url}/{subfolders}/{filename_without_ext}".lower()

            # Creating a markdown link with the prefix "UXR Playbook: "
            markdown_link = f"[{filename_without_ext.capitalize()}]({final_url})"

            if final_url not in seen_urls:
                seen_urls.add(final_url)
                sources_list.append(markdown_link)


    # Join the sources_list with commas
    sources_str = 'Sources: ' + ', '.join(sources_list)

    return result_str, sources_str

# Removing potentially harmful characters from user input
def sanitize_input(input_str):
    # HTML escape to prevent XSS attacks
    sanitized_str = html.escape(input_str)

    # Remove potentially dangerous characters
    dangerous_chars = r'[<>{}]'
    sanitized_str = re.sub(dangerous_chars, '', sanitized_str)

    # Truncate input to a reasonable length
    max_length = 500
    if len(sanitized_str) > max_length:
        sanitized_str = sanitized_str[:max_length]

    return sanitized_str

#
#    FRONT-END : BASIC CHAT QUERY
#

@st.cache_resource
def look_and_feel():
    # Customizing CSS
    st.markdown(""" 
                <style>
                
                /* Hiding Menu elements */
                #MainMenu, footer, .stDeployButton, .en6cib65, div[data-testid="stDecoration"] 
                {visibility: hidden; display: none;}

                /* Typography */
                *, .st-ae  {font-family: "IBM Plex Sans", "Helvetica", Arial, sans-serif !important; font-weight: 300 !important; text-rendering: optimizeLegibility;}
                p, ol, li {font-size: 0.9rem !important;}
                p a {font-weight: 400 !important; text-decoration: none;}
                .main p {font-weight: 300 !important;} 

                /* Header bar */ 
                header { background-color: #000000 !important;}

                /* Header Menu */

                /* Menu icon */
                div[data-testid="stImage"].st-emotion-cache-1v0mbdj img[style*="width: 20px;"] {
                position: fixed; 
                top: 13px; 
                left: 10px;
                z-index: 999990 !important;
                visibility: visible !important;}

                /* Menu Hidden button */
                div.eczjsme1, div.eczjsme2, div.eczjsme3  {
                right:auto !important;
                z-index: 999999 !important;}

                button[data-testid="baseButton-header"],
                button[data-testid="collapsedControl"] {
                border-radius: 0;
                }
                button[data-testid="baseButton-header"]:hover,
                button[data-testid="collapsedControl"]:hover {
                background-color: transparent;
                }
                
                button[data-testid="baseButton-header"] svg,
                button[data-testid="baseButton-headerNoPadding"] svg,
                button[data-testid="collapsedControl"] svg {visibility: hidden !important}

                /* Logo */
                /* div.element-container.st-emotion-cache-xriuio div[data-testid="stStyledFullScreenFrame"] button {display:none} */
                div[data-testid="stImage"].st-emotion-cache-1v0mbdj img[style*="width: 60px;"] {
                position: fixed; 
                top: 15px; 
                left: 70px;
                z-index: 999999 !important;
                visibility: visible !important;}

                /* Side bar content */
                div[data-testid="stSidebarUserContent"] div[data-testid="stImage"] {position: relative;}
                div[data-testid="stSidebarUserContent"] ul li {list-style-type: none !important; margin-left: 0; padding-left: 0;}
                div[data-testid="stSidebarUserContent"] {padding: 7rem 2rem; background-color: rgb(0, 0, 0, 0) !important;}
                @media only screen and (max-width: 600px) {div[data-testid="stSidebarContent"] {background-color: rgb(0, 0, 0, 0.9) !important}}

                /* Spinner */
                div.stSpinner {
                transition: opacity 2s ease-out !important;
                opacity: 1 !important;}

                /* Avatar */
                .stChatMessage > img:first-child {
                border-radius:0px; 
                width: 24px; 
                height: 24px;}

                /* Chat content */
                div[data-testid="stChatMessageContent"] {margin-left: 5px;}
                div[aria-label="Chat message from user"] {
                border-radius: 0px 8px 8px 8px; 
                background: linear-gradient(rgba(57, 57, 57, 1), #393939);
                display: flex;
                max-width: 500px;
                padding: 12px 16px;
                flex-direction: column;
                align-items: flex-start;
                gap: 1rem;
                flex: 0 1 auto;}

                .stChatMessage {
                margin-bottom: 1rem;
                padding: 1rem 0px;
                gap:0.6rem;
                }

                section.main div[data-testid="stVerticalBlock"] {gap:0}

                /* Chat Input */
                .stChatInputContainer, .stChatInputContainer div[data-baseweb="textarea"] {
                border-radius: 8px; 
                background: linear-gradient(to right, rgba(38, 38, 38, 1), #262626); 
                border-image: linear-gradient(linear, rgba(244, 244, 244, 1) 0%, rgba(244, 244, 244, 0) 100%); border-direction: to bottom;}
                .stChatInputContainer {border: 1px solid rgba(244, 244, 244, 0.1); }
                .stChatInputContainer textarea { margin-right: 2rem;}
                .stChatInputContainer button, .stChatInputContainer textarea {border-radius: 8px !important; padding: 1rem; font-size: 0.9rem; caret-color: #0F62FE;} 
                .stChatInputContainer button {margin:0}
                .stChatInputContainer textarea::placeholder {color: #6F6F6F;}
                .stChatInputContainer .e1d2x3se1 {margin-top: -3px;}

                /* Streamlit feedback: Success message */
                .main .stAlert div[data-testid="stNotification"]{ margin: 0 0 40px 50px !important; background-color: rgb(0, 123, 255, 0.1);}

                /* Streamlit feedback: Icon area */
                iframe[title="streamlit_feedback.streamlit_feedback"] {margin-left: 2.3rem}

                /* Global image backround workaround */

                body {background-color: #000000 !important}

                .stChatFloatingInputContainer, 
                .stChatMessage,
                .element-container{background-color: transparent !important; }
                .stApp {
                    background: linear-gradient(0deg, rgba(69, 137, 255, 0.25) 0%, rgba(0, 0, 0, 0.00) 44.17%), linear-gradient(rgba(22, 22, 22, 1), #161616);
                    box-shadow: 0px -80px 70px -65px rgba(69, 137, 255, 0.25) inset;
                }
                
                </style>
                """,
                unsafe_allow_html=True)

    # Customizing Javascript
    st.markdown(""" 
                <script>
                window.onload = function() {
                    var textArea = document.querySelector('textarea[placeholder="Type here..."]');
                    if (textArea) {
                        textArea.focus();
                    }
                };
                </script>
                """,
                unsafe_allow_html=True)

@st.cache_resource
def header():
    # Menu icon
    st.image('./public/icon_menu.svg', width=20)

    # AskUXR logo
    st.image('./public/askuxr-carbonlogo.svg', width=60, output_format="auto")

# Calling Ui functions
look_and_feel()
header()

# Building the sidebar
with st.sidebar:
    st.write('''
            Welcome to AskUXR, a Q&A tool based on the [UX Research Playbook](https://pages.github.ibm.com/reops/ux-research/research-practice-playbook/overview) and designed for quick and efficient access to UXR-related information.
            ''')
    st.info("Please note that AskUXR uses [LLMs](https://www.ibm.com/topics/large-language-models), therefore it can make mistakes. Always confirm the accuracy of the information provided before acting on it.")
    with st.expander('### Know your AI', expanded = False):
        st.write("Let's see what's under the hood.")
        st.markdown(f"> *AI Model:* \n {model} ")
        st.caption(f"The {model} model is hosted on watsonx. [Learn more about watsonx models](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models.html)")
        st.markdown(f"> *Parameters:* \n {parameters} ")
        st.caption("Here we are setting the model to get the answer with highest probability (Greedy decoding) and not so repetitive (1.2 penalty), generating a maximum of 1000 [tokens](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-tokens.html?context=wx&audience=wdp) (around 750 words). [Learn more](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-model-parameters.html?context=wx&audience=wdp)")
        st.markdown(f"> *Prompt template:*\n ``` {rag_prompt} ```")
        st.caption(f"This tool uses a technique called [RAG - Retrieval Augemented Generation](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-rag.html?context=wx&audience=wdp). It dynamically pulls relevant pieces of content from the UXR playbook (variable 'context') that matches the user query (variable 'question'), combines it with the question and instructions, then sends it to the AI model. A 'prompt template' is the structure we use for that.")
    st.markdown("[Share your feedback](https://airtable.com/appButViLQv4vGjfz/pagpZYRk06tX9otLV/form) with us!")


# Initializing custom avatars
avatar_assistant = './public/avatar-assistant.svg' 
avatar_user = './public/avatar-user.svg'

# Initializing session state for holding chat messages
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "What's your question about UX Research at IBM?"}]

# Initializing session state for holding bot responses
if "response" not in st.session_state:
    st.session_state["response"] = None

# Initializing session state for holding user queries
if "question" not in st.session_state:
    st.session_state["question"] = None

# Displaying message history
messages = st.session_state.messages
for message in messages:
    if message['role'] == "assistant":
        with st.chat_message(message['role'], avatar=avatar_assistant):
            st.markdown(message['content'])
    else:
        with st.chat_message(message['role'], avatar=avatar_user):
            st.markdown(message['content'])

# Adding a prompt input template at the bottom and all related interactions
if prompt:= st.chat_input('Type here...'):

    # sanitizing user input
    prompt = sanitize_input(prompt)

    st.session_state["question"] = prompt

    # Storing the user input in the app state
    st.session_state.messages.append(
        {'role':'user','content':prompt})

    # Displaying the prompt
    with st.chat_message('user', avatar=avatar_user):
        st.markdown(prompt)

    #  with a visual status under the Assistant area
    with st.chat_message("assistant", avatar=avatar_assistant):

        # Sending the prompt to the LLM 
        with st.spinner("Generating an answer using watsonx..."):

            # Processing the LLM response
            llm_response = chain(prompt)
            result, sources = process_llm_response(llm_response)

            # Storing the processed LLM response in session state
            st.session_state["response"] = result + "\n\n" + sources

            # Streamming the processed LLM response
            placeholder = st.empty()
            full_response = ''
            for chunk in result:
                full_response += chunk
                placeholder.markdown(full_response + "▌")
                time.sleep(0.001)
            placeholder.markdown(full_response)
            st.markdown(sources)
    st.session_state.messages.append({"role": "assistant", "content": st.session_state["response"]})
    print(llm_response)
    print(timestamp)

#
#    FRONT-END : FEEDBACK SYSTEM
#

# Defining the feedback action
@st.cache_resource
def airtable_table():
    api = Api(os.getenv("AIRTABLE_API_KEY", None))
    # Getting the Airtable base ID and Table ID. 
    # Learn more here: https://support.airtable.com/docs/finding-airtable-ids
    table = api.table('appButViLQv4vGjfz', 'tblkGgvoHDGv5Rjaz')
    return table

# def _submit_feedback(user_response, prompt, result, emoji=None):
def _submit_feedback(user_response, emoji=None):

    # Showing a spinner while the data is stored in Airtable.
    with st.spinner("Submitting your feedback..."):
        
        # If storing user feedback on Airtable, comment this line below:
        st.success("Thanks for providing your feedback for demonstration purposes.", icon=None)

        # If storing user feedback on Airtable: 
        # 1) Create an Airtable base with the following columns: User Question, Chatbot Response, User Feedback, and User score (icon).
        # 2) Change EACH of the colummn IDs below (e.g., "fldB6cofeD5Y6vxWs") to match the respective column IDs in your Airtable base.
        # 3) UNcomment the next 14 lines (465 to 478);
        # try:
            # record = {
            #     'fldB6cofeD5Y6vxWs': st.session_state["question"], # User Question
            #     'fldqzXmwtw1v8NXa6': st.session_state["response"], # Chatbot Response
            #     'fldlkBStDaBaLoXGh': user_response.get("text", ""), # User Feedback
            #     'fld2QF9SSFIuPryuV': user_response.get("score", ""), # User score (icon)
            #     'fldPnlex1Zcgc1els': rag_prompt, # Prompt template used
            # }
            # airtable_response = airtable_table().create(record)
            # st.success(f"**The following feedback was submitted, thanks!**\n\n *{record['fldlkBStDaBaLoXGh']}*", icon=None)
            # print("Record created:", airtable_response)  # Print response for debugging
        # except Exception as e:
            # st.error("Failed to submit feedback")
            # print("Error:", e)

# Adding the feedback component to the chatbot reponses
if st.session_state["response"]:
    streamlit_feedback(
        align="flex-start",
        feedback_type="thumbs",
        optional_text_label="Please give more details.",
        on_submit=_submit_feedback,
        key=f"feedback_{len(messages)}",
    )
