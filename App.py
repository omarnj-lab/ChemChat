import streamlit as st
import openai
from streamlit_extras.colored_header import colored_header
import base64
import webbrowser
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from streamlit.components.v1 import html

load_dotenv()

def open_page(url):
    open_script= """
        <script type="text/javascript">
            window.open('%s', '_blank').focus();
        </script>
    """ % (url)
    html(open_script)

st.set_page_config(
    page_title='ChemChatBot',
    page_icon='⚛️',
)
# Custom CSS for the buttons

button_css = """
<style>
.custom-button {
    color: white;
    background-color: #3F3F3F;
    border: none;
    padding: 15px 32px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 12px;
    transition-duration: 0.4s;
}

.custom-button:hover {
    background-color: darkblue;
}
</style>
"""

st.markdown(button_css, unsafe_allow_html=True)


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())

    st.markdown(
        f'''
        <style>
        .stApp {{
            background-image: url(data:image/jpeg;base64,{encoded_string.decode()});
            background-size: 100%;
            height: auto!important;
        }}
        </style>
        ''',
        unsafe_allow_html=True
    )

#add_bg_from_local('back.png')


# Define the predefined questions
predefined_questions = [
    "What is the primary focus of chemistry as a science?",
    "How does chemistry study the composition and behavior of matter?",
    "What is the significance of atoms and bonds in the study of chemistry?",
    "In which industries does chemistry play a pivotal role?",
    "Which six elements make up nearly 99% of the human body's mass?",
    "What are the five main sections of chemistry?",
    "How does inorganic chemistry differ from organic chemistry?",
    "What is the focus of physical chemistry?",
    "What does biochemistry study?",
    "What is the aim of analytical chemistry?",
]

# Sidebar header
st.sidebar.markdown("<h2 About ChemChatBot ⚛️🧪🔬</h2>", unsafe_allow_html=True)


# Dynamic message (as an example)
import random
dynamic_messages = [
    "Atoms and bonds are my specialty!",
    "Did you know? I can help with chemical equations and reactions!",
    "From organic to inorganic, I've got chemistry covered!"
]
dynamic_message = random.choice(dynamic_messages)

# Creative box with details
st.sidebar.markdown(
    f"""
    <div style='background-color: #3F3F3F; padding: 10px; border-radius: 5px; border: 1px solid #ddd;'>
        <p><strong>Welcome to ChemChatBot</strong></p>
        <p>I'm here to assist you with any questions related chemistry.</p>
        <p>{dynamic_message}</p>
        <ul>
            <li>Ask about chemistry </li>
            <li>Learn about its focus</li>
            <li>Get insights into stoms and elements</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("<h2 style='color: orange'>Quick Questions</h2>", unsafe_allow_html=True)

st.markdown("<h1 style='color: orange'>Chat with ChemChatBot 🤖</h1>", unsafe_allow_html=True)


if "conversation" not in st.session_state.keys():
    st.session_state.conversation = [
        {"role": "assistant", "content": "Ask me a question about Chemsitry to help you!"}
    ]

# Display the predefined questions in creatively designed boxes
for question in predefined_questions:
    if st.sidebar.button(question):
        st.session_state.conversation.append({"role": "user", "content": question})

@st.cache_resource(show_spinner=False)
def create_conversation_chain_from_pdf(pdf_file_path):
    with st.spinner(text="Please wait .. "):
        # Extract text from PDF
        text = ""
        pdf_reader = PdfReader(pdf_file_path)
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=200,
            chunk_overlap=10,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Create vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)

        # Create conversation chain
        llm = ChatOpenAI(temperature=0.5, max_tokens = 512 , model_name="gpt-3.5-turbo")
        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            max_tokens_limit=4097,
            memory=memory
        )

    return conversation_chain

# At the start of the Streamlit app (assuming you've defined the create_conversation_chain_from_pdf function earlier)
pdf_file_path = "Chemistry.pdf"
st.session_state.conversation_chain = create_conversation_chain_from_pdf(pdf_file_path)


if prompt := st.chat_input("Your question"):
    st.session_state.conversation.append({"role": "user", "content": prompt})


st.markdown("*Note: This model provides general answers about Chemsitry and might answer partially.*")

if st.button("Clear Chat"):
    st.session_state.conversation = [
        {"role": "assistant", "content": "Ask me a question about Chemsitry to help you!"}
    ]

for message in st.session_state.conversation:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.conversation[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                last_user_message = st.session_state.conversation[-1]["content"]

                # Using the conversation chain for response
                response = st.session_state.conversation_chain(last_user_message)
                # Display the answer to the user
                st.write(response["answer"])
                message = {"role": "assistant", "content": response["answer"]}
                st.session_state.conversation.append(message)
        except Exception as e:
            st.error(f"An error occurred: {e}")
