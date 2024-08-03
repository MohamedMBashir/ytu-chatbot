import streamlit as st
import os
import openai
from dotenv import load_dotenv, find_dotenv


# from dotenv import load_dotenv, find_dotenv
try:
    _ = load_dotenv(find_dotenv()) # read local .env file
except:
    pass

try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except:
    pass

# Check for OpenAI API Key and prompt for it if not found
if 'OPENAI_API_KEY' not in os.environ:
    api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
        openai.api_key = api_key
else:
    openai.api_key = os.environ['OPENAI_API_KEY']


from logic import ytu_chatbot_chain

# st.title("🌟 YTU Chatbot 🌟")
st.markdown("<h1 style='text-align: center; color: white;'>🌟 YTU Chatbot 🌟</h1>", unsafe_allow_html=True)

# Initialize or retrieve the chatbot logic from session state
if 'ytu_chatbot_chain' not in st.session_state:
    st.session_state['ytu_chatbot_chain'] = ytu_chatbot_chain

# # Main chat interface
# user_query = st.text_input("Soru:", key="user_query")

# if user_query:
#     # Get response from the YTU Chatbot Chain
#     print( user_query )
#     answer = st.session_state['ytu_chatbot_chain'].invoke({"query": user_query})
#     st.text_area("Response:", value=answer['result'], height=200)

# ------------------- DEBUG CODE ----------------------------------------
# Main chat interface
user_query = st.text_input("Soru:", key="user_query")

if user_query:
    st.write(f"DEBUG: User input: {user_query}")
    
    # Get response from the YTU Chatbot Chain
    try:
        input_dict = {"query": user_query}
        st.write(f"DEBUG: Input to chain: {json.dumps(input_dict, indent=2)}")
        
        answer = st.session_state['ytu_chatbot_chain'](input_dict)
        
        st.write(f"DEBUG: Raw output from chain: {json.dumps(answer, indent=2)}")
        
        if isinstance(answer, dict) and 'result' in answer:
            st.text_area("Response:", value=answer['result'], height=200)
        else:
            st.error("Unexpected response format from the chain.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write(f"DEBUG: Error details: {repr(e)}")
