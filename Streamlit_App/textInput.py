import streamlit as st
from streamlit_chat import message


def textInput():
 #st.write("text input:", st.session_state.text_input)
 with st.chat_message("user"):
    #st.markdown(st.session_state.text_input)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": st.session_state.text_input})

if __name__ == "__main__":
  if "messages" not in st.session_state:
    st.session_state.messages = []
  
  for i, (content, is_user) in enumerate(st.session_state["messages"]):
    message(content, is_user=is_user, key=str(i))
  st.text_input("Movie title", "Life of Brian", key="text_input", on_change=textInput)
 

