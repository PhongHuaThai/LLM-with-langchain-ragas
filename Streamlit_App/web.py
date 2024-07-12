import os
import tempfile
from rag import ChatPDF
import streamlit as st
from streamlit_chat import message
import json
from export import Export
from handle_question_list import Handle_question_list


def stream_from_rag(user_query):
  chain = st.session_state.chatPDF.chain.pick("answer")
  return chain.stream({"input":user_query})

def process_input():
  user_text = st.session_state.question_input
  with st.chat_message("assistant"):
    response = st.write_stream(stream_from_rag(user_text))
  st.session_state.messages.append({"role": "user", "content": user_text})
  st.session_state.messages.append({"role": "assistant", "content": response})
      
def update_prompt_template():
  st.session_state.chatPDF.update_template(st.session_state.prompt_template)
  question_key = st.session_state.selected_command
  question = [q for q in st.session_state.question_list if q["question"] == question_key][0]
  question["description"] = st.session_state.prompt_template
  Handle_question_list().save_json(st.session_state.question_list)
  st.success("Description saved successfully!")

def read_and_save_file():
  #upload file
  for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], \
             st.spinner(f"Ingesting {file.name}"):
            st.session_state.chatPDF.ingest(file_path)
        os.remove(file_path)

def Execute_default_question():
  #question, prompt_template = Handle_question_list().extract_question(st.session_state.prompt_template)
  selected_questions = get_selected_question()
  st.session_state.chatPDF.update_template(selected_questions["description"])
  st.session_state.question_input = st.session_state.selected_command#question
  st.session_state.trigger_response = True

def show_sidebar():
    with st.sidebar:
      st.text_input("Embeding model", "sentence-transformers/all-mpnet-base-v2")
      st.text_input("LLM model", "llama3:latest")
      st.markdown('#')
      questions = load_instrution_question_list()
      st.session_state.question_list = [q for q in questions]
      question_list = [q for q in questions if q["isExecute"] == True]
      
      selected_command = st.selectbox("Select your command:",
                                      key="selected_command",
                                      options= [q["question"] for q in question_list])
      
      details = [q for q in question_list if q["question"] == selected_command][0] #question_list[selected_question]
      #description = details["description"]
      is_execute = details["isExecute"]
      # st.text_area(
      #     height=100,
      #     key="prompt_template",
      #     label="Prompt template to ask LLM",
      #     value=description,
      #     on_change=update_prompt_template
      # )

      firstCol, secondCol = st.columns(2)
      with firstCol:
        st.button("Execute command", key="execute",type="secondary", on_click=Execute_default_question, disabled= not is_execute)
      with secondCol:
        #if 'messages' in st.session_state and len(st.session_state.messages) > 0:
        if st.button("Export to PDF", type="primary"):
          export = Export()
          json_data = export.export_to_pdf(st.session_state.messages)
          st.download_button("Download PDF", type="primary", data=json_data, file_name="message_history.pdf", mime='application/pdf')

      
      # st.markdown("""
      #   <style>
      #   .custom-button {
      #       background-color: #4CAF50;
      #       color: white;
      #       padding: 15px 32px;
      #       text-align: center;
      #       text-decoration: none;
      #       display: inline-block;
      #       font-size: 16px;
      #       margin: 4px 2px;
      #       cursor: pointer;
      #       border: none;
      #   }
      #   </style>
      #   """, unsafe_allow_html=True)

      # # Display the button and handle its click event
      # if st.button('Click Me!', key='custom', css_class='custom-button'):
      #     st.write("Button clicked!")

def export_message_history():
  return json.dumps(st.session_state.messages)
def like_message(response):
  st.success("like successfully!")
def get_selected_question():
  selected_question = [q for q in st.session_state.question_list if q["question"] == st.session_state.selected_command][0]
  return selected_question
def load_instrution_question_list():
  data = Handle_question_list().load_json()
  questions = [item for item in data]
  return questions

if __name__ == "__main__":
  show_sidebar()
  st.session_state.ingestion_spinner = st.empty()
  if "messages" not in st.session_state:
    st.session_state.messages = []
  if "chatPDF" not in st.session_state:
    st.session_state.chatPDF = ChatPDF()
  if "trigger_response" not in st.session_state:
    st.session_state.trigger_response = False
  if "question_input" not in st.session_state:
    st.session_state.question_input = ""
  st.header('Chat with your documents (RAG Application)')
  st.file_uploader(
    "Upload document",
    type=["pdf"],
    key="file_uploader",
    on_change=read_and_save_file,
    label_visibility="collapsed",
    accept_multiple_files=True,
  )
  for i, msg in enumerate(st.session_state.messages):
    if 'role' in msg:
      with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
    else:
       print(msg)

  #accept user input
  #st.text_input("Message", key="user_input", on_change=process_input)
  if prompt := st.chat_input("Ask a question on uploaded PDF?", key="user_input"):
    if st.session_state["user_input"] and\
    len(st.session_state["user_input"].strip()) > 0:
      question = [q for q in st.session_state.question_list if q["question"] == "Common Prompt"][0]
      st.session_state.chatPDF.update_template(question["description"])
      st.session_state.question_input = st.session_state["user_input"].strip()
      st.session_state.trigger_response = True

  if st.session_state.trigger_response:
    # Add user message to chat history
    prompt = st.session_state.question_input
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
      st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
      #response = st.write_stream(response_generator())
      response = st.write_stream(stream_from_rag(prompt))
    st.session_state.trigger_response = False       
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
  
  
