import streamlit as st
from rag.retriever import retrieve
from rag.chain import answer

st.set_page_config(page_title="MTG Combo Assistant", page_icon="🃏", layout="centered")
st.title("🃏 MTG Combo Assistant")
st.caption("Ask about card synergies, combos, and interactions.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if question := st.chat_input("Ask about combos, e.g. 'What goes infinite with Melira?'"):
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching cards and combos..."):
            try:
                context = retrieve(question)
                response = answer(question, context)
            except Exception as e:
                response = f"Error: {e}"
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
