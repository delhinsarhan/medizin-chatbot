"""
Chatbot interface for the ESC consensus document on hypertension in
children and adolescents (RAG with Arcana) via the GWDG Chat AI API
(SAIA / Academic Cloud).

Requirements:
    pip install streamlit openai

Run with:
    streamlit run app.py
"""

import os
import re
import streamlit as st
from openai import OpenAI

# --- Configuration -------------------------------------------------------

# The API key is read from the environment variable CHAT_AI_API_KEY
# (set it in the terminal with: export CHAT_AI_API_KEY="your_key")

def lese_api_key() -> str:
    """Online from the Streamlit-Secrets, local from environment."""
    try:
        return st.secrets["CHAT_AI_API_KEY"]
    except Exception:
        return os.environ.get("CHAT_AI_API_KEY", "")

API_KEY = lese_api_key()
ARCANA_ID = "delhin.sarhan/BA_med"
MODEL = "qwen3-30b-a3b-instruct-2507"
BASE_URL = "https://chat-ai.academiccloud.de/v1"

# Full reference, shown once in front of the chapter / table in the source line
CITATION = (
    "de Simone G, Mancusi C, Hanssen H, et al. Hypertension in children and "
    "adolescents. Eur Heart J. 2022;43:3290–3301. doi:10.1093/eurheartj/ehac328"
)

SYSTEM_PROMPT = """You are an assistant for the ESC consensus document on hypertension in children and adolescents (de Simone et al., European Heart Journal 2022). Answer only on the basis of the provided context. If the answer is not contained in the context, say so. Always answer in English.

End every answer with one final line that states where in the document the information comes from, in exactly this format:
SOURCE: Chapter <number>
If the information comes from a table, figure or box, add it:
SOURCE: Chapter <number>, Table <number>
If the information comes from several places, separate them with a semicolon.

Rules for the SOURCE line:
- Only name the chapter a passage belongs to. Do not name chapters that are only referred to with "see Chapter ...".
- Only name a table, figure or box if its title appears in the provided context and your answer uses it.
- Do not write the title of the document or any other reference in the SOURCE line.
- Do not write anything after the SOURCE line.
- If you cannot tell the chapter, write: SOURCE: unspecified
- If the answer is not contained in the context, leave out the SOURCE line.

Chapters of the document:
Chapter 1: Definition and classification
Chapter 2: How to measure BP in children and adolescents
Chapter 3: Clinical evaluation and assessment of hypertension-mediated target organ damage
Chapter 4: Secondary hypertension
Chapter 5: Treatment of hypertension
Chapter 6: Assessment and management of concomitant risk factors
Chapter 7: Implementation of suggestions in the real world

Tables, figures and boxes by chapter:
Chapter 1: Table 1, Table 2, Box 1
Chapter 2: Figure 1
Chapter 3: Table 3, Box 2
Chapter 4: Table 4
Chapter 5: Table 5, Figure 2, Box 3
Chapter 6: Table 6
The Graphical Abstract belongs to no chapter. Cite it as: SOURCE: Graphical Abstract"""

# Matches the source line, also when the model adds Markdown such as **Source:**
SOURCE_LINE = re.compile(
    r"^[\s*_#>]*sources?[*_]*\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE
)

# Arcana appends the retrieved passages as a "References:" block after the answer
REFERENCES_HEADER = re.compile(
    r"^[\s*_#]*references?[*_]*\s*:?[*_\s]*$", re.IGNORECASE | re.MULTILINE
)
# Inline markers such as [RREF1] that point into that block
REF_MARKER = re.compile(r"\s*\[RREF\d+\]")

# --- Client setup (once per session) -------------------------------------

@st.cache_resource
def get_client() -> OpenAI:
    return OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        default_headers={"inference-service": "saia-openai-gateway"},
    )


def ask_question(client: OpenAI, question: str) -> str:
    """Sends a question to the model with RAG context from the Arcana."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
        top_p=0.05,
        extra_body={
            "enable-tools": True,
            "arcana": {"id": ARCANA_ID},
        },
    )
    return response.choices[0].message.content


def split_answer(text: str) -> tuple[str, str, str]:
    """
    Splits the raw model output into three parts:
    the answer, the location from the SOURCE line, and the references block
    that Arcana appends (kept for the evaluation, but not shown).
    """
    header = REFERENCES_HEADER.search(text)
    body = text[: header.start()] if header else text
    references = text[header.start():].strip() if header else ""

    found = [m.group(1).strip(" *_") for m in SOURCE_LINE.finditer(body)]
    if not found and references:
        # In case the SOURCE line ended up after the references block
        found = [m.group(1).strip(" *_") for m in SOURCE_LINE.finditer(references)]
        references = SOURCE_LINE.sub("", references).strip()

    answer = REF_MARKER.sub("", SOURCE_LINE.sub("", body)).rstrip()
    location = "; ".join(s for s in found if s)
    return answer, location, references


def show_answer(answer: str, location: str) -> None:
    st.markdown(answer)
    if not location:
        return  # no source for answers that are not in the guideline
    if location.lower() in ("unspecified", "unknown"):
        st.caption(f"Source: {CITATION}")
    else:
        st.caption(f"Source: {CITATION} – {location}")


# --- Streamlit interface -------------------------------------------------

st.set_page_config(page_title="Paediatric Hypertension Guideline Chatbot", page_icon="🩺")
st.title("🩺 Paediatric Hypertension Guideline Chatbot")

""" def password_ok() -> bool:
    try:
        correct = st.secrets["APP_PASSWORD"]
    except Exception:
        return True   # local without password
    if st.session_state.get("signed_in"):
        return True
    entered = st.text_input("Password", type="password")
    if entered == correct:
        st.session_state.signed_in = True
        st.rerun()
    elif entered:
        st.error("Wrong password.")
    return False

if not password_ok():
    st.stop() """

st.caption(
    "Research prototype for a bachelor's thesis, based on the ESC consensus "
    "document on hypertension in children and adolescents (2022). "
    "Not intended for clinical decision-making."
)

client = get_client()

# Store the chat history in the session so it survives reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the previous conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            show_answer(message["content"], message.get("location", ""))
        else:
            st.markdown(message["content"])

# Accept new user input
question = st.chat_input("Ask a question about the guideline ...")

if question:
    # Show and store the user's question
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate the answer and show it with the source in small print.
    # The references block from Arcana is not shown.
    with st.chat_message("assistant"):
        with st.spinner("Searching the guideline ..."):
            raw = ask_question(client, question)
        answer, location, references = split_answer(raw)
        show_answer(answer, location)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "location": location,
         "references": references}
    )