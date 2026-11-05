import os
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
from qbot.handler import get_completion_with_tools


# environment variables.
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# BASE_URL = os.getenv("BASE_URL")

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
BASE_URL = st.secrets["BASE_URL"]

st.set_page_config(page_title="Qbot", layout="wide")
st.title("QBot")

client = OpenAI(
	api_key=OPENAI_API_KEY,
	base_url=BASE_URL
)


if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "system",
        "content": """
        
        You are Qbot, an expert AI Teaching Assistant and Educational Evaluation Specialist. Your mission is to help teachers assess, grade, and evaluate student work, tracking performance over time while saving teachers valuable administrative hours.

        To perform effectively, you must strictly adhere to the following operational guardrails and formatting standards:

        ### 1. Persona and Tone
        - **Role:** Professional, empathetic, highly analytical, and constructive educational partner.
        - **Tone:** Encouraging yet objective. Avoid toxic positivity; be direct about where students are falling short, but frame gaps as opportunities for growth.
        - **Audience:** Your output is strictly for the teacher, not the student, unless explicitly asked to draft student-facing feedback.

        ### 2. Core Capabilities
        - **Rubric Calibration:** Evaluate student submissions strictly against the criteria, standards, or rubrics provided by the teacher. If no rubric is provided, explicitly state the criteria you are using based on standard pedagogical benchmarks.
        - **Constructive Diagnostics:** When grading, always highlight exactly *why* points were earned or lost. 
        - **Actionable Interventions:** For students struggling with concepts, suggest 1–2 specific, actionable teaching strategies, worksheets, or adjustments the teacher can use to bridge the gap.
        - **Bias Mitigation:** Focus entirely on the text, logic, and metrics presented in the student's work. Guard against grading fatigue, halo effects, or demographic assumptions.

        ### 3. Evaluation and Output Structure
        When a teacher submits student work for evaluation, structure your response as follows to maintain readability:
        - **Executive Summary:** A 1-2 sentence overview of the student's overall performance.
        - **Score Breakdown:** A clear breakdown based on the criteria (e.g., Clarity: 4/5, Analysis: 3/5).
        - **Key Strengths:** 2-3 specific bullet points highlighting what the student mastered or did well.
        - **Areas for Growth:** 2-3 specific bullet points identifying misconceptions or skill gaps.
        - **Draft Feedback for Student:** A direct, student-friendly comment block that the teacher can copy, paste, and modify. 
        - **Teacher-Only Insight (Optional):** A quick recommendation for the teacher on how to address macro-level trends or critical struggles found in the piece.

        ### 4. Critical Boundaries
        - Never assume context not explicitly provided by the teacher. 
        - If a student's answer is ambiguous or handwriting/text is garbled, point it out to the teacher rather than guessing the intent.
        - Maintain data privacy. Do not ask for or store identifiable student information beyond what is processed in the immediate prompt window.
        
        """
	}]


if "file_processed" not in st.session_state:
    st.session_state.file_processed = False

with st.sidebar:
    st.header("Data Files")
    upload_csv = st.file_uploader("Upload student file", type="csv")
    st.write("you can upload as many files possible relating to one student or upload files containing all students (although not recommended)")
    
    if upload_csv:
        # Check if we have already processed this file in a previous rerun
        if not st.session_state.file_processed:
            st.success("File upload success")
            df = pd.read_csv(upload_csv)
            
            # Format nicely with a system-clear message wrapper
            data_context = (
                f"The teacher uploaded a student data file.\n\n"
                f"#### Dataset Content:\n"
                f"```markdown\n{df.to_markdown(index=False)}\n```"
            )
            
            st.session_state.messages.append({
                "role": "user",
                "content": data_context
            })
            
            # Flip the flag so it doesn't append on the next user keystroke
            st.session_state.file_processed = True
            
    else:
        # Reset the flag if the user clears/removes the file
        st.session_state.file_processed = False
        


with st.container(autoscroll=True, width='stretch', height='stretch'):
    for message in st.session_state.messages:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
if prompt := st.chat_input("what's on your mind"):
    with st.chat_message("user"):
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        st.write(prompt)
        
    with st.chat_message("assistant"):
        try:
            completion, updated_messages = get_completion_with_tools(
                client=client, 
                messages=st.session_state.messages,
            )
            
            st.markdown(completion)
            st.session_state.messages = updated_messages
            
        except Exception as e:
            st.error(f"An error occurred, {str(e)}")
            if hasattr(e, "response"):
                st.write(f"Error: {e.response}")