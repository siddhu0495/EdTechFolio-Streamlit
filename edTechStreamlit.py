import streamlit as st
import openai
import ollama
import io
import pandas as pd
import json
import os
import hashlib
# import google.generativeai as genai
from google import genai
from google.genai import types
from fpdf import FPDF
from docx import Document
from pptx import Presentation

# --- PAGE CONFIG ---
st.set_page_config(page_title="EdTech Folio", layout="wide")

# --- LLM INTEGRATION PROVISION ---
def get_llm_response(tool_name, inputs):
    """
    Generates a response using the selected LLM provider (Mock, OpenAI, or Gemini).
    """
    provider = st.session_state.get('api_provider', 'Mock')
    api_key = st.session_state.get('api_key', '')
    ollama_model = st.session_state.get('ollama_model', 'llama3.2')

    # 1. Construct the Prompt based on the tool
    prompt = ""
    if tool_name == "YouTube Generator":
        prompt = f"Generate guiding questions for a YouTube video with this URL: {inputs.get('url')}. If you cannot access the video, provide general questions based on the topic inferred from the URL."
    elif tool_name == "Text Question":
        prompt = f"Generate text-dependent questions based on the following text:\n\n{inputs.get('text')}"
    elif tool_name == "Worksheet Generator":
        prompt = f"Create a comprehensive worksheet for the topic: {inputs.get('topic')}."
    elif tool_name == "MCQ Generator":
        prompt = f"Generate {inputs.get('num_questions')} multiple-choice questions based on this context:\n\n{inputs.get('context')}"
    elif tool_name == "Text Summarizer":
        prompt = f"Summarize the following text. Length: {inputs.get('length')}.\n\nText: {inputs.get('text')}"
    elif tool_name == "Text Rewriter":
        prompt = f"Rewrite the following text based on these criteria: {inputs.get('criteria')}.\n\nText: {inputs.get('text')}"
    elif tool_name == "Proof Read":
        prompt = f"Proofread the following text for grammar, spelling, punctuation, and clarity:\n\n{inputs.get('text')}"
    elif tool_name == "Lesson Plan":
        prompt = f"Create a lesson plan for '{inputs.get('topic')}' with a duration of {inputs.get('duration')} minutes."
    elif tool_name == "Report Card":
        prompt = f"Write report card comments for student '{inputs.get('name')}'. Strengths: {inputs.get('strengths')}. Areas for Growth: {inputs.get('growth_areas')}."
    elif tool_name == "Essay Grader":
        prompt = f"Grade the following essay based on this rubric: {inputs.get('rubric')}.\n\nEssay:\n{inputs.get('essay')}"
    elif tool_name == "PPT Generator":
        prompt = f"Create a presentation outline with {inputs.get('num_slides')} slides for the topic: {inputs.get('topic')}."
    elif tool_name == "Question Paper Creation":
        prompt = f"Create a {inputs.get('exam_type')} question paper for '{inputs.get('topic')}' with {inputs.get('num_questions')} questions."
    elif tool_name == "Paper Correction":
        prompt = f"Provide a correction summary for the answer sheet file named '{inputs.get('file_name')}'. (Note: Real file analysis requires OCR integration)."

    # 2. Handle Real API Calls
    if provider != "Mock":
        if provider != "Ollama" and not api_key:
            return "⚠️ Please enter an API Key in the sidebar settings."
        
        try:
            if provider == "OpenAI":
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
            elif provider == "Gemini":
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model='gemini-2.0-flash', contents=prompt
                )
                return response.text
            elif provider == "Ollama":
                # Ensure Ollama is running locally (default port 11434)
                response = ollama.chat(model=ollama_model, messages=[{'role': 'user', 'content': prompt}])
                return response['message']['content']
        except Exception as e:
            if provider == "Ollama" and "404" in str(e):
                return f"⚠️ Model '{ollama_model}' not found locally. Please run `ollama pull {ollama_model}` in your terminal."
            return f"❌ Error: {str(e)}"
    
    # 3. Fallback to Mock Logic
    if tool_name == "YouTube Generator":
        return "Sample Output: What are the three key points mentioned about photosynthesis in the video?"
    elif tool_name == "Text Question":
        return "Sample Output: Based on the text, why did the character decide to leave?"
    elif tool_name == "Worksheet Generator":
        return "Sample Worksheet:\n\n**The Water Cycle**\n\n1. Fill in the blank: The process of water turning into vapor is called ______.\n2. What is precipitation?"
    elif tool_name == "MCQ Generator":
        return f"Generated {inputs.get('num_questions')} questions.\n\nSample Question: What is the capital of France?\nA) Berlin B) Madrid C) Paris D) Rome"
    elif tool_name == "Text Summarizer":
        return f"This is a {inputs.get('length', '').lower()} summary of the provided text, focusing on the key points."
    elif tool_name == "Text Rewriter":
        return f"Here is the rewritten text, adapted to be '{inputs.get('criteria')}'. "
    elif tool_name == "Proof Read":
        return "Sample Output: The original text had a few grammatical errors which have been corrected for clarity and flow."
    elif tool_name == "Lesson Plan":
        return f"**Lesson Plan: {inputs.get('topic')} ({inputs.get('duration')} mins)**\n\n*   **Objective:** Students will be able to...\n*   **Activities:** Warm-up, Direct Instruction, Guided Practice..."
    elif tool_name == "Report Card":
        return f"{inputs.get('name')} is a pleasure to have in class. He/She consistently demonstrates {inputs.get('strengths')}. To further improve, {inputs.get('name')} should focus on {inputs.get('growth_areas')}."
    elif tool_name == "Essay Grader":
        return "**Grade: 4/5**\n\n**Feedback:** The essay presents a strong argument but could be improved by providing more specific examples. There are minor grammatical errors."
    elif tool_name == "PPT Generator":
        return f"**{inputs.get('topic')} - {inputs.get('num_slides')} Slide Outline**\n\n1.  **Title Slide:** {inputs.get('topic')}\n2.  **Introduction:** What is AI?\n3.  **History:** Early Concepts\n..."
    elif tool_name == "Question Paper Creation":
        return f"**{inputs.get('exam_type')} Exam - {inputs.get('topic')}**\n\n**Total Questions: {inputs.get('num_questions')}**\n\n1.  Discuss the primary causes of {inputs.get('topic')}.\n2.  Define the term 'Blitzkrieg'."
    elif tool_name == "Paper Correction":
        return "**Correction Summary:**\n- Question 1: Correct\n- Question 2: Partially incorrect. Key detail missed.\n\n**Marks: 8/10**"
    
    return "Output generated."

# --- FILE GENERATION UTILS ---
def generate_file_content(content, file_format):
    """Converts text content to the specified file format."""
    if file_format == "txt":
        return content.encode("utf-8"), "text/plain"
    
    elif file_format == "html":
        html_content = f"<html><body><pre>{content}</pre></body></html>"
        return html_content.encode("utf-8"), "text/html"
    
    elif file_format == "pdf":
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        # FPDF doesn't handle unicode well by default, so we replace/encode
        # For production, use a unicode font.
        safe_content = content.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, safe_content)
        return pdf.output(dest='S').encode('latin-1'), "application/pdf"
    
    elif file_format == "docx":
        doc = Document()
        doc.add_paragraph(content)
        bio = io.BytesIO()
        doc.save(bio)
        return bio.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    
    elif file_format == "ppt":
        prs = Presentation()
        slide_layout = prs.slide_layouts[1] # Title and Content
        slide = prs.slides.add_slide(slide_layout)
        slide.shapes.title.text = "Generated Content"
        slide.placeholders[1].text = content
        bio = io.BytesIO()
        prs.save(bio)
        return bio.getvalue(), "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    
    elif file_format == "csv":
        df = pd.DataFrame([{"Content": content}])
        return df.to_csv(index=False).encode('utf-8'), "text/csv"
    
    elif file_format == "xlsx":
        df = pd.DataFrame([{"Content": content}])
        bio = io.BytesIO()
        df.to_excel(bio, index=False)
        return bio.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        
    return None, None

def render_download_section(content, key_suffix):
    """Renders the download options section."""
    st.divider()
    st.subheader("💾 Download Options")
    col1, col2 = st.columns([2, 1])
    with col1:
        file_format = st.selectbox("Select Format", ["txt", "html", "pdf", "docx", "ppt", "csv", "xlsx"], key=f"fmt_{key_suffix}")
    with col2:
        file_data, mime = generate_file_content(content, file_format)
        st.download_button("Download File", data=file_data, file_name=f"generated_content.{file_format}", mime=mime, key=f"btn_{key_suffix}")

# --- DATA PERSISTENCE ---
DATA_FILE = "edtech_data.json"

def load_data():
    if not os.path.exists(DATA_FILE):
        return {"users": {}}
    try:
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    except:
        return {"users": {}}

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    data = load_data()
    if username in data["users"]:
        return False
    data["users"][username] = {
        "password": hash_password(password),
        "history": []
    }
    save_data(data)
    return True

#admin: password123
#testuser: password
#test: sid

def authenticate_user(username, password):
    data = load_data()
    if username not in data["users"]:
        return False
    if data["users"][username]["password"] == hash_password(password):
        return True
    return False

def get_user_history(username):
    data = load_data()
    return data["users"].get(username, {}).get("history", [])

def save_user_history(username, entry):
    data = load_data()
    if username in data["users"]:
        data["users"][username]["history"].append(entry)
        save_data(data)

def add_history_entry(tool, input_data, output):
    entry = {"tool": tool, "input": input_data, "output": output}
    st.session_state.history.append(entry)
    if st.session_state.get('username'):
        save_user_history(st.session_state.username, entry)

# --- INITIALIZE SESSION STATE ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_tool' not in st.session_state:
    st.session_state.current_tool = None

def login_page():
    st.container()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🔐 Login to EdTech Hub")
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login", use_container_width=True):
                if authenticate_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.history = get_user_history(username)
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        
        with tab2:
            new_user = st.text_input("Username", key="signup_user")
            new_pass = st.text_input("Password", type="password", key="signup_pass")
            if st.button("Sign Up", use_container_width=True):
                if new_user and new_pass:
                    if register_user(new_user, new_pass):
                        st.success("Account created! Please login.")
                    else:
                        st.error("Username already exists.")
                else:
                    st.error("Please fill in all fields")

if not st.session_state.logged_in:
    login_page()
else:
    # --- SIDEBAR ---
    with st.sidebar:
        st.write(f"👤 **{st.session_state.username}**")
        if st.button("Logout", key="logout_btn"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.history = []
            st.rerun()
        st.divider()
        st.title("EdTech Hub")
        
        def reset_tool_state():
            st.session_state.current_tool = None
            st.session_state.last_tool_output = None

        nav_option = st.radio("Navigation", ["Home", "Magic Tools", "Output History"], on_change=reset_tool_state)

        st.divider()
        st.subheader("⚙️ Settings")
        st.session_state.api_provider = st.selectbox("Select Model Provider", ["Mock", "Ollama"], key="api_provider_select")
        
        if st.session_state.api_provider == "Ollama":
            # Add more models to this list as needed
            st.session_state.ollama_model = st.selectbox("Select Ollama Model", ["deepseek-r1", "llama3.2", "gpt-oss", "qwen3", "qwen3-vl", "mistral", "phi", "gemma3", "translategemma"], key="ollama_model_select")
        elif st.session_state.api_provider != "Mock":
            st.session_state.api_key = st.text_input(f"Enter {st.session_state.api_provider} API Key", type="password", key="api_key_input")

    # --- MAIN PANEL ---

    if nav_option == "Home":
        st.title("🏠 Welcome to EdTech Folio")
        st.markdown("""
        ### Your AI-Powered Education Assistant
        
        Select **Magic Tools** from the sidebar to access our suite of tools:
        - 🎬 **YouTube Generator**
        - 📝 **Worksheet & MCQ Generators**
        - 🍎 **Lesson Planning**
        - And much more!
        
        Check **Output History** to view your saved generations.
        """)

    elif nav_option == "Output History":
        st.title("📜 Output History")
        if not st.session_state.history:
            st.info("No history available yet. Use the Magic Tools to generate content!")
        else:
            for i, entry in enumerate(reversed(st.session_state.history)):
                with st.expander(f"#{len(st.session_state.history)-i}: {entry['tool']}", expanded=True):
                    st.markdown("**User Input:**")
                    if isinstance(entry.get('input'), dict):
                        for key, value in entry['input'].items():
                            st.markdown(f"- **{key.replace('_', ' ').title()}:** {value}")
                    else:
                        st.write(entry.get('input'))
                    st.markdown("**Model Answer:**")
                    st.write(entry['output'])

    elif nav_option == "Magic Tools":
        # Check if a specific tool is selected
        if st.session_state.current_tool:
            if st.button("← Back to Magic Tools"):
                st.session_state.current_tool = None
                st.rerun()
            
            tool = st.session_state.current_tool
            
            # --- TOOL: YouTube Generator ---
            if tool == "YouTube Generator":
                st.title("🎬 YouTube Generator")
                st.write("Generate guiding questions aligned to a YouTube video.")
                url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
                if st.button("Generate Guiding Questions", key="yt_gen"):
                    if url:
                        with st.spinner("Brewing questions from the video..."):
                            output = get_llm_response(tool, {"url": url})
                            st.session_state.last_tool_output = {'tool': tool, 'content': output}
                            add_history_entry("YouTube Generator", {"url": url}, output)
                    else:
                        st.warning("Please enter a YouTube URL.")

            # --- TOOL: Text Question ---
            elif tool == "Text Question":
                st.title("❓ Text Question Generator")
                st.write("Generate text-dependent questions for students based on any text.")
                text = st.text_area("Paste your text here:", height=250)
                if st.button("Generate Questions", key="txt_q_gen"):
                    if text:
                        with st.spinner("Analyzing text and creating questions..."):
                            output = get_llm_response(tool, {"text": text})
                            st.session_state.last_tool_output = {'tool': tool, 'content': output}
                            add_history_entry("Text Question", text[:100] + "...", output)
                    else:
                        st.warning("Please paste some text.")

            # --- TOOL: Worksheet Generator ---
            elif tool == "Worksheet Generator":
                st.title("📝 Worksheet Generator")
                st.write("Generate a Worksheet based on any topic or text.")
                topic = st.text_input("Topic or Text", placeholder="e.g., The Water Cycle")
                if st.button("Generate Worksheet", key="ws_gen"):
                    if topic:
                        with st.spinner("Designing your worksheet..."):
                            output = get_llm_response(tool, {"topic": topic})
                            st.session_state.last_tool_output = {'tool': tool, 'content': output}
                            add_history_entry("Worksheet Generator", {"topic": topic}, output)
                    else:
                        st.warning("Please enter a topic.")

            # --- TOOL: MCQ Generator ---
            elif tool == "MCQ Generator":
                st.title("📝 MCQ Generator")
                st.write("Create a multiple choice assessment based on any topic, standard(s) or criteria!")
                context = st.text_area("Paste your source text, topic, or criteria here:", height=200)
                num_questions = st.slider("Number of questions", 1, 10, 5)
                if st.button("Generate Quiz", key="mcq_gen"):
                    if context:
                        with st.spinner("Analyzing text and creating questions..."):
                            output = get_llm_response(tool, {"context": context, "num_questions": num_questions})
                            st.session_state.last_tool_output = {'tool': tool, 'content': output}
                            add_history_entry("MCQ Generator", {"context": context[:100] + "...", "num_questions": num_questions}, output)
                    else:
                        st.warning("Please provide some context.")

            # --- TOOL: Text Summarizer ---
            elif tool == "Text Summarizer":
                st.title("✂️ Text Summarizer")
                st.write("Take any text and summarize it in whatever length you choose.")
                text = st.text_area("Paste the text to summarize:", height=250)
                length = st.select_slider("Summary Length", options=["Short", "Medium", "Long"], value="Medium")
                if st.button("Summarize Text", key="sm_gen"):
                    if text:
                        with st.spinner("Summarizing..."):
                            output = get_llm_response(tool, {"text": text, "length": length})
                            st.session_state.last_tool_output = {'tool': tool, 'content': output}
                            add_history_entry("Text Summarizer", {"text": text[:100] + "...", "length": length}, output)
                    else:
                        st.warning("Please paste some text to summarize.")

            # --- TOOL: Text Rewriter ---
            elif tool == "Text Rewriter":
                st.title("✍️ Text Rewriter")
                st.write("Take any text and rewrite it with custom criteria however you’d like!")
                text = st.text_area("Paste the text to rewrite:", height=200)
                criteria = st.text_input("Rewriting Criteria", placeholder="e.g., Make it more professional, simplify for a 5th grader")
                if st.button("Rewrite Text", key="rw_gen"):
                    if text and criteria:
                        with st.spinner("Rewriting..."):
                            output = get_llm_response(tool, {"text": text, "criteria": criteria})
                            st.session_state.last_tool_output = {'tool': tool, 'content': output}
                            add_history_entry("Text Rewriter", {"text": text[:100] + "...", "criteria": criteria}, output)
                    else:
                        st.warning("Please provide both text and criteria.")

            # --- TOOL: Proof Read ---
            elif tool == "Proof Read":
                st.title("🧐 Proofreader")
                st.write("Have any text proofread for grammar, spelling, punctuation, and clarity.")
                text = st.text_area("Paste the text to proofread:", height=250)
                if st.button("Proofread Text", key="pr_gen"):
                    if text:
                        with st.spinner("Checking your text..."):
                            output = get_llm_response(tool, {"text": text})
                            st.session_state.last_tool_output = {'tool': tool, 'content': output}
                            add_history_entry("Proof Read", text[:100] + "...", output)
                    else:
                        st.warning("Please paste some text to proofread.")

            # --- TOOL: Lesson Plan ---
            elif tool == "Lesson Plan":
                st.title("🍎 Lesson Plan Generator")
                st.write("Generate a lesson plan for a topic or objective you’re teaching.")
                topic = st.text_input("Topic/Objective", placeholder="e.g., Introduction to Python loops")
                duration = st.number_input("Class Duration (minutes)", min_value=15, max_value=120, value=45)
                if st.button("Generate Lesson Plan", key="lp_gen"):
                    if topic:
                        with st.spinner("Planning your lesson..."):
                            output = get_llm_response(tool, {"topic": topic, "duration": duration})
                            st.session_state.last_tool_output = {'tool': tool, 'content': output}
                            add_history_entry("Lesson Plan", {"topic": topic, "duration": duration}, output)
                    else:
                        st.warning("Please enter a topic or objective.")

            # --- TOOL: Report Card ---
            elif tool == "Report Card":
                st.title("📊 Report Card Comment Generator")
                st.write("Generate report card comments with a student’s strengths and areas for growth.")
                name = st.text_input("Student's Name")
                strengths = st.text_area("Strengths", placeholder="e.g., Participates well in class, strong analytical skills")
                growth_areas = st.text_area("Areas for Growth", placeholder="e.g., Needs to show work in math, can be distracted")
                if st.button("Generate Comments", key="rc_gen"):
                    if name and strengths and growth_areas:
                        with st.spinner("Writing comments..."):
                            output = get_llm_response(tool, {"name": name, "strengths": strengths, "growth_areas": growth_areas})
                            st.session_state.last_tool_output = {'tool': tool, 'content': output}
                            add_history_entry("Report Card", {"name": name, "strengths": strengths, "growth_areas": growth_areas}, output)
                    else:
                        st.warning("Please fill in all fields.")

            # --- TOOL: Essay Grader ---
            elif tool == "Essay Grader":
                st.title("🎓 Essay Grader")
                st.write("Grade an essay based on a rubric or criteria.")
                essay = st.text_area("Paste the student's essay here:", height=300)
                rubric = st.text_area("Paste the grading rubric/criteria:", height=150, placeholder="e.g., Clarity: /5, Argument: /5, Grammar: /5")
                if st.button("Grade Essay", key="eg_gen"):
                    if essay and rubric:
                        with st.spinner("Grading the essay..."):
                            output = get_llm_response(tool, {"essay": essay, "rubric": rubric})
                            st.session_state.last_tool_output = {'tool': tool, 'content': output}
                            add_history_entry("Essay Grader", {"essay": essay[:100] + "...", "rubric": rubric}, output)
                    else:
                        st.warning("Please provide both the essay and the rubric.")

            # --- TOOL: PPT Generator ---
            elif tool == "PPT Generator":
                st.title("🖼️ PPT Generator")
                st.write("Generate a presentation outline based on a topic.")
                topic = st.text_input("Presentation Topic", placeholder="e.g., The History of Artificial Intelligence")
                num_slides = st.slider("Number of Slides", 5, 20, 10)
                if st.button("Generate PPT Outline", key="ppt_gen"):
                    if topic:
                        with st.spinner("Outlining your presentation..."):
                            output = get_llm_response(tool, {"topic": topic, "num_slides": num_slides})
                            st.session_state.last_tool_output = {'tool': tool, 'content': output}
                            add_history_entry("PPT Generator", {"topic": topic, "num_slides": num_slides}, output)
                    else:
                        st.warning("Please enter a topic.")

            # --- TOOL: Question Paper Creation ---
            elif tool == "Question Paper Creation":
                st.title("📜 Question Paper Creator")
                st.write("Create a question paper for any topic, text, no. of questions and exam-type.")
                topic = st.text_input("Topic/Text", placeholder="e.g., World War II")
                num_q = st.number_input("Number of Questions", min_value=1, max_value=50, value=10)
                exam_type = st.selectbox("Exam Type", ["Semester", "Yearly", "Class Test"])
                if st.button("Create Question Paper", key="qp_gen"):
                    if topic:
                        with st.spinner("Constructing question paper..."):
                            output = get_llm_response(tool, {"topic": topic, "num_questions": num_q, "exam_type": exam_type})
                            st.session_state.last_tool_output = {'tool': tool, 'content': output}
                            add_history_entry("Question Paper Creation", {"topic": topic, "num_questions": num_q, "exam_type": exam_type}, output)
                    else:
                        st.warning("Please enter a topic.")

            # --- TOOL: Paper Correction ---
            elif tool == "Paper Correction":
                st.title("✅ Paper Correction Assistant")
                st.write("Generate a correction, summary, and marks based on the answer sheet provided.")
                answer_sheet = st.file_uploader("Upload Answer Sheet (Image or PDF)", type=['png', 'jpg', 'jpeg', 'pdf'])
                if st.button("Correct Paper", key="pc_gen"):
                    if answer_sheet:
                        with st.spinner("Correcting the paper..."):
                            output = get_llm_response(tool, {"file_name": answer_sheet.name})
                            st.session_state.last_tool_output = {'tool': tool, 'content': output}
                            add_history_entry("Paper Correction", {"file_name": answer_sheet.name}, output)
                    else:
                        st.warning("Please upload an answer sheet.")

            # --- CONSOLIDATED OUTPUT DISPLAY ---
            if st.session_state.get('last_tool_output') and st.session_state['last_tool_output']['tool'] == tool:
                st.success("Content Generated!")
                content = st.session_state['last_tool_output']['content']
                
                # Determine rendering method based on tool type
                markdown_tools = [
                    "Worksheet Generator", "Lesson Plan", "Essay Grader", 
                    "PPT Generator", "Question Paper Creation", "Paper Correction"
                ]
                if tool in markdown_tools:
                    st.markdown(content)
                else:
                    st.write(content)
                
                render_download_section(content, "final_output")

        else:
            # Show Grid of Tools
            st.title("✨ Magic Tools")
            st.write("Select a specialized AI service to get started")
            
            # --- CUSTOM CSS FOR SERVICE CARDS ---
            st.markdown("""
            <style>
            section.main div.stButton > button {
                height: 180px;
                width: 100%;
                background-color: #f8f9fa;
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                color: #333;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                text-align: center;
                white-space: pre-wrap !important;
            }
            </style>
            """, unsafe_allow_html=True)

            tools_list = [
                {"name": "YouTube Generator", "icon": "🎬", "desc": "Generate guiding questions aligned to a YouTube video."},
                {"name": "Text Question", "icon": "❓", "desc": "Generate text-dependent questions for students based on any text."},
                {"name": "Worksheet Generator", "icon": "📝", "desc": "Generate a Worksheet based on any topic or text."},
                {"name": "MCQ Generator", "icon": "📝", "desc": "Create a multiple choice assessment based on any topic."},
                {"name": "Text Summarizer", "icon": "✂️", "desc": "Take any text and summarize it in whatever length you choose."},
                {"name": "Text Rewriter", "icon": "✍️", "desc": "Take any text and rewrite it with custom criteria."},
                {"name": "Proof Read", "icon": "🧐", "desc": "Have any text proofread for grammar, spelling, and clarity."},
                {"name": "Lesson Plan", "icon": "🍎", "desc": "Generate a lesson plan for a topic or objective."},
                {"name": "Report Card", "icon": "📊", "desc": "Generate report card comments based on strengths/growth."},
                {"name": "Essay Grader", "icon": "🎓", "desc": "Grade an essay based on a rubric or criteria."},
                {"name": "PPT Generator", "icon": "🖼️", "desc": "Generate a presentation outline based on a topic."},
                {"name": "Question Paper Creation", "icon": "📜", "desc": "Create a question paper for any topic and exam-type."},
                {"name": "Paper Correction", "icon": "✅", "desc": "Generate a correction and marks based on an answer sheet."}
            ]
            
            # Grid Layout: Iterate in chunks of 3 to ensure proper alignment
            for i in range(0, len(tools_list), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(tools_list):
                        tool = tools_list[i + j]
                        with cols[j]:
                            # Formatting: Title Uppercase for distinction, Description below
                            label = f"{tool['icon']} {tool['name'].upper()}\n\n{tool['desc']}"
                            if st.button(
                                label,
                                key=f"btn_{i+j}",
                                use_container_width=True
                            ):
                                st.session_state.current_tool = tool['name']
                                st.rerun()