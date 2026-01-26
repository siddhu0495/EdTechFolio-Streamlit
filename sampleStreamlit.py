import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(page_title="GenAI Service Hub", layout="wide")

# --- CUSTOM CSS FOR CARDS ---
st.markdown("""
    <style>
    .service-card {
        border-radius: 10px;
        padding: 20px;
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        text-align: center;
        height: 200px;
        transition: 0.3s;
    }
    .service-card:hover {
        border-color: #ff4b4b;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- NAVIGATION LOGIC ---
if 'page' not in st.session_state:
    st.session_state.page = "Home"

def navigate_to(page_name):
    st.session_state.page = page_name

# --- PAGE: HOME (SERVICE CARDS) ---
if st.session_state.page == "Home":
    st.title("🚀 GenAI Service Hub")
    st.subheader("Select a specialized AI service to get started")

    # Define your service cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="service-card"><h3>📝 MCQ Generator</h3><p>Create quizzes from any text or PDF instantly.</p></div>', unsafe_allow_html=True)
        if st.button("Open MCQ Generator", key="btn_mcq"):
            navigate_to("MCQ Generator")

    with col2:
        st.markdown('<div class="service-card"><h3>📈 Marketing AI</h3><p>Generate SEO-optimized blog posts and ad copy.</p></div>', unsafe_allow_html=True)
        if st.button("Open Marketing Tool", key="btn_mkt"):
            navigate_to("Marketing AI")

    with col3:
        st.markdown('<div class="service-card"><h3>📂 Code Upscaler</h3><p>Modernize legacy Python, R, or SAS code.</p></div>', unsafe_allow_html=True)
        if st.button("Open Code Upscaler", key="btn_code"):
            navigate_to("Code Upscaler")

# --- PAGE: MCQ GENERATOR ---
elif st.session_state.page == "MCQ Generator":
    if st.button("← Back to Hub"): navigate_to("Home")
    st.title("📝 GenAI MCQ Generator")
    
    context = st.text_area("Paste your source text here:", height=200)
    num_questions = st.slider("Number of questions", 1, 10, 5)
    
    if st.button("Generate Quiz"):
        with st.spinner("Analyzing text and creating questions..."):
            # Placeholder for your LLM Logic (e.g., LangChain + OpenAI/Gemini)
            st.success(f"Generated {num_questions} questions based on your context!")
            st.info("Sample Question: What is the primary focus of RAG? \n\nA) Retrieval B) Training C) Storage")

# --- PAGE: MARKETING AI ---
elif st.session_state.page == "Marketing AI":
    if st.button("← Back to Hub"): navigate_to("Home")
    st.title("📈 Marketing Content Generator")
    
    product_name = st.text_input("Product/Service Name")
    target_audience = st.text_input("Target Audience (e.g., Tech Professionals)")
    platform = st.selectbox("Platform", ["LinkedIn", "Twitter/X", "Blog Post", "Instagram"])
    
    if st.button("Create Content"):
        with st.spinner("Crafting your copy..."):
            st.markdown(f"**Generated {platform} Post:**")
            st.write(f"Are you a {target_audience} looking to master {product_name}? Check out our latest...")

# --- PAGE: CODE UPSCALER ---
elif st.session_state.page == "Code Upscaler":
    if st.button("← Back to Hub"): navigate_to("Home")
    st.title("📂 AI Code Upscaler")
    st.write("Convert legacy code to modern, optimized Python.")
    
    source_lang = st.selectbox("Source Language", ["SAS", "R", "Legacy Python 2.7", "SQL"])
    code_input = st.text_area("Paste your code here:", height=300)
    
    if st.button("Upscale Code"):
        with st.spinner("Refactoring..."):
            st.code("# Optimized Python Output\nimport pandas as pd\n\ndef optimized_func():\n    pass", language="python")