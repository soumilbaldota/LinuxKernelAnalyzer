import streamlit as st
import requests
import json

# Page configuration
st.set_page_config(
    page_title="LinusSLM: Linux Kernel Log Analyzer",
    page_icon="🐧",
    layout="wide"
)

# Custom CSS for beautiful styling
st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 1rem 2rem;
    }
    
    /* Column borders */
    div[data-testid="column"]:first-child {
        border-right: 2px solid #e1e4e8;
        padding-right: 2rem !important;
        margin-right: 1rem;
    }
    
    div[data-testid="column"]:last-child {
        padding-left: 2rem !important;
    }
    
    /* Text areas */
    .stTextArea textarea {
        font-family: 'Courier New', monospace;
        font-size: 13px;
        background-color: #0d1117;
        color: #c9d1d9;
        border: 1px solid #30363d;
    }
    
    /* AI Analysis Box - Hero style */
    .analysis-hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .analysis-hero h3 {
        margin: 0 0 1rem 0;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    .analysis-content {
        font-size: 1.05rem;
        line-height: 1.6;
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 0.5rem;
        backdrop-filter: blur(10px);
    }
    
    /* Code Reference Cards */
    .code-card {
        background: white;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #e1e4e8;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .code-card:hover {
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    
    .code-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f6f8fa;
    }
    
    .code-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #24292e;
    }
    
    .score-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 2rem;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .file-path {
        background: #f6f8fa;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        color: #586069;
        margin: 0.5rem 0;
    }
    
    /* Stats Cards */
    .stat-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 12px rgba(245, 87, 108, 0.3);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.75rem;
        font-weight: 700;
        color: #24292e;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Input section styling */
    .input-section {
        background: #f6f8fa;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 0.5rem;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }
    
    /* Metrics styling */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #f6f8fa;
        border-radius: 0.5rem;
        font-weight: 600;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #586069;
        font-size: 0.9rem;
        margin-top: 3rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header with gradient
st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; 
                border-radius: 1rem; 
                margin-bottom: 2rem;
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);'>
        <h1 style='color: white; margin: 0; font-size: 2.5rem;'>🐧 Linux Kernel Log Analyzer</h1>
        <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.1rem;'>
            AI-powered kernel diagnostics with RAG-enhanced code analysis
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for API configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    api_endpoint = st.text_input(
        "API Endpoint",
        value="http://localhost:8000/",
        help="Enter your FastAPI endpoint URL"
    )
    
    api_key = st.text_input(
        "API Key (Optional)",
        type="password",
        placeholder="Enter API key if required",
        help="Leave empty if no authentication is needed"
    )
    
    st.divider()
    
    st.markdown("### 📊 Display Settings")
    max_results = st.slider("Max Code References", 1, 10, 5)
    show_similarity = st.checkbox("Show Similarity Scores", value=True)
    sort_by_score = st.checkbox("Sort by Similarity", value=True)
    
    st.divider()
    
    st.markdown("### 📖 Quick Guide")
    st.markdown("""
    1. Paste kernel log
    2. Set your query
    3. Click Analyze
    4. Review AI insights
    """)

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown('<div class="section-header">📝 Kernel Log Input</div>', unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload kernel log file (.log or .txt)",
        type=['log', 'txt'],
        help="Upload a file containing kernel messages"
    )
    
    # Text area for direct input
    kernel_log = st.text_area(
        "Or paste kernel log directly:",
        height=250,
        placeholder="Example:\n[    0.000000] Linux version 5.15.0-73-generic\n[    1.234567] Kernel panic - not syncing: Fatal exception",
        value=uploaded_file.read().decode('utf-8') if uploaded_file else ""
    )
    
    # Query input
    query = st.text_input(
        "🔍 Analysis Query",
        value="what's wrong with the kernel?",
        placeholder="Ask about the kernel issue..."
    )

with col2:
    st.markdown('<div class="section-header">🎯 Actions</div>', unsafe_allow_html=True)
    
    # Analyze button
    analyze_button = st.button("🚀 Analyze Kernel Log", type="primary", use_container_width=True)
    
    # Show current stats if available
    if 'results' in st.session_state:
        st.markdown("---")
        st.markdown("### 📊 Current Analysis")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Code Refs", len(st.session_state.results))
        with col_b:
            st.metric("Status", "✅ Ready")

# Handle analysis
if analyze_button:
    if not kernel_log.strip():
        st.error("⚠️ Please provide a kernel log to analyze.")
    elif not api_endpoint.strip():
        st.error("⚠️ Please configure the API endpoint in the sidebar.")
    else:
        with st.spinner("🔄 Analyzing kernel log with AI..."):
            try:
                # Prepare the request
                headers = {"Content-Type": "application/json"}
                if api_key:
                    headers["x-api-key"] = api_key
                
                payload = {
                    "kernel_log": kernel_log,
                    "query": query,
                    "max_results": max_results
                }
                
                # Make API request
                response = requests.post(
                    api_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    results = response.json()
                    
                    # Extract matches and analysis
                    matches = results.get('matches', results.get('results', []))
                    analysis = results.get('analysis', '')
                    
                    # Store results in session state
                    st.session_state.results = matches
                    st.session_state.analysis = analysis
                    
                    st.success(f"✅ Analysis complete! Found {len(matches)} relevant code snippets")
                    st.rerun()
                    
                else:
                    st.error(f"❌ API Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. Please try again.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Connection error. Please check your API endpoint.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Display AI Analysis if available - HERO SECTION
if 'analysis' in st.session_state and st.session_state.analysis:
    st.markdown('<div class="section-header">🤖 AI-Powered Analysis</div>', unsafe_allow_html=True)
    
    # Format analysis text for HTML display
    analysis_text = st.session_state.analysis.replace('\n', '<br>')
    
    st.markdown(f"""
    <div class="analysis-hero">
        <h3>💡 Kernel Issue Diagnosis</h3>
        <div class="analysis-content">
            {analysis_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Display RAG results if available - CARD LAYOUT
if 'results' in st.session_state and st.session_state.results:
    results = st.session_state.results
    
    st.markdown('<div class="section-header">🔍 Related Kernel Code References</div>', unsafe_allow_html=True)
    
    # Sort results if requested
    if sort_by_score and isinstance(results, list):
        results_sorted = sorted(results, key=lambda x: float(x.get('score', 0)), reverse=True)
    else:
        results_sorted = results
    
    # Display as expandable cards
    for idx, match in enumerate(results_sorted[:max_results], 1):
        content = match.get('content', '')
        metadata = match.get('metadata', {})
        score = match.get('score', 0)
        
        source_file = metadata.get('source', 'Unknown')
        start_line = metadata['start_line']
        filename = source_file.split('/')[-1] if '/' in source_file else source_file
        
        # Create expandable section with custom styling
        with st.expander(
            f"📌 Reference #{idx}: {filename} (Line: {start_line})" + 
            (f" • Score: {score:.4f}" if show_similarity else ""),
            expanded=(idx <= 2)
        ):
            # Metrics in columns
            metric_cols = st.columns(3)
            
            with metric_cols[0]:
                st.metric("📍 Line", start_line)
            
            with metric_cols[1]:
                st.metric("📊 Score", f"{score:.4f}" if show_similarity else "N/A")
            
            with metric_cols[2]:
                st.metric("📁 File", filename[:15] + "..." if len(filename) > 15 else filename)
            
            
            # File path
            st.markdown("**File Path:**")
            st.code(source_file, language=None)
            
            st.markdown("---")
            
            # Display code snippet
            st.markdown("**Code Snippet:**")
            if content:
                st.code(content, language='c', line_numbers=True)
            else:
                st.warning("No code content available")
            
            st.caption("💡 Hover over the code block to copy")

# Footer
st.markdown("""
    <div class="footer">
        <strong>Linux Kernel Log Analyzer</strong><br>
        Powered by RAG (Retrieval-Augmented Generation) + Qwen AI<br>
        <small>Analyzing kernel issues with AI precision 🚀</small>
    </div>
""", unsafe_allow_html=True)

# Example section in sidebar
with st.sidebar:
    st.divider()
    with st.expander("📚 Example Log"):
        st.code("""[    0.000000] Linux version 5.15.0
[    1.234567] Kernel panic - not syncing
[    1.234568] CPU: 0 PID: 1234
[    1.234569] Call Trace:
[    1.234570]  dump_stack+0x5c
[    1.234571]  panic+0x101
[    1.234572]  do_exit+0x1d4""", language=None)