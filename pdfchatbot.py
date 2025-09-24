import streamlit as st
import os
import PyPDF2
import re
from huggingface_hub import InferenceClient

# 🔐 Set your Hugging Face token directly here
os.environ["HF_TOKEN"] = "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"

# Access token
HF_TOKEN = os.getenv("HF_TOKEN")

# Set up the Inference Client
client = InferenceClient(api_key=HF_TOKEN)

st.title("📄📚 PDF-Based Question Answering with Hugging Face")
st.markdown("""
Upload a PDF and ask a question based on its content using 
[deepset/longformer-base-4096-squad2](https://huggingface.co/deepset/longformer-base-4096-squad2).
""")

# --- Function: Clean PDF Text ---
def clean_pdf_text(text):
    text = re.sub(r'\s+', ' ', text)  # remove excessive whitespace
    return text.strip()

# --- Function: Split into chunks ---
def get_relevant_chunk(context, question, chunk_size=1500):
    chunks = [context[i:i+chunk_size] for i in range(0, len(context), chunk_size)]
    for chunk in chunks:
        if any(word.lower() in chunk.lower() for word in question.split()):
            return chunk
    return chunks[0]  # fallback

# --- Upload PDF ---
uploaded_file = st.file_uploader("📤 Upload a PDF file", type=["pdf"])

pdf_text = ""
if uploaded_file is not None:
    try:
        with st.spinner("Extracting text from PDF..."):
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    pdf_text += extracted
        pdf_text = clean_pdf_text(pdf_text)
        st.success("✅ PDF text extracted successfully!")
        st.text_area("📄 Extracted Context (editable):", value=pdf_text[:3000], height=200, key="context_area")  # show partial
    except Exception as e:
        st.error(f"❌ Failed to read PDF: {str(e)}")
else:
    st.info("Please upload a PDF to continue.")

# --- Input question ---
question = st.text_input("❓ Enter your question:")

# --- Get Answer ---
if st.button("Get Answer"):
    if not pdf_text.strip():
        st.warning("⚠️ Please upload a PDF with readable text.")
    elif not question.strip():
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("Thinking... 🤔"):
            try:
                # Get the most relevant chunk from the context
                relevant_context = get_relevant_chunk(pdf_text, question)

                # Use Longformer model
                answer = client.question_answering(
                    question=question,
                    context=relevant_context,
                    model="deepset/roberta-base-squad2",
                )

                st.success("✅ Answer:")
                st.markdown(f"**{answer}**")

            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
