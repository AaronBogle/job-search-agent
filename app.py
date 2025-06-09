import streamlit as st
from sentence_transformers import SentenceTransformer, util

# Load free, open-source model
model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("🧠 Job Search Intelligence Agent")

st.markdown("Compare your resume with a job description to see how well they align.")

# Input fields
resume_text = st.text_area("📄 Paste your resume text here", height=300)
job_description = st.text_area("💼 Paste the job description here", height=300)

if st.button("Compare and Score"):
    if resume_text and job_description:
        # Embed and calculate cosine similarity
        resume_embed = model.encode(resume_text, convert_to_tensor=True)
        jd_embed = model.encode(job_description, convert_to_tensor=True)
        score = util.cos_sim(resume_embed, jd_embed).item()

        st.markdown(f"### 🔍 Match Score: `{round(score * 100, 2)}%`")
        if score > 0.7:
            st.success("Great fit! Your resume aligns well with the job.")
        elif score > 0.5:
            st.warning("Moderate fit. Consider tailoring your resume.")
        else:
            st.error("Low match. Tailoring your resume could help.")

    else:
        st.error("Please paste both resume and job description text.")
