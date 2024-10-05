import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import re

# Load environment variables
load_dotenv()

# Configure the API key for generative AI
genai.configure(api_key=("AIzaSyBTzRq72XY8hR8l3QWXGNxwYA1tAAxj3OI"))

# Initialize FastAPI
app = FastAPI()

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Convert PDF to text
def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in range(len(reader.pages)):
        page = reader.pages[page]
        text += str(page.extract_text())
    return text

# Extract candidate name from resume (if present)
def extract_candidate_name(resume_text):
    name = ""
    # Look for potential name lines in the resume (you can customize this regex pattern)
    lines = resume_text.split('\n')
    for line in lines:
        if re.search(r'Name[: ]', line, re.IGNORECASE):
            name = line.split(":")[-1].strip()  # Extract the text after "Name:"
            break
    return name if name else "Candidate Name Not Found"

# Updated prompt template with matching score calculation logic and table format with full lines
input_prompt = """
### As an advanced Application Tracking System (ATS) with expertise in technology and data science, evaluate the candidate's resume against the provided job description.

### Steps:
1. Parse the job description and create a 'Should Do' list, categorizing required skills into levels: Beginner, Competent, Intermediate, Expert.
2. Parse the candidate's resume and create a 'Can Do' list, categorizing listed skills into the same levels: Beginner, Competent, Intermediate, Expert.
3. Compare the 'Can Do' list with the 'Should Do' list and generate a matching score based on how well the skill levels align using the following rules:
    - Exact match: 100%
    - One level difference: 75%
    - Two level difference: 50%
    - Three level difference: 25%
    - No match or missing skill: 0%
4. Calculate the overall matching score by averaging the scores for all skills.
5. **Format the matching scores in a table as shown below:**

┌──────────────────────────────┬──────────────────┬──────────────────────┬────────────────┐
| Skill                        | 'Can Do' Level   | 'Should Do' Level     | Matching Score |
├──────────────────────────────┼──────────────────┼──────────────────────┼────────────────┤
| HTML                         | Beginner         | Beginner              | 100%           |
| Python                       | Competent        | Intermediate           | 75%            |
| Cloud Computing              | Intermediate     | Expert                | 0%             |
└──────────────────────────────┴──────────────────┴──────────────────────┴────────────────┘

6. **Include the total matching score**:
Total matching score is calculated by averaging the individual skill scores. For example: If 3 skills have scores 100%, 75%, and 0%, the total score is (100 + 75 + 0) / 3 = 58.33%.

7. Generate a detailed report that includes the 'Can Do' list, 'Should Do' list, matching score, analysis of strengths and weaknesses, missing skills, and recommendations for improvement.

### Format the output exactly as below:

Candidate Information: 
• Name: {candidate_name}
• Email: 
'Can Do' List (Skills from Resume):
• Beginner:
• Competent:
• Intermediate:
• Expert:
'Should Do' List (Skills from Job Description):
• Beginner:
• Competent:
• Intermediate:
• Expert:
Matching Score Calculation:
┌──────────────────────────────┬──────────────────┬──────────────────────┬────────────────┐
| Skill                        | 'Can Do' Level   | 'Should Do' Level     | Matching Score |
├──────────────────────────────┼──────────────────┼──────────────────────┼────────────────┤
Overall Matching Score: 
Analysis:
• Strengths: 
• Weaknesses: 
• Recommendations for Improvement: 
Conclusion:
"""

# Function to get the response from the Gemini AI model using GenerationConfig
def get_gemini_response(input):
    model = genai.GenerativeModel('gemini-pro')
    generation_config = genai.GenerationConfig(
        temperature=0.6,  # Set temperature to 0.6 as per requirement
    )
    
    response = model.generate_content(
        contents=input,
        generation_config=generation_config
    )
    
    return response

# API endpoint to generate the report
@app.post("/generate-report/")
async def generate_report(
    resume: UploadFile = File(...),
    jd: UploadFile = File(...)
):
    # Convert both the resume and job description PDFs to text
    resume_text = input_pdf_text(resume.file)
    jd_text = input_pdf_text(jd.file)

    # Extract candidate name from resume
    candidate_name = extract_candidate_name(resume_text)

    # Inject the appropriate prompts into the template
    response = get_gemini_response(
        input_prompt.format(
            text=resume_text,
            jd=jd_text,
            candidate_name=candidate_name
        )
    )

    # Debugging: Print the full LLM response for transparency
    print("LLM Response:", response)

    # Extract the raw content from the LLM response
    candidate_content = response.candidates[0].content.parts[0].text

    # Return the full LLM response, including the matching score, report, and all details
    return {
        "report": candidate_content
    }
