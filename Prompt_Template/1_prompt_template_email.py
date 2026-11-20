from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.7
)

# Create Prompt Template
email_template = PromptTemplate(
    input_variables=["recipient", "purpose", "tone"],
    template="""
Write a professional email.

Recipient: {recipient}
Purpose: {purpose}
Tone: {tone}

Generate a complete email with:
- Subject
- Greeting
- Body
- Closing
"""
)

# Placeholder values as an object (dictionary)
email_data = {
    "recipient": "Manager",
    "purpose": "Request leave for university examinations from 23 Nov 2026 to 3 Dec 2026",
    "tone": "Professional and concise"
}

# Format prompt
formatted_prompt = email_template.invoke(email_data)

# Invoke model
response = llm.invoke(formatted_prompt)

# Print result
print(response.content)
