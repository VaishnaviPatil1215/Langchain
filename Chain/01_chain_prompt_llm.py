from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{question}")
    ]
)

# |-->pipe operator to combine prompt template, LLM and output parser (avoid usage of invoke everytime)

chain = (
    prompt_template
    | ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0.7
    )
    | StrOutputParser()
)

result = chain.invoke(
    {"question": "What is the capital of France?"}
)

print(result)



