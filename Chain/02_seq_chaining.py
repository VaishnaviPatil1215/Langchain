from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.7
)

# Prompt Template
prompt = ChatPromptTemplate.from_template(
    "Generate {fact_count} interesting facts about cats."
)

# Lambda function to translate output to French
translate_to_french = RunnableLambda(
    lambda text: f"Translate the following text into French:\n\n{text}"
)

# Sequential Chain
chain = RunnableSequence(
    prompt,
    llm,
    translate_to_french,
    llm
)

# Invoke Chain
result = chain.invoke(
    {
        "fact_count": 3
    }
)

print(result.content)