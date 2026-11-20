from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(
    # model="gemini-3.8-flash",
    model="gemini-3.5-flash",
    temperature=0.7
)

chat_history = []

system_message =SystemMessage(content="You are a helpful assistant.")
chat_history.append(system_message)

#chat loop
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    human_message = HumanMessage(content=user_input)
    chat_history.append(human_message)

    result = llm.invoke(chat_history)
    print("Assistant:", result.text)

    assistant_message = SystemMessage(content=result.text)
    chat_history.append(assistant_message)
