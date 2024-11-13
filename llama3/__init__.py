# from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferMemory
import uuid

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


template = """Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

model = ChatOllama(
    model="llama3.2:1b",
    temperature=0.1,
)

memory = MemorySaver()

# print(chain.invoke({"question": "what is javascript?"}))
# conversation = ConversationChain(
#     llm=model, verbose=True, memory=ConversationBufferMemory()
# )

app = create_react_agent(
    model,
    tools=[],
    checkpointer=memory,
)

thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}


def main():
    while True:
        # print("ask question?")
        question = input()
        input_message = HumanMessage(content=question)
        print(input_message)

        if question == "":
            print("exit!")
            break

        # print(chain.invoke({"question": question}))
        for event in app.stream(
            {"messages": [input_message]}, config, stream_mode="values"
        ):
            event["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()

# https://python.langchain.com/api_reference/ollama/chat_models/langchain_ollama.chat_models.ChatOllama.html
