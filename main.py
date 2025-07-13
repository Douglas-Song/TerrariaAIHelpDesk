from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model = "llama3.2")

#You are an AI agent that answers questions regarding a popular video game - Terraria. 
#Here are some relavant wiki {webpages}
template = """
You are an expert in answering questions aobut a pizza restaurant.

Here are some relavant reviews: {reviews}

Here is the question to answer: {question}
"""

prompt =  ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n------------------------")
    question = input("Ask your question (q to quit)")
    print("\n\n")
    if question == "q":
        break
    
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)