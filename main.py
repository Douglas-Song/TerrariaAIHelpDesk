from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
from dotenv import load_dotenv

#Load API key from .env file
load_dotenv()

model = ChatOpenAI(model = "gpt-3.5-turbo")

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
    print(result.content)