import os
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
# from speech_input import get_user_query

# import google.generativeai as genai
from google import genai  # Import the NEW library
INDEX_NAME = "rag-index"

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# client = OpenAI(api_key=OPENAI_API_KEY)


def embed_query(query: str):
    return embedding_model.embed_query(query)


def retreive_context(query:str, top_k = 10):
    query_vec = embed_query(query)
    results = index.query(
        vector=query_vec,
        top_k=top_k,
        include_metadata=True,
        # include_values=True,
    )
    
    # chunks retreived 
    # for match in results["matches"]:
    #     print(f"\n{match["metadata"]["text"]} ---->{match["metadata"]["source"]}----> page no.{match["metadata"]["page"]}\n\n")
        
    context = "\n\n".join([m["metadata"]["text"]for m in results['matches']])
    return context


def generate_answer(query: str, context: str):
    prompt = f"""
            You are an intelligent AI assistant and your name is HELPAI.
            You are made by Aman Puskar.

            Use the retrieved context to answer the user query.
            - Do NOT copy sentences exactly.
            - If you dont't understand the question ask for clarification.
            - Rewrite in your own natural words.
            - Combine information if needed.
            - If there are mutiple points, list them in seperate lines.
            - Keep the tone conversational, clear, and human-like.
            - Give answers to all the parts of query the seperatly.
            - If the query is about confidential data or password then say "Cant't help with this request"
            If the answer is not present, say: "Sorry! currently this question is out of scope."

            CONTEXT:
            {context}

            QUESTION:
            {query}
            """
    # llm = pipeline(
    #     "text2text-generation",
    #     model="google/flan-t5-large",
    #     max_new_tokens=600,
    # )
    # response = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[{"role" : "user", "content" : prompt}]
    # )
    
    # return response.choices[0].message["content"]
    # genai.configure(api_key=os.getenv("GEMINI_LLM_KEY"))
    # # model = genai.GenerativeModel(model_name="gemini-1.0-pro")
    # # model = genai.GenerativeModel("gemini-pro")
    # model = genai.GenerativeModel("gemini-1.5-flash")

    client = genai.Client(api_key=os.getenv("GEMINI_LLM_KEY"))

    # response = model.generate_content(prompt)

    response = client.models.generate_content(
       model="gemini-2.5-flash-lite",
        contents=prompt
    )
    answer = response.text
    # return answer[0]["generated_text"].strip()
    return answer

def rag_pipeline(query: str):
    context = retreive_context(query)
    answer = generate_answer(query, context)
    return answer


# if __name__ == "__main__":
#     while True:
        
#         query = input("\nAsk a question: ")
#         # query = get_user_query() 

#         print("\nANSWER:\n", rag_pipeline(query))