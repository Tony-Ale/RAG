import os
from dotenv import load_dotenv
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_cohere import ChatCohere, CohereEmbeddings
from PyPDF2 import PdfReader
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()
llm = ChatCohere()

def load_doc(doc_path):
    loader_pdf = PyPDFLoader(doc_path)
    pages = loader_pdf.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 10
    )

    splits = text_splitter.split_documents(pages)
    return splits

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 400,
        chunk_overlap = 100
    )
    
    splits = text_splitter.split_text(text)
    return splits

def create_pineconedb(splits, dbindex_name):
    key = os.getenv('PINECONE_API_KEY')
    pc = Pinecone(api_key=key)
    index_name = dbindex_name
    pc.create_index(
        name=index_name,
        dimension=1024, # Replace with your model dimensions
        metric="cosine", # Replace with your model metric
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ) 
    )
    embed_model = CohereEmbeddings(model="embed-english-v3.0")

    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embed_model)

    vectorstore.add_documents(splits)
    print("\nDatabase index created")

def search_with_pinecone(index_name):
    embedding_model = CohereEmbeddings(model="embed-english-v3.0")
    
    vectors = PineconeVectorStore(index_name=index_name, embedding=embedding_model)

    return vectors


def history_retriever_chain(vectordb):

    retriever = vectordb.as_retriever()

    # history aware retrieval
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"), 
        ("user", "Use the conversation above as context")
    ])

    # retrieves based on the history and context
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain

#print(vectordb.as_retriever().get_relevant_documents("summarize the document"))
def conversational_chain(retriever_chain):

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context: \n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"), 
        ("user", "{input}")
    ])


    # creating a document chain allows us to passing a list of documents as context to the LLM
    document_chain = create_stuff_documents_chain(llm, prompt)

    # to place the context automatically we have to use pass in the document_chain to a retrieval_chain
    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

    return retrieval_chain

def main(create_database=False):
    if create_database:
        file_path = input('input file path: ')
        splits = load_doc(file_path.strip())
        dbindex_name = input('\ninput your database index name: ')
        create_pineconedb(splits, dbindex_name.strip())
    else:
        dbindex_name = input('Input your index name: ')
    
    chat_history = []
    vectordb = search_with_pinecone(index_name=dbindex_name)

    history_chain = history_retriever_chain(vectordb)
    
    conversation_chain = conversational_chain(history_chain)
    user_input = input("\nI am a Doc bot, ask me any question!\n*****Type exit to stop chat***** \n\nQuery: ")
    while True:
        response = conversation_chain.invoke({
            "chat_history": chat_history,
            "input": user_input
        })

        print(f"\nAI Response: {response['answer']}")

        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response['answer']))

        user_input = input("\nQuery: ")
        if user_input.lower() == 'exit':
            break

if __name__ == '__main__':
    main(create_database=True)