from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document

docs = [
    Document(page_content="ros2 topic list shows all active topics.",
             metadata={"source": "demo", "version": "humble"}),
    Document(page_content="Use colcon build to build a ROS2 workspace.",
             metadata={"source": "demo", "version": "humble"}),
    Document(page_content="Isaac Sim can publish sensor data to ROS2 via the ROS2 bridge extension.",
             metadata={"source": "demo", "version": "n/a"}),
]

emb = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma.from_documents(docs, emb, persist_directory="../vectordb/hello")

question = "How do I build my workspace?"
hits = db.similarity_search(question, k=2)
context = "\n".join(h.page_content for h in hits)

llm = ChatOllama(model="gemma4:31b", num_ctx=8192)
answer = llm.invoke(f"Answer using only this context:\n{context}\n\nQuestion: {question}")
print(answer.content)
print("Sources:", [h.metadata for h in hits])
