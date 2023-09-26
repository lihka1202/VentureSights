from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os
load_dotenv()
print(os.environ.get("OPENAI_API_KEY"))

input_data = PdfReader("testing_deck.pdf")
rawUnformattedText = ''
for i,page in enumerate(input_data.pages):
    text = page.extract_text()
    if text:
        rawUnformattedText += text

splittingMechanism = CharacterTextSplitter(
    separator="\n",
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len,
)

formattedTexts = splittingMechanism.split_text(rawUnformattedText)
print(len(formattedTexts))

embeddings = OpenAIEmbeddings()

docsearch = FAISS.from_texts(texts=formattedTexts, embedding=embeddings)

chain = load_qa_chain(OpenAI(), chain_type="stuff")

query = "who are the team members of this startup?"
docs = docsearch.similarity_search(query)
output = chain.run(input_documents=docs, question=query)
print(output)
