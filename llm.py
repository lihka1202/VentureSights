from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import time

class OpenAIAPI:
    def __init__(self, queries) -> None:
        load_dotenv()
        self.queries = queries
    
    def read_data(self) -> str:
        input_data = PdfReader("testing_deck.pdf")
        rawUnformattedText = ''
        for i,page in enumerate(input_data.pages):
            text = page.extract_text()
            if text:
                rawUnformattedText += text
        return rawUnformattedText
    
    def splitIntoChunks(self, rawUnformattedText : str):
        splittingMechanism = CharacterTextSplitter(
            separator="\n",
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len,
        )

        return splittingMechanism.split_text(rawUnformattedText)
    
    def generateResponses(self, formattedTexts):
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(texts=formattedTexts, embedding=embeddings)
        chain = load_qa_chain(OpenAI(), chain_type="stuff")
        for query in self.queries:
            print(f"The query is: {query}")
            docs = docsearch.similarity_search(query)
            print(f"The answer is: {chain.run(input_documents=docs, question=query)}")
            print("\n")
            time.sleep(10)

    def run(self):
        rawData = self.read_data()
        formattedData = self.splitIntoChunks(rawUnformattedText=rawData)
        self.generateResponses(formattedTexts=formattedData)

# query = "who are the team members of this startup?"
# docs = docsearch.similarity_search(query)
# output = chain.run(input_documents=docs, question=query)
# print(output)

if __name__=="__main__":
    queries = [
        "who are the team members of this startup?",
        "What is the Go To Market Strategy",
        "Explain the Value proposition"
    ]
    testerObj = OpenAIAPI(queries=queries)
    testerObj.run()