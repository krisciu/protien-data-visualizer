# New file for dependency management with FastAPI

from lang_folder.vectorStore.pineconeClient import PineconeClient
from lang_folder.prompts import few_shot_examples

#TODO: make this a singleton
#TODO: remove uneccessary upserts
def get_pinecone_client():
    pinecone_client = PineconeClient()
    pinecone_client.load_few_shot_examples(few_shot_examples)
    return pinecone_client

def get_embedding_model():
    return OpenAIEmbeddings()