# New file for dependency management with FastAPI

from lang_folder.vectorStore.pineconeClient import PineconeClient


def get_pinecone_client():
    return PineconeClient()