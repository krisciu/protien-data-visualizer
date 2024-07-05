import os
from uuid import uuid4
from .vectorStoreClient import VectorStoreClient
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec, PodSpec  
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import time
import logging
from langchain.vectorstores import VectorStore

DEFAULT_NAMESPACE="query-to-sql"
DEFAULT_INDEX_ID="default-index-id"

class PineconeClient(VectorStoreClient, VectorStore):
    def __init__(self, api_key=None, cloud='aws', region='us-east-1', dimension=1536, metric='dotproduct', default_index_id=DEFAULT_INDEX_ID):
        if not api_key:
            api_key = os.environ["PINECONE_API_KEY"]
        
        self.client = Pinecone(api_key=api_key)
        self.spec = ServerlessSpec(cloud=cloud, region=region)
        logging.info("Pinecone client initialized")

        if default_index_id not in self.list_indexes():
            self.create_index(default_index_id, dimension, metric)

    def list_indexes(self):
        try:
            return self.client.list_indexes().names()
        except Exception as e:
            logging.error(f"Error listing indexes: {e}")
            return []

    def delete_index(self, index_id):
        if index_id in self.list_indexes():
            try:
                self.client.delete_index(index_id)
                logging.info(f"Index {index_id} deleted")
            except Exception as e:
                logging.error(f"Error deleting index {index_id}: {e}")

    def create_index(self, index_id, dimension, metric='dotproduct'):
        try:
            self.client.create_index(index_id, dimension=dimension, metric=metric, spec=self.spec)
            self._wait_for_index(index_id)
            logging.info(f"Index {index_id} created with dimension {dimension} and metric {metric}")
        except Exception as e:
            logging.error(f"Error creating index {index_id}: {e}")

    def _wait_for_index(self, index_id):
        try:
            while not self.client.describe_index(index_id).status['ready']:
                time.sleep(1)
        except Exception as e:
            logging.error(f"Error waiting for index {index_id} to be ready: {e}")

    def upsert_data(self, index_id, values, metadata, key=str(uuid4())):
        # TODO: we should only allow string values, as the PineConeVectorStore should handle embedding for us
        try:
            vectorStore = self._get_vector_store_for_index(index_id, OpenAIEmbeddings())
            vectorStore.add_texts(values, metadata)
            logging.info(f"Data upserted to index {index_id}")
        except Exception as e:
            logging.error(f"Error upserting data to index {index_id}: {e}")

    def _get_vector_store_for_index(self, index_id, embedding):
        # TODO: better namespace here?
        return PineconeVectorStore(index_name=index_id, embedding=embedding, namespace="QueryToSQL")

    def similarity_search(self, input="", k=3):
        #TODO: stop hardcoding index
        try:
            vectorStore = self._get_vector_store_for_index(DEFAULT_INDEX_ID, OpenAIEmbeddings())
            results = vectorStore.similarity_search(input, k)
            return results
        except Exception as e:
            logging.error(f"Error performing similarity search on index {DEFAULT_INDEX_ID}: {e}")
            return []
    
    def describe_index(self, index_id):
        try:
            vectorStore = self._get_vector_store_for_index(index_id, OpenAIEmbeddings())
            return vectorStore.describe_index_stats()
        except Exception as e:
            logging.error(f"Error describing index {index_id}: {e}")
            return None

    def add_texts(self, texts):
        # TODO: Implement the method when needed
        raise NotImplementedError("add_texts method is not implemented yet.")

    def from_texts(self, texts):
        # TODO: Implement the method when needed
        raise NotImplementedError("from_texts method is not implemented yet.")

    def load_few_shot_examples(self, examples, index_id=DEFAULT_INDEX_ID):
            try:
                for example in examples:
                    self.upsert_data(index_id, [example["input"]], [{"query": example["query"]}])
                logging.info(f"Few-shot examples loaded into index {index_id}")
            except Exception as e:
                logging.error(f"Error loading few-shot examples: {e}")