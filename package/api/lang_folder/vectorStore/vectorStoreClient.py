from uuid import uuid4
from abc import ABC, abstractmethod


class VectorStoreClient(ABC):
    @abstractmethod
    def list_indexes(self):
        pass

    @abstractmethod
    def delete_index(self, index_id):
        pass

    @abstractmethod
    def create_index(self, index_id, dimension, metric='dotproduct'):
        pass

    @abstractmethod
    def describe_index(self, index_id):
        pass

    @abstractmethod
    def _wait_for_index(self, index_id):
        pass

    @abstractmethod
    def upsert_data(self, index_id, values, metadata, key=str(uuid4())):
        pass

    @abstractmethod
    def similarity_search(self, index_id, query, num_results=3):
        pass
    
    @abstractmethod
    def _get_vector_store_for_index(self,index_id, embedding):
        pass
