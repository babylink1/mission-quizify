import os
from langchain_google_vertexai import VertexAIEmbeddings
from google.cloud import aiplatform


class EmbeddingClient:
    """
    Task: Initialize the EmbeddingClient class to connect to Google Cloud's VertexAI for text embeddings.

    The EmbeddingClient class should be capable of initializing an embedding client with specific configurations
    for model name, project, and location. Your task is to implement the __init__ method based on the provided
    parameters. This setup will allow the class to utilize Google Cloud's VertexAIEmbeddings for processing text queries.

    Steps:
    1. Implement the __init__ method to accept 'model_name', 'project', and 'location' parameters.
       These parameters are crucial for setting up the connection to the VertexAIEmbeddings service.

    2. Within the __init__ method, initialize the 'self.client' attribute as an instance of VertexAIEmbeddings
       using the provided parameters. This attribute will be used to embed queries.

    Parameters:
    - model_name: A string representing the name of the model to use for embeddings.
    - project: The Google Cloud project ID where the embedding model is hosted.
    - location: The location of the Google Cloud project, such as 'us-central1'.

    Instructions:
    - Carefully initialize the 'self.client' with VertexAIEmbeddings in the __init__ method using the parameters.
    - Pay attention to how each parameter is used to configure the embedding client.

    Note: The 'embed_query' method has been provided for you. Focus on correctly initializing the class.
    """

    def __init__(self, model_name, project, location):
        # 设置环境变量指向服务账户密钥文件路径
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/a1/Documents/Project/radicalAI/mission-quizify/application_key.json"

        # 初始化 VertexAIEmbeddings 客户端
        self.client = VertexAIEmbeddings(
            model=model_name,
            project=project,
            location=location
        )

    def embed_query(self, query):
        vectors = self.client.embed_query(query)
        return vectors

    def embed_documents(self, documents):
        try:
            return self.client.embed_documents(documents)
        except AttributeError:
            print("Method embed_documents not defined for the client.")
            return None


if __name__ == "__main__":
    model_name = "textembedding-gecko@003"
    project = "projectjune11"
    location = "us-central1"

    embedding_client = EmbeddingClient(model_name, project, location)
    vectors = embedding_client.embed_query("Hello World!")
    if vectors:
        print(vectors)
        print("Successfully used the embedding client!")
