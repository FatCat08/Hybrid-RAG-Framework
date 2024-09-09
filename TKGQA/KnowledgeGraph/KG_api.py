import os
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from KnowledgeGraph.kb_to_vector import *
from Config.Config import *

class KnowledgeGraph:
    def __init__(self,type,kb):
        if kb == "new_kb_1" : kb = "new-kb-1"
        elif kb == "new_kb_2" : kb = "new-kb-2"

        if type == "neo4j":
            os.environ["NEO4J_URI"] = Config.NEO4J_URI
            os.environ["NEO4J_USERNAME"] = Config.NEO4J_USERNAME
            os.environ["NEO4J_PASSWORD"] = Config.NEO4J_PASSWORD
            os.environ["NEO4J_DATABASE"] = kb
            self.type = type
            self.kg = Neo4jGraph()
            self.kg.refresh_schema()

        elif type == "vector stores":
            vector_store = Chroma(
                embedding_function=OpenAIEmbeddings(),
                persist_directory=f"D:/Desktop/gra_pro/Code/main_project/data/vector_persist/{kb}"
                # Where to save data locally, remove if not neccesary
            )
            self.type = type
            self.kg = vector_store




    def retrieve(self,query):
        if self.type == "neo4j":
            llm = ChatOpenAI(model=Config.Model, temperature=0)
            chain = GraphCypherQAChain.from_llm(graph=self.kg, llm=llm, verbose=True)
            response = chain.invoke(
                {"query": query})
            return response

        elif self.type == "vector stores":
            results = self.kg.similarity_search_with_score(
                query,
                k = 40

            )
            return  [i[0].page_content for i in results[:10]] + [i[0].page_content for i in results[10:] if i[1] < 0.375]





if __name__ == '__main__':
    import os
    os.environ["OPENAI_API_KEY"] = ""
    KG = KnowledgeGraph("neo4j","new_kb_2")
    # ## example01
    # response = KG.retrieve("David Miles")
    # print(response)
    # response = KG.retrieve("The Lonely Villa")

    response = KG.retrieve("What are the movies starred by [Donna Douglas]?")
    # response = KG.retrieve("Frankie and Johnny")

    print(response)