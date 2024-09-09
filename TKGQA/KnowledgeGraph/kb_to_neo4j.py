from py2neo import Graph
from tqdm import tqdm


def kb_to_neo4j(kb):
    if kb == "new-kb-1": kb_path = "../data/new_kb_1.txt"
    elif kb == "new-kb-2": kb_path = "../data/new_kb_2.txt"
    else:
        print("The knowledge graph path is illegal.")
        return

    # Connecting to Neo4j database
    graph = Graph('http://localhost:7474', name=kb, auth=('neo4j', 'wuweixin888'))

    if kb == "new-kb-2":
        # 读取文件并导入数据
        with open(kb_path, "r") as file:
            for line in tqdm(file):
                if line.strip():
                    movie, relation, value = line.strip().split('|')
                    # 创建电影节点
                    graph.run("MERGE (m:Movie {name: $movie})", movie=movie)

                    # 根据关系类型创建相关的节点和关系
                    if relation == 'directed_by':
                        graph.run(f"""
                        MATCH (m:Movie {{name: $movie}})
                        MERGE (p:Director {{name: $value}})
                        MERGE (m)-[:{relation.upper()}]->(p)
                        """, movie=movie, value=value)
                    elif relation == 'written_by':
                            graph.run(f"""
                            MATCH (m:Movie {{name: $movie}})
                            MERGE (p:Writer {{name: $value}})
                            MERGE (m)-[:{relation.upper()}]->(p)
                            """, movie=movie, value=value)
                    elif relation == 'starred_actors':
                        graph.run("""
                        MATCH (m:Movie {name: $movie})
                        MERGE (a:Actor {name: $value})
                        MERGE (m)-[:STARRED_ACTORS]->(a)
                        """, movie=movie, value=value)

    if kb == "new-kb-1":
        # 读取文件并导入数据
        with open(kb_path, "r") as file:
            for line in tqdm(file):
                if line.strip():
                    movie, relation, value = line.strip().split('|')
                    # 创建电影节点
                    graph.run("MERGE (m:Movie {name: $movie})", movie=movie)
                    if relation in ['release_year', 'in_language','has_genre']:
                        graph.run(f"""
                        MATCH (m:Movie {{name: $movie}})
                        SET m.{relation} = $value
                        """, movie=movie, value=value)

    print(f"{kb} has been constructed successfully!")
if __name__ == '__main__':
    kb_to_neo4j("new-kb-1")
    # kb_to_neo4j("new-kb-2")