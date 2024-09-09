import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import chromadb


def kb_to_vector(kb):
    if kb == "new-kb-1": kb_path = "../data/new_kb_1.txt"
    elif kb == "new-kb-2": kb_path = "../data/new_kb_2.txt"
    else:
        print("The knowledge graph path is illegal.")
        return

    movies, relations, tails = [], [], []
    with open(kb_path, 'r') as file:
        for line in file:
            if line.strip():
                movie, relation, tail = line.strip().split('|')
                movies.append(movie)
                relations.append(relation)
                tails.append(tail)
    kb_df = pd.DataFrame({'movie': movies, 'relation': relations, 'tail': tails})

    documents = []
    if kb == "new-kb-2":
        # 只包含actor,writers,directors数据
        filtered_df_1 = kb_df[kb_df['relation'].isin(['starred_actors', 'written_by', 'directed_by'])]
        reshaped_df_1 = filtered_df_1.pivot_table(index='movie', columns='relation', values='tail',
                                                  aggfunc=lambda x: "|".join(x)).reset_index()
        # 填充缺失值
        reshaped_df_1.fillna('', inplace=True)
        table_data_1 = pd.DataFrame(reshaped_df_1.values, columns=['movie', 'director', 'actor', 'writer'])

        def generate_sentence(row):
            directors = row['director'] if row['director'] != '' else 'N/A'
            actors = row['actor'] if row['actor'] != '' else 'N/A'
            writers = row['writer'] if row['writer'] != '' else 'N/A'

            sentence = f"The directors of movie {row['movie']} are {directors}, the actors of movie {row['movie']} are {actors if actors else 'N/A'}, and the writers of movie {row['movie']} are {writers if writers else 'N/A'}."
            document = Document(page_content=sentence,metadata={"source": row['movie']},)
            return document

        documents = table_data_1.apply(generate_sentence, axis=1).to_list()
    elif kb == "new-kb-1":
        filtered_df_2 = kb_df[kb_df['relation'].isin(['release_year', 'has_genre', 'in_language'])]
        reshaped_df_2 = filtered_df_2.pivot_table(index='movie', columns='relation', values='tail',
                                                  aggfunc=lambda x: "|".join(x)).reset_index()
        reshaped_df_2.fillna('', inplace=True)
        table_data_2 = pd.DataFrame(reshaped_df_2.values, columns=['movie', 'has_genre', 'in_language', 'release_year'])
        table_data_2.head()

        def generate_sentence(row):
            genre = row['has_genre'] if row['has_genre'] != '' else 'N/A'
            language = row['in_language'] if row['in_language'] != '' else 'N/A'
            release_year = row['release_year'] if row['release_year'] != '' else 'N/A'

            sentence = f"The movie '{row['movie']}' is in the genre {genre}, is in the language {language}, and was released in the year {release_year}."
            document = Document(page_content=sentence, metadata={"source": row['movie']})
            return document

        # Applying the function to each row
        documents = table_data_2.apply(generate_sentence, axis=1).tolist()

    vectorstore = Chroma.from_documents(
        documents,
        embedding=OpenAIEmbeddings(),
        persist_directory=f"../data/vector_persist/{kb}"
    )
    print(f"The {kb} vector store has successfully created!")
    return vectorstore




if __name__ == '__main__':
    # kb_to_vector("new-kb-1")
    # kb_to_vector("new-kb-2")
    vector_store = Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory=f"../data/vector_persist/new-kb-1"  # Where to save data locally, remove if not neccesary
    )

    results = vector_store.similarity_search(
        "What is the language of The Big Green",
        k=2,
    )
    print(results)