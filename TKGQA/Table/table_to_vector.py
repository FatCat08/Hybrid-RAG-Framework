import json

import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

## store every table of each directory to vector database
def table_to_vector(data):
    df = pd.DataFrame(data["table_data"], columns=data["table_header"])

    def generate_sentence(row):
        sentence = ""
        for column in data['table_header']:
            if column == 'movie': continue
            sent_part = f"The {column} of movie {row['movie']} are {row[column]}."
            sentence += sent_part

        document = Document(page_content=sentence, metadata={"source": row['movie']}, )
        return document

    documents = df.apply(generate_sentence, axis=1).to_list()
    vector_store = Chroma.from_documents(
        documents,
        embedding=OpenAIEmbeddings()
    )
    return vector_store
    # print(f"The {data['qid']}_table vector store has successfully created!")




