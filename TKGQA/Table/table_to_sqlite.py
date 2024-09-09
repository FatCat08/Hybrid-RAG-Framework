import json
import pandas as pd
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from tqdm import tqdm
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
import os

def table_to_sqlite(data):
    if os.path.exists("tmp.db"):
        os.remove("tmp.db")
    engine = create_engine(f"sqlite:///tmp.db")
    df = pd.DataFrame(data["table_data"], columns=data["table_header"])
    df.to_sql(f"table_0", engine, index=False)
    db = SQLDatabase(engine=engine)
    return db,engine

if __name__ == '__main__':
    engine = create_engine(f"sqlite:///tmp.db")
    db = SQLDatabase(engine=engine)

    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = create_sql_query_chain(llm, db)
    response = chain.invoke({"question": "What is the director of Tai-Pan? in table_0"})

    print(db.run(response))
    # # print(table.run("SELECT 'director' FROM table_0 WHERE 'movie' = 'Tai-Pan' LIMIT 5"))
    # print(table.run(response[9:]))