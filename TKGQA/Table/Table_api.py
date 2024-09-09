from Table.table_to_sqlite import *
from Table.table_to_vector import *
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.tools import PythonAstREPLTool
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import BaseOutputParser
from Config.Config import Config
import re

with open("D:/Desktop/gra_pro/Code/main_project/prompts/prompts.json", 'r', encoding='utf-8') as file:
    prompt = json.load(file)['retrieve table from db']
    prompt_template = ChatPromptTemplate.from_messages(
    [("system", prompt)]
)

class BracketRemovalParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        # 去除方括号 [ 和 ]
        # 正则表达式匹配 [] 中的 SELECT 语句
        select_statements = re.findall(r'\[SELECT.*?\]', text)
        try:
            return select_statements[0][1:-1]
        except:
            return ""

class Table:
    def __init__(self,type,data):
        self.data = data
        self.type = type
        if type == "sqlite":
            self.table_data,self.engine = table_to_sqlite(data)

        elif type == "vector stores":
            vector_store = table_to_vector(data)
            self.table_data = vector_store

        elif type == "dataframe":
            df = pd.DataFrame(data["table_data"], columns=data["table_header"])
            self.table_data = df




    def retrieve(self,query):
        if self.type == "sqlite":
            execute_query = QuerySQLDataBaseTool(db=self.table_data)
            llm = ChatOpenAI(model=Config.Model)
            parser = BracketRemovalParser()
            chain = prompt_template | llm | parser | execute_query
            response = chain.invoke({'table_name':'table_0','columns':self.data['table_header'],'query':query})
            # print(response)
            return response

        elif self.type == "vector stores":
            results = self.table_data.similarity_search_with_score(
                query,
                k = 10
            )
            return [i[0].page_content for i in results[:8]] + [i[0].page_content for i in results[8:] if i[1] < 0.4]

        elif self.type == "dataframe":
            tool = PythonAstREPLTool(locals={"df": self.table_data})
            llm = ChatOpenAI(model=Config.Model)
            llm_with_tools = llm.bind_tools([tool], tool_choice=tool.name)
            parser = JsonOutputKeyToolsParser(key_name=tool.name, first_tool_only=True)
            system = f"""You have access to a pandas dataframe `df`. \
                        Here is the output of `df.head().to_markdown()`:
                        {self.table_data.head().to_markdown()}
                        Given a user question, write the Python code to answer it. \
                        Return ONLY the valid Python code and nothing else. \
                        Don't assume you have access to any libraries other than built-in Python ones and pandas."""

            prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])
            chain = prompt | llm_with_tools | parser | tool
            response = chain.invoke({"question": query})
            return response

if __name__ == '__main__':
    import os
    os.environ["OPENAI_API_KEY"] = ""
    with open("D:/Desktop/gra_pro/Code/main_project/data/2-hop-new-split/dev_dataset_sample.json","r",encoding="utf-8") as file:
        dataset = json.load(file)

    data = dataset[0]
    Table_data = Table("sqlite",data)
    info = Table_data.retrieve("Who is the director of [Off the Black]?")

    print(info)
    # print(Table_data.table_data.run(''))
    # print(chain.get_prompts()[0].pretty_print())