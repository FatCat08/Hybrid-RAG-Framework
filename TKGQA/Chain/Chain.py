
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import json
from langchain_core.output_parsers import BaseOutputParser,StrOutputParser
from Parser.Parser import *
from Config.Config import *
from Table.Table_api import Table
from KnowledgeGraph.KG_api import KnowledgeGraph


class Chain:
    def __init__(self,data,table_type="vector stores",kb_type="vector stores"):
        self.llm = ChatOpenAI(model=Config.Model)
        self.kb_type = kb_type
        self.table_type = table_type
        self.kb = KnowledgeGraph(self.kb_type,data['KG_data'])
        self.table = Table(self.table_type,data)
        self.data = data

    def __del__(self):
        if self.table_type == "sqlite":
            self.table.engine.dispose()
    def split_into_subquestion(self):
        data = self.data
        with open(Config.Json_file,'r',encoding='utf-8') as file:
            prompt = json.load(file)['split_into_sub_questions']
        if data['KG_data'] == 'new_kb_1' : kb = "The kb you have is  ‘new-kb-1’ and contain the information about the movie's language, genre, and release year."
        elif data['KG_data'] == 'new_kb_2' : kb = "The kb you have is  ‘new-kb-2’ and contain the information about the movie's actors, directors, and writers"
        user_prompt = '''
            original_question: {question}\n
            {kb}\n
            The table you have only includes information about:\n{table_header}
            
            '''
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", prompt), ("user", user_prompt)]
        )
        llm = self.llm
        parser = Split_sub_question_Parser()

        chain = prompt_template | llm | parser
        response = chain.invoke(
            {"question": data['question'], "kb": kb, "table_header": data['table_header']})

        return response

    def generate_sub_question_with_history(self,history_info,sub_question):
        with open(Config.Json_file, 'r', encoding='utf-8') as file:
            prompt = json.load(file)['generate_new_subquestion']
        user_prompt = '''
        original_question:
        {original_question}
        history_info: 
        {history_info}
        sub_question:
        {sub_question}
        '''
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", prompt), ("user", user_prompt)]
        )
        chain = prompt_template | self.llm | StrOutputParser()
        response = chain.invoke(
            {"original_question":self.data['question'],"history_info":history_info,"sub_question":sub_question})
        return response.split('new_question:')[-1].strip()

    def generate_final_answer(self,history_info):
        history_info = "\n".join(history_info)
        with open(Config.Json_file, 'r', encoding='utf-8') as file:
            prompt = json.load(file)['generate_final_answer']
        user_prompt = '''
                        original_question:
                        {original_question}
                        history_info: 
                        {history_info}
                        '''
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", prompt), ("user", user_prompt)]
        )
        chain = prompt_template | self.llm | StrOutputParser()
        final_answer = chain.invoke({"original_question": self.data['question'], "history_info": history_info})
        return  final_answer.split('answer:')[-1].strip()

    def ask_sub_question(self,retrieve_answer,sub_question):
        with open(Config.Json_file,'r',encoding='utf-8') as file:
            prompt = json.load(file)['ask_sub_question']
        user_prompt = '''
        retrieve_answer:
        {retrieve_answer}
        question: 
        {question}
        '''
        prompt_template = ChatPromptTemplate.from_messages([("system", prompt), ("user", user_prompt)])
        chain = prompt_template | self.llm | StrOutputParser()
        sub_answer = chain.invoke({"retrieve_answer": retrieve_answer, "question": sub_question})
        return sub_answer

    def process(self,verbose=0):

        questions = self.split_into_subquestion()
        if verbose == 1:
            print("Original Question:")
            print("-" * 20)
            print(self.data['question'],'\n')

            print("The Splitting sub-questions:")
            print("-" * 20)
            for question in questions:
                print(question)
            print()


        sub_answer = None
        history_info = []
        for id,q in enumerate(questions):
            if id != 0:
                new_sub_question = self.generate_sub_question_with_history("\n".join(history_info),q['sub_question'])
                q['sub_question'] = new_sub_question

            if q['data_source'] == 'Table':
                if self.table_type == 'vector stores' and sub_answer:
                    answer_list = []
                    for item in sub_answer.split("|"):
                        answer_item = self.table.retrieve(item)
                        answer_list.extend(answer_item)
                    retrieve_answer = "\n".join(answer_list)
                else:
                    retrieve_answer = self.table.retrieve(q['sub_question'])


            elif q['data_source'] == 'KG':
                if self.kb_type == 'vector stores' and sub_answer:
                    answer_list = []
                    for item in sub_answer.split("|"):
                        answer_item = self.kb.retrieve(item)
                        answer_list.extend(answer_item)
                    retrieve_answer = "\n".join(answer_list)
                else:
                    retrieve_answer = self.kb.retrieve(q['sub_question'])

            # print(retrieve_answer)
            sub_answer = self.ask_sub_question(retrieve_answer,q['sub_question'])
            history_info.append(f"The No.{id+1} subquestion is `{q['sub_question']}`,and the answer is `{sub_answer}`")

        final_answer = self.generate_final_answer(history_info)

        return history_info,final_answer





if __name__ == '__main__':
    import os
    os.environ["OPENAI_API_KEY"] = ""
    import time
    with open("../data/2-hop-new-split/dev_dataset_sample.json", 'r', encoding="utf-8") as file:
        data_list = json.load(file)


    # data = data_list[1000]
    ## kb_diff_example: 1002
    for data in data_list[2009:2010]:
        # print(data)
        chain = Chain(data,kb_type="neo4j",table_type="dataframe")

        history_info,answer = chain.process(verbose=1)
        del chain
        print("History Information:")
        print("-" * 20)
        for item in history_info:
            print(item)

        print("\nOriginal Question:")
        print("-" * 20)
        print(data['question'])

        print("\nLLM Answer:")
        print("-" * 20)
        print(answer)

        print("\nStandard Answer:")
        print("-" * 20)
        print(data['answer'])









