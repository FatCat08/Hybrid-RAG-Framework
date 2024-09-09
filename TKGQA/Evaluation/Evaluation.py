import argparse
import json
from Chain.Chain import Chain
from tqdm import tqdm
import numpy as np

def bad_case_record(history_info,output_file,llm_answer,data):
    with open(f'D:/Desktop/gra_pro/Code/main_project/outputs/{output_file}/bad_case_2.txt', 'a+', encoding='utf-8') as file:
        file.write(f"--------------------------qid_{data['qid']}--------------------------\n")
        file.write("original_data:\n")
        file.write(json.dumps(data, indent=4))
        # 写入History Information
        file.write("\nHistory Information:\n")
        file.write("-" * 20 + "\n")
        for item in history_info:
            file.write(item + "\n")

        # 写入Original Question
        file.write("\nOriginal Question:\n")
        file.write("-" * 20 + "\n")
        file.write(data['question'] + "\n")

        # 写入LLM Answer
        file.write("\nLLM Answer:\n")
        file.write("-" * 20 + "\n")
        file.write(llm_answer + "\n")

        # 写入Standard Answer
        file.write("\nStandard Answer:\n")
        file.write("-" * 20 + "\n")
        file.write(data['answer'] + "\n")


def evaluate(i,dataset,kb_type,table_type):
    tb = table_type
    kb = kb_type
    if table_type == "vector stores" : tb = "vs"
    if kb_type == "vector stores": kb = "vs"
    if table_type == "dataframe" : tb = "df"


    output_file = f"{tb}_{kb}"


    avg_em = []
    avg_precision = []
    avg_recall = []
    avg_f1 = []
    avg_acc = []
    avg_hallucination = []
    id = 0
    for data in tqdm(dataset):
        try:
            id += 1
            chain = Chain(data=data,table_type=table_type,kb_type=kb_type)
            history_info,llm_answer = chain.process()
            standard_answer = data['answer']

            llm_answer_set = set(llm_answer.split('|'))
            standard_answer_set = set(standard_answer.split('|'))

            true_positive = len(llm_answer_set & standard_answer_set)
            false_positive = len(llm_answer_set - standard_answer_set)
            false_negative = len(standard_answer_set - llm_answer_set)

            # Exact Match
            em = 1 if llm_answer_set == standard_answer_set else 0

            # F1 Score
            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Accuracy
            accuracy = 1 if true_positive > 0 else 0

            # Hallucination
            hallucination = len(llm_answer_set - standard_answer_set) / len(llm_answer_set) if len(llm_answer_set) > 0 else 0

            avg_em.append(em)
            avg_f1.append(f1)
            avg_acc.append(accuracy)
            avg_hallucination.append(hallucination)
            if em != 1: bad_case_record(history_info,output_file,llm_answer,data)
            del chain
        except Exception as e:
            print(e)



    em_avg = np.average(avg_em)
    f1_avg = np.average(avg_f1)
    acc_avg = np.average(avg_acc)
    hallucination_avg = np.average(avg_hallucination)

    results = f"The results of the method(Table:'{table_type}',KG:'{kb_type}') are\n  q_nums:{id}\n  em:{em_avg}\n  f1:{f1_avg}\n  acc:{acc_avg}\n  hallucination:{hallucination_avg}\n"


    with open(f"D:/Desktop/gra_pro/Code/main_project/outputs/{output_file}/results_{i}.txt",'a+',encoding='utf-8') as file:
        file.write(results)

    return


if __name__ == '__main__':
    import os
    os.environ["OPENAI_API_KEY"] = ""
    parser = argparse.ArgumentParser()
    parser.add_argument('--kb_type', type=str,default="neo4j")
    parser.add_argument('--table_type', type=str,default="dataframe")
    args = parser.parse_args()
    with open("D:/Desktop/gra_pro/Code/main_project/data/2-hop-new-split/dev_dataset_sample.json","r",encoding="utf-8") as file:
        dataset = json.load(file)
    # for i in range(200,3000,100):
    #     # dataset_part = dataset[i:i+100]
    #     dataset_part = dataset[210:220]
    #     evaluate(3,dataset_part,args.kb_type,args.table_type)
    #     break
    # dataset_part = dataset[0:333]
    # evaluate(1, dataset_part, args.kb_type, args.table_type)
    dataset_part = dataset[1000:1100]
    evaluate(2, dataset_part, args.kb_type, args.table_type)
    # dataset_part = dataset[2000:2334]
    # evaluate(3, dataset_part, args.kb_type, args.table_type)
