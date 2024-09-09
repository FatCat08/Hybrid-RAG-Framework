import re
from langchain_core.output_parsers import BaseOutputParser


class Split_sub_question_Parser(BaseOutputParser):
    def parse(self, text: str) -> str:
        # Regular expression patterns
        sub_question_pattern = r'Sub_question_\d+:\s*"([^"]+)"'
        data_source_pattern = r'data_source_\d+:\s*"([^"]+)"'

        # Extract sub_questions and data_sources
        sub_questions = re.findall(sub_question_pattern, text)
        data_sources = re.findall(data_source_pattern, text)

        # Pair sub_questions with their corresponding data_sources
        result = []
        for i in range(len(sub_questions)):
            result.append({
                f'sub_question': sub_questions[i],
                f'data_source': data_sources[i] if i < len(data_sources) else None
            })

        # Print the results
        return result

