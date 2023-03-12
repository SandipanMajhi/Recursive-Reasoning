from typing import Final

class Generation_Prefix:
    def __init__(self, prompt_version = "promptsv_0_2"):
        self.data_src = {
            "com2sense" : {
                "abductive" : f"{prompt_version}/com2sense/abductive_prefix.txt",
                "belief" : f"{prompt_version}/com2sense/belief_prompt.txt",
                "negation" : f"{prompt_version}/com2sense/negation_prefix.txt",
                "question" : f"{prompt_version}/com2sense/question_prefix.txt",
                "nli" : f"{prompt_version}/com2sense/nli_prefix.txt"
            },
            "csqa2" : {
                "question" : f"{prompt_version}/csqa2/question_prefix.txt",
                "abductive" : f"{prompt_version}/csqa2/abductive_prefix.txt",
                "belief" : f"{prompt_version}/csqa2/belief_prefix.txt",
                "negation" : f"{prompt_version}/csqa2/negation_prefix.txt",
                "nli" : f"{prompt_version}/csqa2/nli_prefix.txt"
            },
            "creak" : {
                "question" : f"{prompt_version}/creak/question_prefix.txt",
                "abductive" : f"{prompt_version}/creak/abductive_prefix.txt",
                "belief" : f"{prompt_version}/creak/belief_prefix.txt",
                "negation" : f"{prompt_version}/creak/negation_prefix.txt",
                "nli" : f"{prompt_version}/csqa2/nli_prefix.txt"
            },
            "strategyqa" : {
                "question" : f"{prompt_version}/strategyqa/question_prefix.txt",
                "abductive" : f"{prompt_version}/strategyqa/abductive_prefix.txt",
                "belief" : f"{prompt_version}/strategyqa/belief_prefix.txt",
                "negation" : f"{prompt_version}/strategyqa/negation_prefix.txt",
                "nli" : f"{prompt_version}/strategyqa/nli_prefix.txt"
            }
        }


    def retrieve_prompt_prefix(self,dataset_name: str) -> dict:
        def open_file(data: dict):
            return {
                key: open_file(value) if isinstance(value, dict) else open(value, "r").read()
                for key, value in data.items()
            }
        filename_dict = self.data_src[dataset_name]
        return open_file(filename_dict)


