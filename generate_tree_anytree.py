import itertools
import torch
import math
import dataclasses
from dataclasses import dataclass
from typing import List, Union, Tuple

from prompt import Promptconfig
from generator import Gen_Model
from treelib import Tree
from anytree import Node, RenderTree
from anytree.exporter import DotExporter


@dataclass
class Generation_Config:
    max_new_tokens: int = 64
    do_sample : bool = True
    temperature: float = 1.0
    top_p: float = 1.0
    return_dict_in_generate : bool = True
    output_scores : bool = True
    num_return_sequences: int = 3


class GenerationWrapper:
    def __init__(self, incontext_prefix, belief_prefix,
                 negation_prefix, question_prefix, nli_prefix , depth = 2):
        
        self.prompt_configs = Promptconfig(incontext_prefix, belief_prefix, negation_prefix,
                                            question_prefix, nli_prefix)

        self.abductive_config = Generation_Config()
        self.abductive_config2 = Generation_Config(
            temperature=0.5,
            top_p=1,
            num_return_sequences=1
        )
        self.negation_config = Generation_Config(
            temperature=0.001,
            top_p=1,
            num_return_sequences=3
        )
        self.belief_config = Generation_Config(
            temperature=0.001,
            top_p=1,
            num_return_sequences=1,
        )
        self.question_config = Generation_Config(
            temperature=0.001,
            top_p=1,
            num_return_sequences=1
        )
        self.max_depth = depth
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gen_model = Gen_Model("google/flan-t5-xl", device=self.device)

    def create_graph(self, Q, Q_tilde = None):
        # G = Tree()
        if Q_tilde is None:
            Q, Q_tilde = self.prompt_Q_tilde(Q)
        G_blf, G_int = self.prompt_belief(Q, Q_tilde)
        # G.create_node(Q, "Q", data={
        #     "E": Q,
        #     "E_tilde": Q_tilde,
        #     "blf": G_blf,
        #     "int": G_int
        # })
        G = Node(
            name = f"{Q}",
            E = f"{Q}",
            E_tilde = f"{Q_tilde}",
            blf = G_blf,
            integrity = G_int
        )
        prev_level = [G]
        for depth in range(1, self.max_depth + 1):
            generation_config = self.abductive_config if depth == 1 else self.abductive_config2
            parents_to_generate_from = list(filter(lambda node: not node.data["int"], G.leaves()))
            for parent_node in parents_to_generate_from:
                new_E_T_list = self.prompt_E_T(parent_node.data["E"], dataclasses.replace(generation_config))
                new_E_T_tilde_list = [self.prompt_tilde(E_T) for E_T in new_E_T_list]

                for idx, (E_T, E_T_tilde) in enumerate(zip(new_E_T_list, new_E_T_tilde_list)):
                    node_identifier = f"{parent_node.identifier}T{idx}"
                    E_blf, E_int = self.prompt_belief(E_T, E_T_tilde)
                    # G.create_node(E_T, node_identifier, parent=parent_node.identifier, data={
                    #     "E": E_T,
                    #     "E_tilde": E_T_tilde,
                    #     "blf": E_blf,
                    #     "int": E_int,
                    # })
                    E_T_node = Node(
                        name = f"{E_T}",
                        parent = prev_node,
                        E = f"{E_T}",
                        blf = E_blf,
                        integrity = E_int
                    )

                new_E_F_list = self.prompt_E_F(parent_node.data["E"], dataclasses.replace(generation_config))
                new_E_F_tilde_list = [self.prompt_tilde(E_F) for E_F in new_E_F_list]
                for idx, (E_F, E_F_tilde) in enumerate(zip(new_E_F_list, new_E_F_tilde_list)):
                    node_identifier = f"{parent_node.identifier}F{idx}"
                    E_blf, E_int = self.prompt_belief(E_F, E_F_tilde)
                    # G.create_node(E_F, node_identifier, parent=parent_node.identifier, data={
                    #     "E": E_F,
                    #     "E_tilde": E_F_tilde,
                    #     "blf": E_blf,
                    #     "int": E_int,
                    # })
                    E_F_node = Node(
                        name = f"{E_F}",
                        parent = prev_node,
                        E = f"{E_F}",
                        blf = E_blf,
                        integrity = E_int
                    )

        integral_leaf_nodes = [node.identifier for node in G.leaves() if node.data["int"]]
        paths_to_integral_leaves = [path for path in G.paths_to_leaves() if path[-1] in integral_leaf_nodes]
        nodes_not_to_remove = set(itertools.chain.from_iterable(paths_to_integral_leaves))

        nodes_before_removal = list(G.nodes.keys())
        for node in nodes_before_removal:
            if node in G and node not in nodes_not_to_remove:
                G.remove_node(node)

        return G

    def prompt_E_T(self, Q: str, generation_config):
        prompt_str = self.prompt_configs.create_E_T_prompt(Q)
        num_Es_to_generate = generation_config.num_return_sequences
        E_T_list = []

        while len(E_T_list) < num_Es_to_generate:
            generation_config.num_return_sequences = num_Es_to_generate - len(E_T_list)
            response_inputs = self.gen_model.tokenizer(prompt_str, return_tensors = "pt").to(self.device)
            response = self.gen_model.model.generate(**response_inputs, **self.abductive_config.__dict__)
            response = self.gen_model.tokenizer.batch_decode(response.sequences, skip_special_tokens = True)
            E_T_list.extend(GenerationWrapper.filter_generated_explanations(response))

        return E_T_list[:num_Es_to_generate]

    def prompt_E_F(self, Q, generation_config):
        prompt_str = self.prompt_configs.create_E_F_prompt(Q)
        num_Es_to_generate = generation_config.num_return_sequences
        E_F_list = []

        while len(E_F_list) < num_Es_to_generate:
            generation_config.num_return_sequences = num_Es_to_generate - len(E_F_list)
            response_inputs = self.gen_model.tokenizer(prompt_str, return_tensors = "pt").to(self.device)
            response = self.gen_model.model.generate(**response_inputs, **self.abductive_config.__dict__)
            response = self.gen_model.tokenizer.batch_decode(response.sequences, skip_special_tokens = True)
            E_F_list.extend(GenerationWrapper.filter_generated_explanations(response))

        return E_F_list[:num_Es_to_generate]

    def prompt_tilde(self, E):
        generation_config = self.negation_config
        prompt_str = self.prompt_configs.create_negation_prompt(E)
        num_Es_to_generate = self.negation_config.num_return_sequences
        E_list = []
        while len(E_list) < num_Es_to_generate:
            generation_config.num_return_sequences = num_Es_to_generate - len(E_list)
            response_inputs = self.gen_model.tokenizer(prompt_str, return_tensors = "pt").to(self.device)
            response = self.gen_model.model.generate(**response_inputs, **self.negation_config.__dict__)
            response = self.gen_model.tokenizer.batch_decode(response.sequences, skip_special_tokens = True)
            E_list.extend(GenerationWrapper.filter_generated_explanations(response))
        return E_list[0]

    def prompt_belief(self, Q, Q_tilde):
        Q_E_Q_prob = self.prompt_true_given_Q(Q)
        Q_tilde_E_Q_tilde_prob = self.prompt_true_given_Q(Q_tilde)
        probs = (Q_E_Q_prob, Q_tilde_E_Q_tilde_prob)

        if None not in probs:
            integrity = GenerationWrapper.logical_integrity(Q_E_Q_prob, Q_tilde_E_Q_tilde_prob)
        else:
            integrity = False

        return probs, integrity

    def prompt_true_given_Q(self, Q):
        prompt_str = self.prompt_configs.create_belief_prompt(Q)
        response_inputs = self.gen_model.tokenizer(prompt_str, return_tensors = "pt").to(self.device)
        response = self.gen_model.model.generate(**response_inputs, **self.belief_config.__dict__)
        true_given_Q = self.retrieve_true_prob(response)
        return true_given_Q

    def retrieve_true_prob(self, response):
        generated_text = self.gen_model.tokenizer.batch_decode(response.sequences, skip_special_tokens = True)[0]
        if ". Therefore, the statement is" in generated_text:
            token_index_list = self.gen_model.tokenizer.batch_decode(self.gen_model.tokenizer(generated_text).input_ids)
            true_index = self.gen_model.tokenizer("true", return_tensors = "pt").input_ids.to(self.device)
            false_index = self.gen_model.tokenizer("false", return_tensors = "pt").input_ids.to(self.device)
            # print(f"Generated Text = {generated_text}")
            # print(f"True index = {true_index[0]}")
            # print(f"False index = {false_index[0]}")
            true_or_false_index = len(token_index_list) - 3
            true_score = response.scores[true_or_false_index][0][true_index[0][0].item()]
            false_score = response.scores[true_or_false_index][0][false_index[0][0].item()]
            # print(f"True logprob = {true_logprob}")
            # print(f"False logprob = {false_logprob}")
            swts = torch.nn.functional.softmax(torch.FloatTensor([false_score, true_score]))
            # print(f"Swts = {swts}")
            if not math.isinf(true_score):
                return swts[1]
        return None

    def prompt_Q_tilde(self, question, refine = True):
        if refine and self.prompt_configs.question_prefix is not None:
            prompt_str = self.prompt_configs.create_Q_prompt(question)
            response_inputs = self.gen_model.tokenizer(prompt_str, return_tensors = "pt").to(self.device)
            response = self.gen_model.model.generate(**response_inputs, **self.question_config.__dict__)
            response = self.gen_model.tokenizer.batch_decode(response.sequences, skip_special_tokens = True)
            refined_Q = GenerationWrapper.filter_generated_question(response)[0]
        else:
            refined_Q = question

        prompt_str = self.prompt_configs.create_Q_tilde_prompt(refined_Q)
        response_inputs = self.gen_model.tokenizer(prompt_str, return_tensors = "pt").to(self.device)
        response = self.gen_model.model.generate(**response_inputs, **self.question_config.__dict__)
        response = self.gen_model.tokenizer.batch_decode(response.sequences, skip_special_tokens = True)
        Q_tilde = GenerationWrapper.filter_generated_question(response)[0]

        return refined_Q, Q_tilde

    @staticmethod
    def filter_generated_explanations(explanations: List):
        filtered_explanations = [explanation.strip() for explanation in explanations]
        filtered_explanations = list(filter(lambda exp: len(exp) > 0 and exp.endswith("."), filtered_explanations))
        filtered_explanations = [explanation[0].upper() + explanation[1:] for explanation in filtered_explanations]
        if len(filtered_explanations) == 0:
            filtered_explanations.append(explanations[0].strip())
        filtered_explanations = list(dict.fromkeys(filtered_explanations))

        return filtered_explanations

    @staticmethod
    def filter_generated_question(questions):
        filtered_questions = [question.strip() for question in questions]
        return filtered_questions

    @staticmethod
    def logical_integrity(prob1, prob2):
        return abs(prob1 - prob2) > 0.45