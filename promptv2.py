class Promptconfig:
    def __init__(self, incontext_prompt, belief_prompt, negation_prompt, question_prompt, nli_prompt):
        self.incontext_prefix = incontext_prompt
        self.belief_prefix = belief_prompt
        self.negation_prefix = negation_prompt
        self.question_prefix = question_prompt
        self.nli_prefix = nli_prompt

    
    def create_E_T_prompt(self, sentence):
        generation_prefix = self.incontext_prefix
        question = sentence[:-1] + "?"
        return f"{generation_prefix}" \
                f"Q: {question} This statement is true, because\n" \
                    f"A:"

    def create_E_F_prompt(self, sentence):
        generation_prefix = self.incontext_prefix
        question = sentence[:-1] + "?"
        return f"{generation_prefix}s" \
                f"Q: {question} This statement is false, because\n" \
                    f"A:"
    
    def create_negation_prompt(self, sentence):
        return f"{self.negation_prefix}" \
                    f"Q: {sentence}\n"\
                    f"A:"
    
    def create_new_negation_prompt(self,sentence):
        return f"{self.negation_prefix}" \
                f"Q: {sentence}\n"\
                f"A:"

    def create_belief_prompt(self, sentence):
        belief_prefix = self.belief_prefix
        question = sentence[:-1] + "?"
        return f"{belief_prefix}" \
                f"Q: {question}\n" \
                    f"A:"

    def create_Q_prompt(self, question):
        question = question[:-1] + "."
        return f"{self.question_prefix}" \
                f"Q: {question}\n" \
                    f"A: This statement is true."

    def create_Q_tilde_prompt(self, question):
        question = question[:-1] + "."
        return f"{self.negation_prefix}" \
            f"Q: {question}\n" \
                f"A:"
    
    def create_new_Q_tilde_prompt(self, question):
        question = question[:-1] + "."
        return f"{self.negation_prefix}" \
            f"Q: {question}\n" \
                f"A:"

    def create_nli_prompt(self, premise, hypothesis):
        return f"{self.nli_prefix}" \
            f"Premise: {premise}\n" \
                f"Hypothesis: {hypothesis}\n" \
                    f"Options: entailment, contradiction, neutral.\n" \
                        f"A:"