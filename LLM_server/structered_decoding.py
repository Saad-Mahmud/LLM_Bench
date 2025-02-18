from sd_logistic import logistic_decoding
from sd_belief import belief_decoding
from sd_plain import plain_decoding
from sd_mcq import mcq_decoding
from sd_mcq_logistic import mcq_logit_decoding

def structered_generation(state, query):
    if query.type == "logistic":
        return logistic_decoding(state.model, query.prompt, query.class_names, query.T)
    if query.type == "belief":
        return belief_decoding(state.model, query.prompt, query.class_names, query.n_samples, query.T, query.cot)
    if query.type == "plain":
        return plain_decoding(state.model, query.prompt, query.T)
    if query.type == "mcq":
        return mcq_decoding(state.model, query.prompt, query.class_names, query.T, query.cot, query.cot_tokens, query.eof_tags)
    if query.type == "mcq_logits":
        return mcq_logit_decoding(state.model, state.llama_model, query.prompt, query.class_names, query.T, query.cot, query.cot_tokens, query.eof_tags)
    else:
        print("This feature has not been implimented yet.")