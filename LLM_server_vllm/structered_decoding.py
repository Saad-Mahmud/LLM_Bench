from sd_mcq_logistic import mcq_logit_decoding

def structered_generation(state, query):
    if query.type == "mcq_logits":
        return mcq_logit_decoding(state.model, state.llama_model, query.prompt, query.class_names, query.T, query.cot, query.cot_tokens, query.eof_tags)
    else:
        print("This feature has not been implimented yet.")