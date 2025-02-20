import numpy as np
import torch  # Assuming torch is used for calculating log probabilities

class ConstrainedLogitBias:
    def __init__(self, target_sequence, negative_infinity=-1e10):
        self.target_sequence = target_sequence
        self.current_step = 0
        self.negative_infinity = negative_infinity  # A very low value to force rejection
        self.nll = 0  # Accumulated negative log likelihood

    def reset(self, target_sequence):
        self.target_sequence = target_sequence
        self.current_step = 0
        self.nll = 0

    def calculate_nll(self, logits, target_token):
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=0)
        # Avoid log(0) by adding a tiny constant.
        log_prob_target = torch.log(probs[target_token] + 1e-10)
        self.nll -= log_prob_target.item()
        return self.nll

    def apply_bias(self, input_ids, logits):
        """
        Applies a bias to force the model to select the next token in the target sequence.
        Also accumulates the negative log likelihood.
        """
        if self.current_step < len(self.target_sequence):
            target_token = self.target_sequence[self.current_step]
            self.calculate_nll(logits, target_token)
            # Create a copy of logits biased against all tokens except the target.
            biased_logits = np.full_like(logits, self.negative_infinity)
            biased_logits[target_token] = logits[target_token]
            self.current_step += 1
            return biased_logits
        else:
            return logits
