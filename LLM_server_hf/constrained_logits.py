import torch
import torch.nn.functional as F
from transformers import LogitsProcessor

class ConstrainedLogitBias(LogitsProcessor):
    def __init__(self, target_sequence, negative_infinity=-1e10):
        """
        Initializes with a target sequence and a negative infinity value.
        """
        self.target_sequence = target_sequence
        self.current_step = 0
        self.negative_infinity = negative_infinity  # Default value (may be too low for FP16)
        self.nll = 0  # Accumulated negative log likelihood

    def reset(self, target_sequence):
        """
        Resets the target sequence and counters.
        """
        self.target_sequence = target_sequence
        self.current_step = 0
        self.nll = 0

    def calculate_nll(self, logits, target_token):
        """
        Calculates and accumulates the negative log likelihood for the target token.
        """
        probs = F.softmax(logits, dim=0)
        # Avoid log(0) by adding a small constant.
        log_prob_target = torch.log(probs[target_token] + 1e-10)
        self.nll -= log_prob_target.item()
        return self.nll

    def __call__(self, input_ids, scores):
        """
        Applies a bias to force the model to select the next token in the target sequence.
        Also accumulates the negative log likelihood.
        
        Parameters:
            input_ids: The current sequence of token IDs (not used here).
            scores: Logits tensor of shape (batch_size, vocab_size).
        
        Returns:
            Modified scores with bias applied.
        """
        # Assuming batch_size=1 for simplicity.
        if self.current_step < len(self.target_sequence):
            target_token = self.target_sequence[self.current_step]
            # Calculate NLL using the logits for the first (and only) batch item.
            self.calculate_nll(scores[0], target_token)
            
            # Use a safe negative infinity based on the tensor's dtype.
            if scores.dtype in [torch.float16, torch.bfloat16]:
                safe_negative_infinity = torch.finfo(scores.dtype).min
            else:
                safe_negative_infinity = self.negative_infinity

            # Create a tensor filled with safe negative infinity.
            biased_scores = torch.full_like(scores, safe_negative_infinity)
            # Only allow the target token's original score.
            biased_scores[0, target_token] = scores[0, target_token]
            self.current_step += 1
            return biased_scores
        else:
            return scores

    def apply_bias(self, input_ids, scores):
        """
        Alias to __call__ to support the expected API.
        """
        return self.__call__(input_ids, scores)
