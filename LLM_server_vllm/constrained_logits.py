import torch
import torch.nn.functional as F

class ConstrainedLogitBias:
    def __init__(self, target_sequence, negative_infinity=-1e10):
        self.target_sequence = target_sequence
        self.current_step = 0
        self.negative_infinity = negative_infinity  # Default value (may be too low for FP16)
        self.nll = 0  # Accumulated negative log likelihood

    def reset(self, target_sequence):
        self.target_sequence = target_sequence
        self.current_step = 0
        self.nll = 0

    def calculate_nll(self, logits, target_token):
        # Assumes logits is already a torch tensor.
        probs = F.softmax(logits, dim=0)
        # Avoid log(0) by adding a small constant.
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
            
            # Check if logits are in a lower-precision format (e.g., FP16 or BF16).
            if logits.dtype in [torch.float16, torch.bfloat16]:
                safe_negative_infinity = torch.finfo(logits.dtype).min
            else:
                safe_negative_infinity = self.negative_infinity

            # Create a copy of logits biased against all tokens except the target.
            biased_logits = torch.full_like(logits, safe_negative_infinity)
            biased_logits[target_token] = logits[target_token]
            self.current_step += 1
            return biased_logits
        else:
            return logits