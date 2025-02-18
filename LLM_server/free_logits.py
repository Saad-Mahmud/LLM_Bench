import numpy as np
import torch  # Assuming torch is used for calculating log probabilities

class LogitBias:
    def __init__(self, target_sequence, negative_infinity=-1e6):
        """
        Initializes the LogitBias class.

        :param target_sequence: List of token IDs to force the model to generate in sequence.
        :param negative_infinity: A large negative value to force all logits except the target to be ignored.
        """
        self.target_sequence = target_sequence
        self.current_step = 0
        self.negative_infinity = negative_infinity  # Near negative infinity value
        self.nll = 0  # Accumulated negative log likelihood of the sequence
    
    def reset(self, target_sequence):
        self.target_sequence = target_sequence
        self.current_step = 0
        self.nll = 0

    def calculate_nll(self, logits, target_token):
        """
        Calculates the negative log likelihood for the current target token.

        :param logits: The original logits from the model.
        :param target_token: The current target token to calculate NLL for.
        :return: NLL for the current token.
        """
        # Convert logits to probabilities using softmax
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=0)
        # Calculate log probability of the target token
        log_prob_target = torch.log(probs[target_token] + 1e-6)  # Add epsilon to avoid log(0)
        # Update the cumulative NLL
        self.nll -= log_prob_target.item()  # Subtract to accumulate NLL
        return self.nll

    def apply_bias(self, logits):
        """
        Applies a bias to force the model to select the next token in the target sequence.
        Also calculates the NLL of the current target token.

        :param logits: The original logits from the model.
        :return: Adjusted logits with extreme bias.
        """
        # Check if we are within the target sequence range
        if self.current_step < len(self.target_sequence):
            target_token = self.target_sequence[self.current_step]
            
            # Calculate the NLL for the current target token
            self.calculate_nll(logits, target_token)
            
            # Apply extreme bias to enforce target token selection
            biased_logits = np.full_like(logits, self.negative_infinity)
            biased_logits[target_token] = logits[target_token]  # Keep the target logit as it is

            # Move to the next step
            self.current_step += 1
            return biased_logits
        else:
            return logits  # Return logits as-is if beyond the target sequence length
