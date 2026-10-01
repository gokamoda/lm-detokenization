from .gpt2 import EQGPT2LMHeadModel
from .llama import EQLlamaForCausalLM


def AutoEQCausalLM_from_pretrained(model_name_or_path):
    """
    AutoEQCausalLM is a factory function that returns an appropriate causal language model class based on the model name or path.
    It currently supports Llama and GPT2 models.

    Args:
        model_name_or_path (str): The name or path of the model.

    Returns:
        class: The appropriate causal language model class.
    """
    if "llama" in model_name_or_path.lower() or "llm-jp" in model_name_or_path.lower():
        return EQLlamaForCausalLM.from_pretrained(model_name_or_path)
    elif "gpt2" in model_name_or_path.lower():
        return EQGPT2LMHeadModel.from_pretrained(model_name_or_path)
    else:
        raise ValueError(f"Unsupported model type for {model_name_or_path}")