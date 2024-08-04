import transformers
import re
import time
from functools import wraps

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds\n\n')
        return result
    return timeit_wrapper


# define regex to match all <extra_id_*> tokens, where * is an integer
pattern = re.compile(r"<extra_id_\d+>")


def cut_length(text, max_length=-1):
    if max_length == -1:
        return text
    else:
        text = text.split()[:max_length]
        text = " ".join(text)
        return text


def load_base_model_and_tokenizer(model_path):

    print(f'Loading BASE model {model_path}...')
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path)
    base_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path)
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

    return base_model, base_tokenizer


def load_base_model(base_model, DEVICE):
    print('MOVING BASE MODEL TO GPU...', end='', flush=True)
    start = time.time()

    base_model.to(DEVICE)
    print(f'DONE ({time.time() - start:.2f}s)')
