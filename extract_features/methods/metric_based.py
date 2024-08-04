import numpy as np
import torch
import torch.nn.functional as F
from methods.utils import timeit
from tqdm import tqdm

def get_ll(text, base_model, base_tokenizer, DEVICE):
    with torch.no_grad():
        tokenized = base_tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt").to(DEVICE)
        labels = tokenized.input_ids
        return -base_model(**tokenized, labels=labels).loss.item()
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L1317


def get_lls(texts, base_model, base_tokenizer, DEVICE):
    return [get_ll(_, base_model, base_tokenizer, DEVICE) for _ in texts]


# get the average rank of each observed token sorted by model likelihood
def get_rank(text, base_model, base_tokenizer, DEVICE, log=False):
    with torch.no_grad():
        tokenized = base_tokenizer(
            text,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(DEVICE)
        logits = base_model(**tokenized).logits[:, :-1]
        labels = tokenized.input_ids[:, 1:]

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True)
                   == labels.unsqueeze(-1)).nonzero()

        assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:, -1], matches[:, -2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (timesteps == torch.arange(len(timesteps)).to(
            timesteps.device)).all(), "Expected one match per timestep"

        ranks = ranks.float() + 1  # convert to 1-indexed rank
        if log:
            ranks = torch.log(ranks)

        return ranks.float().mean().item()


def get_ranks(texts, base_model, base_tokenizer, DEVICE, log=False):
    return [get_rank(_, base_model, base_tokenizer, DEVICE, log)
            for _ in texts]


def get_rank_GLTR(text, base_model, base_tokenizer, DEVICE, log=False):
    with torch.no_grad():
        tokenized = base_tokenizer(
            text,
            truncation=True,
            max_length=1024,
            return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:, :-1]
        labels = tokenized.input_ids[:, 1:]

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True)
                   == labels.unsqueeze(-1)).nonzero()

        assert matches.shape[
            1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:, -1], matches[:, -2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (timesteps == torch.arange(len(timesteps)).to(
            timesteps.device)).all(), "Expected one match per timestep"
        ranks = ranks.float()
        res = np.array([0.0, 0.0, 0.0, 0.0])
        for i in range(len(ranks)):
            if ranks[i] < 10:
                res[0] += 1
            elif ranks[i] < 100:
                res[1] += 1
            elif ranks[i] < 1000:
                res[2] += 1
            else:
                res[3] += 1
        if res.sum() > 0:
            res = res / res.sum()

        return res


# get average entropy of each token in the text
def get_entropy(text, base_model, base_tokenizer, DEVICE):
    with torch.no_grad():
        tokenized = base_tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:, :-1]
        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        return -neg_entropy.sum(-1).mean().item()


@timeit
def extract_features(data, criterion_fn, name):
    torch.manual_seed(0)
    np.random.seed(0)

    test_text = data['test']['text']
    # test_label = data['test']['label']
    test_criterion = [criterion_fn(test_text[idx]) for idx in tqdm(range(len(test_text)))]
    x_test = np.array(test_criterion)

    # y_test = np.array(test_label)

    # remove nan values
    if name == 'rank_GLTR':
        return x_test
    
    select_test_index = ~np.isnan(x_test)
    x_test = x_test[select_test_index]
    # y_test = y_test[select_test_index]
    # x_test = np.expand_dims(x_test, axis=-1)
    print(len(select_test_index))

    return x_test, select_test_index