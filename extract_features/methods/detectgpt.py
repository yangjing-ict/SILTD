import numpy as np
import transformers
import re
import torch
import random
import time
from tqdm import tqdm
from methods.metric_based import get_lls, get_ll, get_ranks, get_rank

# define regex to match all <extra_id_*> tokens, where * is an integer
pattern = re.compile(r"<extra_id_\d+>")


def load_mask_model(args, mask_model):
    print('MOVING MASK MODEL TO GPU...', end='', flush=True)
    start = time.time()

    # base_model.cpu()
    if not args.random_fills:
        mask_model.to(args.DEVICE)
    print(f'DONE ({time.time() - start:.2f}s)')


def tokenize_and_mask(text, span_length, buffer_size, pct, ceil_pct=False):
    tokens = text.split(' ')

    # Note that you can also comment these line out if you have enough memory
    if len(tokens) > 1024:
        tokens = tokens[:1024]
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1

    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM
    # increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text


def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")])
            for text in texts]


# replace each masked span with a sample from T5 mask_model
def replace_masks(texts, mask_model, mask_tokenizer, mask_top_p, DEVICE):
    n_expected = count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt",
                            padding=True).to(DEVICE)
    outputs = mask_model.generate(
        **tokens,
        max_length=150,
        do_sample=True,
        top_p=mask_top_p,
        num_return_sequences=1,
        eos_token_id=stop_id)
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)


def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills


def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(
            zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


def perturb_texts_(
        args,
        texts,
        mask_model,
        mask_tokenizer,
        base_tokenizer,
        ceil_pct=False):
    span_length = args.span_length
    buffer_size = args.buffer_size
    mask_top_p = args.mask_top_p
    pct = args.pct_words_masked
    DEVICE = args.DEVICE
    if not args.random_fills:
        masked_texts = [tokenize_and_mask(
            x, span_length, buffer_size, pct, ceil_pct) for x in texts]
        raw_fills = replace_masks(
            masked_texts, mask_model, mask_tokenizer, mask_top_p, DEVICE)
        extracted_fills = extract_fills(raw_fills)
        perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

        # Handle the fact that sometimes the model doesn't generate the right
        # number of fills and we have to try again
        attempts = 1
        while '' in perturbed_texts:
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
            masked_texts = [
                tokenize_and_mask(
                    x,
                    span_length,
                    pct,
                    ceil_pct) for idx,
                x in enumerate(texts) if idx in idxs]
            raw_fills = replace_masks(
                masked_texts, mask_model, mask_tokenizer, mask_top_p, DEVICE)
            extracted_fills = extract_fills(raw_fills)
            new_perturbed_texts = apply_extracted_fills(
                masked_texts, extracted_fills)
            for idx, x in zip(idxs, new_perturbed_texts):
                perturbed_texts[idx] = x
            attempts += 1
    else:
        #FIXME
        FILL_DICTIONARY = set()
        for text in texts:
            FILL_DICTIONARY.update(text.split())
        FILL_DICTIONARY = sorted(list(FILL_DICTIONARY))
        #
        if args.random_fills_tokens:
            # tokenize base_tokenizer
            tokens = base_tokenizer(
                texts, return_tensors="pt", padding=True).to(DEVICE)
            valid_tokens = tokens.input_ids != base_tokenizer.pad_token_id
            replace_pct = args.pct_words_masked * \
                (args.span_length / (args.span_length + 2 * args.buffer_size))

            # replace replace_pct of input_ids with random tokens
            random_mask = torch.rand(
                tokens.input_ids.shape, device=DEVICE) < replace_pct
            random_mask &= valid_tokens
            random_tokens = torch.randint(
                0, base_tokenizer.vocab_size, (random_mask.sum(),), device=DEVICE)
            # while any of the random tokens are special tokens, replace them
            # with random non-special tokens
            while any(base_tokenizer.decode(
                    x) in base_tokenizer.all_special_tokens for x in random_tokens):
                random_tokens = torch.randint(
                    0, base_tokenizer.vocab_size, (random_mask.sum(),), device=DEVICE)
            tokens.input_ids[random_mask] = random_tokens
            perturbed_texts = base_tokenizer.batch_decode(
                tokens.input_ids, skip_special_tokens=True)
        else:
            masked_texts = [tokenize_and_mask(
                x, span_length, pct, ceil_pct) for x in texts]
            perturbed_texts = masked_texts
            # replace each <extra_id_*> with args.span_length random words from
            # FILL_DICTIONARY
            for idx, text in enumerate(perturbed_texts):
                filled_text = text
                for fill_idx in range(count_masks([text])[0]):
                    fill = random.sample(FILL_DICTIONARY, span_length)
                    filled_text = filled_text.replace(
                        f"<extra_id_{fill_idx}>", " ".join(fill))
                assert count_masks([filled_text])[
                    0] == 0, "Failed to replace all masks"
                perturbed_texts[idx] = filled_text

    return perturbed_texts


def perturb_texts(
        args,
        texts,
        mask_model,
        mask_tokenizer,
        base_tokenizer,
        ceil_pct=False):

    outputs = []
    for i in tqdm(
            range(
                0,
                len(texts),
                args.chunk_size),
            desc="Applying perturbations"):
        outputs.extend(perturb_texts_(args,
                                      texts[i:i + args.chunk_size],
                                      mask_model,
                                      mask_tokenizer,
                                      base_tokenizer,
                                      ceil_pct=ceil_pct))
    return outputs


def get_perturbation_results(
        args,
        data,
        mask_model,
        mask_tokenizer,
        base_model,
        base_tokenizer,
        span_length=10,
        n_perturbations=1,
        method="DetectGPT"):
    load_mask_model(args, mask_model)

    torch.manual_seed(0)
    np.random.seed(0)

    test_text = data['test']['text']
    test_label = data['test']['label']

    if method in ["DetectGPT", "NPR"]:  # needs perturbation
        p_test_text = perturb_texts(args, [x for x in test_text for _ in range(
            n_perturbations)], mask_model, mask_tokenizer, base_tokenizer, ceil_pct=False)

        for _ in range(args.n_perturbation_rounds - 1):
            try:
                p_test_text = perturb_texts(
                    args, p_test_text, mask_model, mask_tokenizer, base_tokenizer, ceil_pct=False)
            except AssertionError:
                break

        assert len(p_test_text) == len(test_text) * \
            n_perturbations, f"Expected {len(test_text) * n_perturbations} perturbed samples, got {len(p_test_text)}"
        
        test = []
        for idx in range(len(test_text)):
            test.append({
                "text": test_text[idx],
                "label": test_label[idx],
                "perturbed_text": p_test_text[idx * n_perturbations: (idx + 1) * n_perturbations],
            })

    elif method in ["LRR"]:  # for LRR, we don't need perturbation
        test = []
        for idx in range(len(test_text)):
            test.append({
                "text": test_text[idx],
                "label": test_label[idx],
            })

    if method == "DetectGPT":
        for res in tqdm(test, desc="Computing log likelihoods"):
            p_ll = get_lls(res["perturbed_text"], base_model,
                           base_tokenizer, args.DEVICE)
            res["ll"] = get_ll(res["text"], base_model,
                               base_tokenizer, args.DEVICE)
            res["all_perturbed_ll"] = p_ll
            res["perturbed_ll_mean"] = np.mean(p_ll)
            res["perturbed_ll_std"] = np.std(p_ll) if len(p_ll) > 1 else 1

    elif method == "LRR":
        for res in tqdm(test, desc="LRR Test"):
            res["ll"] = get_ll(res["text"], base_model,
                               base_tokenizer, args.DEVICE)
            res["logrank"] = get_rank(res["text"], base_model,
                                      base_tokenizer, args.DEVICE, log=True)
    elif method == "NPR":
        for res in tqdm(test, desc="NPR Test"):
            res["logrank"] = get_rank(res["text"], base_model,
                                      base_tokenizer, args.DEVICE, log=True)
            res["all_perturbed_logrank"] = get_ranks(
                res["perturbed_text"], base_model, base_tokenizer, args.DEVICE, log=True)
            res["perturbed_logrank_mean"] = np.mean(
                res["all_perturbed_logrank"])
    else:
        raise ValueError("%s method has not been implemented.")

    return test


def run_perturbations(
        args,
        results,
        criterion,
        span_length=10,
        n_perturbations=1,
        method="DetectGPT"):

    if method == "DetectGPT":
        test_predictions = []
        for res in results:
            if criterion == 'd':
                test_predictions.append(res['ll'] - res['perturbed_ll_mean'])
            elif criterion == 'z':
                if res['perturbed_ll_std'] == 0:
                    res['perturbed_ll_std'] = 1
                    # print("WARNING: std of perturbed original is 0, setting to 1")
                    # print(
                    #     f"Number of unique perturbed original texts: {len(set(res['perturbed_text']))}")
                    # print(f"Original text: {res['text']}")

                test_predictions.append(
                    (res['ll'] - res['perturbed_ll_mean']) / res['perturbed_ll_std'])

    elif method == "LRR":
        test_predictions = [(res['ll'] / res['logrank'])
                            for res in results]

    elif method == "NPR":
        test_predictions = [
            (res['perturbed_logrank_mean'] /
             res['logrank']) for res in results]

    else:
        raise ValueError("%s method has not been implemented.")

    x_test = np.array(test_predictions)
    y_test = np.array([_['label'] for _ in results])

    # remove nan values
    select_test_index = ~np.isnan(x_test)
    x_test = x_test[select_test_index]
    y_test = y_test[select_test_index]
    print(len(select_test_index))
    # x_test = np.expand_dims(x_test, axis=-1)

    return x_test, select_test_index


def extract_perturbation_features(
        args,
        data,
        base_model,
        base_tokenizer,
        method="DetectGPT"):
    mask_filling_model_name = args.mask_filling_model_name
    # cache_dir = args.cache_dir

    # get mask filling model (for DetectGPT only)
    if args.random_fills:
        FILL_DICTIONARY = set()
        for texts in data['train'].values() + data['test'].values():
            for text in texts:
                FILL_DICTIONARY.update(text.split())
        FILL_DICTIONARY = sorted(list(FILL_DICTIONARY))

    if not args.random_fills:
        int8_kwargs = {}
        half_kwargs = {}
        if args.int8:
            int8_kwargs = dict(load_in_8bit=True,
                               device_map='auto', torch_dtype=torch.bfloat16)
        elif args.half:
            half_kwargs = dict(torch_dtype=torch.bfloat16)
        print(f'Loading mask filling model {mask_filling_model_name}...')
        mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            mask_filling_model_name, **int8_kwargs, **half_kwargs)
        try:
            n_positions = mask_model.config.n_positions
        except AttributeError:
            n_positions = 512
    else:
        n_positions = 512

    mask_tokenizer = transformers.AutoTokenizer.from_pretrained(
        mask_filling_model_name, model_max_length=n_positions)

    # perturbation_mode = 'd'
    perturbation_mode = 'z'
    n_perturbations = 10
    t1 = time.time()

    perturbation_results = get_perturbation_results(
        args,
        data,
        mask_model,
        mask_tokenizer,
        base_model,
        base_tokenizer,
        args.span_length,
        n_perturbations,
        method=method)

    res, index = run_perturbations(
        args,
        perturbation_results,
        perturbation_mode,
        span_length=args.span_length,
        n_perturbations=n_perturbations,
        method=method)
    print("%s took %.4f sec" % (method, time.time() - t1))
    return res, index