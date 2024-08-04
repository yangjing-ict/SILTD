import argparse
import datetime
import os
import numpy as np
import methods.dataset_loader as dataset_loader
from methods.utils import load_base_model, load_base_model_and_tokenizer
from methods.detectgpt import extract_perturbation_features
from methods.metric_based import get_ll, get_rank, get_entropy, get_rank_GLTR, extract_features
import pickle as pkl

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="Essay", help="Essay, Reuters, WP")
    parser.add_argument('--detectLLM', type=str, default="ChatGLM", help="ChatGPT, ChatGLM, Claude, Dolly, StableLM, GPT4All")
    # parser.add_argument('--method', type=str, default="Log-Likelihood")
    parser.add_argument('--base_model_name', type=str, default="gpt2-medium")
    parser.add_argument('--mask_filling_model_name',
                        type=str, default="/data/yangjing/models/base/t5-base")
    # parser.add_argument('--cache_dir', type=str, default="/data/yangjing/models/base")
    parser.add_argument('--base_model_path', type=str, default="/data/yangjing/models/base/gpt2-medium")
    parser.add_argument('--DEVICE', type=str, default="cuda")

    # params for DetectGPT
    parser.add_argument('--pct_words_masked', type=float, default=0.3)
    parser.add_argument('--span_length', type=int, default=2)
    parser.add_argument('--n_perturbation_list', type=str, default="10")
    parser.add_argument('--n_perturbation_rounds', type=int, default=1)
    parser.add_argument('--chunk_size', type=int, default=20)
    parser.add_argument('--n_similarity_samples', type=int, default=20)
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--mask_top_p', type=float, default=1.0)
    parser.add_argument('--random_fills', action='store_true')
    parser.add_argument('--random_fills_tokens', action='store_true')

    args = parser.parse_args()

    DEVICE = args.DEVICE

    START_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
    START_TIME = datetime.datetime.now().strftime('%H-%M-%S-%f')

    print(f'Loading dataset {args.dataset}...')
    data = dataset_loader.load(args.dataset, detectLLM=args.detectLLM)
    labels = np.array(data['test']['label'])

    base_model_name = args.base_model_name.replace('/', '_')
    SAVE_PATH = f"extract_features/features/{args.dataset}"
    label_path = os.path.join(SAVE_PATH, f"labels_{args.detectLLM}.pkl")
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    print(f"Saving results to absolute path: {os.path.abspath(SAVE_PATH)}")
    print(os.path.join(SAVE_PATH, f"{args.detectLLM}_metric_features.pkl"))

    base_model, base_tokenizer = load_base_model_and_tokenizer(args.base_model_path)
    load_base_model(base_model, DEVICE)

    def ll_criterion(text): return get_ll(
        text, base_model, base_tokenizer, DEVICE)

    def rank_criterion(text): return -get_rank(text,
                                               base_model, base_tokenizer, DEVICE, log=False)

    def logrank_criterion(text): return -get_rank(text,
                                                  base_model, base_tokenizer, DEVICE, log=True)

    def entropy_criterion(text): return get_entropy(
        text, base_model, base_tokenizer, DEVICE)

    def GLTR_criterion(text): return get_rank_GLTR(
        text, base_model, base_tokenizer, DEVICE)


    outputs = []
    # method_list = ["Log-Likelihood", "Rank", "Log-Rank", "Entropy", "GLTR", "DetectGPT", "LRR", "NPR"]

    # Log-Likelihood
    output, index1 = extract_features(data, ll_criterion, "likelihood")
    outputs.append(output)
    # Rank
    output, index2 = extract_features(data, rank_criterion, "rank")
    outputs.append(output)
    # Log-Rank
    output, index3 = extract_features(data, logrank_criterion, "log_rank")
    outputs.append(output)
    # Entropy
    output, index4 = extract_features(data, entropy_criterion, "entropy")
    outputs.append(output)
    # GLTR
    output = extract_features(data, GLTR_criterion, "rank_GLTR")
    outputs.append(output)
    # LRR
    output, index5 = extract_perturbation_features(args, data, base_model, base_tokenizer, method="LRR")
    outputs.append(output)
    # DetectGPT
    output, index6 = extract_perturbation_features(args, data, base_model, base_tokenizer, method="DetectGPT")
    outputs.append(output)
    # NPR
    output, index7 = extract_perturbation_features(args, data, base_model, base_tokenizer, method="NPR")
    outputs.append(output)

    index = index1 & index2 & index3 & index4 & index5 & index6 & index7
    labels = labels[index]
    # save results
    with open(os.path.join(SAVE_PATH, f"metric_features_{args.detectLLM}.pkl"), "wb") as f:
        pkl.dump(outputs, f)

    if not os.path.exists(label_path):
        with open(label_path, "wb") as f:
            pkl.dump(labels, f)

    print("Finish")
