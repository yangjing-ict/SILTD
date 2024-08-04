import random
import tqdm
import pandas as pd

def process_spaces(text):
    return text.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()


def load(name, detectLLM):

    if name in ["Essay", "Reuters", "WP"]:

        f = pd.read_csv(f"datasets/{name}_LLMs.csv")
        a_human = f["human"].tolist()
        a_chat = f[f'{detectLLM}'].fillna("").tolist()

        res = []
        for i in range(len(a_human)):
            if len(a_human[i].split()) > 1 and len(a_chat[i].split()) > 1:
                res.append([a_human[i], a_chat[i]])

        data_new = {
            'train': {
                'text': [],
                'label': [],
            },
            'test': {
                'text': [],
                'label': [],
            }

        }

        index_list = list(range(len(res)))
        random.seed(0)
        random.shuffle(index_list)

        total_num = len(res)
        for i in tqdm.tqdm(range(total_num), desc="parsing data"):
            if i < total_num * 0.8:
                data_partition = 'train'
            else:
                data_partition = 'test'
            data_new[data_partition]['text'].append(
                process_spaces(res[index_list[i]][0]))
            data_new[data_partition]['label'].append(0)
            data_new[data_partition]['text'].append(
                process_spaces(res[index_list[i]][1]))
            data_new[data_partition]['label'].append(1)

        return data_new

    else:
        raise ValueError(f'Unknown dataset {name}')
