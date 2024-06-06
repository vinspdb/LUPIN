import sys
import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader
from neural_network.HistoryDataset import CustomDataset
from neural_network.llamp_multiout_wrapper import BertMultiOutputClassificationHeads
from neural_network.llamp_multiout import BertMultiOutputClassificationHeads as original
from bs4 import BeautifulSoup
from transformers import AutoModel, AutoTokenizer
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization


def reconstruct_prefix(l,c):
    somma = []
    stringa = ''
    new_list_word = []
    new_list_score = []
    check_init = False
    for i, j in zip(l, c):
        if '#' in i:
            if check_init == False:
                init = l[(l.index(i)) - 1]
                somma.append(c[(l.index(i)) - 1])
                somma.append(j)
                stringa = init + stringa + i
                check_init = True
                stringa = stringa.replace('#', '')
            else:
                somma.append(j)
                stringa = stringa + i
                stringa = stringa.replace('#', '')
        else:
            if stringa != '':
                new_list_word.append(stringa)
                new_list_score.append(np.mean(somma))
                stringa = ''
                somma = []
                check_init = False
                if '#' not in l[(l.index(i)) + 1]:
                    new_list_word.append(i)
                    new_list_score.append(j)
            else:
                try:
                    if '#' not in l[(l.index(i)) + 1]:
                        new_list_word.append(i)
                        new_list_score.append(j)
                except:
                    new_list_word.append(i)
                    new_list_score.append(j)
    return new_list_word, new_list_score


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device-->', device)
    csv_log = sys.argv[1]
    TYPE = 'all'
    with open('log_history/' + csv_log + '/' + csv_log + '_test_' + TYPE + '.pkl', 'rb') as f:
        test = pickle.load(f)

    with open('log_history/' + csv_log + '/' + csv_log + '_label_test_' + TYPE + '.pkl', 'rb') as f:
        y_test = pickle.load(f)

    with open('log_history/' + csv_log + '/' + csv_log + '_id2label_' + TYPE + '.pkl', 'rb') as f:
        id2label = pickle.load(f)

    with open('log_history/' + csv_log + '/' + csv_log + '_label2id_' + TYPE + '.pkl', 'rb') as f:
        label2id = pickle.load(f)

    with open('log_history/' + csv_log + '/' + csv_log + '_suffix_train_' + TYPE + '.pkl', 'rb') as f:
        y_train_suffix = pickle.load(f)

    with open('log_history/' + csv_log + '/' + csv_log + '_suffix_test_' + TYPE + '.pkl', 'rb') as f:
        y_test_suffix = pickle.load(f)


    tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-medium', truncation_side='left')
    bertmodel = AutoModel.from_pretrained('prajjwal1/bert-medium').to(device)
    MAX_LEN = 512
    test_dataset = CustomDataset(test, y_test_suffix, tokenizer, MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    n_exp = 1854
    cont = 0
    first_element_list = []
    for f in test_loader:
        first_element_list.append(f)
        cont = cont + 1
        if cont==n_exp:
            break


    first_element = first_element_list[-1]
    output_sizes = []

    dict_pred = {}
    dict_truth = {}
    vis_data_records = []
    for i in range(len(y_train_suffix)):
        output_sizes.append(len(id2label['activity']))

    model_original = original(bertmodel, output_sizes).to(device)
    model_original.load_state_dict(torch.load('models/' + csv_log + '_' + TYPE + '.pth'))
    model_original.eval()
    model_original.zero_grad()
    input_ids = first_element['input_ids'].to(device)
    attention_mask = first_element['attention_mask'].to(device)
    output = model_original(input_ids, attention_mask)

    for j in range(len(y_train_suffix)):
        model = BertMultiOutputClassificationHeads(bertmodel, output_sizes, j)
        pred =output[j].argmax(dim=1).item()
        # Load the state dictionary into the model
        model.load_state_dict(torch.load('models/' + csv_log + '_' + TYPE + '.pth'))
        model = model.to(device)
        # Make sure to set the model in evaluation mode if you're not training it further
        model.eval()
        token_reference = TokenReferenceBase(reference_token_idx=tokenizer.pad_token_id)
        reference_indices = token_reference.generate_reference(512, device=device).unsqueeze(0)

        lig = LayerIntegratedGradients(model, model.gpt_model.embeddings)
        attributions_ig, delta = lig.attribute(input_ids, reference_indices, additional_forward_args=attention_mask, \
                                               n_steps=15, return_convergence_delta=True, target=first_element['labels'][j].item())

        attributions = attributions_ig.sum(dim=2).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.cpu().detach().numpy()
        a = input_ids.cpu().numpy().tolist()

        new_a = [tokenizer.convert_ids_to_tokens(t) for t in a[0]]
        new_prefix, new_score = reconstruct_prefix(new_a, attributions)



        text = []
        for t in new_prefix:
            if t != '[CLS]' and t != '[SEP]' and t != '[PAD]':
                text.append(t)
            else:
                text.append('')
        if id2label['activity'][pred] == id2label['activity'][first_element['labels'][j].item()]:
            pred_res = 'correct'
        else:
            pred_res = 'wrong'

        vis_data_records.append(visualization.VisualizationDataRecord(
            new_score,
            1,#pred
            id2label['activity'][pred],  # Label.vocab.itos[pred_ind],
            id2label['activity'][first_element['labels'][j].item()],
            pred_res,
            attributions.sum(),
            text,
            delta))
        if id2label['activity'][pred]=='ENDactivity' and  id2label['activity'][first_element['labels'][j].item()]=='ENDactivity':
            break

    html = visualization.visualize_text(vis_data_records)
    soup = BeautifulSoup(html.data, 'html.parser')

    with open('html_file.html', 'w') as f:
        f.write(str(soup))
