import torch 
import torch.utils.data
import os
import sys
import time
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from collections import Counter

from xclib.evaluation import xc_metrics
from io import StringIO
import itertools
import pickle

from envyaml import EnvYAML
from XPERT_model import XPERT
from dataset import DatasetMain
from utils import vmap_idx, get_indices_map_embs

def read_config(config_file, section=None):
    """
    Read the yaml config file
    :param config_file:
    :param section: section in the config file that needs to be read
    :return: A dictionary containing configuration for the section. If section is none, return the whole config
    """
    config = EnvYAML(config_file)
#     with open(config_file, 'r') as yaml_handle:
#         config = yaml.safe_load(yaml_handle)
    if section is None:
        return config
    return config[section]

def create_model(config, device='cuda:0'):        # feat_type by default is embeddings
    print(f"Initializing XPERT model")
    model = XPERT(config, device).to(device)
    return model

def evaluate_model(model, val_loader, eval_out_dir,
                   topk=100, k=10, num_batches=-1, verbose=True):
    model.eval()

    epoch_train_start_time = time.time()
    pred_smats, gt_smats = [], []

    print("eval pass started")
    for batch_num, batch_data in tqdm(enumerate(val_loader), disable=(not verbose)):
        with torch.no_grad():
            logits = model(batch_data)  # novel item embeddings, item embeddings

            novel_events = logits[0]
            novel_offsets = batch_data["novel_offsets"]

            personalized_embeddings = novel_events.detach().cpu().numpy()

        nn_retrievals, _ = val_loader.getnns(q=personalized_embeddings,
                                                              novel_offsets=novel_offsets,
                                                              k=k,
                                                              sampling_probabilities=batch_data["novel_importance"])
        pred_smats.append(nn_retrievals)
        gt_smats.append(batch_data['gt_smat'])

        del batch_data, logits
        if num_batches != -1 and batch_num == num_batches:
            break

    val_loader.join()
    print("eval pass ended")
    
    pred_smat = sp.vstack(pred_smats)
    pred_smat.sum_duplicates()
    dataset_labels = sp.vstack(gt_smats)

    outputs = StringIO()
    sys.stdout = outputs

    print(pred_smat.shape, dataset_labels.shape)

    # Saving CSR pred_smat , dataset_labels and debug file
    sp.save_npz(os.path.join(eval_out_dir, "pred_smat.npz"), pred_smat)
    sp.save_npz(os.path.join(eval_out_dir, "dataset_labels.npz"), dataset_labels)

    # Metrics Calculation
    print(f"Run time {time.time() - epoch_train_start_time}s.")

    prec = xc_metrics.precision(pred_smat, dataset_labels, k=topk)
    recall = xc_metrics.recall(pred_smat, dataset_labels, k=topk)
    print(f"P@1: {prec[0]} P@3: {prec[2]} P@{topk}: {prec[-1]}")
    print(f"R@1: {recall[0]} R@3: {recall[2]} R@{topk}: {recall[-1]}")

    sys.stdout = sys.__stdout__

    print(outputs.getvalue())
    with open(os.path.join(eval_out_dir, "Metrics.txt"), "w") as text_file:
        print(outputs.getvalue(), file=text_file)

    return prec, recall


if __name__ == "__main__":
    eval_config_file = read_config(sys.argv[1])
    print("Read Configs")


    sd = torch.load(eval_config_file['evaluation']['model'], map_location=torch.device("cpu"))
    # TO-DO: Need to create model weights with bare min information
    model = create_model(sd['params']['train_config']['model'],
                            device=eval_config_file['evaluation']['device'])
    model.load_state_dict(sd['model_state_dict'])

    print("Created model")

    data_config = eval_config_file['eval_data']

    val_dataloader = DatasetMain(data_config, candidates=True)
    os.makedirs(eval_config_file['evaluation']['eval_out_dir'], exist_ok=True)

    outcome = evaluate_model(model, val_dataloader, eval_config_file['evaluation']['eval_out_dir'],
                                          num_batches=eval_config_file['evaluation']['num_batches'])
