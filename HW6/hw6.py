import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPT2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from typing import Tuple, Dict
from tqdm import tqdm
from datasets import load_dataset, Dataset
import math
import random


from hw6_utils import (
    prompt_zero,
    load_classification_data,
    eval_model,
    TOKENIZER,
    MAX_LEN,
    YES_ID,
    NO_ID,
)







## Problem: Simple Q-Learning

def calculate_q_update(q_value, reward, next_max_q, alpha, gamma):
    """
    Calculate the updated Q-value using the Q-learning formula.
    
    Q(s,a) ← Q(s,a) + α [ r + γ·max_a' Q(s',a') − Q(s,a) ]
    """
    # temporal‐difference error
    td_error = reward + gamma * next_max_q - q_value
    # update
    return q_value + alpha * td_error


def select_best_action(q_values):
    """
    Select the action with the highest Q-value (greedy policy).
    """
    # np.argmax breaks ties by choosing the first max
    return int(np.argmax(q_values))


def simple_training_loop(episodes, alpha, gamma):
    """
    Train a Q-table in the 2×2 grid world:
    
        S 0
        0 G
    
    States 0–3, actions 0=right,1=down, reward −1 per step, +10 at G (state 3).
    """
    # 4 states × 2 actions
    q_table = np.zeros((4, 2))
    
    def step(state, action):
        # action 0 = move right
        if action == 0 and (state % 2) == 0:
            return state + 1
        # action 1 = move down
        if action == 1 and state < 2:
            return state + 2
        # otherwise stay
        return state

    def reward_for(state):
        return 10 if state == 3 else -1

    for _ in range(episodes):
        state = 0
        # run until goal
        while state != 3:
            # (b) greedy action
            action = select_best_action(q_table[state])
            # (c) transition
            next_state = step(state, action)
            # (d) reward
            r = reward_for(next_state)
            # (e) Q-update
            next_max = np.max(q_table[next_state])
            q_table[state, action] = calculate_q_update(
                q_table[state, action], r, next_max, alpha, gamma
            )
            # move on
            state = next_state

    return q_table


## Problem: GPT-2 Finetuning
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED);  random.seed(SEED)
torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
# --------------------------------------------------------------------
# Do Not modify the following Constants
# --------------------------------------------------------------------
MODEL_NAME = "gpt2"
TOKENIZER = GPT2Tokenizer.from_pretrained(MODEL_NAME)
TOKENIZER.pad_token = TOKENIZER.eos_token
TOKENIZER.pad_token_id = TOKENIZER.eos_token_id
YES_ID, NO_ID = TOKENIZER.encode(" yes")[0], TOKENIZER.encode(" no")[0]

MAX_LEN = 200   # maximum total length (prompt + answer + pads)



def build_prompt_with_answer(ex):
    """
    Create a training prompt followed by the correct “ yes” / “ no” answer.
    """
    prompt = prompt_zero(ex["content"])
    answer = " yes" if ex["label"] == 1 else " no"
    return {"text": prompt + answer}


def tokenize_seq2seq(batch):
    """
    Tokenize prompt+answer for causal LM training.  Truncate from the front
    to MAX_LEN, pad to MAX_LEN, and mask all labels except the final token.
    """
    input_ids, attention_mask, labels = [], [], []
    for text, label in zip(batch["content"], batch["label"]):
        # build the full text
        full = prompt_zero(text) + (" yes" if label == 1 else " no")
        # raw tokenization (no special tokens)
        tok = TOKENIZER.encode(full, add_special_tokens=False)
        # truncate from front
        if len(tok) > MAX_LEN:
            tok = tok[-MAX_LEN:]
        # pad
        pad_len = MAX_LEN - len(tok)
        ids = tok + [TOKENIZER.pad_token_id] * pad_len
        mask = [1] * len(tok) + [0] * pad_len
        # labels: ignore all but final actual token
        lab = [-100] * MAX_LEN
        lab[len(tok) - 1] = tok[-1]
        input_ids.append(ids)
        attention_mask.append(mask)
        labels.append(lab)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def train_seq2seq(train_ds, lm: GPT2LMHeadModel):
    """
    Fine-tune GPT2LMHeadModel in a causal seq2seq style:
      1) map via tokenize_seq2seq
      2) Trainer over 1 epoch, lr=2e-4, bs=8, no saving
    """
    tok_ds = train_ds.map(
        tokenize_seq2seq,
        batched=True,
        remove_columns=[c for c in train_ds.column_names]
    )
    collator = DataCollatorWithPadding(TOKENIZER)
    args = TrainingArguments(
        output_dir="./seq2seq",
        per_device_train_batch_size=8,
        learning_rate=2e-4,
        num_train_epochs=1,
        save_strategy="no",
        logging_strategy="no",
    )
    trainer = Trainer(
        model=lm,
        args=args,
        train_dataset=tok_ds,
        data_collator=collator,
    )
    trainer.train()


def tokenize_clf(batch):
    """
    Tokenize for classification.  Simple truncate+pad to MAX_LEN,
    return input_ids, attention_mask, and integer labels.
    """
    enc = TOKENIZER(
        batch["content"],
        truncation=True,
        max_length=MAX_LEN,
        padding=False,           # we let the DataCollator pad
        return_attention_mask=True,
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": batch["label"],
    }


def train_clf(train_ds: Dataset, model: GPT2ForSequenceClassification):
    """
    Fine-tune GPT2ForSequenceClassification:
      1) map via tokenize_clf
      2) Trainer over 1 epoch, lr=2e-4, bs=8, no saving
    """
    tok_ds = train_ds.map(
        tokenize_clf,
        batched=True,
        remove_columns=[c for c in train_ds.column_names]
    )
    collator = DataCollatorWithPadding(TOKENIZER)
    args = TrainingArguments(
        output_dir="./clf",
        per_device_train_batch_size=8,
        learning_rate=2e-4,
        num_train_epochs=1,
        save_strategy="no",
        logging_strategy="no",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tok_ds,
        data_collator=collator,
    )
    trainer.train()


###### You may want to use the following code to debug your implementation
# # test the build prompt function
# print(build_prompt_with_answer({"content": "I love this movie!", "label": 1})["text"])

# # initialize the dataset
# train_set = load_classification_data("classification_train.txt")

# # initialize the seq2seq model
# lm_seq2seq = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
# # train seq2seq
# train_seq2seq(train_set, lm_seq2seq)
# # evaluate seq2seq on the training set to make sure it works
# seq_loss, seq_acc = eval_model(lm_seq2seq)


# # initialize the classification model
# lm_clf = GPT2ForSequenceClassification.from_pretrained(
#     MODEL_NAME, num_labels=2, pad_token_id=TOKENIZER.pad_token_id).to(DEVICE)
# # train classification model
# train_clf(train_set, lm_clf)
# # evaluate classification model on the training set to make sure it works
# clf_loss, clf_acc = eval_model(lm_clf, tokenize_clf)









## Problem: Transformer Attnetion

def scaled_dot_product_attention(q, k, v):
    """
    Compute scaled dot‐product attention.
    Args:
      q: (..., seq_len, d_k)
      k: (..., seq_len, d_k)
      v: (..., seq_len, d_k)
    Returns:
      out: (..., seq_len, d_k)
    """
    d_k = q.size(-1)
    # (..., seq_len, seq_len)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    weights = F.softmax(scores, dim=-1)
    # (..., seq_len, d_k)
    return torch.matmul(weights, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        d_model: total feature dimension
        num_heads: number of heads (must divide d_model)
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # four linear layers: Q, K, V, and output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        """
        q, k, v: (batch_size, seq_len, d_model)
        returns: (batch_size, seq_len, d_model)
        """
        B, L, _ = q.size()

        # 1) linear projections
        q_proj = self.W_q(q)  # (B, L, d_model)
        k_proj = self.W_k(k)
        v_proj = self.W_v(v)

        # 2) split into heads and transpose
        # → (B, num_heads, L, d_k)
        q_heads = q_proj.view(B, L, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        k_heads = k_proj.view(B, L, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        v_heads = v_proj.view(B, L, self.num_heads, self.d_k).permute(0, 2, 1, 3)

        # 3) apply scaled dot‐product attention on each head
        # out_heads: (B, num_heads, L, d_k)
        out_heads = scaled_dot_product_attention(q_heads, k_heads, v_heads)

        # 4) concatenate heads and project
        # → (B, L, num_heads * d_k == d_model)
        out_concat = (
            out_heads
            .permute(0, 2, 1, 3)       # (B, L, num_heads, d_k)
            .contiguous()
            .view(B, L, self.d_model)  # (B, L, d_model)
        )

        # final linear layer
        return self.W_o(out_concat)