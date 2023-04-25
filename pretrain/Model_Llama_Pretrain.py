import torch
import json
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM

model_name = "/llm/models/llama_hf"
dataset = "./dataset/xbtqyj.txt"

torch.manual_seed(42)

tokenizer = AutoTokenizer.from_pretrained(model_name)
print(tokenizer.pad_token_id)
print(tokenizer)
training_args = TrainingArguments(output_dir='./results',
                                  num_train_epochs=2.2,
                                  logging_steps=100,
                                  learning_rate=1e-5,
                                  save_strategy='steps',
                                  save_steps=1000,
                                  evaluation_strategy='epoch',
                                  per_device_train_batch_size=2,
                                  per_device_eval_batch_size=2,
                                  lr_scheduler_type="cosine",
                                  warmup_steps=50,
                                  weight_decay=0.01,
                                  fp16=True,
                                  gradient_checkpointing=True,
                                  # deepspeed='./ds_config.json'
                                  )
model = AutoModelForCausalLM.from_pretrained(model_name).cuda()


# model.resize_token_embeddings(len(tokenizer))

class data_sets(Dataset):
    def __init__(self, corpus_path, max_length):
        self.input_ids = []
        self.attn_masks = []

        end = count_lines(corpus_path)
        pos = 0
        with open(corpus_path, mode="r", encoding="utf-8") as f:
            while True:
                line = f.readline()
                pos += 1

                encodings_dict = tokenizer(line, max_length=max_length, padding="max_length")
                self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
                self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

                if pos >= end:
                    break

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

def count_lines(file_path):
    lines_num = 0
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(2 ** 20)
            if not data:
                break
            lines_num += data.count(b'\n')
    return lines_num

# 362570
dataset = data_sets(dataset, 1024)
train_size = int(0.9995 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])


'''
model training
'''


def the_collate_fn(batch):
    input_ids = torch.stack([f[0] for f in batch])
    attention_mask = torch.stack([f[1] for f in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': input_ids}


class Mytrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=inputs["labels"])
        loss, logits = outputs[:2]
        return (loss, logits) if return_outputs else loss


trainer = Mytrainer(model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    data_collator=the_collate_fn
                    )
trainer.train()
trainer.save_model()

