import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, pipeline
from datasets import load_dataset
import json
import numpy as np
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
# from huggingface_hub import HfFolder
from transformers import Trainer, TrainingArguments
from torch import nn

from prompt import PROMPTS

torch.set_float32_matmul_precision('high')

class CustomTrainer(Trainer):
    def compute_losss(self, model: nn.Module, inputs, return_outputs = False, num_items_in_batch = None):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss for 3 labels with different weights
        reduction = "mean" if num_items_in_batch is not None else "sum"
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9], device=model.device, reduction=reduction))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        if num_items_in_batch is not None:
            loss = loss / num_items_in_batch
        return (loss, outputs) if return_outputs else loss

class bert():
    def __init__(self):
        self.train_data_file = 'entities/train_data_llama_7b.json'
        self.test_data_file = 'entities/test_data_llama_7b.json'
        # self.test_data_file = 'entities/train_data_llama_7b.json'
        self.json_path = 'entities/sent_rel_result_llama_7b.json'

        self.model_id = "model/ModernBERT-Large"

        self.save_model_path = "save_model/modernbert"
        self.save_token = "save_model/modernbert"
        self.training_args_dir = "save_model"

        self.test_model = "code/LRP4RAG-master/save_model/checkpoint-19380"
        self.test_tokenizer = "code/LRP4RAG-master/save_model/modernbert"

        self.tokenizer = None
        self.model = None

    def train(self):
        data_files = {'train': self.train_data_file, 'test': self.test_data_file}
        raw_dataset = load_dataset(path = 'json', data_files = data_files)

        """pos = 0
        neg = 0
        for item in raw_dataset['train']:
            if item['lable'] == 0:
                neg += 1
            else:
                pos += 1
        print(pos / (pos + neg), neg / (pos + neg))"""

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.model_max_length = 512 # set model_max_length to 512 as prompts are not longer than 1024 tokens
        
        # Tokenize helper function
        def tokenize(batch):
            return self.tokenizer(batch['prompt'], padding='max_length', truncation=True, return_tensors="pt")
        
        # Tokenize dataset
        raw_dataset =  raw_dataset.rename_column("lable", "labels") # to match Trainer
        tokenized_dataset = raw_dataset.map(tokenize, batched=True)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, num_labels=2, reference_compile=False,
        )

        training_args = TrainingArguments(
            output_dir= self.training_args_dir,
            per_device_train_batch_size=10,
            per_device_eval_batch_size=10,
            learning_rate=5e-5,
            num_train_epochs=100,
            bf16=True, # bfloat16 training 
            optim="adamw_torch_fused", # improved optimizer 
            # logging & evaluation strategies
            logging_strategy="steps",
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
        )
        
        # Create a Trainer instance
        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            compute_metrics=self.compute_metrics,
        )
        trainer.train()

        self.tokenizer.save_pretrained(self.save_model_path)
        trainer.save_model(self.save_model_path)
        # trainer.create_model_card()
        # trainer.push_to_hub()

    def test(self, ):
        raw_dataset = load_dataset(path = 'json', data_files =  self.test_data_file)
        doc_text = {}
        doc_lable = {}
        for item in raw_dataset['train']:
            if item['id'] in doc_text:
                doc_text[item['id']].append(item['prompt'])
                doc_lable[item['id']].append(item['lable'])
            else:
                doc_text[item['id']] = [item['prompt']]
                doc_lable[item['id']] = [item['lable']]

        self.model =  AutoModelForSequenceClassification.from_pretrained(
            self.test_model, num_labels=2, reference_compile=False,
            )
        self.tokenizer =  AutoTokenizer.from_pretrained(self.test_tokenizer)

        classifier = pipeline("sentiment-analysis", 
                              model=self.test_model, 
                              tokenizer=self.tokenizer,
                              device=0, )
        
        predictions = []
        labels = []
        for doc_id in doc_text:
            pred = classifier(doc_text[doc_id])
            doc_pred = 0
            for item in pred:
                if item['label'][-1] == '1':
                    doc_pred = 1
            predictions.append(doc_pred)
            
            doc_true = 0
            for item in doc_lable[doc_id]:
                if item == 1:
                    doc_true = 1
            labels.append(doc_true)
            print("pred: ", doc_pred, "lable: ", doc_true)

        score = {
            'Accuracy': accuracy_score(labels, predictions),
            'Precision': precision_score(labels, predictions),
            'Recall': recall_score(labels, predictions),
            'F1 Score': f1_score(labels, predictions),
        }
        print(score)
            
    @staticmethod
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        score = f1_score(
                # labels, predictions, labels=labels, pos_label=1, average="weighted"
                labels, predictions
            )
        return {"f1": float(score) if score == 1 else score}

    def process_data(self, top_k = 3):
        with open(self.json_path, 'r') as f:
            data = json.load(f)

        train_data = []
        test_data = []
        train_num = int(len(data) * 0.8)
        num = 0
        for example_id, example in data.items(): 
            num += 1

            graph =  example['graph']
            node_lables = example['node_lables']
            prompts_sentences = example['prompts_sentences']
            response_sentences = example['response_sentences']

            edge_weights = [] 
            graph_max = [max(node) for node in graph]
            globle_ratio = [node_max / sum(graph_max) for node_max in graph_max]
            node_std = []
            for index, (node, ratio) in enumerate(zip(graph, globle_ratio)):
                node = np.array(node)
                connect = node / sum(node)
                # connect = node / sum(node) * ratio * 100
                edge_weights.append(connect)
                node_std.append(np.std(connect))
        
            all_sentences = prompts_sentences + response_sentences
            for index, edge_weight in enumerate(edge_weights):
                edge_indexes = np.argsort(edge_weight)[::-1][:top_k]
                
                context = ""
                for i, edge_index in enumerate(edge_indexes):
                    if edge_index < len(prompts_sentences):
                        source_sent = all_sentences[edge_index][3] + all_sentences[edge_index][0]

                    context += PROMPTS['context_prompt'].format(num=1+i, 
                                                                context=source_sent, 
                                                                contribution=round(edge_weight[edge_index], 2))
                    
                prompt = PROMPTS["hallucination_detect_prompt_bert"].format(context=context, 
                                                            # previous=all_sentences[index + len(prompts_sentences) - 1][0],
                                                            current=response_sentences[index][0])
                
                process_data= {
                    'id': example_id,
                    'prompt': prompt,
                    'lable': 0 if len(node_lables[index]) == 0 else 1
                                     }
                if num < train_num:
                    train_data.append(process_data)
                else:
                    test_data.append(process_data)

        with open(self.train_data_file, "w") as file:
            json.dump(train_data, file, indent=4)

        with open(self.test_data_file, "w") as file:
            json.dump(test_data, file, indent=4)

    def bert_demo(self ):
        # Load model and tokenizer
        torch.set_float32_matmul_precision('high')
        model_name = "model/ModernBERT-Large-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            model = AutoModelForMaskedLM.from_pretrained(model_name)
        else:
            model = AutoModelForMaskedLM.from_pretrained(model_name)

        model.to(device)

        # Format input for classification or multiple choice. This is a random example from MMLU.
        text = """You will be given a question and options. Select the right answer.
        QUESTION: If (G, .) is a group such that (ab)^-1 = a^-1b^-1, for all a, b in G, then G is a/an
        CHOICES:
        - A: commutative semi group
        - B: abelian group
        - C: non-abelian group
        - D: None of these
        ANSWER: [unused0] [MASK]"""

        # Get prediction
        inputs = tokenizer(text, return_tensors="pt").to(device)
        outputs = model(**inputs)
        mask_idx = (inputs.input_ids == tokenizer.mask_token_id).nonzero()[0, 1]
        pred_id = outputs.logits[0, mask_idx].argmax()
        answer = tokenizer.decode(pred_id)
        print(f"Predicted answer: {answer}")  # Outputs: B

if __name__ == '__main__':
    bert_model = bert()
    # bert_model.process_data()
    bert_model.train()
    # bert_model.bert_demo()
    # bert_model.test()
