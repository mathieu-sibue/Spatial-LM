from tqdm import tqdm 
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score,roc_auc_score,classification_report

from transformers import TrainingArguments, Trainer
from transformers.data.data_collator import default_data_collator
import numpy as np


# class WeightedLossTrainer(Trainer):
#     def compute_loss(self,model,inputs,return_outputs=False):
#         outputs = model(**inputs)
#         logits = outputs.get("logits")
#         labels = inputs.get("labels")
#         loss_func = nn.CrossEntropyLoss(weight=class_weights)
#         loss = loss_func(logits,labels)
#         return (loss,outputs) if return_outputs else loss
    
# the model itself is (by default) responsible for computing some sort of loss and returning it in outputs.
def train(opt, model, mydata):
    logging_steps = len(mydata.train_dataset)//opt.batch_size

    training_args = TrainingArguments(
        output_dir = opt.output_dir,
        num_train_epochs = opt.epochs,
        learning_rate = opt.lr,
        per_device_train_batch_size = opt.batch_size,
        per_device_eval_batch_size = opt.batch_size,
        weight_decay = 0.01,
        warmup_ratio = 0.1,
        # fp16 = True,
        push_to_hub = False,
        # push_to_hub_model_id = f"layoutlmv3-finetuned-cord"        
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        logging_steps = logging_steps,
        load_best_model_at_end = True,
        metric_for_best_model = "f1"
    )
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = mydata.train_dataset,
        eval_dataset = mydata.test_dataset,
        tokenizer = opt.processor,
        compute_metrics = compute_metrics,  # evaluaion phase, return dict
    )
    trainer.train()
    trainer.evaluate()

def compute_metrics(p,return_entity_level_metrics=False):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

