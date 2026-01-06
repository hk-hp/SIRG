import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3' 

from pytorch_lightning import Trainer, seed_everything
from alignscore.dataloader import DSTDataLoader
from alignscore.model import BERTAlignModel
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser
import os
from alignscore.Parameter import MODEL_SIZE, TASK, LLM_NAME

def train(datasets, args):
    dm = DSTDataLoader(
        dataset_config=datasets, 
        model_name=args.model_name, 
        sample_mode='graph',
        train_batch_size=args.batch_size,
        eval_batch_size=16,
        num_workers=args.num_workers, 
        train_eval_split=0.95,
        need_mlm=args.do_mlm,
        is_finetune=True,
    )
    dm.setup()

    if args.ckpt_path == '':
        model = BERTAlignModel(model=args.model_name, 
            using_pretrained=args.use_pretrained_model,
            ckpt_path='code/LRP4RAG-master/save_model/AlignScore-' + MODEL_SIZE + '.ckpt', 
            adam_epsilon=args.adam_epsilon,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps_portion=args.warm_up_proportion
        )
    else:
        model = BERTAlignModel(model=args.model_name, 
            using_pretrained=args.use_pretrained_model,
            ckpt_path='/home/ /code/LRP4RAG-master/save_model/AlignScore-' + MODEL_SIZE + '.ckpt', 
            adam_epsilon=args.adam_epsilon,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps_portion=args.warm_up_proportion
        ).load_from_checkpoint(checkpoint_path=args.ckpt_path, strict=False)

    model.need_mlm = args.do_mlm

    checkpoint_name = str(TASK) + '_max_top15_' + LLM_NAME + '_' + MODEL_SIZE
    checkpoint_callback = ModelCheckpoint(
        monitor='f1',
        mode='max',
        dirpath=args.ckpt_save_path,
        filename=checkpoint_name + "_{epoch:02d}_{step}",
        # every_n_train_steps=422,
        every_n_epochs=1,
        save_top_k=1
    )
    trainer = Trainer(
        # logger=Logger(),
        accelerator='gpu', 
        max_epochs=args.num_epoch, 
        devices=args.devices, 
        strategy="dp", 
        precision=32,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=args.accumulate_grad_batch
    )

    trainer.fit(model, datamodule=dm)
    trainer.save_checkpoint(os.path.join(args.ckpt_save_path, f"{checkpoint_name}_final.ckpt"))

    print("Training is finished.")

if __name__ == "__main__":
    ALL_TRAINING_DATASETS = {
        ### NLI
        'graph_3': {'task_type': 'nli', 'data_path': 'entities/train_data_llama_7b.json'},     
        'mnli': {'task_type': 'nli', 'data_path': 'mnli.json'},     
        'doc_nli': {'task_type': 'bin_nli', 'data_path': 'doc_nli.json'},
        'snli': {'task_type': 'nli', 'data_path': 'snli.json'},
        'anli_r1': {'task_type': 'nli', 'data_path': 'anli_r1.json'},
        'anli_r2': {'task_type': 'nli', 'data_path': 'anli_r2.json'},
        'anli_r3': {'task_type': 'nli', 'data_path': 'anli_r3.json'},

        ### fact checking
        'nli_fever': {'task_type': 'fact_checking', 'data_path': 'nli_fever.json'},
        'vitaminc': {'task_type': 'fact_checking', 'data_path': 'vitaminc.json'},

        ### paraphrase
        'paws': {'task_type': 'paraphrase', 'data_path': 'paws.json'},
        'paws_qqp': {'task_type': 'paraphrase', 'data_path': 'paws_qqp.json'},
        'paws_unlabeled': {'task_type': 'paraphrase', 'data_path': 'paws_unlabeled.json'},
        'qqp': {'task_type': 'paraphrase', 'data_path': 'qqp.json'},
        'wiki103': {'task_type': 'paraphrase', 'data_path': 'wiki103.json'},

        ### QA
        'squad_v2': {'task_type': 'qa', 'data_path': 'squad_v2_new.json'},
        'race': {'task_type': 'qa', 'data_path': 'race.json'},
        'adversarial_qa': {'task_type': 'qa', 'data_path': 'adversarial_qa.json'},
        'drop': {'task_type': 'qa', 'data_path': 'drop.json'},
        'hotpot_qa_distractor': {'task_type': 'qa', 'data_path': 'hotpot_qa_distractor.json'},
        'hotpot_qa_fullwiki': {'task_type': 'qa', 'data_path': 'hotpot_qa_fullwiki.json'},
        'newsqa': {'task_type': 'qa', 'data_path': 'newsqa.json'},
        'quoref': {'task_type': 'qa', 'data_path': 'quoref.json'},
        'ropes': {'task_type': 'qa', 'data_path': 'ropes.json'},
        'boolq': {'task_type': 'qa', 'data_path': 'boolq.json'},
        'eraser_multi_rc': {'task_type': 'qa', 'data_path': 'eraser_multi_rc.json'},
        'quail': {'task_type': 'qa', 'data_path': 'quail.json'},
        'sciq': {'task_type': 'qa', 'data_path': 'sciq.json'},
        'strategy_qa': {'task_type': 'qa', 'data_path': 'strategy_qa.json'},

        ### Coreference
        'gap': {'task_type': 'coreference', 'data_path': 'gap.json'},

        ### Summarization
        'wikihow': {'task_type': 'summarization', 'data_path': 'wikihow.json'},

        ### Information Retrieval
        'msmarco': {'task_type': 'ir', 'data_path': 'msmarco.json'},

        ### STS
        'stsb': {'task_type': 'sts', 'data_path': 'stsb.json'},
        'sick': {'task_type': 'sts', 'data_path': 'sick.json'},        
    }

    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--accumulate-grad-batch', type=int, default=1)
    parser.add_argument('--num-epoch', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--warm-up-proportion', type=float, default=0.06)
    parser.add_argument('--adam-epsilon', type=float, default=1e-6)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--val-check-interval', type=float, default=1. / 4)
    parser.add_argument('--devices', nargs='+', type=int, default=0)
    parser.add_argument('--model-name', type=str, default="/home/ /model/roberta-" + MODEL_SIZE)
    parser.add_argument('--ckpt-save-path', type=str, default='/home/ /code/LRP4RAG-master/ckpt')
    parser.add_argument('--ckpt-path', type=str, default='')
    parser.add_argument('--ckpt-comment', type=str, default="")
    parser.add_argument('--trainin-datasets', nargs='+', type=str, default=list(ALL_TRAINING_DATASETS.keys()), choices=list(ALL_TRAINING_DATASETS.keys()))
    parser.add_argument('--max-samples-per-dataset', type=int, default=500000)
    parser.add_argument('--do-mlm', type=bool, default=False)
    parser.add_argument('--use-pretrained-model', type=bool, default=True)
   
    args = parser.parse_args()
    args.devices = [0]

    seed_everything(args.seed)

    """datasets = {
        name: {
            **ALL_TRAINING_DATASETS[name],
            "size": args.max_samples_per_dataset,
            "data_path": os.path.join(args.data_path, ALL_TRAINING_DATASETS[name]['data_path'])
        }
        for name in args.trainin_datasets
    }"""
    datasets = {'train': {'task_type': 'nli', 'data_path': 'entities/align_train_max_top15' + str(TASK) + '_data_' + LLM_NAME + '.json', 'size': 500000}, 
                'test': {'task_type': 'nli', 'data_path': 'entities/align_test_max_top15' + str(TASK) + '_data_' + LLM_NAME + '.json', 'size': 500000}}

    train(datasets, args)

