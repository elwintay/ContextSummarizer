import hydra
from omegaconf import OmegaConf 

import pandas as pd
from dataloader import *
from model import *
import pytorch_lightning as pl
from sklearn.metrics import classification_report  

@hydra.main(config_path='.', config_name="config")
def main(cfg):
    
    from clearml import Task, Dataset, Logger, StorageManager

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    do_train = cfg['do_train']
    do_local = cfg['do_local']

    if do_train:

        if do_local==False:

            Task.force_requirements_env_freeze(force=True, requirements_file='requirements.txt')
            task = Task.init(project_name='ContextSum', task_name='train', output_uri="s3://experiment-logging/storage/")
            task.connect(cfg_dict)
            task.set_base_docker("nvcr.io/nvidia/pytorch:20.08-py3")
            task.execute_remotely(queue_name="compute", exit_process=True)
            logger = task.get_logger()

            import pandas as pd
            from dataloader import *
            from model import *
            import pytorch_lightning as pl
            from sklearn.metrics import classification_report  

            #config
            clearml_cfg = task.get_parameters_as_dict()
            epochs = clearml_cfg['General']["epochs"]
            gpu = clearml_cfg['General']["gpu"]
            save_model_folder = clearml_cfg['General']['save_model_folder']
            save_model_filename = clearml_cfg['General']['save_model_filename']
            batch_size = clearml_cfg['General']["batch_size"]
            workers = clearml_cfg['General']["workers"]
            max_token_len = clearml_cfg['General']["max_token_len"]
            warmup_steps = clearml_cfg['General']["warmup_steps"]

            #data
            dataset = Dataset.get(
                dataset_name="muc-sentence-6-fields",
                dataset_project="datasets/muc4",
                dataset_tags=["sentence-summarizer"],
                only_completed=True,
            )
            data_folder = dataset.get_local_copy()
        
        else:

            #config
            epochs = cfg_dict["epochs"]
            gpu = cfg_dict["gpu"]
            save_model_folder = cfg_dict['save_model_folder']
            save_model_filename = cfg_dict['save_model_filename']
            batch_size = cfg_dict["batch_size"]
            workers = cfg_dict["workers"]
            max_token_len = cfg_dict["max_token_len"]
            warmup_steps = cfg_dict["warmup_steps"]
            data_folder = cfg_dict['data_folder']

        #data
        train = pd.read_csv("{}/train.csv".format(data_folder))
        dev = pd.read_csv("{}/dev.csv".format(data_folder))
        test = pd.read_csv("{}/test.csv".format(data_folder))
        data = MucDataModule(train, dev, test, workers = workers, batch_size=batch_size, max_token_len=max_token_len)

        #model
        steps_per_epoch=len(train) // batch_size
        total_training_steps = steps_per_epoch * epochs
        model = BertTransformer("bert-base-uncased", n_warmup_steps=warmup_steps, n_training_steps=total_training_steps)
        
        #train
        checkpoint_callback = ModelCheckpoint(dirpath=save_model_folder, filename=save_model_filename, save_top_k=1, verbose=True, monitor="val_loss", mode="min")
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)
        logger = TensorBoardLogger("lightning_logs", name="ContextSum")
        trainer = pl.Trainer(logger = logger, checkpoint_callback=checkpoint_callback, callbacks=[early_stopping_callback], max_epochs=epochs, gpus=gpu, progress_bar_refresh_rate=30)
        trainer.fit(model, data)

        #predict
        result = trainer.test(model)
        text = result['text']
        docid = result['docid']
        preds = result['predictions']
        labels = result['labels']
        print(classification_report(labels, preds, zero_division=0))

        #save predictions
        output_df = pd.DataFrame()
        output_df['docid'] = docid
        output_df['text'] = text
        output_df['labels'] = labels
        output_df['preds'] = preds
        output_df.to_csv("output.csv",index=False)
        if do_local==False:
            task.upload_artifact("predictions", output_df)
            task.close()

if __name__ == "__main__":
    main()








        
        # Task.force_requirements_env_freeze(force=True, requirements_file='requirements.txt')
        # task = Task.init(project_name='ContextSum', task_name='train', output_uri="s3://experiment-logging/storage/")
        # task.connect(cfg_dict)
        # task.set_base_docker("nvcr.io/nvidia/pytorch:20.08-py3")
        # task.execute_remotely(queue_name="compute", exit_process=True)
        # logger = task.get_logger()

        # import argparse
        # import pandas as pd
        # from dataloader import *
        # from model import *
        # import pytorch_lightning as pl
        # import torch
        # from sklearn.metrics import classification_report
        # import os
        # import json

        # #config
        # clearml_cfg = task.get_parameters_as_dict()
        # epoch = clearml_cfg['General']["epoch"]
        # gpu = clearml_cfg['General']["gpu"]
        # model_dirpath = clearml_cfg['General']['model_path']
        # model_filename = clearml_cfg['General']['model_name']
        # batch_size = clearml_cfg['General']["batch_size"]
        # workers = clearml_cfg['General']["workers"]
        # max_token_len = clearml_cfg['General']["max_token_len"]

        # dataset_name = clearml_cfg['General']["dataset_name"]
        # dataset_project = clearml_cfg['General']["dataset_project"]
        # pred_proj = clearml_cfg['General']["pred_project"]
        # pred_name = clearml_cfg['General']["pred_name"]
        # model_path = clearml_cfg['General']["model_path"]
        

        # #data and model
        # train = pd.read_csv("data/muc_sentence/train.csv")
        # dev = pd.read_csv("data/muc_sentence/dev.csv")
        # test = pd.read_csv("data/muc_sentence/test.csv")
        # data = MucDataModule(train, dev, test, tokenizer, workers = workers, batch_size=batch_size, max_token_len=max_token_len)
        # model = MucTagger(label_columns, args.model_path, n_classes=len(label_columns), n_warmup_steps=args.warmup_steps, n_training_steps=total_training_steps)
        
        # checkpoint_callback = ModelCheckpoint(dirpath=model_dirpath, filename=model_filename, save_top_k=1, verbose=True, monitor="val_loss", mode="min")
        # early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)
        # logger = TensorBoardLogger("lightning_logs", name="ContextSum")
        # trainer = pl.Trainer(logger = logger, checkpoint_callback=checkpoint_callback, callbacks=[early_stopping_callback], max_epochs=epoch, gpus=gpu, progress_bar_refresh_rate=30)
        # trainer.fit(model, data_module)