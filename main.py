import hydra
from omegaconf import OmegaConf
from clearml import Task, StorageManager, Dataset

import pandas as pd
from dataloader import *
from model import *
import pytorch_lightning as pl
from sklearn.metrics import classification_report  
import os

Task.force_requirements_env_freeze(force=True, requirements_file='requirements.txt')

@hydra.main(config_path='.', config_name="config")
def main(cfg):
    
    from clearml import Task, Dataset, Logger, StorageManager

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    do_train = cfg['do_train']
    do_local = cfg['do_local']

    if do_train:

        if do_local==False:
            
            task = Task.init(project_name='ContextSum', task_name='train', output_uri="s3://experiment-logging/storage/")
            task.connect(cfg_dict)
            task.set_base_docker("nvcr.io/nvidia/pytorch:21.06-py3")
            task.execute_remotely(queue_name="compute", exit_process=True)
            logger = task.get_logger()

            #config
            clearml_cfg = task.get_parameters_as_dict()
            epochs = int(clearml_cfg['General']["epochs"])
            gpu = int(clearml_cfg['General']["gpu"])
            save_model_folder = clearml_cfg['General']['save_model_folder']
            save_model_filename = clearml_cfg['General']['save_model_filename']
            batch_size = int(clearml_cfg['General']["batch_size"])
            workers = int(clearml_cfg['General']["workers"])
            max_token_len = int(clearml_cfg['General']["max_token_len"])
            warmup_steps = int(clearml_cfg['General']["warmup_steps"])

            #data
            dataset = Dataset.get(
                dataset_name="muc4-sentence-6-fields-v2",
                dataset_project="datasets/muc4",
                dataset_tags=["sentence-summarizer"],
                only_published=True,
            )
            data_folder = dataset.get_local_copy()
            print(list(os.walk(data_folder)))
        
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
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        data = MucDataModule(train, dev, test, tokenizer, workers = workers, batch_size=batch_size, max_token_len=max_token_len)

        #model
        steps_per_epoch = len(train) // batch_size
        total_training_steps = steps_per_epoch * epochs
        model = BertTransformer("bert-base-uncased", n_warmup_steps=warmup_steps, n_training_steps=total_training_steps)
        
        #train
        checkpoint_callback = ModelCheckpoint(dirpath=save_model_folder, filename=save_model_filename, save_top_k=1, verbose=True, monitor="val_loss", mode="min")
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)
        logger = TensorBoardLogger("lightning_logs", name="ContextSum")
        trainer = pl.Trainer(logger = logger, checkpoint_callback=checkpoint_callback, callbacks=[early_stopping_callback], max_epochs=epochs, gpus=gpu, progress_bar_refresh_rate=1)
        trainer.fit(model, data)

        #predict
        trainer.test(model, data)
        results = model.test_results
        docid = results['docid']
        text = results['text']
        labels =  results['labels'].squeeze().tolist()
        preds =  torch.round(results['preds'].squeeze()).tolist()
        target_names = ['What was targeted','What was used','Where','Which','Who attack','Who injured or killed']
        print(classification_report(labels, preds, zero_division=0, target_names=target_names))

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

    else:

        if do_local==False:
            
            task = Task.init(project_name='ContextSum', task_name='test', output_uri="s3://experiment-logging/storage/")
            task.connect(cfg_dict)
            task.set_base_docker("nvcr.io/nvidia/pytorch:21.06-py3")
            task.execute_remotely(queue_name="compute", exit_process=True)
            logger = task.get_logger()

            #config
            clearml_cfg = task.get_parameters_as_dict()
            epochs = int(clearml_cfg['General']["epochs"])
            gpu = int(clearml_cfg['General']["gpu"])
            save_model_folder = clearml_cfg['General']['save_model_folder']
            save_model_filename = clearml_cfg['General']['save_model_filename']
            batch_size = int(clearml_cfg['General']["batch_size"])
            workers = int(clearml_cfg['General']["workers"])
            max_token_len = int(clearml_cfg['General']["max_token_len"])
            warmup_steps = int(clearml_cfg['General']["warmup_steps"])
            pretrained_path = clearml_cfg['General']['pretrained_path']
            args = clearml_cfg['General']

            #data
            dataset = Dataset.get(
                dataset_name="muc4-sentence-6-fields-v2",
                dataset_project="datasets/muc4",
                dataset_tags=["sentence-summarizer"],
                only_published=True,
            )
            data_folder = dataset.get_local_copy()
            print(list(os.walk(data_folder)))
        
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
            pretrained_path = cfg_dict['pretrained_path']
            args = cfg_dict

        #data
        train = pd.read_csv("{}/train.csv".format(data_folder))
        dev = pd.read_csv("{}/dev.csv".format(data_folder))
        test = pd.read_csv("{}/test.csv".format(data_folder))
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        data = MucDataModule(train, dev, test, tokenizer, workers = workers, batch_size=batch_size, max_token_len=max_token_len)

        #model
        trained_model_path = StorageManager.get_local_copy(pretrained_path)
        # model = BertTransformer.load_from_checkpoint(trained_model_path, hparams = args)
        model = BertTransformer.load_from_checkpoint(trained_model_path, strict=False)
        
        #train
        checkpoint_callback = ModelCheckpoint(dirpath=save_model_folder, filename=save_model_filename, save_top_k=1, verbose=True, monitor="val_loss", mode="min")
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)
        logger = TensorBoardLogger("lightning_logs", name="ContextSum")
        trainer = pl.Trainer(logger = logger, checkpoint_callback=checkpoint_callback, callbacks=[early_stopping_callback], max_epochs=epochs, gpus=gpu, progress_bar_refresh_rate=1)
        
        #predict
        trainer.test(model, data)
        results = model.test_results
        docid = results['docid']
        text = results['text']
        labels =  results['labels'].squeeze().tolist()
        preds =  torch.round(results['preds'].squeeze()).tolist()
        target_names = ['What was targeted','What was used','Where','Which','Who attack','Who injured or killed']
        print(classification_report(labels, preds, zero_division=0, target_names=target_names))

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