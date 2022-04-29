import hydra
from omegaconf import OmegaConf
from clearml import Task, StorageManager, Dataset

import pandas as pd
from dataloader import *
from model import *
import pytorch_lightning as pl
from sklearn.metrics import classification_report  
import os
import ast

Task.force_requirements_env_freeze(force=True, requirements_file='requirements.txt')

@hydra.main(config_path='.', config_name="config")
def main(cfg):
    
    from clearml import Task, Dataset, Logger, StorageManager

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    do_train = cfg['do_train']
    do_local = cfg['do_local']

    if do_train:

        if do_local==False:
            
            task = Task.init(project_name='ContextSum', task_name='train-context', output_uri="s3://experiment-logging/storage/")
            task.connect(cfg_dict)
            task.set_base_docker("nvcr.io/nvidia/pytorch:21.06-py3")
            task.execute_remotely(queue_name="compute", exit_process=True)
            logger = task.get_logger()

            #config
            clearml_cfg = task.get_parameters_as_dict()
            epochs = int(clearml_cfg['General']["epochs"])
            gpu = int(clearml_cfg['General']["gpu"])
            save_model_folder = clearml_cfg['General']['save_model_folder']
            context_save_model_filename = clearml_cfg['General']['context_save_model_filename']
            batch_size = int(clearml_cfg['General']["batch_size"])
            workers = int(clearml_cfg['General']["workers"])
            max_token_len = int(clearml_cfg['General']["max_token_len"])
            warmup_steps = int(clearml_cfg['General']["warmup_steps"])

            #data
            dataset = Dataset.get(
                dataset_name="muc4-sentence-6-fields-v4",
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
            context_save_model_filename = cfg_dict['context_save_model_filename']
            batch_size = cfg_dict["batch_size"]
            workers = cfg_dict["workers"]
            max_token_len = cfg_dict["max_token_len"]
            warmup_steps = cfg_dict["warmup_steps"]
            data_folder = cfg_dict['data_folder']

        #data
        train = pd.read_csv("{}/train.csv".format(data_folder))
        dev = pd.read_csv("{}/dev.csv".format(data_folder))
        test = pd.read_csv("{}/test.csv".format(data_folder))
        test_qa = pd.read_csv("{}/test_qa.csv".format(data_folder))
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        data = MucDataModule(train, dev, test, tokenizer, workers = workers, batch_size=batch_size, max_token_len=max_token_len)

        #model
        steps_per_epoch = len(train) // batch_size
        total_training_steps = steps_per_epoch * epochs
        model = BertTransformer("bert-base-uncased", n_warmup_steps=warmup_steps, n_training_steps=total_training_steps)
        
        #train
        context_checkpoint_callback = ModelCheckpoint(dirpath=save_model_folder, filename=context_save_model_filename, save_top_k=1, verbose=True, monitor="val_loss", mode="min")
        context_early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)
        context_logger = TensorBoardLogger("lightning_logs", name="ContextSum")
        context_trainer = pl.Trainer(logger = context_logger, checkpoint_callback=context_checkpoint_callback, callbacks=[context_early_stopping_callback], max_epochs=epochs, gpus=gpu, progress_bar_refresh_rate=1)
        context_trainer.fit(model, data)

        #predict
        context_trainer.test(model, data)
        results = model.test_results
        docid = results['docid']
        text = results['text']
        labels =  results['labels'].squeeze().tolist()
        preds =  torch.round(results['preds'].squeeze()).tolist()
        target_names = ["Where is the location?","Who was the attacker?","Which organisation?","What was targeted?","Who injured or killed?","What weapon was used?"]
        print(classification_report(labels, preds, zero_division=0, target_names=target_names))

        output_df = pd.DataFrame()
        output_df['docid'] = docid
        output_df['text'] = text
        pred_data = pd.DataFrame.from_records(preds)
        pred_data.columns = target_names
        pred_df = pd.concat([output_df,pred_data],axis=1)
        pred_df = pd.melt(pred_df, id_vars=['docid','text'], value_vars=target_names)
        pred_df.rename(columns={'variable':'qns', 'value':'label'}, inplace=True)

        pred_df.to_csv("context_pred.csv",index=False)
        if do_local==False:
            task.upload_artifact("context_predictions", 'context_pred.csv')
            task.close()

        # #predict
        # trainer.test(model, data)
        # results = model.test_results
        # docid = results['docid']
        # text = results['text']
        # labels =  results['labels'].squeeze().tolist()
        # preds =  torch.round(results['preds'].squeeze()).tolist()
        # target_names = ["Where is the location?","Who was the attacker?","Which organisation?","What was targeted?","Who injured or killed?","What weapon was used?"]
        # print(classification_report(labels, preds, zero_division=0, target_names=target_names))

        # #save predictions
        # test_qa = pd.read_csv("{}/test_qa.csv".format(data_folder))
        # test_qa['entity'] = test_qa['entity'].apply(lambda x: ast.literal_eval(x))
        # test_qa['text_label'] = test_qa['entity'].apply(lambda x: x['entity'] if x!={} else '')
        # output_df = pd.DataFrame()
        # output_df['docid'] = docid
        # output_df['text'] = text
        # pred_data = pd.DataFrame.from_records(preds)
        # pred_data.columns = target_names
        # pred_df = pd.concat([output_df,pred_data],axis=1)
        # pred_df = pd.melt(pred_df, id_vars=['docid','text'], value_vars=target_names)
        # pred_df.rename(columns={'variable':'qns', 'value':'label'}, inplace=True)
        # question_answering = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        # with open('pred.json', 'w') as outfile:
        #     for id in output_df['docid'].unique():
        #         doc_dict = {}
        #         doc_dict['docid'] = id
        #         doc_dict['gold'] = {}
        #         doc_dict['pred'] = {}
        #         temp_df = pred_df[pred_df['docid']==id].reset_index(drop=True)
        #         temp_label = test_qa[test_qa['docid']==id].reset_index(drop=True)
        #         for qns in target_names:
        #             doc_dict['pred'][qns] = []
        #             doc_dict['gold'][qns] = []
        #         for i in range(len(temp_df)):
        #             if temp_df.loc[i,'label']==1:
        #                 qns = temp_df.loc[i,'qns']
        #                 result = question_answering(question=qns, context=temp_df.loc[i,'text'])
        #                 doc_dict['pred'][qns].append(result['answer'])
        #                 doc_dict['pred'][qns] = list(set(doc_dict['pred'][qns]))
        #                 doc_dict['gold'][qns] += list(set(temp_label.loc[temp_label['question']==qns,'text_label'].tolist()))
        #                 doc_dict['gold'][qns] = list(set(doc_dict['gold'][qns]))
        #             else:
        #                 continue
        #         json.dump(doc_dict, outfile)
        #         outfile.write('\n')
        # if do_local==False:
        #     task.upload_artifact("predictions", 'pred.json')
        #     task.close()

    else:

        if do_local==False:
            
            task = Task.init(project_name='ContextSum', task_name='test-context', output_uri="s3://experiment-logging/storage/")
            task.connect(cfg_dict)
            task.set_base_docker("nvcr.io/nvidia/pytorch:21.06-py3")
            task.execute_remotely(queue_name="compute", exit_process=True)
            logger = task.get_logger()

            #config
            clearml_cfg = task.get_parameters_as_dict()
            epochs = int(clearml_cfg['General']["epochs"])
            gpu = int(clearml_cfg['General']["gpu"])
            save_model_folder = clearml_cfg['General']['save_model_folder']
            context_save_model_filename = clearml_cfg['General']['context_save_model_filename']
            batch_size = int(clearml_cfg['General']["batch_size"])
            workers = int(clearml_cfg['General']["workers"])
            max_token_len = int(clearml_cfg['General']["max_token_len"])
            warmup_steps = int(clearml_cfg['General']["warmup_steps"])
            context_pretrained_path = clearml_cfg['General']['context_pretrained_path']

            #data
            dataset = Dataset.get(
                dataset_name="muc4-sentence-6-fields-v4",
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
            context_save_model_filename = cfg_dict['context_save_model_filename']
            batch_size = cfg_dict["batch_size"]
            workers = cfg_dict["workers"]
            max_token_len = cfg_dict["max_token_len"]
            warmup_steps = cfg_dict["warmup_steps"]
            data_folder = cfg_dict['data_folder']
            context_pretrained_path = cfg_dict['context_pretrained_path']

        #data
        train = pd.read_csv("{}/train.csv".format(data_folder))
        dev = pd.read_csv("{}/dev.csv".format(data_folder))
        test = pd.read_csv("{}/test.csv".format(data_folder))
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        data = MucDataModule(train, dev, test, tokenizer, workers = workers, batch_size=batch_size, max_token_len=max_token_len)

        #model
        trained_model_path = StorageManager.get_local_copy(context_pretrained_path)
        # model = BertTransformer.load_from_checkpoint(trained_model_path, hparams = args)
        model = BertTransformer.load_from_checkpoint(trained_model_path, strict=False)
        
        #train
        context_checkpoint_callback = ModelCheckpoint(dirpath=save_model_folder, filename=context_save_model_filename, save_top_k=1, verbose=True, monitor="val_loss", mode="min")
        context_early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)
        context_logger = TensorBoardLogger("lightning_logs", name="ContextSum")
        context_trainer = pl.Trainer(logger = context_logger, checkpoint_callback=context_checkpoint_callback, callbacks=[context_early_stopping_callback], max_epochs=epochs, gpus=gpu, progress_bar_refresh_rate=1)
        
        #predict
        context_trainer.test(model, data)
        results = model.test_results
        docid = results['docid']
        text = results['text']
        labels =  results['labels'].squeeze().tolist()
        preds =  torch.round(results['preds'].squeeze()).tolist()
        target_names = ["Where is the location?","Who was the attacker?","Which organisation?","What was targeted?","Who injured or killed?","What weapon was used?"]
        print(classification_report(labels, preds, zero_division=0, target_names=target_names))

        output_df = pd.DataFrame()
        output_df['docid'] = docid
        output_df['text'] = text
        pred_data = pd.DataFrame.from_records(preds)
        pred_data.columns = target_names
        pred_df = pd.concat([output_df,pred_data],axis=1)
        pred_df = pd.melt(pred_df, id_vars=['docid','text'], value_vars=target_names)
        pred_df.rename(columns={'variable':'qns', 'value':'label'}, inplace=True)

        pred_df.to_csv("context_pred.csv",index=False)
        if do_local==False:
            task.upload_artifact("context_predictions", 'context_pred.csv')
            task.close()

        # #save predictions
        # test_qa = pd.read_csv("{}/test_qa.csv".format(data_folder))
        # test_qa['entity'] = test_qa['entity'].apply(lambda x: ast.literal_eval(x))
        # test_qa['text_label'] = test_qa['entity'].apply(lambda x: x['entity'] if x!={} else '')
        # output_df = pd.DataFrame()
        # output_df['docid'] = docid
        # output_df['text'] = text
        # pred_data = pd.DataFrame.from_records(preds)
        # pred_data.columns = target_names
        # pred_df = pd.concat([output_df,pred_data],axis=1)
        # pred_df = pd.melt(pred_df, id_vars=['docid','text'], value_vars=target_names)
        # pred_df.rename(columns={'variable':'qns', 'value':'label'}, inplace=True)
        # question_answering = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        # with open('pred.json', 'w') as outfile:
        #     for id in output_df['docid'].unique():
        #         doc_dict = {}
        #         doc_dict['docid'] = id
        #         doc_dict['gold'] = {}
        #         doc_dict['pred'] = {}
        #         temp_df = pred_df[pred_df['docid']==id].reset_index(drop=True)
        #         temp_label = test_qa[test_qa['docid']==id].reset_index(drop=True)
        #         for qns in target_names:
        #             doc_dict['pred'][qns] = []
        #             doc_dict['gold'][qns] = []
        #         for i in range(len(temp_df)):
        #             if temp_df.loc[i,'label']==1:
        #                 qns = temp_df.loc[i,'qns']
        #                 result = question_answering(question=qns, context=temp_df.loc[i,'text'])
        #                 doc_dict['pred'][qns].append(result['answer'])
        #                 doc_dict['pred'][qns] = list(set(doc_dict['pred'][qns]))
        #                 doc_dict['gold'][qns] += list(set(temp_label.loc[temp_label['question']==qns,'text_label'].tolist()))
        #                 doc_dict['gold'][qns] = list(set(doc_dict['gold'][qns]))
        #             else:
        #                 continue
        #         json.dump(doc_dict, outfile)
        #         outfile.write('\n')
        # if do_local==False:
        #     task.upload_artifact("predictions", 'pred.json')
        #     task.close()
            

    

if __name__ == "__main__":
    main()