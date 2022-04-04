import pandas as pd
import json
import spacy
import ast
import os

class DataPreprocess:

    def __init__(self,spacy_path):
        self.nlp = spacy.load(spacy_path) #'modules/spacy/en_core_web_sm-3.2.0/en_core_web_sm/en_core_web_sm-3.2.0'

    def get_data(self,data_path):
        data = []
        with open(data_path) as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def convert_to_sentences(self, data, qns_path, output_path, data_type):
        # with open(output_path, mode='w') as outfile:
        #     print("Output file is created at {}".format(output_path))
        with open(qns_path, 'r') as f:
            qns = json.load(f)
        
        template_list = []
        sentence_idx_list = []
        docid_list = []
        sentence_list = []
        label_list = []
        qns_list = []
        entity_full_list = []
        
        for doc in data:
            cur_doc = self.nlp(doc['doctext'])
            sentences = [sent.text for sent in cur_doc.sents]
            for template in doc['templates']:
                event = template['incident_type']
                for key in template.keys():
                    if key=="incident_type":
                        continue
                    else:
                        new_key = qns[event][key]
                        if template[key]==[]:
                            for i,sent in enumerate(sentences):
                                sent = sent.strip()
                                template_list.append(event)
                                docid_list.append(doc['docid'])
                                entity_full_list.append({})
                                qns_list.append(new_key)
                                sentence_list.append(sent)
                                sentence_idx_list.append(i)
                                label_list.append(0)
                        else:
                            for partial_ls in template[key]:
                                for partial_ls_2 in partial_ls:
                                    cur_text = partial_ls_2[0]
                                    for i,sent in enumerate(sentences):
                                        sent = sent.strip()
                                        template_list.append(event)
                                        docid_list.append(doc['docid'])
                                        qns_list.append(new_key)
                                        sentence_list.append(sent)
                                        sentence_idx_list.append(i)
                                        if cur_text in sent:
                                            label_list.append(1)
                                            start_idx = sent.find(cur_text.strip())
                                            entity_dict = {}
                                            entity_dict['entity'] = cur_text.strip()
                                            entity_dict['start'] = start_idx
                                            entity_dict['end'] = start_idx + len(cur_text.strip())
                                            entity_full_list.append(entity_dict)
                                        else:
                                            label_list.append(0)
                                            entity_full_list.append({})

        #class output df
        output_df_class = pd.DataFrame()
        output_df_class['docid'] = docid_list
        output_df_class['template'] = template_list
        output_df_class['question'] = qns_list
        output_df_class['sentence_idx'] = sentence_idx_list
        output_df_class['sentence'] = sentence_list
        output_df_class['label'] = label_list
        output_df_class = output_df_class[output_df_class['label']==1].reset_index(drop=True)
        output_df_class = output_df_class.pivot_table(index=['docid','template','sentence_idx','sentence'],columns='question',values='label', aggfunc='max')
        output_df_class.columns.name = None
        output_df_class = output_df_class.reset_index()
        output_df_class = output_df_class.fillna(0)
        if data_type=="train":
            output_path_class = os.path.join(output_path,"train.csv")
            output_df_class.to_csv(output_path_class, index=False)
        elif data_type=="dev":
            output_path_class = os.path.join(output_path,"dev.csv")
            output_df_class.to_csv(output_path_class, index=False)
        else:
            output_path_class = os.path.join(output_path,"test.csv")
            output_df_class.to_csv(output_path_class, index=False)

        #qa output df
        output_df_qa = pd.DataFrame()
        output_df_qa['docid'] = docid_list
        output_df_qa['template'] = template_list
        output_df_qa['question'] = qns_list
        output_df_qa['sentence_idx'] = sentence_idx_list
        output_df_qa['sentence'] = sentence_list
        output_df_qa['entity'] = entity_full_list
        # output_df_qa['entity'] = output_df_qa['entity'].apply(lambda x: ast.literal_eval(x))
        output_df_qa = output_df_qa.reset_index(drop=True)     
        if data_type=="train":
            output_path_qa = os.path.join(output_path,"train_qa.csv")
            output_df_qa.to_csv(output_path_qa, index=False)
        elif data_type=="dev":
            output_path_qa = os.path.join(output_path,"dev_qa.csv")
            output_df_qa.to_csv(output_path_qa, index=False)
        else:
            output_path_qa = os.path.join(output_path,"test_qa.csv")
            output_df_qa.to_csv(output_path_qa, index=False)   


        


if __name__ == "__main__":
    data_obj = DataPreprocess('modules/spacy/en_core_web_sm-3.2.0/en_core_web_sm/en_core_web_sm-3.2.0')
    train, dev, test = data_obj.get_data("data/muc_6_fields/processed/train.json"), data_obj.get_data("data/muc_6_fields/processed/dev.json"), data_obj.get_data("data/muc_6_fields/processed/test.json")
    data_obj.convert_to_sentences(train,"data/muc_6_fields/processed/qns_list.json","data/muc_sentence_6_fields","train")
    data_obj.convert_to_sentences(dev,"data/muc_6_fields/processed/qns_list.json","data/muc_sentence_6_fields","dev")
    data_obj.convert_to_sentences(test,"data/muc_6_fields/processed/qns_list.json","data/muc_sentence_6_fields","test")
                    
