import pandas as pd
import json
import spacy

class DataPreprocess:

    def __init__(self,spacy_path):
        self.nlp = spacy.load(spacy_path) #'modules/spacy/en_core_web_sm-3.2.0/en_core_web_sm/en_core_web_sm-3.2.0'

    def get_data(self,data_path):
        data = []
        with open(data_path) as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def convert_to_sentences(self, data, qns_path, output_path):
        with open(output_path, mode='w') as outfile:
            print("Output file is created at {}".format(output_path))
        with open(qns_path, 'r') as f:
            qns = json.load(f)
        
        template_list = []
        sentence_idx_list = []
        docid_list = []
        sentence_list = []
        label_list = []
        qns_list = []
        
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
                                template_list.append(event)
                                docid_list.append(doc['docid'])
                                qns_list.append(new_key)
                                sentence_list.append(sent.strip())
                                sentence_idx_list.append(i)
                                label_list.append(0)
                        else:
                            for partial_ls in template[key]:
                                for partial_ls_2 in partial_ls:
                                    cur_text = partial_ls_2[0]
                                    for i,sent in enumerate(sentences):
                                        template_list.append(event)
                                        docid_list.append(doc['docid'])
                                        qns_list.append(new_key)
                                        sentence_list.append(sent.strip())
                                        sentence_idx_list.append(i)
                                        if cur_text in sent:
                                            label_list.append(1)
                                        else:
                                            label_list.append(0)

        output_df = pd.DataFrame()
        output_df['docid'] = docid_list
        output_df['template'] = template_list
        output_df['question'] = qns_list
        output_df['sentence_idx'] = sentence_idx_list
        output_df['sentence'] = sentence_list
        output_df['label'] = label_list
        output_df = output_df[output_df['label']==1].reset_index(drop=True)
        output_df = output_df.pivot_table(index=['docid','template','sentence_idx','sentence'],columns='question',values='label', aggfunc='max')
        output_df.columns.name = None
        output_df = output_df.reset_index()
        output_df = output_df.fillna(0)


        output_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    data_obj = DataPreprocess('modules/spacy/en_core_web_sm-3.2.0/en_core_web_sm/en_core_web_sm-3.2.0')
    train, dev, test = data_obj.get_data("data/muc_6_fields/processed/train.json"), data_obj.get_data("data/muc_6_fields/processed/dev.json"), data_obj.get_data("data/muc_6_fields/processed/test.json")
    data_obj.convert_to_sentences(train,"data/muc_6_fields/processed/qns_list.json","data/muc_sentence_6_fields/train.csv")
    data_obj.convert_to_sentences(dev,"data/muc_6_fields/processed/qns_list.json","data/muc_sentence_6_fields/dev.csv")
    data_obj.convert_to_sentences(test,"data/muc_6_fields/processed/qns_list.json","data/muc_sentence_6_fields/test.csv")
                    
