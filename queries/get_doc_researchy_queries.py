## Get the documents for each sampled researchy query
import os
import gzip
import json
import csv
import zipfile
from tqdm import tqdm
from datasets import load_dataset

class ClueWeb22Api:
    def __init__(self, cw22id, cw22root_path='/data/datasets/clueweb22/ClueWeb22_B'):
        self.cw22id = cw22id
        self.cw22root_path = cw22root_path

    def get_base_filename_by_id(self, cw22id, cw22root_path, file_type='html'):
        html_path = self.cw22root_path + os.sep + file_type
        id_parts = cw22id.split('-')
        doc = int(id_parts[len(id_parts) - 1])

        language = id_parts[1][:2]
        segment = id_parts[1][:4]
        directory = id_parts[1]
        base_path = html_path + os.sep + language + os.sep + segment + os.sep + directory + os.sep
        base_filename = base_path + id_parts[1] + '-' + id_parts[2]

        print(f"Base filename for {cw22id}: {base_filename}")
        return base_filename

    def get_json_record(self, record_type):
        cw22id = self.cw22id
        cw22root_path = self.cw22root_path
        base_filename = self.get_base_filename_by_id(cw22id, cw22root_path, file_type=record_type)

        json_path = base_filename + '.json.gz'
        offset_path = base_filename + '.offset'

        id_parts = cw22id.split('-')
        doc = int(id_parts[len(id_parts) - 1])

        offset_length = len('{:010d}\n'.format(0, 0))
        with open(json_path, 'rb') as f_json:
            with open(offset_path, 'r') as f_offset:
                f_offset.seek(int(doc) * int(offset_length))
                start_bytes = int(f_offset.read(offset_length).strip())
                end_bytes = int(f_offset.read(offset_length).strip())
                f_json.seek(start_bytes)
                record = f_json.read(end_bytes - start_bytes)
                record = gzip.decompress(record).decode('utf-8')
                return record

    def get_clean_text(self):
        record = self.get_json_record('txt')
        return record

    def get_inlinks(self):
        record = self.get_json_record('inlink')
        return record

    def get_outlinks(self):
        record = self.get_json_record('outlink')
        return record


with open("queries/researchy_queries_sample_clueweb.jsonl", "r", encoding="utf-8") as f:
    sampled_queries = [json.loads(line) for line in f]



# Get the documents in ClueWeb for each sampled query 
for sampled_query in sampled_queries:
    doc_stream = sampled_query["DocStream"]
    for doc in doc_stream:
        clueweb_hash = doc["CluewebURLHash"]
        clueweb_id = doc["CluewebID"]

        # skip if clueweb_id is None or the language is not English
        if clueweb_id is not None and "clueweb22-en" in clueweb_id:
            clueweb_api = ClueWeb22Api(clueweb_id)
            clueweb_document = clueweb_api.get_clean_text()
            entry = json.loads(clueweb_document)
            clean_text = entry["CluewebDocument"]
            doc["CluewebDocument"] = clean_text
        else:
            doc["CluewebDocument"] = None


    
output_path = "queries/researchy_queries_sample_clueweb.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for sampled_query in sampled_queries:
        json.dump(sampled_query, f)
        f.write('\n')
print(f"Saved sampled queries with ClueWeb documents to {output_path}")