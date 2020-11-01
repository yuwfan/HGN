import json
import sqlite3
import sys
import unicodedata
import re
import pickle

from urllib.parse import unquote
from tqdm import tqdm

#input:
input_file = sys.argv[1]
db_path = sys.argv[2]
#output
output_file = sys.argv[3]

EDGE_XY = re.compile(r'<a href="(.*?)">(.*?)</a>')
def get_edges(sentence):
    #ret = EDGE_XY.findall(sentence)
    ret = EDGE_XY.findall(sentence + '</a>')
    return [(unquote(x), y) for x, y in ret]

def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


class DocDB(object):
    """Sqlite backed document storage.

    Implements get_doc_text(doc_id).
    """

    def __init__(self, db_path):
        self.path = db_path
        self.connection = sqlite3.connect(self.path, check_same_thread=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def path(self):
        """Return the path to the file that backs this database."""
        return self.path

    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def get_doc_ids(self):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

    def _get_doc_key(self, doc_id, key):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT {} FROM documents WHERE id = ?".format(key),
            (normalize(doc_id),)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]

    def get_doc_text(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        return self._get_doc_key(doc_id, 'text')


    def get_doc_sent_num(self, doc_id):
        return int(self._get_doc_key(doc_id, 'sent_num'))

    def get_doc_text_with_links(self, doc_id):
        return self._get_doc_key(doc_id, 'text_with_links')

    def get_doc_ner(self, doc_id):
        return self._get_doc_key(doc_id, 'text_ner')

    def get_doc_url(self, doc_id):
        return self._get_doc_key(doc_id, 'url')

    def get_doc_title(self, doc_id):
        return self._get_doc_key(doc_id, 'title')

doc_db = DocDB(db_path)

# 1. map title to ID
title_to_id = {}
doc_ids = doc_db.get_doc_ids()
for doc_id in doc_ids:
    title = doc_db.get_doc_title(doc_id)

    if title not in title_to_id:
        title_to_id[title] = doc_id

# 2. extract hyperlink and NER
input_data = json.load(open(input_file, 'r'))
output_data = {}
for data in input_data:
    context = dict(data['context'])
    for title in context.keys():
        if title not in title_to_id:
            print("{} not exist in DB".format(title))
        else:
            doc_id = title_to_id[title]
            text_with_links = pickle.loads(doc_db.get_doc_text_with_links(doc_id))
            text_ner = pickle.loads(doc_db.get_doc_ner(doc_id))

            hyperlink_titles, hyperlink_spans = [], []
            hyperlink_paras = []
            for i, sentence in enumerate(text_with_links):
                _lt, _ls, _lp = [], [], []

                t = get_edges(sentence)
                if len(t) > 0:
                    for link_title, mention_entity in t:
                        if link_title in title_to_id:
                            _lt.append(link_title)
                            _ls.append(mention_entity)
                            doc_text = pickle.loads(doc_db.get_doc_text(title_to_id[link_title]))
                            _lp.append(doc_text)

                hyperlink_titles.append(_lt)
                hyperlink_spans.append(_ls)
                hyperlink_paras.append(_lp)

            output_data[title] = {'hyperlink_titles': hyperlink_titles, 
                                  'hyperlink_paras': hyperlink_paras, 
                                  'hyperlink_spans': hyperlink_spans,
                                  'text_ner': text_ner}

json.dump(output_data, open(output_file, 'w'))
