import spacy
from spacy.lang.en import English
from spacy.pipeline import EntityRuler
import csv
import pandas as pd
import re
import json

# provident fund
PF_ACCOUNT = set(['Provident Fund', 'provident fund'])
PF_TERM = set(['balance', 'current' , 'account number', 'contributed', 'interest rate', 'breakdown', 'statement'])
PF_PARTICIPANT = set(['employer'])
#TIME_PERIOD = set(['so far', 'current', 'this year'])

# expense report
EXPENSE_TYPE = set([
    'business expenses', 'travel expenses', 'expenses', 'trip', 
    'work trip', 'out-patient'
])
TIME_PERIOD = re.compile(
    r'\b(?:\d+ (?:days|weeks|months|years) ago|last (?:week|month|year)|'
    r'this (?:week|month|year)|next (?:week|month|year)|\b\d{4}\b|'
    r'from \d{1,2}(?:st|nd|rd|th)? (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|'
    r'Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|'
    r'Nov(?:ember)?|Dec(?:ember)?) to \d{1,2}(?:st|nd|rd|th)? (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|'
    r'Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|'
    r'Nov(?:ember)?|Dec(?:ember)?) \d{4})\b',
    re.IGNORECASE
)
EXPENSE_COUNT = re.compile(
    r'\b(?:last \d+ expenses|first \d+ expenses|top \d+ expenses|'
    r'\d+ recent expenses|recent \d+ expenses|all expenses|total expenses)\b',
    re.IGNORECASE
)
EXPENSE_STATUS = set([
    'status', 'progress', 'approval', 'pending', 'submitted', 
    'completed', 'rejected', 'processed', 'authorized', 'pending approval', 
    'awaiting approval', 'under review', 'reviewed', 'approved'
])
EXPENSE_CATEGORY = set([
    'category', 'type', 'kind', 'group', 'department', 'division', 
    'unit', 'sector', 'segment', 'team', 'branch', 'section', 'unit'
])
TRAVEL_DETAILS = set([
    'travel details', 'departure times', 'arrival times', 'departure', 
    'arrival', 'start time', 'end time', 'check-in', 'check-out', 
    'flight details', 'train details', 'bus details', 'transport details'
])
FINANCIAL_DETAILS = set([
    'financial details', 'total amount', 'net payable amount', 
    'advance amount', 'payment details', 'financial summary', 
    'financial information', 'expense amount'
])
RECEIPT_DETAILS = set([
    'receipt details', 'receipt numbers', 'receipts', 
    'invoice numbers', 'billing details'
])

# Define your entity patterns list
all_patterns = [
    (PF_ACCOUNT, 'PF_ACCOUNT'),
    (PF_TERM, 'PF_TERM'),
    (PF_PARTICIPANT, 'PF_PARTICIPANT'),
    (TIME_PERIOD, 'TIME_PERIOD'),
    (EXPENSE_TYPE, 'EXPENSE_TYPE'),
    (EXPENSE_COUNT, 'EXPENSE_COUNT'),
    (EXPENSE_STATUS, 'EXPENSE_STATUS'),
    (EXPENSE_CATEGORY, 'EXPENSE_CATEGORY'),
    (TRAVEL_DETAILS, 'TRAVEL_DETAILS'),
    (FINANCIAL_DETAILS, 'FINANCIAL_DETAILS'),
    (RECEIPT_DETAILS, 'RECEIPT_DETAILS')
]

class RulerModel():
    def __init__(self, PF_ACCOUNT, PF_TERM, PF_PARTICIPANT, TIME_PERIOD, 
                 EXPENSE_TYPE, EXPENSE_COUNT, EXPENSE_STATUS,
                 EXPENSE_CATEGORY, TRAVEL_DETAILS, FINANCIAL_DETAILS, RECEIPT_DETAILS):
        # self.ruler_model = spacy.blank('en')
        # self.entity_ruler = self.ruler_model.add_pipe('entity_ruler')
        self.nlp = spacy.load('en_core_web_lg')  # Load pre-trained model
        self.entity_ruler = self.nlp.add_pipe('entity_ruler', before='ner')

        total_patterns = []
        
        # Provident Fund patterns
        patterns = self.create_patterns(PF_ACCOUNT, 'PF_ACCOUNT')
        total_patterns.extend(patterns)

        patterns = self.create_patterns(PF_TERM, 'PF_TERM')
        total_patterns.extend(patterns)

        patterns = self.create_patterns(PF_PARTICIPANT, 'PF_PARTICIPANT')
        total_patterns.extend(patterns)

        patterns = self.create_patterns(TIME_PERIOD, 'TIME_PERIOD')
        total_patterns.extend(patterns)

        # Expense patterns
        patterns = self.create_patterns(EXPENSE_TYPE, 'EXPENSE_TYPE')
        total_patterns.extend(patterns)

        patterns = self.create_patterns(EXPENSE_COUNT, 'EXPENSE_COUNT')
        total_patterns.extend(patterns)

        patterns = self.create_patterns(EXPENSE_STATUS, 'EXPENSE_STATUS')
        total_patterns.extend(patterns)

        patterns = self.create_patterns(EXPENSE_CATEGORY, 'EXPENSE_CATEGORY')
        total_patterns.extend(patterns)

        patterns = self.create_patterns(TRAVEL_DETAILS, 'TRAVEL_DETAILS')
        total_patterns.extend(patterns)

        patterns = self.create_patterns(FINANCIAL_DETAILS, 'FINANCIAL_DETAILS')
        total_patterns.extend(patterns)

        patterns = self.create_patterns(RECEIPT_DETAILS, 'RECEIPT_DETAILS')
        total_patterns.extend(patterns)

        self.add_patterns_into_ruler(total_patterns)

        self.save_ruler_model()

    def create_patterns(self, entity_type_set, entity_type):
        patterns = []
        if isinstance(entity_type_set, re.Pattern):
            pattern = {'label': entity_type, 'pattern': entity_type_set.pattern}
            patterns.append(pattern)
        else:
            for item in entity_type_set:
                pattern = {'label': entity_type, 'pattern': item}
                patterns.append(pattern)
        return patterns

    def add_patterns_into_ruler(self, total_patterns):
        self.entity_ruler.add_patterns(total_patterns)

    def save_ruler_model(self):
        self.nlp.to_disk('BE')


class GenerateDataset(object):
    def __init__(self, ruler_model):
        self.ruler_model = ruler_model

    def find_entitytypes(self, text, entity_patterns):
        ents = []
        #doc = self.ruler_model.ruler_model(str(text))
        doc = self.ruler_model.nlp(str(text))
        for ent in doc.ents:
            ents.append((ent.start_char, ent.end_char, ent.label_))
        return ents

    def assign_labels_to_documents(self, df, entity_patterns):
        dataset = []
        text_list = df['Questions'].values.tolist()
        for text in text_list:
            ents = self.find_entitytypes(text, entity_patterns)
            if len(ents) > 0:
                dataset.append((text, {'entities': ents}))
            else:
                continue
        return dataset

def Annotate():
    df=pd.read_csv('BE\For_annotation.csv',encoding = "ISO-8859-1")
    rulerModel = RulerModel(PF_ACCOUNT, PF_TERM, PF_PARTICIPANT, TIME_PERIOD,
                        EXPENSE_TYPE, EXPENSE_COUNT, EXPENSE_STATUS,
                        EXPENSE_CATEGORY, TRAVEL_DETAILS, FINANCIAL_DETAILS, RECEIPT_DETAILS)
    generateDataset = GenerateDataset(rulerModel)
    annotated_data = generateDataset.assign_labels_to_documents(df, all_patterns)
    with open(r'BE\annoted_data.json', 'w') as fp:
        for item in annotated_data:
            # write each item on a new line
            json.dump(item, fp)
        print("Done writing JSON data into .json file")
    print("done annotation")
    return annotated_data