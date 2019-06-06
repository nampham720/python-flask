from docx import Document
import string

def readFile(doc):
    '''
    Return a string of text
    '''
    doc = Document(doc)
    fullText = []
    translator = str.maketrans('', '', '\t\n')
    for para in doc.paragraphs:
        fullText.append(para.text)
        
    return '\n'.join(fullText).translate(translator)    

class Extract(object):
    def __init__(self, document):
        self.document = document

    def get_info(self):
        doc = self.document 
        doc = readFile(doc)   
        return doc
