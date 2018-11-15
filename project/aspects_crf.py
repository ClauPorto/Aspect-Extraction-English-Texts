import os
import nltk
import spacy
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

import sklearn_crfsuite
from sklearn_crfsuite import metrics

class AspectsCRF:
    def __init__(self,text,path_train,path_label_test, path_label):
        self.text = text
        self.path_label = path_label
        # self.format_flag = format_flag
        self.train_sents = self.open_train(path_train)
#         print("TRAIN_SENTNCES",self.train_sents)
        self.tokens_text = self.process("text")
        self.tokens_corpus = self.process("corpus")
        
        self.tagged_text = []
        self.tagged_corpus = []

#         print("TOKENS_TEXT",self.tokens_text)
#         print("TOKENS_CORPUS",self.tokens_corpus)
        
        self.candidates_text = self.get_candidates("text")
        self.candidates_corpus = self.get_candidates("corpus")
        
        self.fdist_text = self.get_frequency("text")
        self.fdist_corpus = self.get_frequency("corpus")
        
#         print("CANDIDATES_TEXT",self.candidates_text)
#         print("CANDIDATES_CORPUS",self.candidates_corpus)

#         print("FDIST_TEXT",self.fdist_text)
#         print("FDIST_CORPUS",self.fdist_corpus)

        
        self.X_train = [self.sent2features(self.tokens_corpus[index],index,"corpus") for index in range(len(self.tokens_corpus))]
        self.y_train = [self.sent2labels(s, path_label) for s in self.tokens_corpus]
#         print("XTRAIN",self.X_train)
#         print("YTRAIN",self.y_train)

        self.X_test = [self.sent2features(self.tokens_text[index],index,"text") for index in range(len(self.tokens_text))]
        self.y_test = [self.sent2labels(s, path_label_test) for s in self.tokens_text]
#         print("XTEST",self.X_test)
#         print("YTEST",self.y_test)
        
        self.train_crf()
        self.evaluation()
        
    def process(self,flag):
        sentences = []
        if flag == "text":
            sentences = self.text.split("\n")
        else:
            sentences = self.train_sents
        
        tokens_text = []
        for s in sentences:
            temp = ""
            if flag =="corpus":
                temp = ""
                for x in s:
                    temp += x+" "
            else:
                temp = s
            tokenizer = RegexpTokenizer(r'\w+')
            t = tokenizer.tokenize(temp)
            stopword = set(stopwords.words('english'))
            t = list(filter(lambda x : x.lower() not in stopword,t))
            t =[x.lower() for x in t]
            tokens_text.append(t)
        return tokens_text      
    
        
    def open_train(self,path):
        result = []
        fd = open(path,'r')
        text = fd.read()
        text = text.split("\n")
        fd.close()
        for s in text:
            tokenizer = RegexpTokenizer(r'\w+')
            t = tokenizer.tokenize(s)
            stopword = set(stopwords.words('english'))
            t = list(filter(lambda x : x.lower() not in stopword,t))
            t =[x.lower() for x in t]
            result.append(t)

        return result
        
    def get_candidates(self,flag):
        nlp = spacy.load('en')
        sentences = []
        tagged = []
#         print(self.tokens_text)
        if flag == "text":
            sentences = self.tokens_text
        else: 
            sentences = self.train_sents

        for s in sentences:
            temp = nltk.pos_tag(s)
            tagged.append(temp)
        nouns = []
        nchunks = []
        for s in tagged:
            nouns = nouns + list(filter(lambda x:x[1].__contains__("NN"),s))
#             print(nouns)
            temp = ""
            for x in s:
                temp += x[0] + " "
            
            nchunks = nchunks + list(nlp(temp).noun_chunks)
        nouns = [x[0] for x in nouns]
#         print("NOUNS",nouns)
        nc = []
        for x in nchunks:
            if str(x) not in nouns:
                nc.append(str(x))
        
        candidates = list(set(nouns + nc))
        nc_tagged = []
        for x in list(set(nc)):
            if not x in nouns:
                nc_tagged.append((x,"NN"))
            
            
        if flag == "text":
            self.tagged_text = tagged
            self.tokens_text.append(list(set(nc)))
            self.tagged_text.append(nc_tagged)
        else: 
            self.tagged_corpus = tagged
            self.tokens_corpus.append(list(set(nc)))
            self.tagged_corpus.append(nc_tagged)
        return candidates
    
    def get_frequency(self,flag):
        fdist = {}
        sentences = []
        if flag == "text":
            sentences = self.tokens_text
        else:
            sentences = self.tokens_corpus
        
        for s in sentences:
            for w in s:
                fdist[w.lower()] = 0
        
        for s in sentences:
            t = FreqDist(word.lower() for word in s)
#             print("FRECUENCIAS")
            for x in t.keys():
#                 print("{}->{}".format(x,t.get(x)))
                fdist[x] += t.get(x)
        return fdist
    
                  
    def word2features(self,sentence,index,index_sent,flag):
#         print("EN WORD@FEATURE======================================================")
#         print(index,index_sent,sentence)
        word = sentence[index]
        postag = ""
        fdis = {}

        if flag == "text":
            postag = self.tagged_text[index_sent][index][1]
            freq = self.fdist_text.get(word)
        else:
            postag = self.tagged_corpus[index_sent][index][1]
            freq = self.fdist_corpus.get(word)
        
        if freq is None:
            freq = 0
        
        features = {
            'word.lower()': word.lower(),
            'postag': postag,
            'freq':float(freq)
        }

#         features = {
#             'postag': postag,
#             'freq':float(freq),
# #             'lenght':len(word)
#         }
        return features
    
    def sent2features(self,sent,index_sent,flag):
#         print("SENT",sent)
#         print(len(sent))
#         print(index_sent)
        return [self.word2features(sent, i,index_sent,flag) for i in range(len(sent))]

    def sent2labels(self,sent, path):
        fd = open(path,'r')
        text = fd.read()
        fd.close()
        labels = ["Aspect","NoAspect"]
#         print("SENTENCE",sent)
        
        return [labels[0] if token in text else labels[1] for token in sent]
    

    def train_crf(self):
        self.crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=20,
        all_possible_transitions=False,
        )
        self.X_train.remove([])
        self.y_train.remove([])
        self.X_test.remove([])
        self.y_test.remove([])
        self.crf.fit(self.X_train, self.y_train)
        
    def evaluation(self):
        labels = list(self.crf.classes_)
        y_pred = self.crf.predict(self.X_test)
        a = 0
        for x in range(len(y_pred)):
            for t in range(len(y_pred[x])):
                if y_pred[x][t]=="Aspect":
                    a+=1
        print(a)
        f = metrics.flat_f1_score(self.y_test,y_pred,average='weighted',labels=labels)

        sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
        print(metrics.flat_classification_report(self.y_test, y_pred, labels=sorted_labels, digits=3))
        
        self.aspects = []
        print(y_pred)
        for i_sent in range(len(y_pred)):
            for i_word in range(len(y_pred[i_sent])):
#                 print(len(self.tokens_text[i_sent]),i_word,len(y_pred[i_sent][i_word]))

                if y_pred[i_sent][i_word] == "Aspect":
                    self.aspects.append(self.tokens_text[i_sent][i_word])
        self.aspects = list(set(self.aspects))
        print("ASPECTS:",set(self.aspects))
        
            

def text_txt_extractor(path):
    fd = open(path)
    text = fd.read()
    fd.close()
    return text
# text = text_txt_extractor("./corpus_test.txt")
# path_train = "./corpus_train.txt"
# path_label= "./corpus_test_aspect.txt"
# path_label_train = "./corpus_train_aspect.txt"
# asp = AspectsCRF(text,path_train,path_label, path_label_train,False)

# text = text_txt_extractor("./beer/test.txt")
# path_train = "./train.txt"
# path_label= "./beer/test_label.txt"
# path_label_train = "./corpus_train_aspect.txt"
# asp = AspectsCRF(text,path_train,path_label, path_label_train,False)

