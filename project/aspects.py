# coding: utf8
# from PyPDF2 import PdfFileReader
import os
import nltk
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords



class AspectsSensor:

	def __init__(self,text):
		self.text = text
		self.process_text()
	
	def process_text(self):
		self.sentences = self.text.split(".")
		self.tokens = []
		for s in self.sentences:
			tokenizer = RegexpTokenizer(r'\w+')
			t = tokenizer.tokenize(s)
			stopword = set(stopwords.words('english'))
			t = list(filter(lambda x : x.lower() not in stopword,t))
			self.tokens.append(t)
		# print("TOKENSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS")
		# print(self.tokens)

	def FREQ(self,threshold):
		tagged = []
		nouns = []
		noun_phrases = []
		sorted_fdist = []
		result = []
		for s in self.tokens:
			print(s)
			temp = nltk.pos_tag(s)
			print(temp)
			tagged.append(temp)
			nouns  = nouns + list(filter(lambda x:x[1].__contains__("NN"),temp))
			noun_phrases = noun_phrases + self.get_noun_phrases(s)
		
			fdist = FreqDist(word.lower() for word in s)
			for x in fdist.keys():
				sorted_fdist.append((fdist.get(x),x))
		sorted_fdist.sort()
		

		nouns_r = set([x[0] for x in nouns])
		noun_phrases = set(noun_phrases)
		print("=================================")
		print("NOUNS:",nouns)
		print("NOUNSPHRA:",noun_phrases)
		print("FREQ:",sorted_fdist)
		t = list(filter(lambda x: x[0]>=threshold and x[1] in nouns_r,sorted_fdist))
		print(t)
		t_r = [x[1] for x in t]
		print("T_R",t_r)
		result = t_r + list(noun_phrases)
		print("RESULT",set(result))
		return set(result)

	def get_noun_phrases(self,sentence):
# 		grammar = "NP: {<DT>?<JJ>*<NN>}"
		grammar = "NP: {<DT>?<NN>*}"
		tagged = nltk.pos_tag(sentence)
		cp = nltk.RegexpParser(grammar)
		
		result = cp.parse(tagged)
		noun_phrase = []
		temp = []
		for t in result.subtrees():
			if t.label() == 'NP':
#               print(t)
#               print(type(t))
				temp.append(t.leaves())
			# print(temp)
		for x in temp:
			np = [t[0] for t in x]
			noun_phrase.append(np)
		result = []
		for x in noun_phrase:
			temp =""
			for s in x:
				if x.index(s)==len(x)-1:
					temp+=s
					continue
				temp+=s+" "

			result.append(temp)
		# print(len(np))
#         print(noun_phrase)
		return result


	def read_aspects():
		self.aspect_text = self.text_txt_extractor(self.pathA)
		self.aspects = nltk.word_tokenize(self.aspect_text)
		print(self.aspects)
		
	def get_precision(self,rr,ri):
		if (rr + ri) == 0:
			return 0
		return (rr / (rr + ri)) * 100
   
	def get_recall(self,rr,nr):
#         if (self.rr + self.nr) == 0:
#             return 0
		return (rr / (rr + nr)) * 100

	def f_measurement(self,r,p):
		if self.r==0 or p== 0:
			return 0
		return 2 / ((1/r) + (1/p))

	def get_pair_nouns(self,sentences, terms):
		temp = set()
		for s in sentences:
			for t1 in terms:
				if t1 in s:
					for t2 in terms:
						if t2 in s and s.index(t1) < s.index(t2):
							temp.add(t1 + " " + t2)
		terms = terms.union(list(temp))
		return terms
	
	def get_trio_nouns(self,sentences, terms):
		temp = set()
		for s in sentences:
			for t1 in terms:
				if t1 in s:
					for t2 in terms:
						if t2 in s and s.index(t1) < s.index(t2):
							for t3 in terms:
								if t3 in s and s.index(t2) < s.index(t3):
									temp.add(t1 + " " + t2 + " " + t3)
		terms = terms.union(list(temp))
		return terms
	
	def _q(self,s, t, terms, psupport):
		for t1 in terms:
			if t1 in s and t1.__contains__(t) and psupport[t] < 3:
				return True
		return False
	
	def _q1(self,s, terms):
		for t1 in terms:
			if t1 in s: #Aqui estaba t
				return True
		return False

	def maxPairDistance(self,term, sentence): 
		if len(term)==1:
			return 0
		
		if len(term) < 3 and term[0] in sentence and term[1] in sentence:
			max_len = len(list(filter(lambda x: sentence.index(x) >= sentence.index(term[0])
							 and sentence.index(x) <= sentence.index(term[1]), sentence)))
			return max_len
		elif term[0] in sentence and term[1] in sentence and term[2] in sentence:
			max_len = max(len(list(filter(lambda x:sentence.index(x) >= sentence.index(term[0])
							 and sentence.index(x) <= sentence.index(term[1]), sentence))),
						  max(len(list(filter(lambda x: sentence.index(x) >= sentence.index(term[0])
							 and sentence.index(x) <= sentence.index(term[2]), sentence))),
							 len(list(filter(lambda x: sentence.index(x) >= sentence.index(term[1])
							 and sentence.index(x) <= sentence.index(term[2]), sentence)))))
			return max_len
		else:
			return 0

	def my_contains(self, s, t):
		for i in t:
			if not i in s:
				return False
		return True
		
	def FrecuencyBasedHL(self):
		terms = set()
		psupport = {}
		support = {}
		m = 5 #minima frecuencia
		for s in self.tokens:
			dic_word = nltk.pos_tag(s)
			# print(dic_word)
			nouns = [x[0] for x in dic_word if x[1].__contains__('NN')]
			# print(nouns)
			nps = self.get_noun_phrases(s)
			terms = terms.union(nouns + nps)
			# print(terms)
		# print('184', terms)
		terms = self.get_pair_nouns(self.tokens, terms)
		# print('186', terms)
		terms = self.get_trio_nouns(self.tokens, terms)
		# print(len(terms))
		# print('188', terms)

		for t in terms:
			support[t] = 0

		for s in self.tokens:
			for t in terms:
				if t in s:
					support[t] += 1

		nonCompact = {}
		for t in terms:
			psupport[t] = 0
			nonCompact[t] = 0


		w = 3 #max distance to be nonCompact
		# print('205', terms)
		# print(self.tokens)
		for s in self.tokens:
			for t in terms:
				# psupport[t] = 0
				if self.maxPairDistance(t,s) > w:
					nonCompact[t] += 1
				# print('s', s, 't', t)

				if t in s:
					# print('s', s)
					_ok = len(list(filter(lambda term2: self.my_contains(s, t) and t != term2 and term2.__contains__(t), terms))) <= 0
					# tt = list(filter(lambda x: x.__contains__(t), terms))
					# print('tt', tt)
					psupport[t] = psupport[t] + 1 if _ok else psupport[t]
		# print(psupport)
		l = psupport.items()
		for it in l:
			if it[1] < m:
				terms.remove(it[0])
				# psupport.pop(it[0])
		l1 = [t for t in l]

		for t in l1:
			if not t[0] in terms:
				psupport.pop(t[0])
		c = 3
		p = 2
		# print('226', terms)
		temp = terms.union([])
		for t in temp:
			if nonCompact[t] > c and t in terms:
				terms.remove(t)
			if psupport[t] < p + 1 and t in terms:
				for t1 in temp:
					if t1 in terms and t1 != t:
						if t.__contains__(t1):
							terms.remove(t1)
							psupport.pop(t1)
		l = sorted(psupport.items())
		l = [x[0] for x in l]
		return terms

		# for t in terms:
		# 	for s in self.tokens:
		# 		if self.maxPairDistance(t, s) > w:
		# 			# nonCompact[t] = 0
		# 			nonCompact[t] += 1
		# temp = terms.union([])
		# for t in temp:
		# 	if t in nonCompact.keys() and nonCompact[t] > 1 or self._q(s, t, terms, psupport):
		# 		terms.remove(t)
		# adjs = set()
		# for s in self.tokens:
		# 	if self._q1(s, terms):
		# 		# tok = nltk.word_tokenize(s)
		# 		# dic_word = nltk.pos_tag(tok, tagset='universal')
		# 		dic_word = nltk.pos_tag(s, tagset='universal')
		# 		nouns = [x[1] for x in dic_word if x[1] == 'NOUN']
		# 		_adjs = [x[1] for x in dic_word if x[1] == 'ADJ']
		# 		_min_adj = None
		# 		for t in terms:
		# 			_min = 10**10
		# 			for a in _adjs:
		# 				if len(t) > 1:
		# 					for t1 in t:
		# 						if s.index(a) < s.index(t1) and _min > (s.index(t1) - s.index(a)):
		# 							_min = (s.index(t1) - s.index(a))
		# 							_min_adjpsupport = a
		# 				else:
		# 					if _min > (s.index(t) - s.index(a)):
		# 							_min = (s.index(t) - s.index(a))
		# 							_min_adj = a
		# 		adjs.add(_min_adj)
		# for s in self.tokens:
		# 	for t in terms:
		# 		if t in s:
		# 			for a in adjs:
		# 				if a in s:
		# 					dic_word = nltk.pos_tag(tok, tagset='universal')
		# 					nouns = [x[1] for x in dic_word if x[1] == 'NOUN']
		# 					_adjs = [x[1] for x in dic_word if x[1] == 'ADJ']
		# 					_min_noun = None
		# 					for n in nouns:
		# 						_min = 10**10
		# 						if s.index(a) < s.index(n) and _min > (s.index(n) - s.index(a)):
		# 							_min = (s.index(n) - s.index(a))
		# 							_min_noun = n
		# 					terms.add(_min_noun)
		# l_res = psupport.items()
		# l_res = sorted(l_res)
		# return set(l_res)


# =======================================================================================================
# TESTER
# path = "C:\\Users\\Claudia\\Desktop\\Proyecto Mineria\\Corpus\\sentences\\s1.txt"
# t=TextProcessing(path)
# real_aspects = ["appetizers","salads","steak","pasta","food"]
# result = t.FREQ(5)

# dir = list(os.listdir("./sentences"))
# sentences = list(filter(lambda x:not x.__contains__("asp"),dir))
# # print(sentences)
# aspects = list(filter(lambda x:x.__contains__("asp"),dir))
# # print(len(sentences))
# # print(aspects)
# for x in sentences:
# #     print(x)
#     pathS = "./sentences/"+x
#     temp = x.split(".")
# #     print(temp)
#     pathA="./sentences/"+temp[0]+"asp.txt"
# #     print(pathA)
#     t = TextProcessing(pathS,pathA)
#     print(t.FREQ(10))
#     break
	
	