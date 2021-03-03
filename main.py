from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import casual_tokenize
from nlpia.data.loaders import get_data
import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation as LDiA #生成LDiA分布
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA#LDA分类
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

#
# 获取数据
pd.options.display.width = 120
sms = get_data('sms-spam')
# print(sms)
index = ['sms{}{}'.format(i,'!'*j) for (i,j) in zip(range(len(sms)),sms.spam)]
sms = pd.DataFrame(sms.values,columns = sms.columns,index = index)
sms['spam'] = sms.spam.astype(int)
#生成原始词袋词频向量4837*9232
np.random.seed(42)
counter = CountVectorizer(tokenizer=casual_tokenize)
bow_docs = pd.DataFrame(counter.fit_transform(raw_documents=sms.text).toarray(),index = index)
column_nums,terms = zip(*sorted(zip(counter.vocabulary_.values(),counter.vocabulary_.keys())))
bow_docs.columns = terms
print(sms.loc['sms0'].text)
print(bow_docs.loc['sms0'][bow_docs.loc['sms0']>0].head())
print(bow_docs.head(5))
#LDiA由词袋向量生成主题词矩阵16*9232
ldia = LDiA(n_components=16,learning_method='batch')
ldia = ldia.fit(bow_docs)
print (ldia.components_.shape)
pd.set_option('display.width',75)
columns = ['topic{}'.format(i) for i in range(ldia.n_components)]
components = pd.DataFrame(ldia.components_.T,index=terms,columns=columns)
print(components.round(2).head(5))
#每个主题中的最主要词（排序）
print (components.topic3.sort_values(ascending=False)[:10])
#生成文档主题向量
ldia16_topic_vectors = ldia.transform(bow_docs)
ldia16_topic_vectors = pd.DataFrame(ldia16_topic_vectors,index=index,columns=columns)
print(ldia16_topic_vectors.round().head())


#垃圾分类 LDiA+LDA 0.94
X_train,X_test,y_train,y_test = train_test_split(ldia16_topic_vectors,sms.spam,test_size =0.5,random_state =271828)
lda = LDA(n_components=1)
lda = lda.fit(X_train,y_train)
sms['ldia16_spam'] = lda.predict(ldia16_topic_vectors)
print(round(float(lda.score(X_train,y_train)),2))
print(round(float(lda.score(X_test,y_test)),2))

#垃圾分类tfidf+LDA 0.75

#生成tfidf高维向量4000*9000
tfidf = TfidfVectorizer(tokenizer=casual_tokenize)
tfidf_docs = tfidf.fit_transform(raw_documents=sms.text).toarray()
tfidf_docs = tfidf_docs - tfidf_docs.mean(axis=0)

#分类
X_train,X_test,y_train,y_test = train_test_split(tfidf_docs,sms.spam.values,test_size =0.5,random_state =271828)
lda = LDA(n_components=1)
lda = lda.fit(X_train,y_train)
print(round(float(lda.score(X_train,y_train)),2))
print(round(float(lda.score(X_test,y_test)),2))


#LSA（SVD,PCA）+LDA 0.964
#生成pca主题向量4000*16(由tfidf生成)
pca = PCA(n_components=16)
pca =pca.fit(tfidf_docs)
pca_topicvectors = pca.transform(tfidf_docs)
pca_columns = ['topic{}'.format(i) for i in range(pca.n_components)]
pca_topicvectors = pd.DataFrame(pca_topicvectors,index=index,columns=pca_columns)
print(pca_topicvectors.round(3).head(6))
X_train,X_test,y_train,y_test = train_test_split(pca_topicvectors.values,sms.spam.values,test_size =0.3,random_state =271828)
lda = LDA(n_components=1)
lda = lda.fit(X_train,y_train)
print(round(float(lda.score(X_train,y_train)),3))
print(round(float(lda.score(X_test,y_test)),3))