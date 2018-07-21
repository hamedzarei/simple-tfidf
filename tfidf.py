from sklearn.feature_extraction.text import CountVectorizer
import operator, numpy, os
from sklearn.feature_extraction.text import TfidfTransformer


titles = [
    "صادرات بنزین جمهوری اسلامی",
    "سرمربی تیم ملی ایران افشین قطبی",
    "پژوهشگر و جهادگر سلولهای بنیادی",
    "دین و علم در کلام شهردار تهران",
    "پژوهشگر جهادگر سلولهای بنیادی",
    "دین علم کلام شهردار تهران",
    "پژوهشگر جهادگر سلول بنیادی",
]

def writeListToFile(data, filename):
    try:
        os.remove(filename)
    except OSError:
        pass
    file = open(filename, 'a+')
    # print(data)
    for item in data:
        # print(item[0])
        tmp = ''
        for itm in item:
            tmp += str(itm)
            tmp += '  '
        file.write("%s\n" % tmp)

        
all_files = []
files_name = []
categories = []
for cat in os.listdir("Data"):
    files = os.listdir("Data/"+cat)
    for file in files:
        files_name.append(file)
        with open('Data/'+cat+'/'+file, 'r', encoding="utf-8") as content_file:
            content = content_file.read()
            all_files.append(content)
            categories.append(cat)
            
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(all_files)

transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(X.toarray())

print("tf:")
count = 0
for title in titles:
#     file = open('tf_'+title+'.txt', 'a+')
    t = []
    t = vectorizer.transform([title])
    scores = []
    for idx in range(len(X.toarray())):
        doc = numpy.array(X.toarray()[idx])
        scores.append(numpy.dot(doc, t.toarray()[0]))
    valToWrite = [(10000000, title)]
    t = list(zip(scores, files_name, categories))
    v = sorted(t, key=lambda t: t[0], reverse=True)
    valToWrite += v
    writeListToFile(valToWrite, 'tf_'+str(count)+'.txt')
    index, value = max(enumerate(scores), key=operator.itemgetter(1))
    print(title+": "+categories[index]+" value: "+str(value))
    count += 1
print("tfidf: ")
count = 0
for title in titles:
    t = []
    t = vectorizer.transform([title])
    scores = []
    for idx in range(len(tfidf.toarray())):
        doc = numpy.array(tfidf.toarray()[idx])
        scores.append(numpy.dot(doc, t.toarray()[0]))
    valToWrite = [(10000000, title)]
    t = list(zip(scores, files_name, categories))
    v = sorted(t, key=lambda t: t[0], reverse=True)
    valToWrite += v
    writeListToFile(valToWrite, 'tfidf_'+str(count)+'.txt')
    index, value = max(enumerate(scores), key=operator.itemgetter(1))
    print(title+": "+categories[index]+" value: "+str(value))
    count += 1
