from pyspark.ml.feature import HashingTF as MLHashingTF
from pyspark.ml.feature import IDF as MLIDF
from pyspark.ml.feature import NGram, Tokenizer
from pyspark.sql.functions import split, explode, col
from pyspark.sql import SQLContext, Row
from pyspark import SparkConf, SparkContext
import re
import sys

class TfIdf:

    def main(self, sc, sqlContext, input_file, output_file):
        fileRDD = sc.wholeTextFiles(input_file)
        wordsRDD = fileRDD.map(lambda x: (x[0].split('/')[-1],
                               re.sub('[^a-z| ]', '',
                               x[1].strip().lower())))
        contentRDD = wordsRDD.toDF(['file', 'contents'])
        tokenizer = Tokenizer(inputCol='contents', outputCol='wordList')
        self.wordsDF = tokenizer.transform(contentRDD)
        self.wordsDF.cache()
        self.unigrams_tfidf()
        self.bigrams_tfidf()

    def unigrams_tfidf(self):
        unigram = NGram(n=1, inputCol='wordList', outputCol='unigrams')
        unigramDF = unigram.transform(self.wordsDF)
        unigram_htf = MLHashingTF(inputCol='unigrams',
                                  outputCol='unigrams_tf')
        unigram_tf = unigram_htf.transform(unigramDF)
        unigram_idf = MLIDF(inputCol='unigrams_tf',
                            outputCol='unigrams_tfidf')
        unigrams_tf_idf = \
            unigram_idf.fit(unigram_tf).transform(unigram_tf)
        unigrams_tf_idf.select(col('file'), col('unigrams'),
                               col('unigrams_tfidf'
                               )).rdd.saveAsTextFile(sys.argv[2])

    def bigrams_tfidf(self):
        bigram = NGram(n=2, inputCol='wordList', outputCol='bigrams')
        bigramDF = bigram.transform(self.wordsDF)
        bigram_htf = MLHashingTF(inputCol='bigrams',
                                 outputCol='bigrams_tf')
        bigram_tf = bigram_htf.transform(bigramDF)
        bigram_idf = MLIDF(inputCol='bigrams_tf',
                           outputCol='bigrams_tfidf')
        bigrams_tf_idf = bigram_idf.fit(bigram_tf).transform(bigram_tf)
        bigrams_tf_idf.select(col('file'), col('bigrams'),
                              col('bigrams_tfidf'
                              )).rdd.saveAsTextFile(sys.argv[3])

if __name__ == '__main__':
    conf = SparkConf().setAppName('Part 2: TF-IDF').set('spark.executor.memory', '4g').set('spark.ui.port', '4090')
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    tf_idf = TfIdf()
    tf_idf.main(sc, sqlContext, input_file, output_file)
