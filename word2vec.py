#!/usr/bin/python
# -*- coding: utf-8 -*-

from pyspark.sql.functions import *
from pyspark.ml.feature import Word2Vec
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql import SQLContext, Row
from pyspark import SparkConf, SparkContext
import sys
import re


class Word2VecApp:

    def main(
        self,
        sc,
        sqlContext,
        input_file,
        output_file,
        synonyms_ouput_path
        ):
        fileRDD = sc.wholeTextFiles(input_file)
        self.documentDF = fileRDD.map(lambda x: (x[0].split('/')[-1],
                re.sub('[^a-z| ]', '',
                x[1].strip().lower()).split())).toDF(['file', 'text'])
        self.documentDF.cache()
        word2vec = Word2Vec(vectorSize= 200, inputCol='text', outputCol='result')
        model = word2vec.fit(self.documentDF)
        result = model.transform(self.documentDF)
        wordListForSynonyms = [
            'chad',
            'stockmann',
            'jimmie',
            'bernick',
            'dicaeopolis',
            'rupert',
            'tesman',
            'barabas',
            'mosby',
			'forster',
            ]
        result.rdd.saveAsTextFile(output_file)
        self.getWordsToFindSynonyms(synonyms_ouput_path)
        for w in wordListForSynonyms:
            synonyms = model.findSynonyms(w, 10)
            print(synonyms)
            synonyms.rdd.saveAsTextFile("/bigd21/synonyms/"+w)

    def getWordsToFindSynonyms(self, synonyms_ouput_path):
        wordPerDocDF = self.documentDF.select(self.documentDF.file,
                explode(self.documentDF.text).alias('words'))
        tf = wordPerDocDF.groupBy('file', 'words').count().select('file'
                , 'words', col('count').alias('tf'))
        idf = wordPerDocDF.distinct().groupBy('words'
                ).count().select('words', 'count', log10(216
                                 / col('count')).alias('idf'))
        tf_idf_tmp = tf.join(idf, tf.words == idf.words, 'inner')
        tf_idf = tf_idf_tmp.select('file', tf['words'], (col('tf')
                                   * col('idf')).alias('tf_idf'))
        tf_idf.registerTempTable('tf_idf_table')
        topWords = \
            sqlContext.sql('select words,tf_idf from tf_idf_table distinct where tf_idf > 0 ORDER BY tf_idf desc limit 10'
                           )
        topWords.rdd.saveAsTextFile(synonyms_ouput_path)


if __name__ == '__main__':
    conf = SparkConf().setAppName('Part 3: Word2Vec'
                                  ).set('spark.executor.memory', '4g'
            ).set('spark.ui.port', 5400)
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    word2vec_input_file = sys.argv[1]
    word2vec_output_file = sys.argv[2]
    synonyms_ouput_path = sys.argv[3]
    word2Vec = Word2VecApp()
    word2Vec.main(sc, sqlContext, word2vec_input_file,
                  word2vec_output_file, synonyms_ouput_path)

			
