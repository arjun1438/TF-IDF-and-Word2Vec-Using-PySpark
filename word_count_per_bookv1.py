#!/usr/bin/python
# -*- coding: utf-8 -*-

from pyspark import SparkConf, SparkContext
from operator import add
import re
import sys


class WordCountApp:

    def processFile(self, documentAndContents):
        document = documentAndContents[0].split('/')[-1]
        wordList = re.compile('\w+').findall(documentAndContents[1])
        return map(lambda word: ((str(document), str(word.lower())),
                   1), wordList)

    def main(self, sc, input_file, output_file):
        wholeTextFilesRDD = sc.wholeTextFiles(input_file)
        wholeTextFilesRDD.cache()
        word_count = \
            wholeTextFilesRDD.flatMap(self.processFile).reduceByKey(add, numPartitions=1).sortByKey()
        word_count.saveAsTextFile(output_file)


if __name__ == '__main__':
    conf = SparkConf().setAppName('Part 1: Word Count Per Book'
                                  ).set('spark.executor.memory', '2g').set('spark.ui.port', 6200)
    sc = SparkContext(conf=conf)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    wordCountApp = WordCountApp()
    wordCountApp.main(sc, input_file, output_file)

			
