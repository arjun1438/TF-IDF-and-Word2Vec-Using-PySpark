step1()
{
	echo "Running Step1: word Count per book"
	echo -n "Enter number of executors > "
    read executors_count
	hadoop fs -rm -R /bigd21/hw2_step1_output
	spark-submit --master yarn --num-executors "$executors_count" --executor-cores 8 --executor-memory 8g word_count_per_bookv1.py  /cosc6339_s17/books-longlist /bigd21/hw2_step1_output
	hadoop fs -getmerge /bigd21/hw2_step1_output/* hw2_step1_output.txt
}

step2()
{
	echo "Running Step2: TF_IDF"
	echo -n "Enter number of executors > "
    read executors_count
	hadoop fs -rm -R /bigd21/hw2_step2_unigram_output
	hadoop fs -rm -R /bigd21/hw2_step2_bigram_output
	spark-submit --master yarn --num-executors "$executors_count" --executor-cores 8 --executor-memory 8g tf_idf.py  /cosc6339_s17/books-longlist /bigd21/hw2_step2_unigram_output /bigd21/hw2_step2_bigram_output
	hadoop fs -getmerge /bigd21/hw2_step2_unigram_output/* hw2_step2_unigram_output.txt
	hadoop fs -getmerge /bigd21/hw2_step2_bigram_output/* hw2_step2_bigram_output.txt
}

step3()
{
	echo "Running Step3: Word2Vec and find synonyms"
	echo -n "Enter number of executors > "
    read executors_count
	hadoop fs -rm -R /bigd21/step3_word2vec_model
	hadoop fs -rm -R /bigd21/synonyms_input
	hadoop fs -rm -R /bigd21/synonyms/*
	spark-submit --master yarn --num-executors "$executors_count" --executor-cores 8 --executor-memory 8g word2vec.py  /cosc6339_s17/books-longlist /bigd21/step3_word2vec_model /bigd21/synonyms_input
	hadoop fs -getmerge /bigd21/step3_word2vec_model/* hw2_step3_word2vec_model.txt
	hadoop fs -getmerge /bigd21/synonyms_input/* top10words.txt
}

runAll()
{
    step1
    step2
	step3
}   

PS3='Please enter your choice: '
options=("Step 1" "Step 2" "Step 3" "runAll" "Quit")
select opt in "${options[@]}"
do
    case $opt in
        "Step 1")
            step1
            ;;
        "Step 2")
            step2
            ;;
        "Step 3")
            step3
            ;;
        "runAll")
            runAll
            ;;
        "Quit")
            break
            ;;
        *) echo invalid option;;
    esac
done
