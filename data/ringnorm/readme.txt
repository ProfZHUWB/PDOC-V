Summarized by Prof. Wenbin Zhu (i@zhuwb.com)
date: 2022 May 17

Files:
    ringnorm.tar.gz     Downloaded on 17 May 2022 from http://www.cs.toronto.edu/~delve/data/ringnorm/desc.html
       Source/gen       The gawk code for generating data, where x_{ij} is rounded to 4th decimial places
       Dataset.data.gz  Generated data
    phpWfYmlu.arff      Downloaded on 17 May 2022 from https://www.openml.org/d/1496
                        The uploader claim the data is originated from http://www.cs.toronto.edu/~delve/data/ringnorm/desc.html
                        However compare to ringnorm.tar.gz#Dataset.data.gz, there are following changes:
                            x_{ij} is rounded to 3rd decimial places then multipled by 1000 and converted into an integer
                            class 0 -> 1
                            class 1 -> 2
    ringnorm.csv        Extracted data from phpWfYmlu.arff and changed class labels as follows by Miss Ying Fu
                            ringnorm.tar.gz   phpWfYmlu.arff   ringnorm.csv
                                class 0             1                1
                                class 1             2               -1

    	
