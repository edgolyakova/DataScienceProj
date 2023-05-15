# Running instructions

1. Please install all the needed packages from `requirements.txt`.
2. The notebook `Part 0.ipynb` covers the scrapping part: extracting the articles about writers and astronauts and storing them.
3. The notebook `Part 1.ipynb` covers the data analysis of 2 classes: number of tokens per article/sentence, most frequent words and the classification.
4. The notebook `Part 2.ipynb` covers the comparison of Spacy and Stanza: segmentation, tokenization, PoS tagging.

# Folder structure

1. All scrapped article are stored in the corresponding folders, `Writers` or `Astronauts`.
2. `Data` folder contains: 
	- a DataFrame with a random sample of 200 writer and 200 astronaut articles (`df_with_texts.csv`);
	- a DataFrame with a random sample of 400 writer and 400 astronaut articles (`df_with_texts_800.csv`);
	- a DataFrame with a list of all writers with their subcategory (`all_writers_list.csv`);
	- a DataFrame with a list of all astronauts with their subcategory (`all_astronauts_list.csv`).
3. `FrequentWords` contains the following:
	- a DataFrame with top 50 most frequent Spacy tokens by category (Writers, Astronauts) (`top50_freq_word_by_cat.csv`);
	- a DataFrame with top 50 most frequent Spacy tokens by subcategory (e.g. English writers, French writers, etc) (`top50_freq_word_by_subcat.csv`);
	- a folder `Wordclouds` contaning jpg images of wordclouds for Writers and Astronauts categories.
4. `SpacyStanza` folder contains pickle and csv files of previous tokenization runs which allows you not to re-run the whole tokenization process.
5. `TokenizedDF` contains the tokenization results for the sample from  `df_with_texts.csv`.
6. `util.py` contains helper functions (e.g. tokenization, pre-cleaning, etc) that are re-used in the notebooks.
