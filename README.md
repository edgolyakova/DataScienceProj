**Running instructions

1. Please install all the needed packages from `requirements.txt`.
2. The notebook `Part 0.ipynb` covers the scrapping part: extracting the articles about writers and astronauts and storing them.
3. The notebook `Part 1.ipynb` covers the data analysis of 2 classes: number of tokens per article/sentence, most frequent words and the classification.
4. The notebook `Part 2.ipynb` covers the comparison of Spacy and Stanza: segmentation, tokenization, PoS tagging.

**Folder structure

1. All scrapped article are stored in the corresponding folders, `Writers` or `Astronauts`.
2. `Data` folder contains: 
	- a file with a random sample of 200 writers and 200 astronaut articles (`df_with_texts.csv`);
	- list of all writers with their subcategory (`all_writers_list.csv`);
	- and a list of all astronauts with their subcategory (`all_astronauts_list.csv`).
3. `FrequentWords` contains the following: