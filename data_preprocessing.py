import csv

import requests
from bs4 import BeautifulSoup

import spacy


##########
# Webscraper für Plenarprotokolle aus dem Bundestag
# Einfach lesbare Form
#
def pp_scraper():
    urls = ['https://www.bundestag.de/resource/blob/804766/cfcc95e747d4cba9d852b370171eb8db/19190-data.xml',
            'https://www.bundestag.de/resource/blob/804430/1e443a03536c32755fe4f72c61ef4d2a/19189-data.xml',
            'https://www.bundestag.de/resource/blob/804180/1f2e5154bc1f7532f5852312d2dc0940/19188-data.xml',
            'https://www.bundestag.de/resource/blob/803022/3f477bb03c4cf4ecfa6f8220bbd5d8d5/19187-data.xml',
            'https://www.bundestag.de/resource/blob/802712/b9b525cdd1c7c6a4283ef18531608355/19186-data.xml',
            'https://www.bundestag.de/resource/blob/802072/0ac793838e2c55c0a524fd67bba4bf6c/19185-data.xml',
            'https://www.bundestag.de/resource/blob/798796/9e06eb24f83aee446c723262b2d80441/19184-data.xml',
            'https://www.bundestag.de/resource/blob/798236/ee037da35ea0afefb372c290bd05fe3e/19183-data.xml',
            'https://www.bundestag.de/resource/blob/797960/3291329b781d9b4c5bdbbeebf7d4e46b/19182-data.xml',
            'https://www.bundestag.de/resource/blob/797962/7b33e13c8e82b98ec4eff0e455f9363c/19181-data.xml',
            'https://www.bundestag.de/resource/blob/796178/c9b9519f9edc060532d381075ec86d3f/19180-data.xml',
            'https://www.bundestag.de/resource/blob/795706/b678f2795499c7dbc778b49f63131950/19179-data.xml',
            'https://www.bundestag.de/resource/blob/795500/9277183d3156feb00570b2e1eb6a86ad/19178-data.xml',
            'https://www.bundestag.de/resource/blob/793652/b0102b7ba8a4e59375641b6e72cdf5ad/19177-data.xml']

    filename = "pp_177_bis_190"

    tag_list = ['kopfdaten', 'sitzungsbeginn', 'rede']
    exclude_tags_list = ['name', 'redner']

    with open(str(filename) + '.txt', 'w', encoding='utf-8') as outfile:
        for url in urls:
            website = requests.get(url)
            soup = BeautifulSoup(website.content, "lxml")

            for ex_tag in soup.find_all(exclude_tags_list):
                ex_tag.decompose()

            tags = soup.find_all(tag_list)
            text = [''.join(s.findAll(text=True)) for s in tags]

            text_len = len(text)

            for item in text:
                print(item, file=outfile)

    print("Fertig! Textlänge: " + str(text_len))


##########
# liest gesammelte Daten aus TXT, schreibt jede Zeile aus TXT in jede row in column 1
# und in Parameter gewähltes Label in column 2 einer neuen CSV
# Labels: 'OTHER' oder 'RACISM'
def cleaned_labeled_data_to_csv(label='OTHER'):
    with open('experiments/more_data/data/Nicht_Rassistische_Saetze_moredata.txt', 'r',
              encoding='utf-8') as ifile:
        labeled_list = []
        for line in ifile:
            line_list = [clean_sentence(line).strip(), label]
            labeled_list.append(line_list)
    with open('experiments/more_data/data/Nicht_Rassistische_Saetze_moredata.csv', 'w',
              newline='',
              encoding='utf-8') as ofile:
        writer = csv.writer(ofile, delimiter=';')
        writer.writerows(labeled_list)


##########
# Zwei CSV Files zusammenfügen
#
#
def merge_csv_files(dir1="data/test_stand_13032021/Rassistische_Saetze_13032021_labeled.csv",
                    dir2="data/test_stand_13032021/Nicht_Rassistische_Saetze_13032021_labeled.csv"):
    file1 = open(dir1, "a")
    file2 = open(dir2, "r")

    for line in file2:
        file1.write(line)

    file1.close()
    file2.close()


##########
# String tokenisieren, lemmatisieren und toLowerCase
#
#
def clean_sentence(snt):

    # uebergebliebene linebreaks entfernen
    no_linebreaks = snt.replace(" ", " ").replace(" ", " ")

    nlp = spacy.load('de_core_news_sm')
    snt_spacy = nlp(no_linebreaks)

    # lemmatisieren
    cleaned = [tok.lemma_ for tok in snt_spacy]
    # Zahlen, Punktion und Stoppwörter entfernen -->  if (not tok.is_digit and not tok.is_punct and not tok.is_stop)

    # to lower case
    normalized = [tok.lower() for tok in cleaned]

    # from list to string
    sentence = ' '.join(normalized)

    return sentence

