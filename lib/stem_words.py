from nltk.stem import PorterStemmer

from typing import List

# this function stems words to their root form
# ex -> running => run
stemmer = PorterStemmer()  # stemmer class instance from nltk


# actual function responsible for stemming words
# in the list
def stem_words(
    word_list: List[str],
) -> List[str]:
    words: List[str] = []
    for word in word_list:
        if word:
            words.append(
                stemmer.stem(word),
            )

    return words
