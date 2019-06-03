
from nltk.corpus import wordnet as wn


def main():
    print(wn.synsets('dog')[0].path_similarity(wn.synsets('cat')[0]))    


if __name__=="__main__":
    main()
