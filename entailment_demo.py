from allennlp.predictors.predictor import Predictor
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive

#model = PretrainedModel('./esim-elmo-2018.05.17.tar.gz','textual-entailment')
#predictor = model.predictor()


class PretrainedModel:
    """
    A pretrained model is determined by both an archive file
    (representing the trained model)
    and a choice of predictor.
    """
    def __init__(self, archive_file: str, predictor_name: str) -> None:
        self.archive_file = archive_file
        self.predictor_name = predictor_name

    def predictor(self) -> Predictor:
        archive = load_archive(self.archive_file)
        return Predictor.from_archive(archive, self.predictor_name)




if __name__=="__main__":
    model = PretrainedModel('./esim-elmo-2018.05.17.tar.gz','textual-entailment')
    predictor = model.predictor()

    p = "Anna did a lot better than her good friend Lucy on the test because Anna had studied so hard ."
    h = "Anna succeeded because Anna studied hard ."
    score = predictor.predict(hypothesis=h,premise=p)["label_probs"]
    print("SCORE: ",score)
