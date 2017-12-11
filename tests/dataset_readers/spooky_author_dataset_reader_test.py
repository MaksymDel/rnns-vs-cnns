from allennlp.common.testing import AllenNlpTestCase
from spooky_author_identification.dataset_readers import SpookyAuthorsDatasetReader

class TestSpookyAuthorsDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):
        reader = SpookyAuthorsDatasetReader()
        dataset = reader.read('tests/fixtures/spooky_lines.txt')

        instance1 = {"sentence": ["the", "cold", "is", "Merely", "nothing", "."],
                     "author": "EAP"}

        instance2 = {"sentence": ["but", "it", "is", "not", "to", 'this', 'fact', 'that', 'I', 'now',
                     'especially', 'advert', '.'],
                     "author": "EAP"}

        instance3 = {"sentence": ["no", 'father', 'could', 'claim', 'the', 'gratitude', 'of', 'his',
                     'child', 'so', 'completely', 'as', 'I', "should", "deserve", 'theirs', '.'],
                     "author": "MWS"}

        assert len(dataset.instances) == 5
        fields = dataset.instances[0].fields
        assert [t.text for t in fields["sentence"].tokens] == instance1["sentence"]
        assert fields["label"].label == instance1["author"]
        fields = dataset.instances[1].fields
        assert [t.text for t in fields["sentence"].tokens[:13]] == instance2["sentence"]
        assert fields["label"].label == instance2["author"]
        fields = dataset.instances[2].fields
        assert [t.text for t in fields["sentence"].tokens[:17]] == instance3["sentence"]
        assert fields["label"].label == instance3["author"]

        reader = SpookyAuthorsDatasetReader(cnn_paper_dataset = True)
        dataset = reader.read('tests/fixtures/cnn_paper_lines.txt')
        instance1 = {"sentence": ['How', 'did', 'serfdom', 'develop', 'in', 'and', 'then', 'leave', 'Russia', '?'],
                     "author": "0"}
        assert len(dataset.instances) == 2
        fields = dataset.instances[0].fields
        assert [t.text for t in fields["sentence"].tokens] == instance1["sentence"]
        assert fields["label"].label == instance1["author"]