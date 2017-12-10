# pylint: disable=invalid-name,protected-access
from allennlp.common.testing import ModelTestCase


class SpookyAuthorsClassifierTest(ModelTestCase):
    def setUp(self):
        super(SpookyAuthorsClassifierTest, self).setUp()
        self.set_up_model('tests/fixtures/spooky_author_classifier.json',
                          'tests/fixtures/spooky_lines.txt')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)