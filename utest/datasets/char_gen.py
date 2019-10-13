import sys

from PIL import ImageFont

sys.path.append('/Users/huanghaihun/Documents/CenterNet/src/lib')
from datasets.sample.char_gen import CTNumberDataset
import unittest


class TestStringMethods(unittest.TestCase):

    def test_CTNumberDataset(self):
        font = ImageFont.truetype("../../resources/fonts/Hack-Regular.ttf", 50)
        dataset = CTNumberDataset(font=font)
        a = dataset[0]
        self.assertEqual(len(a), 7)


if __name__ == '__main__':
    unittest.main()
