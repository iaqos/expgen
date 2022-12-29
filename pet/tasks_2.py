# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains the logic for loading training and test data for all tasks.
"""

import csv
import json
import os
import random
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable, Union, Tuple

from datasets import load_dataset

import log
from pet import task_helpers
from pet.utils import InputExample, GenerativeInputExample

logger = log.get_logger('root')


def _shuffle_and_restrict(examples: List[InputExample], num_examples: int, seed: int = 42) -> List[InputExample]:
    """
    Shuffle a list of examples and restrict it to a given maximum size.

    :param examples: the examples to shuffle and restrict
    :param num_examples: the maximum number of examples
    :param seed: the random seed for shuffling
    :return: the first ``num_examples`` elements of the shuffled list
    """
    if 0 < num_examples < len(examples):
        random.Random(seed).shuffle(examples)
        examples = examples[:num_examples]
    return examples


class LimitedExampleList:
    def __init__(self, labels: List[str], max_examples=-1):
        """
        Implementation of a list that stores only a limited amount of examples per label.

        :param labels: the set of all possible labels
        :param max_examples: the maximum number of examples per label. This can either be a fixed number,
               in which case `max_examples` examples are loaded for every label, or a list with the same size as
               `labels`, in which case at most `max_examples[i]` examples are loaded for label `labels[i]`.
        """
        self._labels = labels
        self._examples = []
        self._examples_per_label = defaultdict(int)

        if isinstance(max_examples, list):
            self._max_examples = dict(zip(self._labels, max_examples))
        else:
            self._max_examples = {label: max_examples for label in self._labels}

    def is_full(self):
        """Return `true` iff no more examples can be added to this list"""
        for label in self._labels:
            if self._examples_per_label[label] < self._max_examples[label] or self._max_examples[label] < 0:
                return False
        return True

    def add(self, example: InputExample) -> bool:
        """
        Add a new input example to this list.

        :param example: the example to add
        :returns: `true` iff the example was actually added to the list
        """
        label = example.label
        if self._examples_per_label[label] < self._max_examples[label] or self._max_examples[label] < 0:
            self._examples_per_label[label] += 1
            self._examples.append(example)
            return True
        return False

    def to_list(self):
        return self._examples


class DataProcessor(ABC):
    """
    Abstract class that provides methods for loading training, testing, development and unlabeled examples for a given
    task
    """

    @abstractmethod
    def get_train_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the train set."""
        pass

    @abstractmethod
    def get_dev_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the dev set."""
        pass

    @abstractmethod
    def get_test_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the test set."""
        pass

    @abstractmethod
    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the unlabeled set."""
        pass

    @abstractmethod
    def get_labels(self) -> List[str]:
        """Get the list of labels for this data set."""
        pass


class GenerativeDataProcessor(DataProcessor, ABC):
    @abstractmethod
    def get_train_examples(self, data_dir) -> List[GenerativeInputExample]:
        """Get a collection of `InputExample`s for the train set."""
        pass

    @abstractmethod
    def get_dev_examples(self, data_dir) -> List[GenerativeInputExample]:
        """Get a collection of `InputExample`s for the dev set."""
        pass

    @abstractmethod
    def get_test_examples(self, data_dir) -> List[GenerativeInputExample]:
        """Get a collection of `InputExample`s for the test set."""
        pass

    @abstractmethod
    def get_unlabeled_examples(self, data_dir) -> List[GenerativeInputExample]:
        """Get a collection of `InputExample`s for the unlabeled set."""
        pass

    def get_labels(self) -> List[str]:
        return []


class AeslcProcessor(GenerativeDataProcessor):
    DATASET_NAME = 'aeslc'
    INPUT_NAME = 'email_body'
    OUTPUT_NAME = 'subject_line'

    def __init__(self):
        self.ds = None

    def _init_ds(self, data_dir: str):
        if self.ds is not None:
            return

        if self.DATASET_NAME == 'newsroom':
            meta_info = {'path': 'newsroom', 'data_dir': data_dir, 'cache_dir': os.path.join(data_dir, 'cache')}
        else:
            meta_info = self.DATASET_NAME
        self.ds = self.load_and_split_dataset(meta_info)

    def get_train_examples(self, data_dir) -> List[GenerativeInputExample]:
        self._init_ds(data_dir)
        return self._convert_json_to_examples(self.ds['train'], 'train')

    def get_dev_examples(self, data_dir) -> List[GenerativeInputExample]:
        self._init_ds(data_dir)
        return self._convert_json_to_examples(self.ds['validation'], 'dev')

    def get_test_examples(self, data_dir) -> List[GenerativeInputExample]:
        self._init_ds(data_dir)
        return self._convert_json_to_examples(self.ds['test'], 'test')

    def get_unlabeled_examples(self, data_dir) -> List[GenerativeInputExample]:
        self._init_ds(data_dir)
        return self._convert_json_to_examples(self.ds['train'], 'unlabeled', remove_labels=True)

    def _convert_json_to_examples(self, json, set_type: str, remove_labels=False) -> List[GenerativeInputExample]:
        examples = []
        for idx, (input_text_a, input_text_b, lab_tag, output_text) in enumerate(zip(json[self.INPUT_NAME_A], json[self.INPUT_NAME_B], json[self.LABEL_TAG], json[self.OUTPUT_NAME])):
            if lab_tag == '0' or lab_tag == 0:
                text_a_tmp="'{}' entails '{}'".format(input_text_a,input_text_b)
            elif lab_tag == '1' or lab_tag == 1:
                text_a_tmp="'{}' does not entail '{}'".format(input_text_a,input_text_b)
            elif lab_tag == '2' or lab_tag == 2:
                text_a_tmp="'{}' contradicts '{}'".format(input_text_a,input_text_b)
            else:
                text_a_tmp=input_text_a + '. ' + input_text_b
            examples.append(GenerativeInputExample(
                idx=idx,
                guid=f'{set_type}-{idx}',
                text_a=text_a_tmp,
                output_text=output_text if not remove_labels else None
            ))
        print('GUARDA GLI ESEMPI 0-4', examples[0:4])
        '''with open('examplefile.txt', 'a') as f:
            for ex in examples:
                f.write(str(ex))'''
        return examples
    '''
    def _convert_json_to_examples(self, json, set_type: str, remove_labels=False) -> List[GenerativeInputExample]:
        examples = []
        for idx, (input_text, output_text) in enumerate(zip(json[self.INPUT_NAME], json[self.OUTPUT_NAME])):
            examples.append(GenerativeInputExample(
                idx=idx,
                guid=f'{set_type}-{idx}',
                text_a=input_text,
                output_text=output_text if not remove_labels else None
            ))
        return examples
    '''
    @staticmethod
    def load_and_split_dataset(dataset_name: Union[str, Tuple[str]]):
        if isinstance(dataset_name, tuple):
            ds = load_dataset(*dataset_name)
        elif isinstance(dataset_name, dict):
            ds = load_dataset(**dataset_name)
        else:
            ds = load_dataset(dataset_name)

        if 'test' not in ds and 'validation' not in ds:
            total_len = len(ds['train'])
            ds_first_split = ds['train'].train_test_split(test_size=int(total_len * 0.1), seed=42)
            ds_second_split = ds_first_split['train'].train_test_split(test_size=int(total_len * 0.1), seed=42)
            ds = {
                'train': ds_second_split['train'],
                'validation': ds_second_split['test'],
                'test': ds_first_split['test']
            }
        elif 'validation' not in ds:
            ds_split = ds['train'].train_test_split(test_size=0.1, seed=42)
            ds = {
                'train': ds_split['train'],
                'validation': ds_split['test'],
                'test': ds['test']
            }
        return ds



class CnnDailymailProcessor(AeslcProcessor):
    DATASET_NAME, INPUT_NAME_A, INPUT_NAME_B, OUTPUT_NAME, LABEL_TAG = 'esnli', 'premise', 'hypothesis', 'explanation_1', 'label'


PROCESSORS = {
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "agnews": AgnewsProcessor,
    "yahoo": YahooAnswersProcessor,
    "yelp-polarity": YelpPolarityProcessor,
    "yelp-full": YelpFullProcessor,
    "xstance-de": lambda: XStanceProcessor("de"),
    "xstance-fr": lambda: XStanceProcessor("fr"),
    "xstance": XStanceProcessor,
    "wic": WicProcessor,
    "rte": RteProcessor,
    "cb": CbProcessor,
    "wsc": WscProcessor,
    "boolq": BoolQProcessor,
    "copa": CopaProcessor,
    "multirc": MultiRcProcessor,
    "record": RecordProcessor,
    "ax-g": AxGProcessor,
    "ax-b": AxBProcessor,
    "aeslc": AeslcProcessor,
    "xsum": XSumProcessor,
    "gigaword": GigawordProcessor,
    "reddit-tifu": RedditTifuProcessor,
    "cnn-dailymail": CnnDailymailProcessor,
    "newsroom": NewsroomProcessor,
}  # type: Dict[str,Callable[[],DataProcessor]]

TASK_HELPERS = {
    "wsc": task_helpers.WscTaskHelper,
    "multirc": task_helpers.MultiRcTaskHelper,
    "copa": task_helpers.CopaTaskHelper,
    "record": task_helpers.RecordTaskHelper,
}

METRICS = {
    "cb": ["acc", "f1-macro"],
    "multirc": ["acc", "f1", "em"],
    "aeslc": ["rouge1", "rouge2", "rougeL"],
    "xsum": ["rouge1", "rouge2", "rougeL"],
    "gigaword": ["rouge1", "rouge2", "rougeL"],
    "reddit-tifu": ["rouge1", "rouge2", "rougeL"],
    "cnn-dailymail": ["rouge1", "rouge2", "rougeL"],
    "newsroom": ["rouge1", "rouge2", "rougeL"],
}

DEFAULT_METRICS = ["acc"]

TRAIN_SET = "train"
DEV_SET = "dev"
TEST_SET = "test"
UNLABELED_SET = "unlabeled"

SET_TYPES = [TRAIN_SET, DEV_SET, TEST_SET, UNLABELED_SET]


def load_examples(task, data_dir: str, set_type: str, *_, num_examples: int = None,
                  num_examples_per_label: int = None, seed: int = 42, shuffle: bool = True) -> List[InputExample]:
    """Load examples for a given task."""
    assert (num_examples is not None) ^ (num_examples_per_label is not None), \
        "Exactly one of 'num_examples' and 'num_examples_per_label' must be set."
    assert (not set_type == UNLABELED_SET) or (num_examples is not None), \
        "For unlabeled data, 'num_examples_per_label' is not allowed"

    if num_examples == 0:
        return []

    processor = PROCESSORS[task]()

    ex_str = f"num_examples={num_examples}" if num_examples is not None \
        else f"num_examples_per_label={num_examples_per_label}"
    logger.info(
        f"Creating features from dataset file at {data_dir} ({ex_str}, set_type={set_type})"
    )

    if set_type == DEV_SET:
        examples = processor.get_dev_examples(data_dir)
    elif set_type == TEST_SET:
        examples = processor.get_test_examples(data_dir)
    elif set_type == TRAIN_SET:
        examples = processor.get_train_examples(data_dir)
    elif set_type == UNLABELED_SET:
        examples = processor.get_unlabeled_examples(data_dir)
        for example in examples:
            example.label = processor.get_labels()[0] if processor.get_labels() else None
    else:
        raise ValueError(f"'set_type' must be one of {SET_TYPES}, got '{set_type}' instead")

    logger.info(f"Task processor has returned {len(examples)} {set_type} examples")

    if num_examples is not None:
        if shuffle:
            examples = _shuffle_and_restrict(examples, num_examples, seed)
        else:
            examples = examples[:num_examples]

    elif num_examples_per_label is not None:
        limited_examples = LimitedExampleList(processor.get_labels(), num_examples_per_label)
        for example in examples:
            limited_examples.add(example)
        examples = limited_examples.to_list()

    label_distribution = Counter(example.label for example in examples)
    logger.info(f"Returning {len(examples)} {set_type} examples with label dist.: {list(label_distribution.items())}")

    return examples
