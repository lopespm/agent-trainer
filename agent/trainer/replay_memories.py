from collections import namedtuple, deque, OrderedDict
from itertools import islice
import os.path
import cPickle as pickle
import random
import numpy as np
import logging

class ReplayMemory(namedtuple("ReplayMemory", ["initial_state", "action_index", "consequent_reward", "consequent_state"])):
    def __eq__(self, other):
        return isinstance(other, self.__class__) \
               and np.array_equal(self.initial_state, other.initial_state) \
               and self.action_index == other.action_index \
               and self.consequent_reward == other.consequent_reward \
               and np.array_equal(self.consequent_state, other.consequent_state)

class ReplayMemories(object):

    def __init__(self,
                 previous_memories_path,
                 max_current_memories_in_ram,
                 max_previous_memories_per_file=None,
                 prefetch_into_ram=False):
        self.logger = logging.getLogger(__name__)
        self.previous_memories = IndexedItemsArchive(previous_memories_path, max_items_per_file=max_previous_memories_per_file)
        self.current_memories = deque(maxlen=max_current_memories_in_ram)
        self.max_current_memories_in_ram = max_current_memories_in_ram
        self._num_current_memories_not_saved = 0
        if prefetch_into_ram:
            self._prefetch_into_ram()

    def append(self, replay_memory):
        self.current_memories.append(replay_memory)
        self._num_current_memories_not_saved = self._num_current_memories_not_saved + 1

        if self._num_current_memories_not_saved >= self.max_current_memories_in_ram:
            self.save()

    def _prefetch_into_ram(self):
        self.logger.info("Prefetching replay memories")
        num_memories_prefetched = min(len(self.previous_memories), self.max_current_memories_in_ram)
        num_memories_not_prefetched = len(self.previous_memories) - num_memories_prefetched
        for memory_index in xrange(num_memories_not_prefetched, len(self.previous_memories)):
            memory = self.previous_memories.fetch([memory_index])
            self.current_memories.append(memory[0])
        self.logger.info("Prefetch finished")

    def save(self, purge_min_recent_memories_to_keep=None):
        self.logger.info("Starting to save replay memories")
        self.previous_memories.append_and_save(self._current_memories_not_yet_saved(),
                                               purge_min_recent_items_to_keep=purge_min_recent_memories_to_keep)
        self._num_current_memories_not_saved = 0
        self.logger.info("Saved replay memories")

    def sample(self, sample_size, recent_memories_span=None, seed=None):
        """Performant sampling which only accesses the necessary memories.
        This is particulary inportant on very large 'previous_memories' sets which are backed by the file system"""
        random.seed(seed)
        len_all_memories = len(self)
        if recent_memories_span is None:
            random_indexes = random.sample(range(len_all_memories), sample_size)
        else:
            sampling_range = range(len_all_memories - min(recent_memories_span, len_all_memories), len_all_memories)
            random_indexes = random.sample(sampling_range, sample_size)

        current_memories_start_global_index = len_all_memories - len(self.current_memories)
        previous_memories_indexes = [index for index in random_indexes if index < current_memories_start_global_index]
        current_memories_indexes = [index for index in random_indexes if index >= current_memories_start_global_index]
        previous_memories_samples = self.previous_memories.fetch(previous_memories_indexes)
        current_memories_samples = [self.current_memories[index-current_memories_start_global_index] for index in current_memories_indexes]

        return previous_memories_samples + current_memories_samples

    def _current_memories_not_yet_saved(self):
        selection_start = (len(self.current_memories) - self._num_current_memories_not_saved)
        return deque(islice(self.current_memories, selection_start, None))

    def __iter__(self):
        num_current_memories_saved = (len(self.current_memories) - self._num_current_memories_not_saved)
        previous_memories_to_retrieve_len = len(self.previous_memories) - num_current_memories_saved
        previous_memories_iterator = iter(self.previous_memories)
        for i in xrange(previous_memories_to_retrieve_len):
            yield next(previous_memories_iterator)

        for current_memory in self.current_memories:
            yield current_memory

    def __len__(self):
        return self._num_current_memories_not_saved + len(self.previous_memories)


class IndexedItemsArchive(object):

    INDEXES_FILE_SUFFIX = ".index"

    def __init__(self, file_path, max_items_per_file=None):
        self.logger = logging.getLogger(__name__)
        self.file_path = file_path
        self.indexes_path = self.file_path + self.INDEXES_FILE_SUFFIX
        self.max_items_per_file = max_items_per_file
        if os.path.exists(self.indexes_path):
            with open(self.indexes_path, 'rb') as file:
                self.indexes = pickle.load(file)
                self._override_max_items_per_file_if_different_from_last()
        else:
            self.indexes = OrderedDict([(0, [])])

    def _override_max_items_per_file_if_different_from_last(self):
        if len(self.indexes.keys()) > 1:
            first_index_group_values = next(iter(self.indexes.items()))[1]
            last_max_items_per_file = len(first_index_group_values)
            if last_max_items_per_file != self.max_items_per_file:
                self.logger.warn("Max items per file is different from the last session. Will set max_items_per_file to {0} "
                      "to override the current value of {1}".format(last_max_items_per_file, self.max_items_per_file))
                self.max_items_per_file = last_max_items_per_file

    def append_and_save(self, items, purge_min_recent_items_to_keep=None):
        number_of_items_to_store_by_file = self._number_of_items_to_store_by_file(items)
        num_stored_items = 0
        for placement in number_of_items_to_store_by_file:
            index_key = placement[0]
            num_items_to_store = placement[1]
            with open(self._item_file_path(index_key), 'a') as file:
                for i in xrange(num_stored_items, num_items_to_store + num_stored_items):
                    item = items[i]
                    encoded_item = self._encode(item)
                    file.write(encoded_item)
                    encoded_item_size = len(encoded_item)
                    last_index_key, last_index_value = self._last_index()
                    last_index_value_on_current_file = last_index_value if last_index_key == index_key else 0
                    self._add_index(index_key, last_index_value_on_current_file + encoded_item_size)
                    num_stored_items =  num_stored_items + 1

        if purge_min_recent_items_to_keep is not None:
            self._purge_older_items(purge_min_recent_items_to_keep)

        with open(self.indexes_path, 'wb') as file:
            pickle.dump(self.indexes, file, protocol=2)

    def _number_of_items_to_store_by_file(self, items):
        num_items = len(items)
        if self.max_items_per_file is None:
            return [(0, num_items)]
        else:
            result = []

            last_index_key, last_index_value = self._last_index()
            if len(self.indexes[last_index_key]) == self.max_items_per_file:
                initial_index_key = last_index_key + 1
            else:
                initial_index_key = last_index_key

            num_stored_items = 0
            while num_stored_items < num_items:
                vacant_index_entries_current_file = self.max_items_per_file - ((len(self) + num_stored_items) % self.max_items_per_file)
                remaining_items_to_store = num_items - num_stored_items
                num_items_to_store_in_current_file = min(remaining_items_to_store, vacant_index_entries_current_file)
                result.append((initial_index_key + len(result), num_items_to_store_in_current_file))
                num_stored_items = num_stored_items + num_items_to_store_in_current_file

            return result

    def _add_index(self, index_key, index_value):
        if index_key not in self.indexes:
            self.indexes.update({index_key: []})
        self.indexes[index_key].append(index_value)

    def _last_index(self):
        last_index_key = self.indexes.keys()[-1] if self.indexes else 0
        last_index_value = self.indexes[last_index_key][-1] if self.indexes[last_index_key] else 0
        return (last_index_key, last_index_value)

    def _item_file_path(self, index_key):
        return self.file_path + str(index_key)

    def fetch(self, index_positions):
        result = []
        indexes_grouped_by_file = self._indexes_grouped_by_file(index_positions)

        for index_key in indexes_grouped_by_file:
            with open(self._item_file_path(index_key), 'rb') as file:
                for item_pointer in indexes_grouped_by_file[index_key]:
                    start_byte_offset = item_pointer[0]
                    size_encoded_item = item_pointer[1]
                    file.seek(start_byte_offset)
                    encoded_item = file.read(size_encoded_item)
                    result.append(self._decode(encoded_item))
        return result

    def _indexes_grouped_by_file(self, index_global_positions):
        result_dict = {}
        for index_global_position in sorted(index_global_positions):
            first_index_key = self.indexes.keys()[0]
            index_key = (index_global_position // self.max_items_per_file if self.max_items_per_file else 0) + first_index_key
            index_position_by_file = index_global_position % self.max_items_per_file if self.max_items_per_file else index_global_position

            start_byte_offset = self.indexes[index_key][index_position_by_file - 1] if index_position_by_file > 0 else 0
            end_byte_offset = self.indexes[index_key][index_position_by_file]
            size_encoded_item = end_byte_offset - start_byte_offset

            if index_key not in result_dict:
                result_dict[index_key] = []

            result_dict[index_key].append((start_byte_offset, size_encoded_item))

        return result_dict

    def _purge_older_items(self, min_recent_items_to_keep):
        if self.max_items_per_file is None:
            self.logger.info("All items are stored under one single file, no purge will be made")
        else:
            deletion_boundary = len(self) - min_recent_items_to_keep
            keys_marked_for_deletion = []
            for idx, index_key in enumerate(self.indexes.keys()):
                max_items_until_this_index_key = (self.max_items_per_file * (idx + 1))
                if deletion_boundary > max_items_until_this_index_key:
                    keys_marked_for_deletion.append(index_key)

            for index_key in keys_marked_for_deletion:
                del self.indexes[index_key]
                item_file_path = self._item_file_path(index_key)
                self.logger.warn("Deleting items file {0}".format(item_file_path))
                try:
                    os.remove(item_file_path)
                except Exception:
                    self.logger.exception("Error deleting items file")

    def __iter__(self):
        for index in xrange(len(self)):
            yield self.fetch([index])[0]

    def __len__(self):
        first_index_key = self.indexes.keys()[0]
        last_index_key = self.indexes.keys()[-1]
        last_file_num_indexes = len(self.indexes[last_index_key])
        if self.max_items_per_file is None:
            return last_file_num_indexes
        else:
            num_index_keys = (last_index_key - first_index_key)
            return ((num_index_keys * self.max_items_per_file) + last_file_num_indexes)

    def _encode(self, item):
        return pickle.dumps(item, protocol=2)

    def _decode(self, encoded_item):
        return pickle.loads(encoded_item)


