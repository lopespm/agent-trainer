import os
from unittest import TestCase
from tempdir import TempDir
import numpy as np

from agent.trainer.replay_memories import IndexedItemsArchive, ReplayMemories, ReplayMemory


class TestReplayMemories(TestCase):

    maxDiff = None

    def test_len(self):
        replay_memories = ReplayMemories("non_existing_previous_memories.dat", max_current_memories_in_ram=100)
        replay_memories.append(self._create_replay_memory())
        replay_memories.append(self._create_replay_memory())
        self.assertEquals(len(replay_memories), 2)

    def test_len_loaded_memories(self):
        with TempDir() as temp_directory:
            replay_memories1_file_name = os.path.join(temp_directory, 'replay_memories.dat')
            replay_memories1 = ReplayMemories(replay_memories1_file_name, max_current_memories_in_ram=100)
            replay_memories1.append(self._create_replay_memory(consequent_reward=34.0))
            replay_memories1.append(self._create_replay_memory(consequent_reward=35.0))
            replay_memories1.save()

            replay_memories2 = ReplayMemories(replay_memories1_file_name, max_current_memories_in_ram=100)
            self.assertEquals(len(replay_memories2), 2)

    def test_len_after_appending_to_loaded_memories(self):
        with TempDir() as temp_directory:
            replay_memories1_file_name = os.path.join(temp_directory, 'replay_memories.dat')
            replay_memories1 = ReplayMemories(replay_memories1_file_name, max_current_memories_in_ram=100)
            replay_memories1.append(self._create_replay_memory(consequent_reward=34.0))
            replay_memories1.append(self._create_replay_memory(consequent_reward=35.0))
            replay_memories1.save()

            replay_memories2 = ReplayMemories(replay_memories1_file_name, max_current_memories_in_ram=100)
            replay_memories2.append(self._create_replay_memory(consequent_reward=36.0))
            self.assertEquals(len(replay_memories2), 3)

    def test_retrieval_after_appending_to_loaded_memories(self):
        with TempDir() as temp_directory:
            replay_memories1_file_name = os.path.join(temp_directory, 'replay_memories.dat')
            replay_memories1 = ReplayMemories(replay_memories1_file_name, max_current_memories_in_ram=100)
            replay_memory1 = self._create_replay_memory(action_index=11)
            replay_memory2 = self._create_replay_memory(action_index=22)
            replay_memories1.append(replay_memory1)
            replay_memories1.append(replay_memory2)
            replay_memories1.save()

            replay_memories2 = ReplayMemories(replay_memories1_file_name, max_current_memories_in_ram=100)
            replay_memory3 = self._create_replay_memory(action_index=33)
            replay_memories2.append(replay_memory3)
            expected_replay_memories = [replay_memory1, replay_memory2, replay_memory3]
            self.assertSequenceEqual(list(replay_memories2), expected_replay_memories)

    def test_multiple_saves(self):
        with TempDir() as temp_directory:
            replay_memories1_file_name = os.path.join(temp_directory, 'replay_memories.dat')
            replay_memories1 = ReplayMemories(replay_memories1_file_name, max_current_memories_in_ram=100)
            replay_memory1 = self._create_replay_memory(action_index=11)
            replay_memory2 = self._create_replay_memory(action_index=22)
            replay_memories1.append(replay_memory1)
            replay_memories1.append(replay_memory2)
            replay_memories1.save()
            replay_memories1.save()

            replay_memories2 = ReplayMemories(replay_memories1_file_name, max_current_memories_in_ram=100)
            expected_replay_memories = [replay_memory1, replay_memory2]
            self.assertSequenceEqual(list(replay_memories2), expected_replay_memories)

    def test_sample(self):
        with TempDir() as temp_directory:
            replay_memories_file_name = os.path.join(temp_directory, 'replay_memories.dat')
            replay_memories = ReplayMemories(replay_memories_file_name, max_current_memories_in_ram=100)
            replay_memory1 = self._create_replay_memory(action_index=11)
            replay_memory2 = self._create_replay_memory(action_index=22)
            replay_memory3 = self._create_replay_memory(action_index=32)
            replay_memory4 = self._create_replay_memory(action_index=42)
            replay_memory5 = self._create_replay_memory(action_index=52)
            replay_memory6 = self._create_replay_memory(action_index=62)
            replay_memory7 = self._create_replay_memory(action_index=72)
            replay_memories.append(replay_memory1)
            replay_memories.append(replay_memory2)
            replay_memories.append(replay_memory3)
            replay_memories.append(replay_memory4)
            replay_memories.append(replay_memory5)
            replay_memories.append(replay_memory6)
            replay_memories.append(replay_memory7)

            sampled_replay_memories = replay_memories.sample(5, seed=3)

            expected_replay_memories = [replay_memory2, replay_memory4, replay_memory7, replay_memory3, replay_memory5]
            self.assertItemsEqual(sampled_replay_memories, expected_replay_memories)

    def test_sample_loaded_memories(self):
        with TempDir() as temp_directory:
            replay_memories_file_name = os.path.join(temp_directory, 'replay_memories.dat')
            replay_memories1 = ReplayMemories(replay_memories_file_name, max_current_memories_in_ram=100)
            replay_memory1 = self._create_replay_memory(action_index=11)
            replay_memory2 = self._create_replay_memory(action_index=22)
            replay_memory3 = self._create_replay_memory(action_index=32)
            replay_memory4 = self._create_replay_memory(action_index=42)
            replay_memory5 = self._create_replay_memory(action_index=52)
            replay_memory6 = self._create_replay_memory(action_index=62)
            replay_memory7 = self._create_replay_memory(action_index=72)
            replay_memories1.append(replay_memory1)
            replay_memories1.append(replay_memory2)
            replay_memories1.append(replay_memory3)
            replay_memories1.append(replay_memory4)
            replay_memories1.append(replay_memory5)
            replay_memories1.append(replay_memory6)
            replay_memories1.append(replay_memory7)
            replay_memories1.save()

            replay_memories2 = ReplayMemories(replay_memories_file_name, max_current_memories_in_ram=100)
            sampled_replay_memories = replay_memories2.sample(5, seed=3)

            expected_replay_memories = [replay_memory2, replay_memory4, replay_memory7, replay_memory3, replay_memory5]
            self.assertItemsEqual(sampled_replay_memories, expected_replay_memories)

    def test_sample_after_appending_to_loaded_memories(self):
        with TempDir() as temp_directory:
            replay_memories_file_name = os.path.join(temp_directory, 'replay_memories.dat')
            replay_memories1 = ReplayMemories(replay_memories_file_name, max_current_memories_in_ram=100)
            replay_memory1 = self._create_replay_memory(action_index=11)
            replay_memory2 = self._create_replay_memory(action_index=22)
            replay_memory3 = self._create_replay_memory(action_index=32)
            replay_memory4 = self._create_replay_memory(action_index=42)
            replay_memory5 = self._create_replay_memory(action_index=52)
            replay_memory6 = self._create_replay_memory(action_index=62)
            replay_memory7 = self._create_replay_memory(action_index=72)
            replay_memories1.append(replay_memory1)
            replay_memories1.append(replay_memory2)
            replay_memories1.append(replay_memory3)
            replay_memories1.save()

            replay_memories2 = ReplayMemories(replay_memories_file_name, max_current_memories_in_ram=100)
            replay_memories2.append(replay_memory4)
            replay_memories2.append(replay_memory5)
            replay_memories2.append(replay_memory6)
            replay_memories2.append(replay_memory7)

            sampled_replay_memories = replay_memories2.sample(5, seed=3)

            expected_replay_memories = [replay_memory2, replay_memory4, replay_memory7, replay_memory3, replay_memory5]
            self.assertItemsEqual(sampled_replay_memories, expected_replay_memories)

    def test_sample_recent_memories(self):
        with TempDir() as temp_directory:
            replay_memories_file_name = os.path.join(temp_directory, 'replay_memories.dat')
            replay_memories = ReplayMemories(replay_memories_file_name, max_current_memories_in_ram=100)
            replay_memory1 = self._create_replay_memory(action_index=11)
            replay_memory2 = self._create_replay_memory(action_index=22)
            replay_memory3 = self._create_replay_memory(action_index=32)
            replay_memory4 = self._create_replay_memory(action_index=42)
            replay_memory5 = self._create_replay_memory(action_index=52)
            replay_memory6 = self._create_replay_memory(action_index=62)
            replay_memory7 = self._create_replay_memory(action_index=72)
            replay_memories.append(replay_memory1)
            replay_memories.append(replay_memory2)
            replay_memories.append(replay_memory3)
            replay_memories.append(replay_memory4)
            replay_memories.append(replay_memory5)
            replay_memories.append(replay_memory6)
            replay_memories.append(replay_memory7)

            sampled_replay_memories = replay_memories.sample(2, recent_memories_span=2)

            print(sampled_replay_memories)

            self.assertTrue(replay_memory6 in sampled_replay_memories)
            self.assertTrue(replay_memory7 in sampled_replay_memories)

    def test_sample_recent_memories_loaded_memories(self):
        with TempDir() as temp_directory:
            replay_memories_file_name = os.path.join(temp_directory, 'replay_memories.dat')
            replay_memories1 = ReplayMemories(replay_memories_file_name, max_current_memories_in_ram=100)
            replay_memory1 = self._create_replay_memory(action_index=11)
            replay_memory2 = self._create_replay_memory(action_index=22)
            replay_memory3 = self._create_replay_memory(action_index=32)
            replay_memory4 = self._create_replay_memory(action_index=42)
            replay_memory5 = self._create_replay_memory(action_index=52)
            replay_memory6 = self._create_replay_memory(action_index=62)
            replay_memory7 = self._create_replay_memory(action_index=72)
            replay_memories1.append(replay_memory1)
            replay_memories1.append(replay_memory2)
            replay_memories1.append(replay_memory3)
            replay_memories1.append(replay_memory4)
            replay_memories1.append(replay_memory5)
            replay_memories1.append(replay_memory6)
            replay_memories1.append(replay_memory7)
            replay_memories1.save()

            replay_memories2 = ReplayMemories(replay_memories_file_name, max_current_memories_in_ram=100)
            sampled_replay_memories = replay_memories2.sample(5, recent_memories_span=5)

            self.assertTrue(replay_memory3 in sampled_replay_memories)
            self.assertTrue(replay_memory4 in sampled_replay_memories)
            self.assertTrue(replay_memory5 in sampled_replay_memories)
            self.assertTrue(replay_memory6 in sampled_replay_memories)
            self.assertTrue(replay_memory7 in sampled_replay_memories)

    def test_sample_recent_memories_after_appending_to_loaded_memories(self):
        with TempDir() as temp_directory:
            replay_memories_file_name = os.path.join(temp_directory, 'replay_memories.dat')
            replay_memories1 = ReplayMemories(replay_memories_file_name, max_current_memories_in_ram=100)
            replay_memory1 = self._create_replay_memory(action_index=11)
            replay_memory2 = self._create_replay_memory(action_index=22)
            replay_memory3 = self._create_replay_memory(action_index=32)
            replay_memory4 = self._create_replay_memory(action_index=42)
            replay_memory5 = self._create_replay_memory(action_index=52)
            replay_memory6 = self._create_replay_memory(action_index=62)
            replay_memory7 = self._create_replay_memory(action_index=72)
            replay_memories1.append(replay_memory1)
            replay_memories1.append(replay_memory2)
            replay_memories1.append(replay_memory3)
            replay_memories1.save()

            replay_memories2 = ReplayMemories(replay_memories_file_name, max_current_memories_in_ram=100)
            replay_memories2.append(replay_memory4)
            replay_memories2.append(replay_memory5)
            replay_memories2.append(replay_memory6)
            replay_memories2.append(replay_memory7)

            sampled_replay_memories = replay_memories2.sample(5, recent_memories_span=5)

            self.assertTrue(replay_memory3 in sampled_replay_memories)
            self.assertTrue(replay_memory4 in sampled_replay_memories)
            self.assertTrue(replay_memory5 in sampled_replay_memories)
            self.assertTrue(replay_memory6 in sampled_replay_memories)
            self.assertTrue(replay_memory7 in sampled_replay_memories)

    def test_sample_size_smaller_than_recent_memories(self):
        with TempDir() as temp_directory:
            replay_memories_file_name = os.path.join(temp_directory, 'replay_memories.dat')
            replay_memories1 = ReplayMemories(replay_memories_file_name, max_current_memories_in_ram=100)
            replay_memory1 = self._create_replay_memory(action_index=11)
            replay_memory2 = self._create_replay_memory(action_index=22)
            replay_memory3 = self._create_replay_memory(action_index=32)
            replay_memory4 = self._create_replay_memory(action_index=42)
            replay_memories1.append(replay_memory1)
            replay_memories1.append(replay_memory2)
            replay_memories1.append(replay_memory3)
            replay_memories1.save()

            replay_memories2 = ReplayMemories(replay_memories_file_name, max_current_memories_in_ram=100)
            replay_memories2.append(replay_memory4)

            sampled_replay_memories = replay_memories2.sample(2, recent_memories_span=3, seed=4)

            expected_replay_memories = [replay_memory2, replay_memory4]
            self.assertItemsEqual(sampled_replay_memories, expected_replay_memories)

    def test_save_before_max_current_memories(self):
        with TempDir() as temp_directory:
            replay_memories_file_name = os.path.join(temp_directory, 'replay_memories.dat')
            replay_memories = ReplayMemories(replay_memories_file_name, max_current_memories_in_ram=4)
            replay_memory1 = self._create_replay_memory(action_index=11)
            replay_memory2 = self._create_replay_memory(action_index=22)
            replay_memory3 = self._create_replay_memory(action_index=33)
            replay_memories.append(replay_memory1)
            replay_memories.append(replay_memory2)
            replay_memories.append(replay_memory3)
            replay_memories.save()

            expected_replay_memories = [replay_memory1, replay_memory2, replay_memory3]
            self.assertSequenceEqual(list(replay_memories), expected_replay_memories)

    def test_save_after_max_current_memories(self):
        with TempDir() as temp_directory:
            replay_memories_file_name = os.path.join(temp_directory, 'replay_memories.dat')
            replay_memories = ReplayMemories(replay_memories_file_name, max_current_memories_in_ram=2)
            replay_memory1 = self._create_replay_memory(action_index=11)
            replay_memory2 = self._create_replay_memory(action_index=22)
            replay_memory3 = self._create_replay_memory(action_index=33)
            replay_memories.append(replay_memory1)
            replay_memories.append(replay_memory2)
            replay_memories.append(replay_memory3)
            replay_memories.save()

            expected_replay_memories = [replay_memory1, replay_memory2, replay_memory3]
            self.assertSequenceEqual(list(replay_memories), expected_replay_memories)

    def test_retrieval_when_saved_before_max_current_memories(self):
        with TempDir() as temp_directory:
            replay_memories1_file_name = os.path.join(temp_directory, 'replay_memories1.dat')
            replay_memories1 = ReplayMemories(replay_memories1_file_name, max_current_memories_in_ram=4)
            replay_memory1 = self._create_replay_memory(action_index=11)
            replay_memory2 = self._create_replay_memory(action_index=22)
            replay_memory3 = self._create_replay_memory(action_index=33)
            replay_memories1.append(replay_memory1)
            replay_memories1.append(replay_memory2)
            replay_memories1.append(replay_memory3)
            replay_memories1.save()

            replay_memories2 = ReplayMemories(replay_memories1_file_name, max_current_memories_in_ram=100)
            expected_replay_memories = [replay_memory1, replay_memory2, replay_memory3]
            self.assertSequenceEqual(list(replay_memories2), expected_replay_memories)

    def test_retrieval_when_saved_after_max_current_memories(self):
        with TempDir() as temp_directory:
            replay_memories1_file_name = os.path.join(temp_directory, 'replay_memories1.dat')
            replay_memories1 = ReplayMemories(replay_memories1_file_name, max_current_memories_in_ram=2)
            replay_memory1 = self._create_replay_memory(action_index=11)
            replay_memory2 = self._create_replay_memory(action_index=22)
            replay_memory3 = self._create_replay_memory(action_index=33)
            replay_memories1.append(replay_memory1)
            replay_memories1.append(replay_memory2)
            replay_memories1.append(replay_memory3)
            replay_memories1.save()

            replay_memories2 = ReplayMemories(replay_memories1_file_name, max_current_memories_in_ram=100)
            expected_replay_memories = [replay_memory1, replay_memory2, replay_memory3]
            self.assertSequenceEqual(list(replay_memories2), expected_replay_memories)

    def test_retrieval_when_max_current_memories_not_reached(self):
        with TempDir() as temp_directory:
            replay_memories1_file_name = os.path.join(temp_directory, 'replay_memories.dat')
            replay_memories1 = ReplayMemories(replay_memories1_file_name, max_current_memories_in_ram=4)
            replay_memory1 = self._create_replay_memory(action_index=11)
            replay_memory2 = self._create_replay_memory(action_index=22)
            replay_memory3 = self._create_replay_memory(action_index=33)
            replay_memories1.append(replay_memory1)
            replay_memories1.append(replay_memory2)
            replay_memories1.append(replay_memory3)

            replay_memories2 = ReplayMemories(replay_memories1_file_name, max_current_memories_in_ram=100)
            expected_replay_memories = []
            self.assertSequenceEqual(list(replay_memories2), expected_replay_memories)

    def test_retrieval_when_max_current_memories_is_reached(self):
        with TempDir() as temp_directory:
            replay_memories1_file_name = os.path.join(temp_directory, 'replay_memories.dat')
            replay_memories1 = ReplayMemories(replay_memories1_file_name, max_current_memories_in_ram=3)
            replay_memory1 = self._create_replay_memory(action_index=11)
            replay_memory2 = self._create_replay_memory(action_index=22)
            replay_memory3 = self._create_replay_memory(action_index=33)
            replay_memories1.append(replay_memory1)
            replay_memories1.append(replay_memory2)
            replay_memories1.append(replay_memory3)

            replay_memories2 = ReplayMemories(replay_memories1_file_name, max_current_memories_in_ram=100)
            expected_replay_memories = [replay_memory1, replay_memory2, replay_memory3]
            self.assertSequenceEqual(list(replay_memories2), expected_replay_memories)

    def test_retrieval_when_max_current_memories_is_surpassed(self):
        with TempDir() as temp_directory:
            replay_memories1_file_name = os.path.join(temp_directory, 'replay_memories.dat')
            replay_memories1 = ReplayMemories(replay_memories1_file_name, max_current_memories_in_ram=2)
            replay_memory1 = self._create_replay_memory(action_index=11)
            replay_memory2 = self._create_replay_memory(action_index=22)
            replay_memory3 = self._create_replay_memory(action_index=33)
            replay_memories1.append(replay_memory1)
            replay_memories1.append(replay_memory2)
            replay_memories1.append(replay_memory3)

            replay_memories2 = ReplayMemories(replay_memories1_file_name, max_current_memories_in_ram=100)
            expected_replay_memories = [replay_memory1, replay_memory2]
            self.assertSequenceEqual(list(replay_memories2), expected_replay_memories)

    def test_prefetch(self):
        with TempDir() as temp_directory:
            replay_memories1_file_name = os.path.join(temp_directory, 'replay_memories.dat')
            replay_memories1 = ReplayMemories(replay_memories1_file_name, max_current_memories_in_ram=100)
            replay_memory1 = self._create_replay_memory(action_index=11)
            replay_memory2 = self._create_replay_memory(action_index=22)
            replay_memory3 = self._create_replay_memory(action_index=33)
            replay_memories1.append(replay_memory1)
            replay_memories1.append(replay_memory2)
            replay_memories1.append(replay_memory3)
            replay_memories1.save()

            replay_memories2 = ReplayMemories(replay_memories1_file_name, max_current_memories_in_ram=100, prefetch_into_ram=True)

        # This block should be idented outside the TempDir block, since we want to test that memories are not fetched from disk
        expected_replay_memories = [replay_memory1, replay_memory2, replay_memory3]
        print(list(replay_memories2))
        print(expected_replay_memories)
        self.assertSequenceEqual(list(replay_memories2), expected_replay_memories)

    def test_prefetch_max_memories_in_ram_less_than_total_memories(self):
        with TempDir() as temp_directory:
            replay_memories1_file_name = os.path.join(temp_directory, 'replay_memories.dat')
            replay_memories1 = ReplayMemories(replay_memories1_file_name, max_current_memories_in_ram=100)
            replay_memory1 = self._create_replay_memory(action_index=11)
            replay_memory2 = self._create_replay_memory(action_index=22)
            replay_memory3 = self._create_replay_memory(action_index=33)
            replay_memories1.append(replay_memory1)
            replay_memories1.append(replay_memory2)
            replay_memories1.append(replay_memory3)
            replay_memories1.save()

            replay_memories2 = ReplayMemories(replay_memories1_file_name, max_current_memories_in_ram=2, prefetch_into_ram=True)

            expected_replay_memories = [replay_memory1, replay_memory2, replay_memory3]
            self.assertSequenceEqual(list(replay_memories2), expected_replay_memories)


    def _create_replay_memory(self,
                              initial_state=np.array([0, 1, 2]),
                              action_index=2,
                              consequent_reward=33.0,
                              consequent_state=np.array([3, 4, 5])):
        return ReplayMemory(initial_state=initial_state,
                            action_index=action_index,
                            consequent_reward=consequent_reward,
                            consequent_state=consequent_state)


class TestIndexedItemsArchive(TestCase):

    def test_len_after_append_and_save(self):
        with TempDir() as temp_directory:
            temp_file_name = os.path.join(temp_directory, 'temp_file.name')
            items = IndexedItemsArchive(temp_file_name)
            items.append_and_save(["item0", "item1"])
            self.assertEqual(len(items), 2)

    def test_retrieval_single_item(self):
        with TempDir() as temp_directory:
            temp_file_name = os.path.join(temp_directory, 'temp_file.name')
            items = IndexedItemsArchive(temp_file_name)
            items.append_and_save(["item0", "item1"])
            self.assertEqual(items.fetch([0])[0], "item0")
            self.assertEqual(items.fetch([1])[0], "item1")

    def test_retrieval_multiple_items(self):
        with TempDir() as temp_directory:
            temp_file_name = os.path.join(temp_directory, 'temp_file.name')
            items = IndexedItemsArchive(temp_file_name)
            items.append_and_save(["item0", "item1", "item2", "item3"])
            self.assertEqual(items.fetch([1,3]), ["item1", "item3"])

    def test_retrieval_multiple_items_on_new_session(self):
        with TempDir() as temp_directory:
            temp_file_name = os.path.join(temp_directory, 'temp_file.name')
            items1 = IndexedItemsArchive(temp_file_name)
            items1.append_and_save(["item0", "item1", "item2", "item3"])

            items2 = IndexedItemsArchive(temp_file_name)
            self.assertEqual(items2.fetch([1,3]), ["item1", "item3"])

    def test_retrieval_multiple_items_on_new_session_with_sequential_order(self):
        with TempDir() as temp_directory:
            temp_file_name = os.path.join(temp_directory, 'temp_file.name')
            items1 = IndexedItemsArchive(temp_file_name)
            items1.append_and_save(["item0", "item1", "item2", "item3"])

            items2 = IndexedItemsArchive(temp_file_name)
            self.assertEqual(items2.fetch([0,1,2,3]), ["item0", "item1", "item2", "item3"])

    def test_retrieval_multiple_items_on_new_session_with_non_sequential_order(self):
        with TempDir() as temp_directory:
            temp_file_name = os.path.join(temp_directory, 'temp_file.name')
            items1 = IndexedItemsArchive(temp_file_name)
            items1.append_and_save(["item0", "item1", "item2", "item3"])

            items2 = IndexedItemsArchive(temp_file_name)
            self.assertEqual(items2.fetch([2,0,1,3]), ["item0", "item1", "item2", "item3"])

    def test_len_on_new_session(self):
        with TempDir() as temp_directory:
            temp_file_name = os.path.join(temp_directory, 'temp_file.name')
            items1 = IndexedItemsArchive(temp_file_name)
            items1.append_and_save(["item0", "item1", "item2", "item3"])

            items2 = IndexedItemsArchive(temp_file_name)
            self.assertEqual(len(items2), 4)

    def test_retrieval_complex_items_on_new_session(self):
        with TempDir() as temp_directory:
            temp_file_name = os.path.join(temp_directory, 'temp_file.name')
            items1 = IndexedItemsArchive(temp_file_name)
            items1.append_and_save([np.array([0, 1, 2]), np.array([3, 4, 5]), np.array([6, 7, 8])])

            items2 = IndexedItemsArchive(temp_file_name)
            np.testing.assert_array_equal(items2.fetch([0, 2]), [np.array([0, 1, 2]), np.array([6, 7, 8])])

    def test_len_complex_items_on_new_session(self):
        with TempDir() as temp_directory:
            temp_file_name = os.path.join(temp_directory, 'temp_file.name')
            items1 = IndexedItemsArchive(temp_file_name)
            items1.append_and_save([np.array([0, 1, 2]), np.array([3, 4, 5]), np.array([6, 7, 8])])

            items2 = IndexedItemsArchive(temp_file_name)
            self.assertEqual(len(items2), 3)

    def test_iteration(self):
        with TempDir() as temp_directory:
            temp_file_name = os.path.join(temp_directory, 'temp_file.name')
            items = IndexedItemsArchive(temp_file_name)
            items.append_and_save(["item0", "item1", "item2", "item3"])
            self.assertEqual(list(items), ["item0", "item1", "item2", "item3"])

    def test_items_in_multiple_files(self):
        with TempDir() as temp_directory:
            temp_file_name = os.path.join(temp_directory, 'temp_file.name')
            items = IndexedItemsArchive(temp_file_name, max_items_per_file=2)
            items.append_and_save(["item0", "item1", "item2", "item3", "item4"])
            self.assertEqual(list(items), ["item0", "item1", "item2", "item3", "item4"])

    def test_items_in_multiple_files_on_new_session_with_vacant_positions_on_first_file(self):
        with TempDir() as temp_directory:
            temp_file_name = os.path.join(temp_directory, 'temp_file.name')
            items1 = IndexedItemsArchive(temp_file_name, max_items_per_file=2)
            items1.append_and_save(["item0", "item1", "item2", "item3", "item4"])

            items2 = IndexedItemsArchive(temp_file_name, max_items_per_file=2)
            items2.append_and_save(["item5", "item6"])

            self.assertEqual(list(items2), ["item0", "item1", "item2", "item3", "item4", "item5", "item6"])


    def test_items_in_multiple_files_on_new_session_with_filled_positions_on_first_file(self):
        with TempDir() as temp_directory:
            temp_file_name = os.path.join(temp_directory, 'temp_file.name')
            items1 = IndexedItemsArchive(temp_file_name, max_items_per_file=2)
            items1.append_and_save(["item0", "item1", "item2", "item3"])

            items2 = IndexedItemsArchive(temp_file_name, max_items_per_file=2)
            items2.append_and_save(["item4", "item5"])

            self.assertEqual(list(items2), ["item0", "item1", "item2", "item3", "item4", "item5"])

    def test_items_in_multiple_files_tolerate_different_max_items_in_between_sessions(self):
        with TempDir() as temp_directory:
            temp_file_name = os.path.join(temp_directory, 'temp_file.name')
            items1 = IndexedItemsArchive(temp_file_name, max_items_per_file=2)
            items1.append_and_save(["item0", "item1", "item2", "item3"])

            items2 = IndexedItemsArchive(temp_file_name, max_items_per_file=3)
            items2.append_and_save(["item4", "item5"])

            self.assertEqual(list(items2), ["item0", "item1", "item2", "item3", "item4", "item5"])

    def test_purge_older_items(self):
        with TempDir() as temp_directory:
            temp_file_name = os.path.join(temp_directory, 'temp_file.name')
            items = IndexedItemsArchive(temp_file_name, max_items_per_file=2)

            items.append_and_save(["item0", "item1", "item2", "item3", "item4", "item5", "item6", "item7"],
                                  purge_min_recent_items_to_keep=3)

            self.assertEqual(list(items), ["item4", "item5", "item6", "item7"])

    def test_multiples_purges_older_items(self):
        with TempDir() as temp_directory:
            temp_file_name = os.path.join(temp_directory, 'temp_file.name')
            items = IndexedItemsArchive(temp_file_name, max_items_per_file=2)

            items.append_and_save(["item0", "item1", "item2", "item3", "item4", "item5", "item6", "item7"],
                                  purge_min_recent_items_to_keep=3)
            items.append_and_save(["item8", "item9", "item10"],
                                  purge_min_recent_items_to_keep=4)

            self.assertEqual(list(items), ["item6","item7", "item8", "item9", "item10"])

    def test_purge_older_items_min_recent_items_to_keep_equal_to_len(self):
        with TempDir() as temp_directory:
            temp_file_name = os.path.join(temp_directory, 'temp_file.name')
            items = IndexedItemsArchive(temp_file_name, max_items_per_file=2)

            items.append_and_save(["item0", "item1"], purge_min_recent_items_to_keep=2)

            self.assertEqual(list(items), ["item0", "item1"])

    def test_purge_older_items_min_recent_items_to_keep_greater_then_to_len(self):
        with TempDir() as temp_directory:
            temp_file_name = os.path.join(temp_directory, 'temp_file.name')
            items = IndexedItemsArchive(temp_file_name, max_items_per_file=2)

            items.append_and_save(["item0", "item1"], purge_min_recent_items_to_keep=3)

            self.assertEqual(list(items), ["item0", "item1"])

    def test_purge_older_items_does_not_purge_when_items_stored_single_file(self):
        with TempDir() as temp_directory:
            temp_file_name = os.path.join(temp_directory, 'temp_file.name')
            items = IndexedItemsArchive(temp_file_name)

            items.append_and_save(["item0", "item1", "item2", "item3", "item4", "item5", "item6", "item7"],
                                  purge_min_recent_items_to_keep=2)

            self.assertEqual(list(items), ["item0", "item1", "item2", "item3", "item4", "item5", "item6", "item7"])


    def test_purge_older_items_and_reload(self):
        with TempDir() as temp_directory:
            temp_file_name = os.path.join(temp_directory, 'temp_file.name')
            items1 = IndexedItemsArchive(temp_file_name, max_items_per_file=3)

            items1.append_and_save(["item0", "item1", "item2", "item3", "item4", "item5", "item6", "item7"],
                                   purge_min_recent_items_to_keep=4)
            items2 = IndexedItemsArchive(temp_file_name)

            self.assertEqual(list(items2), ["item3", "item4", "item5", "item6", "item7"])