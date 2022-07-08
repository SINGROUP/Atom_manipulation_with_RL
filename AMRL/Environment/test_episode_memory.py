from AMRL.Environment.episode_memory import Episode_Memory


class TestEpisode_Memory():

    def test_init(self):

       em = Episode_Memory()
       em.memory_init(episode=1)

       assert(em.episode == 1)
