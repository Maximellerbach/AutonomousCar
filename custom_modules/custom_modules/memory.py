
class Memory():
    def __init__(self, queue_size=50):
        self.memory = [{}]  # stored batch of memory
        self.queue_size = queue_size

    def append(self, value):
        if self.__len__() >= self.queue_size:
            del self.memory[0]
        self.memory.append(value)

    def __call__(self):
        return self.memory[-1]

    def __sizeof__(self):
        return self.memory.__sizeof__()

    def __setitem__(self, key, value):
        return self.memory[-1].__setitem__(key, value)

    def __getslice__(self, i, j):
        return self.memory.__getslice__(i, j)

    def __getitem__(self, key):
        return self.memory.__getitem__(key)

    def __str__(self):
        return self.memory.__str__()

    def __len__(self):
        return len(self.memory)

    def __iter__(self):
        return iter(self.memory)

    def __eq__(self, other):
        return self.memory.__eq__(other)


if __name__ == "__main__":
    # just some tests
    mem = Memory(2)

    mem['img_path'] = '/images.dzhaduaudz.png'
    assert(mem == [{'img_path': '/images.dzhaduaudz.png'}])

    mem.append({})

    mem[-1]['duzhadjzahdaz'] = 12
    mem['blublu'] = 52
    assert(mem == [{'img_path': '/images.dzhaduaudz.png'},
                   {'duzhadjzahdaz': 12, 'blublu': 52}])

    mem.append({})
    mem['img_path'] = '/images.dzhaduaudz.png'
    assert(mem == [{'duzhadjzahdaz': 12, 'blublu': 52},
                   {'img_path': '/images.dzhaduaudz.png'}])

    assert(mem() == {'img_path': '/images.dzhaduaudz.png'})

