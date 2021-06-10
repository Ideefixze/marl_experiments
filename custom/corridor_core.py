import random

from custom.core import Creature, Grid


class CorridorGrid(Grid):
    def __init__(self, size, top_block, bot_block):
        self.size = size
        self.creatures = []
        self.blocked = set()
        self.top_block = top_block
        self.bot_block = bot_block

    def randomize_position(self, creature):
        if random.random()<=0.5:
            creature.x = 0
        else:
            creature.x = self.size-1

        creature.y = random.randint(self.top_block+1, self.bot_block-1)

    def init(self):
        for i in range(0, self.size):
            self.add_new_blocked((i, self.top_block))
            self.add_new_blocked((i, self.bot_block))

        """
        self.creatures[0].x = 0
        self.creatures[0].y = self.top_block + 1
        self.creatures[1].x = 0
        self.creatures[1].y = self.bot_block - 1

        self.creatures[2].x = self.size-1
        self.creatures[2].y = self.bot_block - 1
        self.creatures[3].x = self.size-1
        self.creatures[3].y = self.top_block + 1
        """
        for creature in self.creatures:
            self.randomize_position(creature)
            while creature.position in self.blocked or creature.position in {(c.x, c.y) for c in self.creatures if c != creature}:
               self.randomize_position(creature)

class CreatureCameleon(Creature):

    def __init__(self, color_list=[(255,0,0), (0,255,0), (0,0,255)]):
        super(CreatureCameleon, self).__init__(color_list[0])
        self.__colors = color_list

    def action(self, choice):
        if super(CreatureCameleon, self).action(choice) == False:
            self.change_color(min(choice - 4, len(self.__colors)-1))

    def change_color(self, color_id):
        self.color = self.__colors[color_id]