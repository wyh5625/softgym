from softgym.envs.cloth_pick import ClothPickEnv
import pyflex
import numpy as np
import pandas as pd


# A class defining basic operation on the cloth


Env_Setting = {
    'TshirtPokeZp': {
        'polygon_name': 'tshirt_poke_zp',
        'scene_id': 8,
    }

}

class ClothPickAndDropEnv(ClothPickEnv):
    def __init__(self, model_name="Pants", **kwargs):
        self.polygon_name = Env_Setting[model_name]['polygon_name']
        self.scene_id = Env_Setting[model_name]['scene_id']
        super().__init__(**kwargs)

    ### Part 2. Class functions ###

    def get_default_config(self):
        config = super().get_default_config()
        config['env_idx'] = self.scene_id
        config['ClothSize'] = [0.5, 0.5]
        return config


    def snap_to_center(self, center=[0, 0, 0]):
        corners = self.read_polygon_corners()
        # print(corners)
        # print(pyflex.get_positions())
        mean = np.mean(corners, axis=0)

        mean -= np.array(center)
        # mean[0] = 10
        particle_pos = pyflex.get_positions().reshape(-1, 4)
        # print(particle_pos)
        print("size: ", len(particle_pos))
        particle_pos[:, :3] -= mean
        pyflex.set_positions(particle_pos.flatten())




