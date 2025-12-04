import warp as wp

from skrl import config


tile_dim_0 = wp.constant(config.warp.tile_dim_0)
tile_dim_1 = wp.constant(config.warp.tile_dim_1)
tile_dim_2 = wp.constant(config.warp.tile_dim_2)
block_dim = wp.constant(config.warp.block_dim)

nn_transposed_computation = False
