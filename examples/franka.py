import argparse

from autolab_core import YamlConfig

from isaacgym import gymapi
from isaacgym_utils.scene import GymScene
from isaacgym_utils.assets import GymFranka, GymBoxAsset
from isaacgym_utils.policy import RandomDeltaJointPolicy
from isaacgym_utils.draw import draw_transforms
from isaacgym_utils.math_utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='cfg/franka.yaml')
    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)

    scene = GymScene(cfg['scene'])
    
    table = GymBoxAsset(scene, **cfg['table']['dims'], 
                        shape_props=cfg['table']['shape_props'], 
                        asset_options=cfg['table']['asset_options']
                        )
    franka = GymFranka(cfg['franka'], scene)

    table_transform = gymapi.Transform(p=gymapi.Vec3(cfg['table']['dims']['sx']/3, 0, cfg['table']['dims']['sz']/2))
    franka_transform = gymapi.Transform(p=gymapi.Vec3(0, 0, cfg['table']['dims']['sz'] + 0.01))
    
    def setup(scene, _):
        scene.add_asset('table', table, table_transform)
        scene.add_asset('franka', franka, franka_transform, collision_filter=1) # avoid self-collision
    scene.setup_all_envs(setup)

    def custom_draws(scene):
        for env_idx in scene.env_idxs:
            transforms = [
                franka.get_base_transform(env_idx, 'franka'),
                franka.get_ee_transform(env_idx, 'franka'),
            ]
            transforms.extend(franka.get_finger_transforms(env_idx, 'franka'))
            transforms.extend(franka.get_links_transforms(env_idx, 'franka'))

            # import IPython; IPython.embed()
            draw_transforms(scene, [env_idx], transforms, length=0.1)

    policy = RandomDeltaJointPolicy(franka, 'franka')
    scene.run(policy=policy, custom_draws=custom_draws)

    # scene.run()

    # GymURDFAsset(rel_path_to_urdf, assets_root=path_to_asset_root)

    


