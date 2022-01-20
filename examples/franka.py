import argparse

from autolab_core import YamlConfig

from isaacgym import gymapi
from isaacgym_utils.scene import GymScene
from isaacgym_utils.assets import GymFranka, GymBoxAsset
from isaacgym_utils.policy import RandomDeltaJointPolicy
from isaacgym_utils.draw import draw_transforms


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
        draw_transforms(scene, scene.env_idxs, [franka_transform], length=0.2)

    policy = RandomDeltaJointPolicy(franka, 'franka')
    scene.run(policy=policy, custom_draws=custom_draws)
