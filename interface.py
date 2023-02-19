import os
import random
import torch
import hydra
import numpy as np
import zipfile

from typing import Any
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from huggingface_hub import hf_hub_download

from utils.misc import compute_model_dim
from datasets.base import create_dataset
from datasets.misc import collate_fn_general, collate_fn_squeeze_pcd_batch
from models.base import create_model
from models.visualizer import create_visualizer
from models.environment import create_enviroment

def pretrain_pointtrans_weight_path():
    return hf_hub_download('SceneDiffuser/SceneDiffuser', 'weights/POINTTRANS_C_32768/model.pth')

def model_weight_path(task, has_observation=False):
    if task == 'pose_gen':
        return hf_hub_download('SceneDiffuser/SceneDiffuser', 'weights/2022-11-09_11-22-52_PoseGen_ddm4_lr1e-4_ep100/ckpts/model.pth')
    elif task == 'motion_gen' and has_observation == True:
        return hf_hub_download('SceneDiffuser/SceneDiffuser', 'weights//ckpts/model.pth')
    elif task == 'motion_gen' and has_observation == False:
        return hf_hub_download('SceneDiffuser/SceneDiffuser', 'weights//ckpts/model.pth')
    elif task == 'path_planning':
        return hf_hub_download('SceneDiffuser/SceneDiffuser', 'weights/2022-11-25_20-57-28_Path_ddm4_LR1e-4_E100_REL/ckpts/model.pth')
    else:
        raise Exception('Unexcepted task.')

def pose_motion_data_path():
    zip_path = hf_hub_download('SceneDiffuser/SceneDiffuser', 'hf_data/pose_motion.zip')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(zip_path))
    
    rpath = os.path.join(os.path.dirname(zip_path), 'pose_motion')

    return (
        os.path.join(rpath, 'PROXD_temp'),
        os.path.join(rpath, 'models_smplx_v1_1/models/'),
        os.path.join(rpath, 'PROX'),
        os.path.join(rpath, 'PROX/V02_05')
    )

def path_planning_data_path():
    zip_path = hf_hub_download('SceneDiffuser/SceneDiffuser', 'hf_data/path_planning.zip')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(zip_path))
    
    return os.path.join(os.path.dirname(zip_path), 'path_planning')

def load_ckpt(model: torch.nn.Module, path: str) -> None:
    """ load ckpt for current model

    Args:
        model: current model
        path: save path
    """
    assert os.path.exists(path), 'Can\'t find provided ckpt.'

    saved_state_dict = torch.load(path)['model']
    model_state_dict = model.state_dict()

    for key in model_state_dict:
        if key in saved_state_dict:
            model_state_dict[key] = saved_state_dict[key]
        ## model is trained with ddm
        if 'module.'+key in saved_state_dict:
            model_state_dict[key] = saved_state_dict['module.'+key]
    
    model.load_state_dict(model_state_dict)

def _sampling(cfg: DictConfig, scene: str) -> Any:
    ## compute modeling dimension according to task
    cfg.model.d_x = compute_model_dim(cfg.task)
    
    if cfg.gpu is not None:
        device = f'cuda:{cfg.gpu}'
    else:
        device = 'cpu'
    
    dataset = create_dataset(cfg.task.dataset, 'test', cfg.slurm, case_only=True, specific_scene=scene)
    
    if cfg.model.scene_model.name == 'PointTransformer':
        collate_fn = collate_fn_squeeze_pcd_batch
    else:
        collate_fn = collate_fn_general
    
    dataloader = dataset.get_dataloader(
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=True,
    )
    
    ## create model and load ckpt
    model = create_model(cfg, slurm=cfg.slurm, device=device)
    model.to(device=device)
    load_ckpt(model, path=model_weight_path(cfg.task.name, cfg.task.has_observation if 'has_observation' in cfg.task else False))
    
    ## create visualizer and visualize
    visualizer = create_visualizer(cfg.task.visualizer)
    results = visualizer.visualize(model, dataloader)
    return results

def _planning(cfg: DictConfig, scene: str) -> Any:
    ## compute modeling dimension according to task
    cfg.model.d_x = compute_model_dim(cfg.task)
    
    if cfg.gpu is not None:
        device = f'cuda:{cfg.gpu}'
    else:
        device = 'cpu'
    
    dataset = create_dataset(cfg.task.dataset, 'test', cfg.slurm, case_only=True, specific_scene=scene)
    
    if cfg.model.scene_model.name == 'PointTransformer':
        collate_fn = collate_fn_squeeze_pcd_batch
    else:
        collate_fn = collate_fn_general
    
    dataloader = dataset.get_dataloader(
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=True,
    )
    
    ## create model and load ckpt
    model = create_model(cfg, slurm=cfg.slurm, device=device)
    model.to(device=device)
    load_ckpt(model, path=model_weight_path(cfg.task.name, cfg.task.has_observation if 'has_observation' in cfg.task else False))
    
    ## create environment for planning task and run
    env = create_enviroment(cfg.task.env)
    results = env.run(model, dataloader)
    return results


## interface for five task
## real-time model: pose generation, path planning
def pose_generation(scene, count, seed, opt, scale) -> Any:
    scene_model_weight_path = pretrain_pointtrans_weight_path()
    data_dir, smpl_dir, prox_dir, vposer_dir = pose_motion_data_path()
    override_config = [
        "diffuser=ddpm",
        "model=unet",
        f"model.scene_model.pretrained_weights={scene_model_weight_path}",
        "task=pose_gen",
        "task.visualizer.name=PoseGenVisualizerHF",
        f"task.visualizer.ksample={count}",
        f"task.dataset.data_dir={data_dir}",
        f"task.dataset.smpl_dir={smpl_dir}",
        f"task.dataset.prox_dir={prox_dir}",
        f"task.dataset.vposer_dir={vposer_dir}",
    ]

    if opt == True:
        override_config += [
            "optimizer=pose_in_scene",
            "optimizer.scale_type=div_var",
            f"optimizer.scale={scale}",
            "optimizer.vposer=false",
            "optimizer.contact_weight=0.02",
            "optimizer.collision_weight=1.0"
        ]
    
    initialize(config_path="./scenediffuser/configs", version_base=None)
    config = compose(config_name="default", overrides=override_config)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    res = _sampling(config, scene)

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    return res

def motion_generation(scene):
    assert isinstance(scene, str)
    cnt = {
        'MPH1Library': 3,
        'MPH16': 6,
        'N0SittingBooth': 7,
        'N3OpenArea': 5
    }[scene]

    res = f"./results/motion_generation/results/{scene}/{random.randint(0, cnt-1)}.gif"
    if not os.path.exists(res):
        results_path = hf_hub_download('SceneDiffuser/SceneDiffuser', 'results/motion_generation/results.zip')
        os.makedirs('./results/motion_generation/', exist_ok=True)
        with zipfile.ZipFile(results_path, 'r') as zip_ref:
            zip_ref.extractall('./results/motion_generation/')
    
    return res

def grasp_generation(case_id):
    assert isinstance(case_id, str)
    res = f"./results/grasp_generation/results/{case_id}/{random.randint(0, 19)}.glb"
    if not os.path.exists(res):
        results_path = hf_hub_download('SceneDiffuser/SceneDiffuser', 'results/grasp_generation/results.zip')
        os.makedirs('./results/grasp_generation/', exist_ok=True)
        with zipfile.ZipFile(results_path, 'r') as zip_ref:
            zip_ref.extractall('./results/grasp_generation/')
    
    return res

def path_planning(scene, mode, count, seed, opt, scale_opt, pla, scale_pla):

    scene_model_weight_path = pretrain_pointtrans_weight_path()
    data_dir = path_planning_data_path()

    override_config = [
        "diffuser=ddpm",
        "model=unet",
        "model.use_position_embedding=true",
        f"model.scene_model.pretrained_weights={scene_model_weight_path}",
        "task=path_planning",
        "task.visualizer.name=PathPlanningRenderingVisualizerHF",
        f"task.visualizer.ksample={count}",
        f"task.dataset.data_dir={data_dir}",
        "task.dataset.repr_type=relative",
        "task.env.name=PathPlanningEnvWrapperHF",
        "task.env.inpainting_horizon=16",
        "task.env.robot_top=3.0",
        "task.env.env_adaption=false"
    ]

    if opt == True:
        override_config += [
            "optimizer=path_in_scene",
            "optimizer.scale_type=div_var",
            "optimizer.continuity=false",
            f"optimizer.scale={scale_opt}",
        ]
    if pla == True:
        override_config += [
            "planner=greedy_path_planning",
            f"planner.scale={scale_pla}",
            "planner.scale_type=div_var",
            "planner.greedy_type=all_frame_exp"
        ]
    
    initialize(config_path="./scenediffuser/configs", version_base=None)
    config = compose(config_name="default", overrides=override_config)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if mode == 'Sampling':
        img = _sampling(config, scene)
        res = (img, 0)
    elif mode == 'Planning':
        res = _planning(config, scene)
    else:
        res = (None, 0)

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    return res
