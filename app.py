import os
import gradio as gr
import random
import pickle
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

def pose_generation(scene, count):
    assert isinstance(scene, str)
    results_path = hf_hub_download('SceneDiffuser/SceneDiffuser', 'results/pose_generation/results.pkl')
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    images = [Image.fromarray(results[scene][random.randint(0, 19)]) for i in range(count)]
    return images

def path_planning(case_id):
    assert isinstance(case_id, str)
    results_path = hf_hub_download('SceneDiffuser/SceneDiffuser', 'results/path_planning/results.pkl')
    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    case = results[case_id]
    steps = case['step']
    image = Image.fromarray(case['image'])
    return image, steps

with gr.Blocks() as demo:
    gr.Markdown("# **<p align='center'>Diffusion-based Generation, Optimization, and Planning in 3D Scenes</p>**")
    gr.HTML(value="<img src='file/figures/teaser.png' alt='Teaser' width='710px' height='284px' style='display: block; margin: auto;'>")
    gr.HTML(value="<p align='center' style='font-size: 1.25em; color: #485fc7;'><a href='' target='_blank'>Paper</a> | <a href='' target='_blank'>Project Page</a> | <a href='' target='_blank'>Github</a></p>")
    gr.Markdown("<p align='center'><i>\"SceneDiffuser provides a unified model for solving scene-conditioned generation, optimization, and planning.\"</i></p>")

    ## five task
    ## pose generation
    with gr.Tab("Pose Generation"):
        with gr.Row():
            with gr.Column():
                input1 = [
                    gr.Dropdown(choices=['MPH16', 'MPH1Library', 'N0SittingBooth', 'N3OpenArea'], label='Scenes'),
                    gr.Slider(minimum=1, maximum=4, step=1, label='Count', interactive=True)
                ]
                button1 = gr.Button("Generate")
            with gr.Column():
                output1 = [
                    gr.Gallery(label="Result").style(grid=[1], height="auto")
                ]
    button1.click(pose_generation, inputs=input1, outputs=output1)

    ## motion generation
    with gr.Tab("Motion Generation"):
        gr.Markdown('Coming soon!')
    
    ## grasp generation
    with gr.Tab("Grasp Generation"):
        gr.Markdown('Coming soon!')
    
    ## path planning
    with gr.Tab("Path Planing"):
        with gr.Row():
            with gr.Column():
                input4 = [
                    gr.Dropdown(choices=['scene0603_00_N0pT', 'scene0621_00_cJ4H', 'scene0634_00_48Y3', 'scene0634_00_gIRH', 'scene0637_00_YgjR', 'scene0640_00_BO94', 'scene0641_00_3K6J', 'scene0641_00_KBKx', 'scene0641_00_cb7l', 'scene0645_00_35Hy', 'scene0645_00_47D1', 'scene0645_00_XfLE', 'scene0667_00_DK4F', 'scene0667_00_o7XB', 'scene0667_00_rUMp', 'scene0672_00_U250', 'scene0673_00_Jyw8', 'scene0673_00_u1lJ', 'scene0678_00_QbNL', 'scene0678_00_RrY0', 'scene0678_00_aE1p', 'scene0678_00_hnXu', 'scene0694_00_DgAL', 'scene0694_00_etF5', 'scene0698_00_tT3Q'], label='Scenes'),
                ]
                button4 = gr.Button("Run")
            with gr.Column():
                # output4 = gr.Gallery(label="Result").style(grid=[1], height="auto")
                output4 = [
                    gr.Image(label="Result"),
                    gr.Number(label="Steps", precision=0)
                ]
    button4.click(path_planning, inputs=input4, outputs=output4)

    ## arm motion planning
    with gr.Tab("Arm Motion Planning"):
        gr.Markdown('Coming soon!')
    
demo.launch()