import os
os.environ['PYOPENGL_PLATFORM'] = "osmesa"
import sys
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(root_dir, 'scenediffuser'))
import gradio as gr
import interface as IF

with gr.Blocks(css='style.css') as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("<p align='center' style='font-size: 1.5em;'>Diffusion-based Generation, Optimization, and Planning in 3D Scenes</p>")
        gr.HTML(value="<img src='file/figures/teaser.png' alt='Teaser' width='710px' height='284px' style='display: block; margin: auto;'>")
        gr.HTML(value="<p align='center' style='font-size: 1.2em; color: #485fc7;'><a href='https://arxiv.org/abs/2301.06015' target='_blank'>arXiv</a> | <a href='https://scenediffuser.github.io/' target='_blank'>Project Page</a> | <a href='https://github.com/scenediffuser/Scene-Diffuser' target='_blank'>Code</a></p>")
        gr.Markdown("<p align='center'><i>\"SceneDiffuser provides a unified model for solving scene-conditioned generation, optimization, and planning.\"</i></p>")

        ## five task
        ## pose generation
        with gr.Tab("Pose Generation"):
            with gr.Row():
                with gr.Column(scale=2):
                    selector1 = gr.Dropdown(choices=['MPH16', 'MPH1Library', 'N0SittingBooth', 'N3OpenArea'], label='Scenes', value='MPH16', interactive=True)
                    with gr.Row():
                        sample1 = gr.Slider(minimum=1, maximum=8, step=1, label='Count', interactive=True, value=1)
                        seed1 = gr.Slider(minimum=0, maximum=2 ** 16, step=1, label='Seed', interactive=True, value=2023)
                    opt1 = gr.Checkbox(label='Optimizer Guidance', interactive=True, value=True)
                    scale1 = gr.Slider(minimum=0.1, maximum=9.9, step=0.1, label='Scale', interactive=True, value=1.1)
                    button1 = gr.Button("Run")
                with gr.Column(scale=3):
                    image1 = gr.Gallery(label="Image [Result]").style(grid=[1], height="50")
                    # model1 = gr.Model3D(clear_color=[255, 255, 255, 255], label="3D Model [Result]")
        input1 = [selector1, sample1, seed1, opt1, scale1]
        button1.click(IF.pose_generation, inputs=input1, outputs=[image1])

        ## motion generation
        # with gr.Tab("Motion Generation"):
        #     with gr.Row():
        #         with gr.Column(scale=2):
        #             selector2 = gr.Dropdown(choices=['MPH16', 'MPH1Library', 'N0SittingBooth', 'N3OpenArea'], label='Scenes', value='MPH16', interactive=True)
        #             with gr.Row():
        #                 sample2 = gr.Slider(minimum=1, maximum=8, step=1, label='Count', interactive=True, value=1)
        #                 seed2 = gr.Slider(minimum=0, maximum=2 ** 16, step=1, label='Seed', interactive=True, value=2023)
        #             with gr.Row():
        #                 withstart = gr.Checkbox(label='With Start', interactive=True, value=False)
        #             opt2 = gr.Checkbox(label='Optimizer Guidance', interactive=True, value=True)
        #             scale_opt2 = gr.Slider(minimum=0.1, maximum=9.9, step=0.1, label='Scale', interactive=True, value=1.1)
        #             button2 = gr.Button("Run")
        #         with gr.Column(scale=3):
        #             image2 = gr.Image(label="Result")
        # input2 = [selector2, sample2, seed2, withstart, opt2, scale_opt2]
        # button2.click(IF.motion_generation, inputs=input2, outputs=image2)
        with gr.Tab("Motion Generation"):
            with gr.Row():
                with gr.Column(scale=2):
                    input2 = [
                        gr.Dropdown(choices=['MPH16', 'MPH1Library', 'N0SittingBooth', 'N3OpenArea'], label='Scenes')
                    ]
                    button2 = gr.Button("Generate")
                    gr.HTML("<p style='font-size: 0.9em; color: #555555;'>Notes: the output results are pre-sampled results. We will deploy a real-time model for this task soon.</p>")
                with gr.Column(scale=3):
                    output2 = gr.Image(label="Result")
        button2.click(IF.motion_generation, inputs=input2, outputs=output2)
        
        ## grasp generation
        with gr.Tab("Grasp Generation"):
            with gr.Row():
                with gr.Column(scale=2):
                    input3 = [
                        gr.Dropdown(choices=['contactdb+apple', 'contactdb+camera', 'contactdb+cylinder_medium', 'contactdb+door_knob', 'contactdb+rubber_duck', 'contactdb+water_bottle', 'ycb+baseball', 'ycb+pear', 'ycb+potted_meat_can', 'ycb+tomato_soup_can'], label='Objects')
                    ]
                    button3 = gr.Button("Run")
                    gr.HTML("<p style='font-size: 0.9em; color: #555555;'>Notes: the output results are pre-sampled results. We will deploy a real-time model for this task soon.</p>")
                with gr.Column(scale=3):
                    output3 = [
                        gr.Model3D(clear_color=[255, 255, 255, 255], label="Result")
                    ]
        button3.click(IF.grasp_generation, inputs=input3, outputs=output3)
        
        ## path planning
        with gr.Tab("Path Planing"):
            with gr.Row():
                with gr.Column(scale=2):
                    selector4 = gr.Dropdown(choices=['scene0603_00', 'scene0621_00', 'scene0626_00', 'scene0634_00', 'scene0637_00', 'scene0640_00', 'scene0641_00', 'scene0645_00', 'scene0653_00', 'scene0667_00', 'scene0672_00', 'scene0673_00', 'scene0678_00', 'scene0694_00', 'scene0698_00'], label='Scenes', value='scene0621_00', interactive=True)
                    mode4 = gr.Radio(choices=['Sampling', 'Planning'], value='Sampling', label='Mode', interactive=True)
                    with gr.Row():
                        sample4 = gr.Slider(minimum=1, maximum=8, step=1, label='Count', interactive=True, value=1)
                        seed4 = gr.Slider(minimum=0, maximum=2 ** 16, step=1, label='Seed', interactive=True, value=2023)
                    with gr.Box():
                        opt4 = gr.Checkbox(label='Optimizer Guidance', interactive=True, value=True)
                        scale_opt4 = gr.Slider(minimum=0.02, maximum=4.98, step=0.02, label='Scale', interactive=True, value=1.0)
                    with gr.Box():
                        pla4 = gr.Checkbox(label='Planner Guidance', interactive=True, value=True)
                        scale_pla4 = gr.Slider(minimum=0.02, maximum=0.98, step=0.02, label='Scale', interactive=True, value=0.2)
                    button4 = gr.Button("Run")
                with gr.Column(scale=3):
                    image4 = gr.Gallery(label="Image [Result]").style(grid=[1], height="50")
                    number4 = gr.Number(label="Steps", precision=0)
                    gr.HTML("<p style='font-size: 0.9em; color: #555555;'>Notes: 1. It may take a long time to do planning in <b>Planning</b> mode. 2. The <span style='color: #cc0000;'>red</span> balls represent the planning result, starting with the lightest red ball and ending with the darkest red ball. The <span style='color: #00cc00;'>green</span> ball indicates the target position.</p>")
        input4 = [selector4, mode4, sample4, seed4, opt4, scale_opt4, pla4, scale_pla4]
        button4.click(IF.path_planning, inputs=input4, outputs=[image4, number4])

        ## arm motion planning
        with gr.Tab("Arm Motion Planning"):
            gr.Markdown('Coming soon!')

demo.launch()
