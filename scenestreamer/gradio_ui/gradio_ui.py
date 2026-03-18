import pathlib

import gradio as gr
import requests

agent_1_id, agent_1_turn, agent_2_id, agent_2_turn = None, None, None, None


def fetch_next_scene():
    resp = requests.get("http://127.0.0.1:5001/next_scene")
    if resp.status_code == 200:
        print(resp.json())
        return pathlib.Path(f"./{resp.json()['original_image']}")
        # return pathlib.Path(f"./{resp.json()['original_gif']}")
    else:
        raise LookupError()


# Mock function to simulate plotting the new modes based on the form input.
def plot_modes(agent_1_id, agent_1_turn, agent_2_id, agent_2_turn):
    # Replace this with actual logic to generate images.
    value_map = {"STOP": 0, "Go Straight": 1, "Turn Left": 2, "Turn Right": 3, "U-TURN": 4}
    resp = requests.post(
        "http://127.0.0.1:5001/plot",
        json={
            "agent_1_id": agent_1_id,
            "agent_1_turn": value_map[agent_1_turn],
            "agent_2_id": agent_2_id,
            "agent_2_turn": value_map[agent_2_turn]
        }
    )
    if resp.status_code == 200:
        print(resp.json())
        return resp.json()
    raise LookupError()


def update_and_visualize(agent_1_id, agent_1_turn, agent_2_id, agent_2_turn):
    mode_images = plot_modes(agent_1_id, agent_1_turn, agent_2_id, agent_2_turn)
    return [pathlib.Path(f"./{image}") for image in mode_images]


# Creating the Gradio interface
with gr.Blocks() as demo:

    gr.Markdown("# Action-conditioned MotionLM Visualization with Collision and Turn Injection")

    with gr.Row():
        # Button to fetch next scene
        next_scene_btn = gr.Button("Next Scene")

    with gr.Row():
        # Display original scenario image
        original_image = gr.Image(label="Original Scenario")

    with gr.Row():
        # Input fields for agent 1 and agent 2 IDs and turn actions
        agent_1_id = gr.Textbox(label="Enter Agent 1 ID", placeholder="Agent 1 ID")
        agent_1_turn = gr.Dropdown(
            choices=["STOP", "Go Straight", "Turn Left", "Turn Right", "U-TURN"], label="Agent 1 Turn Action"
        )
        agent_2_id = gr.Textbox(label="Enter Agent 2 ID", placeholder="Agent 2 ID")
        agent_2_turn = gr.Dropdown(
            choices=["STOP", "Go Straight", "Turn Left", "Turn Right", "U-TURN"], label="Agent 2 Turn Action"
        )

    with gr.Row():
        # Button to update and visualize decoded modes
        update_btn = gr.Button("Update and Visualize Decoded Modes")

    with gr.Column():
        # Display decoded mode images
        mode_images = [gr.Image(label=f"Mode {i}") for i in range(6)]

    # Define button actions
    next_scene_btn.click(fn=fetch_next_scene, outputs=[original_image])

    update_btn.click(
        fn=update_and_visualize, inputs=[agent_1_id, agent_1_turn, agent_2_id, agent_2_turn], outputs=mode_images
    )

# Launch the Gradio app
demo.launch(share=False, server_port=7860)
