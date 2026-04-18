import gradio as gr
from gradio_client import Client as HfClient, handle_file
import base64
import os
import glob
from PIL import Image
from openai import OpenAI
import shutil
import uuid

VOLUME_PATH = "/Volumes/workspace/default/faor"

os.environ['DATABRICKS_TOKEN'] = ''
DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')

hf_client = HfClient("imperiusrex/FAOR")

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url="https://7474660314620622.ai-gateway.cloud.databricks.com/mlflow/v1"
)

def save_to_memory(temp_video_path, temp_image_path, explanation):
    """Saves the files to a new subfolder in the Volume using read/write to avoid permission issues."""
    try:
        if not os.path.exists(VOLUME_PATH):
            os.makedirs(VOLUME_PATH, exist_ok=True)
            
        run_id = str(uuid.uuid4())[:8]
        
        run_folder = f"{VOLUME_PATH}/run_{run_id}"
        os.makedirs(run_folder, exist_ok=True)
        
        persisted_video = f"{run_folder}/video.mp4"
        persisted_image = f"{run_folder}/image.jpg"
        persisted_exp = f"{run_folder}/exp.txt"
        
        print(f"Saving video from {temp_video_path}...")
        with open(temp_video_path, "rb") as src:
            with open(persisted_video, "wb") as dst:
                dst.write(src.read())
        
        print(f"Saving image from {temp_image_path}...")
        with open(temp_image_path, "rb") as src:
            with open(persisted_image, "wb") as dst:
                dst.write(src.read())
        
        print(f"Saving explanation...")
        with open(persisted_exp, "w") as f:
            f.write(explanation)
        
        print(f"✓ Successfully saved to {run_folder}")
        return True
        
    except Exception as e:
        print(f"Error in save_to_memory: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def load_history_from_volume():
    """Scans the volume subfolders for saved triplets and returns them for the UI."""
    history = []
    if not os.path.exists(VOLUME_PATH):
        return history
        
    run_folders = glob.glob(f"{VOLUME_PATH}/run_*")
    
    for folder in run_folders:
        run_id = os.path.basename(folder).replace("run_", "")
        
        video_path = f"{folder}/video.mp4"
        image_path = f"{folder}/image.jpg"
        exp_path = f"{folder}/exp.txt"
        
        if os.path.exists(video_path) and os.path.exists(image_path) and os.path.exists(exp_path):
            with open(exp_path, "r") as f:
                explanation = f.read()
            
            history.append({
                "run_id": run_id,
                "video": video_path,
                "image": image_path,
                "text": explanation
            })
            
    return history

def process_lecture_video(video_path):
    """Encapsulates your pipeline logic for the Gradio button."""
    if not video_path:
        return None, "Please upload a video."
    
    try:
        print(f"Processing video: {video_path}")
        
        print("Sending to HF Space...")
        image_result = hf_client.predict(
            video_path=handle_file(video_path),
            api_name="/predict"
        )
        print(f"Received image: {image_result}")
        
        with open(image_result, "rb") as image_file:
            raw_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        clean_base64 = raw_base64.replace('\n', '').replace('\r', '').strip()
        
        print("Calling Databricks VLM...")
        chat_completion = client.chat.completions.create(
            model="lecture-description",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please describe the concepts shown on the board in detail."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{clean_base64}"}}
                    ]
                }
            ],
            max_tokens=1024
        )
        
        explanation = chat_completion.choices[0].message.content
        print("VLM response received")
        
        print("Saving to Volume...")
        save_success = save_to_memory(video_path, image_result, explanation)
        if not save_success:
            explanation = f"⚠️ Warning: Failed to save to Volume.\n\n{explanation}"
        
        return image_result, explanation

    except Exception as e:
        print(f"Error in process_lecture_video: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, f"An error occurred: {str(e)}"

# ---------------------------------------------------------
# Gradio UI Design
# ---------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎓 AI Lecture Assistant")
    
    # Load history before building the layout
    history_runs = load_history_from_volume()
    
    with gr.Tabs():
        # ==========================================
        # The Main Active Tab
        # ==========================================
        with gr.Tab("New Analysis"):
            
            # TOP ROW: Video and Image side-by-side
            with gr.Row():
                video_input = gr.Video(label="Upload Lecture Video")
                image_output = gr.Image(label="Extracted Blackboard", type="filepath")
                
            # MIDDLE ROW: The Action Button
            with gr.Row():
                analyze_btn = gr.Button("Extract & Analyze Board", variant="primary")
            
            # BOTTOM ROW: Large Explanation Box
            with gr.Row():
                text_output = gr.Textbox(
                    label="VLM Explanation", 
                    lines=20,
                )

            analyze_btn.click(
                fn=process_lecture_video,
                inputs=[video_input],
                outputs=[image_output, text_output]
            )

        # ==========================================
        # Historical Tabs
        # ==========================================
        for item in history_runs:
            with gr.Tab(f"Run {item['run_id']}"):
                
                # TOP ROW: Archived Media
                with gr.Row():
                    gr.Video(value=item['video'], label="Archived Video", interactive=False)
                    gr.Image(value=item['image'], label="Archived Blackboard", type="filepath", interactive=False)
                
                # BOTTOM ROW: Archived Explanation
                with gr.Row():
                    gr.Textbox(
                        value=item['text'], 
                        label="Archived Explanation", 
                        lines=20, 
                        interactive=False,
                    )

demo.launch(share=True)
