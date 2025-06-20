import os
import sys
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ['GRADIO_ANALYTICS_ENABLED'] = '0'
sys.path.insert(0, os.getcwd())
import gradio as gr
import yaml
import json
from slugify import slugify
from gradio_logsview import LogsView, LogsViewRunner
from huggingface_hub import hf_hub_download, HfApi

with open('models.yaml', 'r') as file:
    models = yaml.safe_load(file)

def resolve_path(p):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    norm_path = os.path.normpath(os.path.join(current_dir, p))
    return f"\"{norm_path}\""
def resolve_path_without_quotes(p):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    norm_path = os.path.normpath(os.path.join(current_dir, p))
    return norm_path

def set_repo(lora_rows):
    selected_name = os.path.basename(lora_rows)
    return gr.update(value=selected_name)

def get_loras():
    try:
        outputs_path = resolve_path_without_quotes(f"outputs")
        files = os.listdir(outputs_path)
        folders = [os.path.join(outputs_path, item) for item in files if os.path.isdir(os.path.join(outputs_path, item)) and item != "sample"]
        folders.sort(key=lambda file: os.path.getctime(file), reverse=True)
        return folders
    except Exception as e:
        return []

def get_samples(lora_name):
    output_name = slugify(lora_name)
    try:
        samples_path = resolve_path_without_quotes(f"outputs/{output_name}/sample")
        files = [os.path.join(samples_path, file) for file in os.listdir(samples_path)]
        files.sort(key=lambda file: os.path.getctime(file), reverse=True)
        return files
    except:
        return []

def pre_encode_dataset(
    base_model,
    lora_name,
    data_root_dir,
    caption_type
):
    #######################################################
    # Download the autoencoder checkpoint and model config
    #######################################################
    
    model = models["diffusion"][base_model]
    ae_folder = "models/autoencoder"
    ae_model_id = model["autoencoder"]
    ae_model = models["autoencoder"][ae_model_id]
    ae_model_name = ae_model["file"]
    ae_model_config_name = ae_model["config_file"]
    ae_model_repo = ae_model["repo"]

    output_name = slugify(lora_name)

    if base_model == "stable-audio-open-small":
        ae_path = os.path.join(ae_folder, ae_model_name)
        ae_config_path = os.path.join(ae_folder, ae_model_config_name)
    else:
        ae_path = os.path.join(ae_folder, ae_model_repo, ae_model_name)
        ae_config_path = os.path.join(ae_folder, ae_model_repo, ae_model_config_name)
    
    if not os.path.exists(ae_path):
        os.makedirs(ae_folder, exist_ok=True)
        gr.Info(f"Downloading autoencoder")
        hf_hub_download(repo_id=ae_model_repo, local_dir=ae_folder, filename=ae_model_name)

    if not os.path.exists(ae_config_path):
        hf_hub_download(repo_id=ae_model_repo, local_dir=ae_folder, filename=ae_model_config_name)

    #########################################
    # Create the pre-encoding dataset config
    #########################################

    if caption_type == "paths":
        metadata_module_path = resolve_path_without_quotes("metadata_modules/paths_md_pre_encode.py")
    else:
        raise ValueError(f"Unknown caption type: {caption_type}")

    pre_encode_dataset_config_json = f'''\
{{
    "dataset_type": "audio_dir",
    "datasets": [{{
        "id": "audio",
        "path": "{data_root_dir}",
        "custom_metadata_module": "{metadata_module_path}"
    }}]
}}\
'''

    pre_encode_dataset_config = resolve_path_without_quotes(f"outputs/{slugify(lora_name)}/dataset/pe_dataset_config.json")
    if not os.path.exists(os.path.dirname(pre_encode_dataset_config)):
        os.makedirs(os.path.dirname(pre_encode_dataset_config), exist_ok=True)

    with open(pre_encode_dataset_config, 'w', encoding="utf-8") as file:
        file.write(pre_encode_dataset_config_json)
    gr.Info(f"Generated pre-encoding dataset config at {pre_encode_dataset_config}")

    ##############################
    # Run the pre-encoding script
    ##############################

    pre_encode_output_path = resolve_path_without_quotes(f"outputs/{output_name}/dataset/pre_encoded/")
    
    line_break = "\\"
    file_type = "sh"
    if sys.platform == "win32":
        line_break = "^"
        file_type = "bat"

    command = f"""python stable-audio-tools/pre_encode.py {line_break}
    --dataset-config {pre_encode_dataset_config} {line_break}
    --model-config {resolve_path_without_quotes(ae_config_path)} {line_break}
    --ckpt-path {resolve_path_without_quotes(ae_path)} {line_break}
    --model-half {line_break}
    --output-path {pre_encode_output_path} {line_break}
    --batch-size 16 {line_break}
    --sample-size {524288} {line_break}
    """

    sh_filename = f"pre-encode.{file_type}"
    sh_filepath = resolve_path_without_quotes(f"outputs/{output_name}/{sh_filename}")
    with open(sh_filepath, 'w', encoding="utf-8") as file:
        file.write(command)
    gr.Info(f"Generated pre-encode script at {sh_filename}")

    # Train
    if sys.platform == "win32":
        command = sh_filepath
    else:
        command = f"bash \"{sh_filepath}\""

    # Use Popen to run the command and capture output in real-time
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['LOG_LEVEL'] = 'DEBUG'
    runner = LogsViewRunner()
    cwd = os.path.dirname(os.path.abspath(__file__))
    gr.Info(f"Started pre-encoding dataset")
    yield from runner.run_command([command], cwd=cwd)
    yield runner.log(f"Runner: {runner}")

    gr.Info(f"Pre-encoding Complete. Check the outputs folder for the pre-encoded dataset.", duration=None)

def start_training(
    base_model,
    lora_name,
    caption_type,
    sample_prompts,
    workers,
    seed,
    save_every_n_steps,
    sample_every_n_steps,
    learning_rate,
    batch_size
):
    #################################################
    # Download the model checkpoint and model config
    #################################################
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)
    if not os.path.exists("outputs"):
        os.makedirs("outputs", exist_ok=True)
    output_name = slugify(lora_name)
    output_dir = resolve_path_without_quotes(f"outputs/{output_name}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    model = models["diffusion"][base_model]
    model_file = model["file"]
    model_config_file = model["config_file"]
    repo = model["repo"]

    # download diffusion model
    if base_model == "stable-audio-open-small":
        dit_folder = "models/dit"
    else:
        dit_folder = f"models/dit/{repo}"

    dit_path = os.path.join(dit_folder, model_file)
    dit_config_path = os.path.join(dit_folder, model_config_file)
    if not os.path.exists(dit_path):
        os.makedirs(dit_folder, exist_ok=True)
        gr.Info(f"Downloading base model: {base_model}. Please wait. (You can check the terminal for the download progress)", duration=None)
        hf_hub_download(repo_id=repo, local_dir=dit_folder, filename=model_file)
    
    if not os.path.exists(dit_config_path):
        hf_hub_download(repo_id=repo, local_dir=dit_folder, filename=model_config_file)

    output_dir = resolve_path(f"outputs/{output_name}")
    
    #############################
    # Modify the training config
    #############################

    dit_config_path = resolve_path_without_quotes(dit_config_path)
    with open(dit_config_path, 'r', encoding="utf-8") as file:
        dit_config_data = json.load(file)
    dit_config_data["training"]["pre_encoded"] = True
    dit_config_data["training"]["demo"]["demo_every"] = sample_every_n_steps
    dit_config_data["training"]["optimizer_configs"]["diffusion"]["optimizer"]["config"]["lr"] = float(learning_rate)
    
    training_config_path = resolve_path_without_quotes(f"outputs/{output_name}/training_model_config.json")

    if not os.path.exists(training_config_path):
        os.makedirs(os.path.dirname(training_config_path), exist_ok=True)

    with open(training_config_path, 'w', encoding="utf-8") as file:
        json.dump(dit_config_data, file, indent=4)
    gr.Info(f"Modified training model config at {training_config_path}")

    #################################
    # Create training dataset config
    #################################
    if caption_type == "paths":
        metadata_module_path = resolve_path_without_quotes("metadata_modules/paths_md.py")
    else:
        raise ValueError(f"Unknown caption type: {caption_type}")

    print(f"Found metadata module")

    pre_encoded_dataset_dir = resolve_path_without_quotes(f"outputs/{output_name}/dataset/pre_encoded/")
    if not os.path.exists(pre_encoded_dataset_dir):
        gr.Error(f"Pre-encoded dataset not found at {pre_encoded_dataset_dir}. Please run the pre-encoding step first.")
        return

    print(f"Found pre-encoded dataset at {pre_encoded_dataset_dir}")

    pre_encode_dataset_config_json = f'''\
{{
    "dataset_type": "pre_encoded",
    "datasets": [{{
        "id": "audio_pre_encoded",
        "path": "{pre_encoded_dataset_dir}",
        "custom_metadata_module": "{metadata_module_path}"
    }}]
}}\
'''

    pre_encode_dataset_config = resolve_path_without_quotes(f"outputs/{output_name}/dataset_config.json")
    if not os.path.exists(os.path.dirname(pre_encode_dataset_config)):
        os.makedirs(os.path.dirname(pre_encode_dataset_config), exist_ok=True)

    with open(pre_encode_dataset_config, 'w', encoding="utf-8") as file:
        file.write(pre_encode_dataset_config_json)

    gr.Info(f"Generated training dataset config at {pre_encode_dataset_config}")

    ##########################
    # Run the training script
    ##########################
    
    line_break = "\\"
    file_type = "sh"
    if sys.platform == "win32":
        line_break = "^"
        file_type = "bat"
    model_path = os.path.join(dit_folder, model_file)
    pretrained_model_path = resolve_path(model_path)
    
    train_command = f"""python stable-audio-tools/train.py {line_break}
--name {output_name} {line_break}
--pretrained-ckpt-path {pretrained_model_path} {line_break}
--model-config {training_config_path} {line_break}
--batch-size {batch_size} {line_break}
--num-workers {workers} {line_break}
--seed {seed} {line_break}
--config-file stable-audio-tools/defaults.ini {line_break}
--checkpoint-every {save_every_n_steps} {line_break}
--dataset-config {resolve_path(f"outputs/{output_name}/dataset_config.json")} {line_break}
--save-dir {output_dir} {line_break}
--precision 16-mixed {line_break}
"""

    sh_filename = f"train.{file_type}"
    sh_filepath = resolve_path_without_quotes(f"outputs/{output_name}/{sh_filename}")
    with open(sh_filepath, 'w', encoding="utf-8") as file:
        file.write(train_command)
    gr.Info(f"Generated train script at {sh_filename}")

    # Train
    if sys.platform == "win32":
        command = sh_filepath
    else:
        command = f"bash \"{sh_filepath}\""

    # Use Popen to run the command and capture output in real-time
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['LOG_LEVEL'] = 'DEBUG'
    runner = LogsViewRunner()
    cwd = os.path.dirname(os.path.abspath(__file__))
    gr.Info(f"Started training")
    yield from runner.run_command([command], cwd=cwd)
    yield runner.log(f"Runner: {runner}")

    gr.Info(f"Training Complete. Check the outputs folder for the LoRA files.", duration=None)

theme = gr.themes.Monochrome(
    text_size=gr.themes.Size(lg="18px", md="15px", sm="13px", xl="22px", xs="12px", xxl="24px", xxs="9px"),
    font=[gr.themes.GoogleFont("Source Sans Pro"), "ui-sans-serif", "system-ui", "sans-serif"],
)
css = """
@keyframes rotate {
    0% {
        transform: rotate(0deg);
    }
    100% {
        transform: rotate(360deg);
    }
}
#advanced_options .advanced:nth-child(even) { background: rgba(0,0,100,0.04) !important; }
h1{font-family: georgia; font-style: italic; font-weight: bold; font-size: 30px; letter-spacing: -1px;}
h3{margin-top: 0}
.tabitem{border: 0px}
.group_padding{}
nav{position: fixed; top: 0; left: 0; right: 0; z-index: 1000; text-align: center; padding: 10px; box-sizing: border-box; display: flex; align-items: center; backdrop-filter: blur(10px); }
nav button { background: none; color: firebrick; font-weight: bold; border: 2px solid firebrick; padding: 5px 10px; border-radius: 5px; font-size: 14px; }
nav img { height: 40px; width: 40px; border-radius: 40px; }
nav img.rotate { animation: rotate 2s linear infinite; }
.flexible { flex-grow: 1; }
.tast-details { margin: 10px 0 !important; }
.toast-wrap { bottom: var(--size-4) !important; top: auto !important; border: none !important; backdrop-filter: blur(10px); }
.toast-title, .toast-text, .toast-icon, .toast-close { color: black !important; font-size: 14px; }
.toast-body { border: none !important; }
#terminal { box-shadow: none !important; margin-bottom: 25px; background: rgba(0,0,0,0.03); }
#terminal .generating { border: none !important; }
#terminal label { position: absolute !important; }
.tabs { margin-top: 50px; }
.hidden { display: none !important; }
.codemirror-wrapper .cm-line { font-size: 12px !important; }
label { font-weight: bold !important; }
#start_training.clicked { background: silver; color: black; }
"""

js = """
function() {
    let autoscroll = document.querySelector("#autoscroll")
    if (window.iidxx) {
        window.clearInterval(window.iidxx);
    }
    window.iidxx = window.setInterval(function() {
        let text=document.querySelector(".codemirror-wrapper .cm-line").innerText.trim()
        let img = document.querySelector("#logo")
        if (text.length > 0) {
            autoscroll.classList.remove("hidden")
            if (autoscroll.classList.contains("on")) {
                autoscroll.textContent = "Autoscroll ON"
                window.scrollTo(0, document.body.scrollHeight, { behavior: "smooth" });
                img.classList.add("rotate")
            } else {
                autoscroll.textContent = "Autoscroll OFF"
                img.classList.remove("rotate")
            }
        }
    }, 500);
    console.log("autoscroll", autoscroll)
    autoscroll.addEventListener("click", (e) => {
        autoscroll.classList.toggle("on")
    })
    function debounce(fn, delay) {
        let timeoutId;
        return function(...args) {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => fn(...args), delay);
        };
    }

    function handleClick() {
        console.log("refresh")
        document.querySelector("#refresh").click();
    }
    const debouncedClick = debounce(handleClick, 1000);
    document.addEventListener("input", debouncedClick);

    document.querySelector("#start_training").addEventListener("click", (e) => {
      e.target.classList.add("clicked")
      e.target.innerHTML = "Training..."
    })

}
"""

with gr.Blocks(elem_id="app", theme=theme, css=css, fill_width=True) as demo:
    with gr.Tabs() as tabs:
        with gr.TabItem("Gym"):
            output_components = []
            with gr.Row():
                gr.HTML("""<nav>
            <img id='logo' src='/file=icon.png' width='80' height='80'>
            <div class='flexible'></div>
            <button id='autoscroll' class='on hidden'></button>
        </nav>
        """)
            with gr.Row(elem_id='container'):
                with gr.Column():
                    gr.Markdown(
                        """# Step 1. Model Info
        <p style="margin-top:0">Configure your model train settings.</p>
        """, elem_classes="group_padding")
                    lora_name = gr.Textbox(
                        label="The name of your model",
                        info="This has to be a unique name",
                        placeholder="e.g.: Drum Loops, Synth Pads",
                    )
                    model_names = list(models["diffusion"].keys())
                    print(f"model_names={model_names}")
                    base_model = gr.Dropdown(label="Base model (edit the models.yaml file to add more to this list)", choices=model_names, value=model_names[0])
                    sample_prompts = gr.Textbox("", lines=5, label="Sample Audio Prompts (Separate with new lines)", interactive=True)
                    sample_every_n_steps = gr.Number(250, precision=0, label="Sample Audio Every N Steps", interactive=True)
                with gr.Column():
                    gr.Markdown(
                        """# Step 2. Dataset""", elem_classes="group_padding")
                    pre_encode_button = gr.Button("Pre-encode dataset", visible=True, elem_id="pre_encode")
                    with gr.Group():
                        data_root_dir = gr.Textbox(label="Data root directory")
                        caption_type_dropdown = gr.Dropdown(label="Caption type", choices=["paths"], value="paths")

                with gr.Column():
                    gr.Markdown(
                        """# Step 3. Train
        <p style="margin-top:0">Press start to start training.</p>
        """, elem_classes="group_padding")
                    refresh = gr.Button("Refresh", elem_id="refresh", visible=False)
                    start = gr.Button("Start training", visible=True, elem_id="start_training")
                    output_components.append(start)
            with gr.Accordion("Advanced options", elem_id='advanced_options', open=False):
                with gr.Row():
                    with gr.Column(min_width=300):
                        seed = gr.Number(label="Seed", value=42, interactive=True)
                    with gr.Column(min_width=300):
                        workers = gr.Number(label="Number of CPU workers", value=2, interactive=True)
                    with gr.Column(min_width=300):
                        learning_rate = gr.Textbox(label="Learning rate", value="1e-4", interactive=True)
                    with gr.Column(min_width=300):
                        save_every_n_steps = gr.Number(label="Save every N steps", value=1000, interactive=True)
                    with gr.Column(min_width=300):
                        batch_size = gr.Number(label="Batch size", value=16, interactive=True)
            with gr.Row():
                terminal = LogsView(label="Train log", elem_id="terminal")
            with gr.Row():
                gallery = gr.Gallery(get_samples, inputs=[lora_name], label="Samples", every=10, columns=6)
         
    dataset_folder = gr.State()

    pre_encode_button.click(
        fn=pre_encode_dataset,
        inputs=[
            base_model,
            lora_name,
            data_root_dir,
            caption_type_dropdown
        ],
        outputs = terminal
    )

    start.click(
        fn=start_training,
        inputs=[
            base_model,
            lora_name,
            caption_type_dropdown,
            sample_prompts,
            workers,
            seed,
            save_every_n_steps,
            sample_every_n_steps,
            learning_rate,
            batch_size
        ],
        outputs=terminal
    )
    
if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    demo.launch(debug=True, show_error=True, allowed_paths=[cwd], share=True)
