from huggingface_hub import create_repo, upload_folder
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Load the Hugging Face API token from an environment variable
HUGGINGFACE_TOKEN = os.environ.get('HF_TOKEN')

if not HUGGINGFACE_TOKEN:
    raise ValueError("Please set the 'HF_TOKEN' environment variable.")

# Your Hugging Face username
your_username = "withmartian"  # Replace with your username or set as an environment variable

# List of local model paths
local_model_names = [
    #"sft_sql_interp_TinyStories-2Layers-33M_cs1_experiment_1.1",
    #"sft_sql_interp_TinyStories-2Layers-33M_cs2_experiment_2.3",
    #"sft_sql_interp_TinyStories-2Layers-33M_cs3_experiment_3.3",
    #"sft_sql_interp_Qwen2.5-0.5B_cs1_experiment_4.1",
    #"sft_sql_interp_Qwen2.5-0.5B_cs2_experiment_5.1",
    #"sft_sql_interp_Qwen2.5-0.5B_cs3_experiment_6.1",
    #"sft_sql_interp_Llama3.2-1B_cs1_experiment_7.1",
    #"sft_sql_interp_Llama3.2-1B_cs2_experiment_8.1",
    #"sft_sql_interp_Llama3.2-1B_cs3_experiment_9.1",
    "sft_sql_interp_TinyStories-2Layers-33M_cs1_experiment_1.6",
    "sft_sql_interp_TinyStories-2Layers-33M_cs2_experiment_2.6",
    "sft_sql_interp_TinyStories-2Layers-33M_cs3_experiment_3.6",
    # Add more paths as needed
]

# Corresponding list of new repository names
new_repo_names = [
    #"sql_interp_bm1_cs1_experiment_1.1",
    #"sql_interp_bm1_cs2_experiment_2.3",
    #"sql_interp_bm1_cs3_experiment_3.3",
    #"sql_interp_bm2_cs1_experiment_4.1",
    #"sql_interp_bm2_cs2_experiment_5.1",
    #"sql_interp_bm2_cs3_experiment_6.1",
    #"sql_interp_bm3_cs1_experiment_7.1",
    #"sql_interp_bm3_cs2_experiment_8.1",
    #"sql_interp_bm3_cs3_experiment_9.1",
    "sql_interp_bm1_cs1_experiment_1.6",
    "sql_interp_bm1_cs2_experiment_2.6",
    "sql_interp_bm1_cs3_experiment_3.6",
    # Add more names as needed
]

# Ensure the lists have the same length
assert len(local_model_names) == len(new_repo_names), "The lists must be of the same length."

# Loop over the lists and upload each model
for local_model_name, new_repo_name in zip(local_model_names, new_repo_names):
    repo_id = f"{your_username}/{new_repo_name}"
    local_model_path = os.path.join("models", local_model_name)
    print(f"Uploading model from '{local_model_path}' to '{repo_id}'..........")

    # Step 1: Create the new repository
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=True,
            token=HUGGINGFACE_TOKEN,
            exist_ok=False
        )
        print(f"Repository '{repo_id}' created!")
    except Exception as e:
        print(f"Error creating repository '{repo_id}': {e}")
        #continue  # Skip to the next model if repo creation fails

    # Step 2: Upload the local model files to the new repository
    try:
        upload_folder(
            folder_path=local_model_path,
            path_in_repo=".",  # Upload to the root directory of the repo
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {local_model_name} files",
            token=HUGGINGFACE_TOKEN
        )
        print(f"Model files from '{local_model_path}' have been uploaded to https://huggingface.co/{repo_id}\n")
    except Exception as e:
        print(f"Error uploading model from '{local_model_path}' to '{repo_id}': {e}\n")

