from huggingface_hub import HfApi
from pprint import pprint

api = HfApi()

# Fetch all models for the user
username = 'dhruvnathawani'
user_models = api.list_models(author=username)

# Extract model IDs
#model_names = [model.modelId for model in user_models]
prefix = "dhruvnathawani/sft_sql_interp_TinyStories-2Layers-33M_cs3_experiment_3."
model_names = [model.modelId for model in user_models if model.modelId.startswith(prefix)]
for model in user_models:
    import ipdb; ipdb.set_trace()
    print(model.modelId)

import ipdb; ipdb.set_trace()
# Confirm deletion
print("The following models will be deleted:")
pprint(model_names)

confirm = input("Are you sure you want to delete all these models? Type 'yes' to confirm: ")

if confirm.lower() == 'yes':
    for model in model_names:
        try:
            # Delete the repository
            api.delete_repo(repo_id=model, repo_type='model')
            print(f"Deleted model: {model}")
        except Exception as e:
            print(f"Failed to delete model: {model}. Error: {e}")
else:
    print("Deletion canceled.")
