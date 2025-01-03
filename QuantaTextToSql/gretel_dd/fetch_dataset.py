# from gretel_client import Gretel
# from gretel_client.rest_v1.api.workflows_api import WorkflowsApi

# from gretel_client.navigator.workflow import BatchWorkflowRun
# from gretel_client.navigator.client.utils import get_navigator_client


# project_name = "dhruv-sql-interp-cs12"
# workflow_run_id = "wr_2qGEzBS2isKxm1sVvxxqabtELOY"
# # step_name = "evaluate-dataset-7"
# step_name = "judge-with-llm-6"

# gretel = Gretel(project_name=project_name)
# project = gretel.get_project()
# workflow_api = project.session.get_v1_api(WorkflowsApi)

# workflow_run = workflow_api.get_workflow_run(
#     workflow_run_id=workflow_run_id,
#     expand=["actions"],
# )

# client = get_navigator_client()
# batch_workflow_run = BatchWorkflowRun(project=project, workflow_id=workflow_run.workflow_id, workflow_run_id=workflow_run_id, client=client)

# import ipdb; ipdb.set_trace()

# step_output = batch_workflow_run.get_step_output(step_name)
# step_output.to_parquet("cs12_dataset.parquet")

# # This line is specifically for getting the pdf report
# # path = batch_workflow_run.download_step_output(step_name, format="pdf")

from gretel_client.navigator.data_designer.interface import DataDesigner
from datasets import Dataset

workflow_run_id = "wr_2q9xTNIbiqXPzdDRo6sDCRiewus"
# cs12 workflow_run_id = "wr_2qL9MapVCbvUBQz2reOFlEIU7ES"
# https://console-eng.gretel.ai/workflows/w_2qL9MNk8CCOoFksDl5ADiQmnExA/runs/wr_2qL9MapVCbvUBQz2reOFlEIU7ES
dataset = DataDesigner.fetch_dataset(workflow_run_id=workflow_run_id)
hf_dataset = Dataset.from_pandas(dataset)
#import ipdb; ipdb.set_trace()
config_name = 'cs11'
hf_dataset.push_to_hub(f"withmartian/{config_name}_dataset")
#dataset.to_parquet("cs12_dataset.parquet")
