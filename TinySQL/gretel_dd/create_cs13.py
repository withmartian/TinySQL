import os

folder_name = "cs13"
os.makedirs(folder_name, exist_ok=True)

# Read the base template
with open("cs13_base.yaml", "r") as f:
    base_yaml = f.read()

i = 1
for num_selected_columns in range(1, 4):
    # Start num_columns from num_selected_columns so that
    # we never have fewer columns than selected columns.
    for num_columns in range(num_selected_columns, 7):
        # Create a modified YAML
        yaml_content = base_yaml.replace("{{num_columns}}", str(num_columns))

        # Handle pluralization
        plural = "s" if num_selected_columns > 1 else ""
        yaml_content = yaml_content.replace("{{num_selected_columns}}", str(num_selected_columns))
        yaml_content = yaml_content.replace("{{plural}}", plural)
        
        config_name = f"cs13_{i}.yaml"
        file_path = os.path.join(folder_name, config_name)
        
        with open(file_path, "w") as out_file:
            out_file.write(yaml_content)
        
        i += 1