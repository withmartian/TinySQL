# Base Models
BM_TUTORIAL = "openai-community/gpt2"
BM_TINY_STORIES = "roneneldan/TinyStories-Instruct-2Layers-33M"
BM_QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
BM_LLAMA = "withmartian/Llama-3.2-1B-Instruct"
BM_GRANITE = "ibm-granite/granite-3.0-1b-a400m-instruct"
BM_SMOL = "HuggingFaceTB/SmolLM-360M-Instruct"

# Model 1 Semantic Variants
BM1_CS1_SEMANTIC = "withmartian/sql_interp_bm1_cs1_experiment_1.10"
BM1_CS2_SEMANTIC = "withmartian/sql_interp_bm1_cs2_experiment_2.10"
BM1_CS3_SEMANTIC = "withmartian/sql_interp_bm1_cs3_experiment_3.10"
BM1_CS4_SEMANTIC = "withmartian/sql_interp_bm1_cs4_dataset_synonyms_experiment_1.1"
BM1_CS5_SEMANTIC = "withmartian/sql_interp_bm1_cs5_dataset_synonyms_experiment_1.2"

# Model 2 Semantic Variants
BM2_CS1_SEMANTIC = "withmartian/sql_interp_bm2_cs1_experiment_4.3"
BM2_CS2_SEMANTIC = "withmartian/sql_interp_bm2_cs2_experiment_5.3"
BM2_CS3_SEMANTIC = "withmartian/sql_interp_bm2_cs3_experiment_6.3"

# Model 3 Semantic Variants
BM3_CS1_SEMANTIC = "withmartian/sql_interp_bm3_cs1_experiment_7.3"
BM3_CS2_SEMANTIC = "withmartian/sql_interp_bm3_cs2_experiment_8.3"
BM3_CS3_SEMANTIC = "withmartian/sql_interp_bm3_cs3_experiment_9.3"

# Model 1 Non-semantic Variants
BM1_CS1_NONSEMANTIC = "withmartian/sql_interp_bm1_cs1_experiment_1.8"
BM1_CS2_NONSEMANTIC = "withmartian/sql_interp_bm1_cs2_experiment_2.8"
BM1_CS3_NONSEMANTIC = "withmartian/sql_interp_bm1_cs3_experiment_3.8"
BM1_CS1_NONSEMANTIC_1_7 = "withmartian/sql_interp_bm1_cs1_experiment_1.7" # Deprecated
BM1_CS1_NONSEMANTIC_1_6 = "withmartian/sql_interp_bm1_cs1_experiment_1.6" # Deprecated
BM1_CS2_NONSEMANTIC_2_7 = "withmartian/sql_interp_bm1_cs2_experiment_2.7" # Deprecated
BM1_CS2_NONSEMANTIC_2_6 = "withmartian/sql_interp_bm1_cs2_experiment_2.6" # Deprecated
BM1_CS2_NONSEMANTIC_2_3 = "withmartian/sql_interp_bm1_cs2_experiment_2.3" # Deprecated
BM1_CS3_NONSEMANTIC_3_7 = "withmartian/sql_interp_bm1_cs3_experiment_3.7" # Deprecated
BM1_CS3_NONSEMANTIC_3_6 = "withmartian/sql_interp_bm1_cs3_experiment_3.6" # Deprecated
BM1_CS3_NONSEMANTIC_3_3 = "withmartian/sql_interp_bm1_cs3_experiment_3.3" # Deprecated

# Model 2 Non-semantic Variants
BM2_CS1_NONSEMANTIC = "withmartian/sql_interp_bm2_cs1_experiment_4.2"
BM2_CS2_NONSEMANTIC = "withmartian/sql_interp_bm2_cs2_experiment_5.2"
BM2_CS3_NONSEMANTIC = "withmartian/sql_interp_bm2_cs3_experiment_6.2"

# Model 3 Non-semantic Variants
BM3_CS1_NONSEMANTIC = "withmartian/sql_interp_bm3_cs1_experiment_7.2"
BM3_CS2_NONSEMANTIC = "withmartian/sql_interp_bm3_cs2_experiment_8.2"
BM3_CS3_NONSEMANTIC = "withmartian/sql_interp_bm3_cs3_experiment_9.2"

# All available models (for validation)
AVAILABLE_MODELS = {
    BM_TUTORIAL,
    BM_TINY_STORIES,
    BM_QWEN,
    BM_LLAMA,
    BM_GRANITE,
    BM_SMOL,
    BM1_CS1_SEMANTIC,
    BM1_CS2_SEMANTIC,
    BM1_CS3_SEMANTIC,
    BM1_CS1_NONSEMANTIC,
    BM1_CS2_NONSEMANTIC,
    BM1_CS3_NONSEMANTIC,
    BM1_CS4_SEMANTIC,
    BM1_CS5_SEMANTIC,
    BM2_CS1_NONSEMANTIC,
    BM2_CS2_NONSEMANTIC,
    BM2_CS3_NONSEMANTIC,
    BM3_CS1_NONSEMANTIC,
    BM3_CS2_NONSEMANTIC,
    BM3_CS3_NONSEMANTIC
}

def is_valid_model(model_path: str) -> bool:
    """Validate if a model path is in the set of available models."""
    return model_path in AVAILABLE_MODELS
# Load the tokenizer and trained model for model 1, 2, or 3 and command set 0 (base model), 1, 2, or 3
