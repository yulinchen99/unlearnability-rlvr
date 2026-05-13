from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import os

class TrainingMode(Enum):
    FULL_FINETUNE = "full_finetune"
    LORA = "lora"

@dataclass
class PromptConfig:
    # Default prompt template for generation
    default_template: str = """Below is an instruction that describes a task, along with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Answer the following question accurately and completely.

### Input:
{question}

### Response:"""
    
    # Math-specific prompt template for generation
    math_template: str = """Below is a math problem. Solve it step by step and provide the final answer in the format '#### answer'.

### Problem:
{question}

### Solution:"""
    
    # Custom templates for specific question types
    custom_templates: Dict[str, str] = None
    
    # Whether to include metadata in training examples
    include_metadata: bool = False
    
    # Template for including metadata (if enabled)
    metadata_template: str = """### Metadata:
Score: {score}
Verification feedback: {feedback}
"""
    
    def __post_init__(self):
        if self.custom_templates is None:
            self.custom_templates = {
                "general": self.default_template,
                "math": self.math_template
            }
    
    def get_training_prompt(self, template: str, question: str, response: str, metadata: Optional[Dict[str, str]] = None) -> str:
        """Get the full training prompt by combining the template with response."""
        # Add metadata if available and enabled
        prefix = ""
        if self.include_metadata and metadata:
            prefix = self.metadata_template.format(**metadata)
        
        # Get the base prompt without response
        base_prompt = prefix + template.format(question=question)
        
        # For training, append the response and an end marker
        return f"{base_prompt} {response}"
    
    def get_generation_prompt(self, template: str, question: str, metadata: Optional[Dict[str, str]] = None) -> str:
        """Get the generation prompt without response."""
        if self.include_metadata and metadata:
            return self.metadata_template.format(**metadata) + template.format(question=question)
        return template.format(question=question)

@dataclass
class ModelConfig:
    base_model: str = "meta-llama/Llama-2-7b-hf"  # Base model to use
    max_length: int = 4096  # Maximum sequence length
    temperature: float = 0.7  # Sampling temperature
    top_p: float = 0.95  # Nucleus sampling parameter
    num_samples_per_question: int = 5  # Default number of samples to generate per question
    # Generation parameters that can be overridden per question type
    type_specific_params: Dict[str, Dict[str, Any]] = None
    # Default prompt type to use for each question type
    default_prompt_types: Dict[str, str] = None
    
    def __post_init__(self):
        if self.type_specific_params is None:
            self.type_specific_params = {
                "math": {
                    "temperature": 0.3,  # Lower temperature for math questions
                    "num_samples": 8,    # More samples for math questions
                    "max_length": 1024   # Shorter responses for math
                },
                "general": {
                    "temperature": 0.7,
                    "num_samples": 5,
                    "max_length": 2048
                }
            }
        if self.default_prompt_types is None:
            self.default_prompt_types = {
                "math": "math",
                "general": "general"
            }

@dataclass
class DataConfig:
    questions_file: str = "data/questions.json"  # Path to questions dataset
    output_dir: str = "data/generated"  # Directory for generated responses
    filtered_data_dir: str = "data/filtered"  # Directory for filtered data
    train_split: float = 0.9  # Train/val split ratio

@dataclass
class TrainingConfig:
    # Basic training parameters
    output_dir: str = "models/fine_tuned"  # Directory to save fine-tuned models
    learning_rate: float = 2e-5
    num_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    max_grad_norm: float = 1.0
    
    # Iterative training parameters
    num_iterations: int = 3  # Number of generate-filter-train iterations
    save_each_iteration: bool = True  # Whether to save model after each iteration
    iteration_dir_template: str = "iteration_{}"  # Template for iteration directories
    cumulative_training: bool = False  # Whether to include data from previous iterations
    
    # Training mode configuration
    training_mode: TrainingMode = TrainingMode.FULL_FINETUNE
    # LoRA parameters (used only if training_mode is LORA)
    lora_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.lora_config is None:
            self.lora_config = {
                "r": 8,
                "lora_alpha": 32,
                "lora_dropout": 0.1
            }
    
    def get_iteration_dir(self, iteration: int) -> str:
        """Get the directory for a specific iteration."""
        return os.path.join(self.output_dir, self.iteration_dir_template.format(iteration))
    
@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42
    device: str = "cuda"  # or "cpu"
    
config = Config() 
config.training.training_mode = TrainingMode.FULL_FINETUNE  # or TrainingMode.LORA
config.training.num_iterations = 3
config.training.cumulative_training = False  # Include previous iterations' data
config.training.save_each_iteration = True  # Save model after each iteration
config.prompt.custom_templates["none"] = """{question}"""
config.prompt.custom_templates["math"] = """Solve the following math problem by reasoning step by step. Put the final answer in $\\boxed{{ANSWER}}$, where ANSWER is just the final number or expression that solves the problem:\n{question}"""
config.prompt.custom_templates["reasoning"] = """You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first verbalize your reasoning process in the form of an internal monologue and then provide the user with the final answer in $\\boxed{{ANSWER}}$, where ANSWER is just the final number or expression that solves the problem:\n{question}"""
config.prompt.custom_templates["qwen"] = """{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."""
config.prompt.custom_templates["simplerl_base"] = """Question:\n{question}\nAnswer:\nLet's think step by step."""
config.prompt.custom_templates["octo"] = """Question: {question}\nAnswer:"""