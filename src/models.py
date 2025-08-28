"""
This file is used for evaluating closed and open source LLMs.

Proprietary LLMs:
1. [OpenAI GPT-4o](https://platform.openai.com/docs/models/gpt-4o)
2. [Claude Sonnet-4](https://www.anthropic.com/claude/sonnet)
3. [Gemini 2.5-Flash](https://deepmind.google/models/gemini/flash/)

Open-source LLMs:
1. [CodeLlama-7b-Instruct](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf)
2. [Deepseek-Coder-7b-Instruct-v1.5](https://huggingface.co/deepseek-ai/deepseek-coder-7b-instruct-v1.5)
3. [Fine-tuned-Codegen-6B-Verilog](https://huggingface.co/shailja/fine-tuned-codegen-6B-Verilog)
4. [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Aug 14th, 2025
Place:  Boston, MA
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List
from src.logger import Logger
from openai import OpenAI
import anthropic
from google import genai
import torch
import torch.distributed as dist
import deepspeed
from google.genai import types
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from src.rl_policy import ActorCriticModel


SYSTEM_PROMPT = "You are a Verilog code generator. Output only Verilog code."


class ProprietaryLLM(ABC):
    """
    Base class for Proprietary LLMs like:
    1. Claude
    2. Gemini
    3. OpenAI
    """

    @abstractmethod
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        """
        Parameters:
        -----------
        model:  str
            Name of the model
            Default: Claude Sonnet 4 - Balanced performance and cost

        Returns:
        --------
        None
        """
        self.log = Logger("base_class").get_logger()
        self.model = model
        self.client = None

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection with Claude API

        Returns:
        --------
        Bool value to confirm connection
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 300,
        n_samples: int = 2,
    ) -> Dict:
        """
        Generate answer via Claude API
        prompt:         str
            Natural language description of the Verilog module to generate
        temperature:    float, default=0.2
            Controls randomness in generation (0=deterministic, 1=very random)
            Lower values produce more consistent, focused code
        max_tokens:     int, default=300
            Maximum number of tokens to generate per response
            Sufficient for most Verilog modules
        n_samples:      int, default=2
            Number of different code samples to generate
        """
        pass


class ClaudeAPIClient(ProprietaryLLM):
    """
    SDK client for Claude LLM
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        """
        Parameters:
        -----------
        model:  str
            Name of the model
            Default: Claude Sonnet 4 - Balanced performance and cost

        Returns:
        --------
        None
        """
        self.log = Logger("claude_api_client").get_logger()
        self.model = model
        self.client = None

    def connect(self) -> bool:
        """
        Establish connection with Claude API

        Returns:
        --------
        Bool value to confirm connection
        """
        try:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")

            self.client = anthropic.Anthropic(api_key=api_key)
            self.log.info(f"Model chosen for Claude: {self.model}")
            return True
        except Exception as err:
            self.log.error(
                f"Error while establishing connection with model: {self.model} | Error: {err}"
            )
            return False

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 300,
        n_samples: int = 2,
    ) -> Dict:
        """
        Generate answer via Claude API

        Parameters:
        -----------
        prompt:         str
            Natural language description of the Verilog module to generate
        temperature:    float, default=0.2
            Controls randomness in generation (0=deterministic, 1=very random)
            Lower values produce more consistent, focused code
        max_tokens:     int, default=300
            Maximum number of tokens to generate per response
            Sufficient for most Verilog modules
        n_samples:      int, default=2
            Number of different code samples to generate

        Returns:
        --------
        Dict of all responses

        Raises:
        -------
        anthropic.OpenAIError
            If API call fails (rate limits, invalid API key, etc.)
        """
        self.log.info("Running chat")

        # Container for all generated samples
        outputs = []
        # Token tracking
        total_input_tokens = 0
        total_output_tokens = 0
        tokens_per_sample = []
        # Time tracking
        start_time = time.time()

        # Generate n_samples by making multiple API calls
        # Claude API doesn't support n>1 in a single call unlike other LLM APIs
        for idx in range(n_samples):
            # Make API call to Claude
            response = self.client.messages.create(
                model=self.model,
                # System prompt specifically instructs Claude to generate Verilog
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": prompt}]},
                ],
                temperature=temperature,  # Control randomness
                max_tokens=max_tokens,  # Limit response length
            )

            # Extract text from Claude's response
            # Claude returns content as a list, we take the first text block
            outputs.append(response.content[0].text)

            # Extract token usage from the response
            # Claude provides usage information in the response object
            if hasattr(response, "usage"):
                # For the first sample, count input tokens (same for all samples)
                if idx == 0:
                    total_input_tokens = response.usage.input_tokens

                # Add output tokens for this sample
                output_tokens_this_sample = response.usage.output_tokens
                total_output_tokens += output_tokens_this_sample
                tokens_per_sample.append(output_tokens_this_sample)

        end_time = time.time()
        generation_time = end_time - start_time
        # Calculate tokens per second metrics
        output_tokens_per_second = (
            total_output_tokens / generation_time if generation_time > 0 else 0
        )
        total_tokens_per_second = (
            (total_input_tokens + total_output_tokens) / generation_time
            if generation_time > 0
            else 0
        )

        # Log generation completion and timing
        self.log.info(
            f"Time taken to generate for '{prompt}': {generation_time:.2f} sec"
        )
        self.log.info("Successfully ran chat")

        response_data = {
            "question": prompt,  # Original prompt for reference
            "outputs": outputs,  # List of generated code samples
            "config": {  # Configuration used for generation
                "model": self.model,
                "system_instruction": SYSTEM_PROMPT,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "samples": n_samples,
            },
            "time": generation_time,
            # Token usage information
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "tokens_per_sample": tokens_per_sample,
            "avg_tokens_per_sample": total_output_tokens / n_samples
            if n_samples > 0
            else 0,
            "output_tokens_per_second": output_tokens_per_second,
            "total_tokens_per_second": total_tokens_per_second,
        }

        return response_data


class GeminiAPIClient:
    """
    SDK client for Gemini LLM
    """

    def __init__(self, model: str = "gemini-2.5-flash"):
        """
        Parameters:
        -----------
        model:  str
            Name of the model
            Default: Gemini Flash 2.5 - Optimized for speed with good performance

        Returns:
        --------
        None
        """
        self.log = Logger("gemini_api_client").get_logger()
        self.model = model
        self.client = None

    def connect(self) -> bool:
        """
        Establish connection with Gemini API

        Returns:
        --------
        Bool value to confirm connection
        """
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment")

            self.client = genai.Client(api_key=api_key)
            self.log.info(f"Model chosen for Gemini: {self.model}")
            return True
        except Exception as err:
            self.log.error(
                f"Error while establishing connection with model: {self.model} | Error: {err}"
            )
            return False

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 300,
        n_samples: int = 2,
    ) -> Dict:
        """
        Generate answer via Gemini API

        Parameters:
        -----------
        prompt:         str
            Natural language description of the Verilog module to generate
        temperature:    float, default=0.2
            Controls randomness in generation (0=deterministic, 1=very random)
            Lower values produce more consistent, focused code
        max_tokens:     int, default=300
            Maximum number of tokens to generate per response
            Sufficient for most Verilog modules
        n_samples:      int, default=2
            Number of different code samples to generate

        Returns:
        --------
        Dict of all responses

        Raises:
        -------
        openai.OpenAIError
            If API call fails (rate limits, invalid API key, etc.)

        Note:
        -----
        Unlike other LLM APIs, token info and time taken weren't added as the API was unusable at the time of benchmarking
        """
        self.log.info("Running chat")

        start_time = time.time()

        # Make single API call for generation
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                # System instruction for Verilog generation
                system_instruction=SYSTEM_PROMPT,
                temperature=temperature,
                maxOutputTokens=max_tokens,  # Note: camelCase for Gemini API
                candidateCount=n_samples,  # Generate multiple candidates at once
            ),
        )

        end_time = time.time()
        generation_time = end_time - start_time
        # Log generation completion and timing
        self.log.info(
            f"Time taken to generate for '{prompt}': {generation_time:.2f} sec"
        )
        self.log.info("Successfully ran chat")

        # Format response to match expected structure
        response_data = {
            "question": prompt,
            # Extract text from each candidate response
            "outputs": [
                candidate.content.parts[0].text for candidate in response.candidates
            ],
            "config": {
                "model": self.model,
                "system_instruction": SYSTEM_PROMPT,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "samples": n_samples,
            },
        }

        return response_data


class OpenAIAPIClient:
    """
    SDK client for OpenAI LLM
    """

    def __init__(self, model: str = "gpt-4o"):
        """
        Parameters:
        -----------
        model:  str
            Name of the model
            Default: OpenAI GPT 4o

        Returns:
        --------
        None
        """
        self.log = Logger("gpt_api_client").get_logger()
        self.model = model
        self.client = None

    def connect(self) -> bool:
        """
        Establish connection with OpenAI API

        Returns:
        --------
        Bool value to confirm connection
        """
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")

            self.client = OpenAI(api_key=api_key)
            self.log.info(f"Model chosen for OpenAI: {self.model}")
            return True
        except Exception as err:
            self.log.error(
                f"Error while establishing connection with model: {self.model} | Error: {err}"
            )
            return False

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 300,
        n_samples: int = 2,
    ) -> Dict:
        """
        Generate answer via OpenAI API

        Parameters:
        -----------
        prompt:         str
            Natural language description of the Verilog module to generate
        temperature:    float, default=0.2
            Controls randomness in generation (0=deterministic, 1=very random)
            Lower values produce more consistent, focused code
        max_tokens:     int, default=300
            Maximum number of tokens to generate per response
            Sufficient for most Verilog modules
        n_samples:      int, default=2
            Number of different code samples to generate

        Returns:
        --------
        Dict of all responses

        Raises:
        -------
        openai.OpenAIError
            If API call fails (rate limits, invalid API key, etc.)
        """
        self.log.info("Running chat")

        start_time = time.time()

        # Make API call to OpenAI
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    # System message instructs the model to behave as a Verilog generator
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,  # The actual Verilog generation request
                },
            ],
            temperature=temperature,  # Control randomness
            max_tokens=max_tokens,  # Limit response length
            n=n_samples,  # Generate multiple samples at once
        )

        end_time = time.time()
        generation_time = end_time - start_time
        # Log generation completion and timing
        self.log.info(
            f"Time taken to generate for '{prompt}': {generation_time:.2f} sec"
        )

        self.log.info("Successfully ran chat")

        # Extract token usage information from the response
        usage = response.usage

        # Calculate tokens per sample (OpenAI gives total, so divide by n_samples)
        # Note: Input tokens are the same for all samples
        tokens_per_sample = []
        if n_samples > 0:
            avg_completion_tokens = usage.completion_tokens / n_samples
            tokens_per_sample = [avg_completion_tokens] * n_samples

        # Calculate tokens per second metrics
        tokens_per_second = (
            usage.completion_tokens / generation_time if generation_time > 0 else 0
        )
        total_tokens_per_second = (
            usage.total_tokens / generation_time if generation_time > 0 else 0
        )

        # Format response to match expected structure for evaluation
        # This ensures consistency across different LLM providers
        response_data = {
            "question": prompt,  # Original prompt for reference
            # Extract generated text from each completion choice
            "outputs": [choice.message.content for choice in response.choices],
            "config": {  # Configuration used for generation
                "model": self.model,
                "system_instruction": SYSTEM_PROMPT,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "samples": n_samples,
            },
            "time": generation_time,
            # Token usage information
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "tokens_per_sample": tokens_per_sample,
            "avg_tokens_per_sample": usage.completion_tokens / n_samples
            if n_samples > 0
            else 0,
            "output_tokens_per_second": tokens_per_second,
            "total_tokens_per_second": total_tokens_per_second,
        }

        return response_data


class OpenSourceLLMClient:
    """
    SDK client for all OSS LLM
    """

    def __init__(
        self,
        model_id: str = "codellama",
        model_name: str = "meta-llama/CodeLlama-7b-Instruct-hf",
        device: str = "cuda",
        training_mode: bool = False,
    ):
        """
        Initializes the tokenizer and model once for efficiency.

        Parameters:
        -----------
        model_id:   str
            Short identifier for the model (used for logging)
        model_name: str
            Full HuggingFace model path/name
        device:     str
            Device for GPU acceleration (if available)
        training_mode:    bool
            Used in training (RLFT) the LLM

        Notes:
        ------
        - Models are loaded in float16 precision to save memory
        - Uses device_map="auto" for automatic GPU/CPU distribution
        - First run will download the model (~13GB)
        """
        self.model_id = model_id
        self.model_name = model_name
        self.device = device
        self.training_mode = training_mode
        if self.training_mode:
            SYSTEM_PROMPT = "You are a Verilog code generator. Output your reasoning on the workflow of the code in <reason>...</reason> format and generate the Verilog code enclosed in ```verilog...```."

        # Initialize logger with model-specific name
        self.log = Logger(model_id).get_logger()

    def connect(self) -> bool:
        """
        Establish connection with LLM

        Returns:
        --------
        Bool value to confirm connection
        """
        # Load HuggingFace API token from environment file
        # Required for accessing gated models like CodeLlama
        try:
            api_key = os.getenv("HUGGINGFACE_API_KEY")
            if not api_key:
                raise ValueError("HUGGINGFACE_API_KEY not found in environment")

            login(token=api_key)
            self.log.info(f"Model chosen for HUGGING_FACE: {self.model_name}")
            # Load tokenizer - converts text to tokens the model understands
            self.log.info(f"Loading tokenizer from {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
            )

            # Load the actual model weights
            self.log.info(f"Loading model from {self.model_name}...")
            self.log.info(
                "This may take several minutes on first run as it is downloading"
            )

            # Initialize model with memory-efficient settings
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,  # Use half precision to save memory
                device_map="auto",  # Automatically distribute across available devices
                low_cpu_mem_usage=True,  # Optimize CPU memory usage during loading
            )

            self.log.info("Model loaded successfully!")
            # Set model to evaluation mode (disables dropout, etc.)
            if not self.training_mode:
                self.model.eval()
            return True
        except Exception as err:
            self.log.error(
                f"Error while establishing connection with model: {self.model_name} | Error: {err}"
            )
            return False

    def lora_adapters(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: List[str] = None,
        task_type: str = "CAUSAL_LM",
    ):
        """
        Apply LoRA adapters to the loaded model for efficient fine-tuning

        Parameters:
        -----------
        r : int
            Rank of the LoRA decomposition (lower = fewer parameters)
        lora_alpha : int
            Scaling parameter for LoRA
        lora_dropout : float
            Dropout probability for LoRA layers
        target_modules : List[str]
            Specific modules to apply LoRA to (None = auto-detect)
        task_type : str
            Type of task (CAUSAL_LM for generation)

        Returns:
        --------
        self.model : PeftModel
            Model wrapped with LoRA adapters
        """
        try:
            self.log.info("Configuring LoRA adapters...")

            # Auto-detect target modules if not specified
            if target_modules is None:
                # For DeepSeek-Coder and similar models
                target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]

            # Create LoRA configuration
            lora_config = LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )

            # Apply LoRA to the model
            self.model = get_peft_model(self.model, lora_config)
            # Enable gradient checkpointing for memory efficiency
            self.model.enable_input_require_grads()
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
                self.log.info("Gradient checkpointing enabled")

            # Print trainable parameters info
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            all_params = sum(p.numel() for p in self.model.parameters())
            trainable_percent = 100 * trainable_params / all_params

            self.log.info(
                f"LoRA applied successfully!\n"
                f"Trainable params: {trainable_params:,} || "
                f"All params: {all_params:,} || "
                f"Trainable%: {trainable_percent:.2f}%"
            )

            return self.model

        except Exception as e:
            self.log.error(f"Error applying LoRA adapters: {e}")
            raise

    def apply_deepspeed(
        self,
        gradient_accumulation_steps: int = 4,
        train_batch_size: int = 2,
        gradient_clipping: float = 1.0,
        zero_stage: int = 2,
        offload_optimizer: bool = True,
        offload_param: bool = False,
        bf16: bool = False,
        lr: float = 2e-5,
    ):
        """
        Apply DeepSpeed optimization to the loaded model for distributed training

        Parameters:
        -----------
        gradient_accumulation_steps : int
            Number of steps to accumulate gradients
        train_batch_size : int
            Training batch size per GPU
        gradient_clipping : float
            Max gradient norm for clipping
        zero_stage : int
            ZeRO optimization stage (1, 2, or 3)
        offload_optimizer : bool
            Offload optimizer states to CPU
        offload_param : bool
            Offload parameters to CPU (for ZeRO-3)
        bf16 : bool
            Use bfloat16 precision training
        lr : float
            Learning rate

        Returns:
        --------
        model_engine : DeepSpeed model engine
        optimizer : DeepSpeed optimizer
        lr_scheduler : Learning rate scheduler
        """
        try:
            self.log.info("Configuring DeepSpeed...")

            # DeepSpeed configuration
            ds_config = {
                "train_batch_size": train_batch_size * gradient_accumulation_steps,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "gradient_clipping": gradient_clipping,
                "steps_per_print": 10,
                "zero_optimization": {
                    "stage": zero_stage,
                    "offload_optimizer": {
                        "device": "cpu" if offload_optimizer else "none",
                        "pin_memory": True,
                    },
                    "offload_param": {
                        "device": "cpu" if offload_param else "none",
                        "pin_memory": True,
                    },
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                    "sub_group_size": 1e9,
                    "reduce_bucket_size": "auto",
                    "stage3_prefetch_bucket_size": "auto",
                    "stage3_param_persistence_threshold": "auto",
                    "stage3_max_live_parameters": 1e9,
                    "stage3_max_reuse_distance": 1e9,
                    "gather_16bit_weights_on_model_save": True,
                },
                "bf16": {
                    "enabled": bf16,
                },
                "fp16": {
                    "enabled": not bf16,
                    "auto_cast": False,
                    "loss_scale": 0,
                    "initial_scale_power": 16,
                    "loss_scale_window": 1000,
                    "hysteresis": 2,
                    "min_loss_scale": 1,
                },
                "optimizer": {
                    "type": "AdamW",
                    "params": {
                        "lr": lr,
                        "betas": [0.9, 0.999],
                        "eps": 1e-8,
                        "weight_decay": 0.01,
                    },
                },
                "scheduler": {
                    "type": "WarmupDecayLR",
                    "params": {
                        "warmup_min_lr": 0,
                        "warmup_max_lr": lr,
                        "warmup_num_steps": 100,
                        "total_num_steps": 1000,  # Adjust based on your training
                    },
                },
                "zero_allow_untested_optimizer": True,
                "wall_clock_breakdown": False,
            }

            # Initialize DeepSpeed
            self.model_engine, self.optimizer, self.lr_scheduler, _ = (
                deepspeed.initialize(
                    model=self.model,
                    config=ds_config,
                )
            )

            self.log.info(
                f"DeepSpeed initialized successfully!\n"
                f"ZeRO Stage: {zero_stage}\n"
                f"Offload Optimizer: {offload_optimizer}\n"
                f"Offload Parameters: {offload_param}\n"
                f"Precision: {'bf16' if bf16 else 'fp16'}"
            )

            # Set up for distributed training if available
            if dist.is_initialized():
                world_size = dist.get_world_size()
                rank = dist.get_rank()
                self.log.info(f"Distributed training: Rank {rank}/{world_size}")

            return self.model_engine, self.optimizer, self.lr_scheduler

        except Exception as e:
            self.log.error(f"Error applying DeepSpeed: {e}")
            raise

    def prepare_for_rlft(
        self,
        # LoRA parameters
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        # DeepSpeed parameters
        gradient_accumulation_steps: int = 4,
        train_batch_size: int = 4,
        zero_stage: int = 2,
        lr: float = 2e-5,
        # For PPO policy
        use_actor_critic: bool = True,
        use_deepspeed: bool = False,
    ):
        """
        Convenience method to prepare model for RLFT by applying both LoRA and DeepSpeed

        Usage:
        ------
        client = OpenSourceLLMClient(
            model_id="deepseek-coder",
            model_name="deepseek-ai/deepseek-coder-7b-instruct-v1.5"
        )
        client.connect()
        model_engine, optimizer, scheduler = client.prepare_for_rlft()
        """
        # First apply LoRA adapters
        self.lora_adapters(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        # Wrap with actor-critic if using PPO
        if use_actor_critic:
            self.model = self.prepare_actor_critic_model()
            self.log.info("Using actor-critic model for PPO")

        # Apply DeepSpeed or use simple optimizer
        if use_deepspeed:
            return self.apply_deepspeed(
                gradient_accumulation_steps=gradient_accumulation_steps,
                train_batch_size=train_batch_size,
                zero_stage=zero_stage,
                lr=lr,
            )
        else:
            # Simple optimizer setup without DeepSpeed
            self.log.info("Using standard optimizer without DeepSpeed")
            
            # Create optimizer
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01
            )
            
            # Create scheduler
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=100,
                gamma=0.95
            )
            
            # Create a mock model engine for compatibility
            class SimpleModelEngine:
                def __init__(self, model, optimizer, device):
                    self.module = model
                    self.optimizer = optimizer
                    self.device = device
                    self.tokenizer = None  # Will be set by caller
                    
                def backward(self, loss):
                    loss.backward()
                    
                def step(self):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            model_engine = SimpleModelEngine(self.model, optimizer, self.device)
            
            return model_engine, optimizer, lr_scheduler
    
    def prepare_actor_critic_model(self):
        """
        Wrap the base model with actor-critic architecture for PPO
        """
        # Create actor-critic model
        actor_critic_model = ActorCriticModel(self.model, self.tokenizer)
        
        # Get device from base model
        device = next(self.model.parameters()).device
        
        # Move value head to same device with float16
        actor_critic_model.value_head = actor_critic_model.value_head.to(device=device, dtype=torch.float16)
        
        # Initialize value head weights with float16
        with torch.no_grad():
            actor_critic_model.value_head.summary.weight.data.normal_(mean=0.0, std=0.02)
            actor_critic_model.value_head.summary.bias.data.zero_()
            actor_critic_model.value_head.value.weight.data.normal_(mean=0.0, std=0.02)
            actor_critic_model.value_head.value.bias.data.zero_()
        
        self.log.info(f"Actor-critic model prepared with value head on {device} with dtype float16")
        return actor_critic_model

    def batch_generate(
        self,
        prompts: List[str],
        temperature: float = 0.2,
        max_tokens: int = 300,
        n_samples: int = 2,
        training_mode: bool = False,
    ) -> List[Dict]:
        """
        Batch generate responses from loaded LLM

        Parameters:
        -----------
        prompts : List[str]
            Input prompt(s) describing the Verilog module to generate
        temperature : float
            Sampling temperature (lower = more deterministic, higher = more creative)
        max_tokens : int
            Maximum number of new tokens to generate
        n_samples : int
            Number of different outputs to generate for the same prompt
        training_mode : bool
            Placeholder for training mode; if True -> activate eval mode for response generation

        Returns:
        --------
        List[Dict]
            List of Dictionaries containing:
            - question: The input prompt
            - outputs: List of generated Verilog code samples
            - config: Generation configuration used
        """
        # Format messages in the chat template expected by the model
        all_messages = []
        for prompt in prompts:
            all_messages.append(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
            )

        # Convert messages to model input format
        inputs = self.tokenizer.apply_chat_template(
            all_messages,
            add_generation_prompt=True,  # Add the prompt that signals the model to generate
            tokenize=True,  # Convert to tokens
            padding=True,
            truncation=True,
            return_dict=True,  # Return as dictionary
            return_tensors="pt",  # Return PyTorch tensors
        ).to(self.device)  # Move to same device as model

        batch_size = len(prompts)
        input_token_counts = []
        # Count input tokens
        for i in range(batch_size):
            # Count non-padded tokens for each prompt
            input_token_counts.append((inputs["attention_mask"][i] == 1).sum().item())

        # Log generation start
        self.log.info(f"Running chat: {self.model_id} ...")
        start_time = time.time()

        # Generate responses
        # Before that ensure model.evals() is ran -> if in training mode
        if training_mode:
            self.model.eval()
        # Uses sampling with temperature to create diverse outputs
        # Generate for the batch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,  # Unpack input tensors
                do_sample=True,  # Enable sampling (vs greedy decoding)
                temperature=temperature,  # Control randomness
                max_new_tokens=max_tokens,  # Limit response length
                num_return_sequences=n_samples,  # Generate multiple samples
            )

        end_time = time.time()
        generation_time = end_time - start_time

        # Log generation completion and timing
        self.log.info(
            f"Time taken to generate for '{prompt}': {generation_time:.2f} sec"
        )
        self.log.info(f"Generating response: {self.model_id} ...")

        # Store for all responses
        responses = []
        for prompt_idx, prompt in enumerate(prompts):
            # Extract outputs for this specific prompt
            # outputs shape: [batch_size * n_samples, sequence_length]
            start_idx = prompt_idx * n_samples
            end_idx = start_idx + n_samples
            # Calculate output tokens for each sample
            prompt_outputs = []
            tokens_per_sample = []
            total_output_tokens = 0

            # Get the input length for THIS specific prompt
            input_length = inputs["input_ids"][prompt_idx].shape[0]

            for sample_idx in range(start_idx, end_idx):
                # Get the generated output for this sample
                output = outputs[sample_idx]

                # Extract only the generated portion (after input)
                generated_ids = output[input_length:]

                # Count actual generated tokens (excluding padding)
                output_token_count = (
                    (generated_ids != self.tokenizer.pad_token_id).sum().item()
                )
                tokens_per_sample.append(output_token_count)
                total_output_tokens += output_token_count

                # Decode to text
                generated_text = self.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
                prompt_outputs.append(generated_text)

            # Calculate metrics for this specific prompt
            prompt_time = generation_time / batch_size  # Average time per prompt
            output_tokens_per_second = (
                total_output_tokens / prompt_time if prompt_time > 0 else 0
            )
            total_tokens_per_second = (
                (input_token_counts[prompt_idx] + total_output_tokens) / prompt_time
                if prompt_time > 0
                else 0
            )

            # Log generation completion and timing
            self.log.info(
                f"Time taken to generate for '{prompt}': {generation_time:.2f} sec"
            )
            self.log.info(f"Generating response: {self.model_id} ...")

            # Create response dict for this prompt
            response = {
                "question": prompt,
                "outputs": prompt_outputs,
                "config": {
                    "model": self.model_name,
                    "system_instruction": SYSTEM_PROMPT,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "samples": n_samples,
                },
                "time": prompt_time,
                "input_tokens": input_token_counts[prompt_idx],
                "output_tokens": total_output_tokens,
                "total_tokens": input_token_counts[prompt_idx] + total_output_tokens,
                "tokens_per_sample": tokens_per_sample,
                "avg_tokens_per_sample": total_output_tokens / n_samples
                if n_samples > 0
                else 0,
                "output_tokens_per_second": output_tokens_per_second,
                "total_tokens_per_second": total_tokens_per_second,
            }
            responses.append(response)

        return responses

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 300,
        n_samples: int = 2,
    ) -> Dict:
        """
        Generates responses for a given prompt using the loaded LLM.

        Parameters:
        -----------
        prompt : str
            Input prompt describing the Verilog module to generate
        temperature : float
            Sampling temperature (lower = more deterministic, higher = more creative)
        max_tokens : int
            Maximum number of new tokens to generate
        n_samples : int
            Number of different outputs to generate for the same prompt

        Returns:
        --------
        Dict
            Dictionary containing:
            - question: The input prompt
            - outputs: List of generated Verilog code samples
            - config: Generation configuration used
        """

        # Format messages in the chat template expected by the model
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # Convert messages to model input format
        # apply_chat_template formats the conversation according to model's expectations
        if self.model_id == "verigen-finetuned":
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
            ).to(self.device)
        else:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,  # Add the prompt that signals the model to generate
                tokenize=True,  # Convert to tokens
                return_dict=True,  # Return as dictionary
                return_tensors="pt",  # Return PyTorch tensors
            ).to(self.device)  # Move to same device as model

        # Count input tokens
        input_token_count = inputs["input_ids"].shape[-1]

        # Log generation start
        self.log.info(f"Running chat: {self.model_id} ...")
        start_time = time.time()

        # Generate responses
        # Uses sampling with temperature to create diverse outputs
        outputs = self.model.generate(
            **inputs,  # Unpack input tensors
            do_sample=True,  # Enable sampling (vs greedy decoding)
            temperature=temperature,  # Control randomness
            max_new_tokens=max_tokens,  # Limit response length
            num_return_sequences=n_samples,  # Generate multiple samples
        )

        end_time = time.time()
        generation_time = end_time - start_time

        # Log generation completion and timing
        self.log.info(
            f"Time taken to generate for '{prompt}': {generation_time:.2f} sec"
        )
        self.log.info(f"Generating response: {self.model_id} ...")

        # Calculate output tokens for each sample
        tokens_per_sample = []
        total_output_tokens = 0

        for output in outputs:
            # Count tokens in this sample (excluding input tokens)
            output_tokens_in_sample = len(output) - input_token_count
            tokens_per_sample.append(output_tokens_in_sample)
            total_output_tokens += output_tokens_in_sample

        # Calculate tokens per second metrics
        output_tokens_per_second = (
            total_output_tokens / generation_time if generation_time > 0 else 0
        )
        total_tokens_per_second = (
            (input_token_count + total_output_tokens) / generation_time
            if generation_time > 0
            else 0
        )

        # Log generation completion and timing
        self.log.info(
            f"Time taken to generate for '{prompt}': {generation_time:.2f} sec"
        )
        self.log.info(f"Generating response: {self.model_id} ...")

        # Decode generated tokens back to text
        responses = {
            "question": prompt,
            "outputs": [
                self.tokenizer.decode(
                    output[inputs["input_ids"].shape[-1] :],
                    skip_special_tokens=True,
                )
                for output in outputs
            ],
            "config": {
                "model": self.model_name,
                "system_instruction": SYSTEM_PROMPT,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "samples": n_samples,
            },
            "time": generation_time,
            "input_tokens": input_token_count,
            "output_tokens": total_output_tokens,
            "total_tokens": input_token_count + total_output_tokens,
            "tokens_per_sample": tokens_per_sample,
            "avg_tokens_per_sample": total_output_tokens / n_samples
            if n_samples > 0
            else 0,
            "output_tokens_per_second": output_tokens_per_second,
            "total_tokens_per_second": total_tokens_per_second,
        }

        return responses
