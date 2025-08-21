"""
Main driver for all python scripts

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Aug 14th, 2025
Place:  Boston, MA
"""

import glob
import json
import re
import os
import sys
import pandas as pd
from dotenv import load_dotenv
from typing import Dict, List
from src.evals import Evals, ResponseEvals
from src.gcp import GoogleStorageClient
from src.logger import Logger
from src.models import (
    ClaudeAPIClient,
    GeminiAPIClient,
    OpenAIAPIClient,
    OpenSourceLLMClient,
)


class ENVLoader:
    """
    Loads all the ENV required in the local machine
    """

    def __init__(self):
        """
        Loads when object is instantiated
        """
        self.log = Logger("env_loader").get_logger()
        self._run()  # run when object is instantiated

    def _run(self):
        """
        Private method to load ENV
        """
        try:
            load_dotenv("secrets/models-api.env")
            self.log.info("ENV loaded successfully")
        except Exception as err:
            self.log.critical(f"Fatal Error on ENV loading: {err}")
            self.log.info("Exiting ...")
            sys.exit(-1)


class RunLLMPrompts:
    """
    Handles operation to run prompts for each model
    Helps in deducing model's accuracy in generating Verilog code with eval metrics
    """

    def __init__(self):
        """
        Loads when object is instantiated
        """
        self.log = Logger("llm_runner").get_logger()
        self.dataset_path = "dataset/testbench/verigen"
        self.models = [
            {"id": "code-llama", "name": "meta-llama/CodeLlama-7b-Instruct-hf"},
            {
                "id": "deepseek-coder",
                "name": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
            },
            {
                "id": "verireason-qwen-coder",
                "name": "Nellyw888/VeriReason-Qwen2.5-7b-RTLCoder-Verilog-GRPO-reasoning-tb",
            },
            {
                "id": "verigen-finetuned",
                "name": "shailja/fine-tuned-codegen-6B-Verilog",
            },
            {"id": "qwen-coder", "name": "Qwen/Qwen2.5-Coder-7B-Instruct"},
        ]
        self.claude_api = ClaudeAPIClient()
        self.gemini_api = GeminiAPIClient()
        self.openai_api = OpenAIAPIClient()
        self.prompts = []  # empty list
        self.answers = {}  # empty dict

    def collect_prompts(self) -> List[Dict]:
        """
        From the dataset, collect all the prompts

        Returns:
        --------
        List of Dict of all prompts and testbench code (from VeriGen Repo)
        """
        # Get all directories in test folder
        # Each directory contains one problem with metadata.json
        dirs = os.listdir(self.dataset_path)
        dirs.sort()
        self.log.info(f"Sorted DIRS from path: {self.dataset_path}")

        # Process each problem directory
        for dir in dirs:
            # Skip macOS metadata file
            if dir in [
                ".DS_Store",
                "prompts-summary.txt",
                "prompts-templates.txt",
                "test_ex.v",
            ]:
                self.log.info(f"Omiting misc DIR: {dir}")
                continue
            else:

                def create_pattern(file_pattern: str) -> str:
                    """
                    Inner function for small stuff

                    Parameters:
                    -----------
                    file_pattern: str
                        String-like pattern of the directory

                    Returns:
                    --------
                    String value representing directory path
                    """
                    
                    return os.path.join(self.dataset_path, dir, file_pattern)

                # answer_filepath = (glob.glob(pattern("answer_*"), recursive=True))[0]
                prompt_filepath = glob.glob(
                    create_pattern("prompt1_*"), recursive=True
                )[0]
                tb_filepath = glob.glob(create_pattern("tb_*"), recursive=True)[0]

                # answer = None
                prompt = None
                testbench = None

                # with open(answer_filepath, "r") as fs:
                #     answer = fs.read()
                with open(prompt_filepath, "r") as fs:
                    prompt = fs.read()
                with open(tb_filepath, "r") as fs:
                    testbench = fs.read()

                self.log.info(f"Prompt: {prompt}")
                self.prompts.append(
                    {
                        # "answer": answer,
                        "prompt": prompt,
                        "testbench": testbench,
                    }
                )

        self.log.info("Created all prompts!")
        prompts = self.prompts
        return prompts

    def run_all_models(
        self,
        temperature: float = 0.2,
        max_tokens: int = 300,
        n_samples: int = 2,
        prompt_size: int = None,
        run_gemini_api: bool = False,
    ) -> List[Dict]:
        """
        Runner method for all LLMs

        Parameters:
        -----------
        temperature:        float, default=0.2
            Controls randomness in generation (0=deterministic, 1=very random)
            Lower values produce more consistent, focused code
        max_tokens:         int, default=300
            Maximum number of tokens to generate per response
            Sufficient for most Verilog modules
        n_samples:          int, default=2
            Number of different code samples to generate
        prompt_size:        int, default=None
            Size of array of the prompts from the Problem set
        run_gemini_api:     bool, False
            Run Gemini API flag

        Returns:
        --------
        List of Dict of all answers generated by all models, and all filenames of the output
        """
        exception = ""  # For final log note
        filenames = []  # For filenames to copy to GCP storage

        # First case: Claude API
        res = self.claude_api.connect()
        if res:
            self.answers["claude"] = []
            for data in self.prompts[:prompt_size]:
                response = self.claude_api.generate(
                    prompt=data["prompt"],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n_samples=n_samples,
                )
                self.answers["claude"].append(
                    {"response": response, "testbench": data["testbench"]}
                )
            with open("prompts-claude.json", "w") as fs:
                json.dump(self.answers["claude"], fs, indent=4)
            self.log.info("Written file for claude")
            filenames.append("prompts-claude.json")
        else:
            self.log.error("Claude connection unavailable; Moving to next case...")
            exception = "Except Claude"

        # Second case: Gemini API
        if run_gemini_api:
            res = self.gemini_api.connect()
            if res:
                self.answers["gemini"] = []
                for data in self.prompts[:prompt_size]:
                    response = self.gemini_api.generate(
                        prompt=data["prompt"],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        n_samples=n_samples,
                    )
                    self.answers["gemini"].append(
                        {"response": response, "testbench": data["testbench"]}
                    )
                with open("prompts-gemini.json", "w") as fs:
                    json.dump(self.answers["gemini"], fs, indent=4)
                self.log.info("Written file for gemini")
                filenames.append("prompts-gemini.json")
            else:
                self.log.error("Gemini connection unavailable; Moving to next case...")
                if len(exception) == 0:
                    exception = "Except Gemini"
                else:
                    exception += ", Gemini"

        # Third case: OpenAI API
        res = self.openai_api.connect()
        if res:
            self.answers["openai"] = []
            for data in self.prompts[:prompt_size]:
                response = self.openai_api.generate(
                    prompt=data["prompt"],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n_samples=n_samples,
                )
                self.answers["openai"].append(
                    {"response": response, "testbench": data["testbench"]}
                )
            with open("prompts-openai.json", "w") as fs:
                json.dump(self.answers["openai"], fs, indent=4)
            self.log.info("Written file for openai")
            filenames.append("prompts-openai.json")
        else:
            self.log.error("OpenAI connection unavailable; Moving to next case...")
            if len(exception) == 0:
                exception = "Except OpenAI"
            else:
                exception += ", OpenAI"

        # Last case: OSS LLM local API
        for model_info in self.models:
            oss_llm_api = OpenSourceLLMClient(
                model_id=model_info["id"], model_name=model_info["name"]
            )
            res = oss_llm_api.connect()
            if res:
                self.answers[model_info["id"]] = []
                for data in self.prompts[:prompt_size]:
                    response = oss_llm_api.generate(
                        prompt=data["prompt"],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        n_samples=n_samples,
                    )
                    self.answers[model_info["id"]].append(
                        {"response": response, "testbench": data["testbench"]}
                    )
                with open(f"prompts-{model_info['id']}.json", "w") as fs:
                    json.dump(self.answers[model_info["id"]], fs, indent=4)
                self.log.info(f"Written file for {model_info['id']}")
                filenames.append(f"prompts-{model_info['id']}.json")
            else:
                self.log.error(
                    f"{model_info['id']} connection unavailable; Moving to next case..."
                )
                if len(exception) == 0:
                    exception = f"Except {model_info['id']}"
                else:
                    exception += f", {model_info['id']}"

        if len(exception) == 0:
            with open("prompts-all.json", "w") as fs:
                json.dump(self.answers, fs, indent=4)
            # filenames = []
            filenames.append("prompts-all.json")

        self.log.info("Collected all prompts!")
        answers = self.answers
        return (answers, filenames)


def summarize_eval(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Summarize metrics from an eval DataFrame into a single row with both
    overall and grouped-by-difficulty metrics.

    Parameters:
    -----------
    df: pd.DataFrame
        Dataframe of the metric per question
    name: name of the model

    Returns:
    -------
    Dataframe containing overall metric of a model
    """
    summary = {"name": name}

    # --- Pass@k overall & grouped ---
    summary["passk_overall"] = df["pass_k_metric_overall"].mean()
    grouped_passk = df.groupby("difficulty")["pass_k_metric_overall"].mean()
    for diff, val in grouped_passk.items():
        summary[f"passk_{diff}"] = val

    # --- Success rate overall & grouped ---
    df["success"] = df["first_correct_idx"] != -1
    summary["success_rate_overall"] = df["success"].mean()
    grouped_success = df.groupby("difficulty")["success"].mean()
    for diff, val in grouped_success.items():
        summary[f"success_rate_{diff}"] = val

    # --- Avg attempts overall & grouped ---
    df["attempts_required"] = df["first_correct_idx"].apply(
        lambda x: x + 1 if x != -1 else None
    )
    summary["avg_attempts_overall"] = int(df["attempts_required"].dropna().mean())
    grouped_attempts = df.groupby("difficulty")["attempts_required"].mean()
    for diff, val in grouped_attempts.items():
        summary[f"avg_attempts_{diff}"] = int(val)

    # --- Token metrics overall ---
    summary["avg_tokens_per_sample"] = df["metadata_avg_tokens_per_sample"].mean()
    summary["output_tokens_per_second"] = df["metadata_output_tokens_per_second"].mean()
    summary["total_tokens_per_second"] = df["metadata_total_tokens_per_second"].mean()

    return pd.DataFrame([summary])


if __name__ == "__main__":
    args = sys.argv
    log = Logger("main").get_logger()
    gcp_storage = GoogleStorageClient()
    gcp_storage.connect()

    if len(args) == 0:
        log.warning("No arguments passed")

    # Run Evals on response
    elif len(args) == 5 and (
        "evals" in args and "--response" in args and "--type" in args
    ):
        response_evals = ResponseEvals()
        dir_name = args[sys.argv.index("--type") + 1]
        results_path = "dataset/results"
        response_path = os.path.join(
            results_path, dir_name
        )  # path where all the prompts-*.json resides

        json_files = os.listdir(response_path)
        json_files.remove(
            "prompts-all.json"
        ) if "prompts-all.json" in json_files else None  # Temporary hack
        # Create directory to store response eval results
        evals_result_path = os.path.join(response_path, "evals")
        if not os.path.exists(evals_result_path):
            os.mkdir(
                evals_result_path
            )  # NOTE: ensure necessary permission is enabled before running script
        for json_file in json_files:
            if (
                os.path.isdir(os.path.join(response_path, json_file))
                or json_file == ".DS_Store"
            ):
                continue

            log.info(f"Running for model-file: {json_file}")
            responses = None
            with open(os.path.join(response_path, json_file), "r") as fs:
                responses = json.load(fs)

            # Empty list to capture evals for all question
            overall_evals = []
            # Iterated over questions for each model
            for data in responses:
                model = data["response"]["config"]["model"]
                question = data["response"]["question"]
                temperature = data["response"]["config"]["temperature"]
                max_tokens = data["response"]["config"]["max_tokens"]
                testbench = data["testbench"]
                # Metadata
                time_taken = data["response"]["time"]
                input_tokens = data["response"]["input_tokens"]
                output_tokens = data["response"]["output_tokens"]
                avg_tokens_per_sample = data["response"]["avg_tokens_per_sample"]
                total_tokens = data["response"]["total_tokens"]
                output_tokens_per_second = data["response"]["output_tokens_per_second"]
                total_tokens_per_second = data["response"]["total_tokens_per_second"]

                # Empty dict to capture eval per question
                question_eval = {}
                question_eval["question"] = question
                question_eval["metadata"] = {
                    "time": time_taken,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "avg_tokens_per_sample": avg_tokens_per_sample,
                    "total_tokens": total_tokens,
                    "output_tokens_per_second": output_tokens_per_second,
                    "total_tokens_per_second": total_tokens_per_second,
                }
                question_eval["evals"] = []
                # Iterated over each question in a model
                log.info(f"Running for question: {question}")
                for output in data["response"]["outputs"]:
                    payload = {
                        "model": model,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "output": output,
                        "testbench": testbench,
                    }
                    eval = response_evals.evaluate_response(response=payload)
                    question_eval["evals"].append(eval)
                overall_evals.append(question_eval)
            filename_pattern = r"prompts-(.*).json"
            match = re.match(
                filename_pattern, json_file, re.DOTALL
            )  # MATCH 100% possible (if conditions are correct)
            filepath = os.path.join(evals_result_path, f"evals-{match.group(1)}.json")
            with open(filepath, "w") as fs:
                json.dump(overall_evals, fs, indent=4)
                log.info(f"Written {match.group(1)} evals to: {filepath}")

    # Create CSV files containing useful metrics (per question and per model)
    elif len(args) == 5 and (
        "evals" in args and "--metrics" in args and "--type" in args
    ):
        evals = Evals()
        dir_name = args[sys.argv.index("--type") + 1]
        results_path = "dataset/results"
        response_path = os.path.join(
            results_path, dir_name
        )  # path where all the prompts-*.json resides

        # Create directory to store metrics in CSV file(s)
        metrics_result_path = os.path.join(response_path, "metrics")
        if not os.path.exists(metrics_result_path):
            os.mkdir(
                metrics_result_path
            )  # NOTE: ensure necessary permission is enabled before running script

        evals_path = os.path.join(
            response_path, "evals"
        )  # path where all the evals-*.json resides
        json_files = os.listdir(evals_path)

        ks = [1, 5, 8, 10]  # default sampling number
        # Looping over k's
        for k in ks:
            summary_df = pd.DataFrame()
            for json_file in json_files:
                if (
                    os.path.isdir(os.path.join(evals_path, json_file))
                    or json_file == ".DS_Store"
                    # TODO: handle the issue
                    # Issue: The output seems to be erroneous:
                    #   - No module at the beginning of the output
                    #   - However multiple answers found in each sample
                    # Possible Solution:
                    #   - Use prompt-style2 instead of prompt-style1
                    or json_file == "evals-verigen-finetuned.json"
                ):
                    continue

                with open(os.path.join(evals_path, json_file), "r") as fs:
                    responses = json.load(fs)
                    # Build per model for each questions
                    model_metrics = []
                    for idx, response in enumerate(responses):
                        eval = evals.per_question_eval(response, k)
                        eval["question"] = response["question"]
                        # Naive method of setting question difficulty
                        eval["difficulty"] = None
                        if idx in range(0, 5):
                            eval["difficulty"] = "advanced"
                        elif idx in range(5, 9):
                            eval["difficulty"] = "basic"
                        else:
                            eval["difficulty"] = "intermediate"
                        eval["metadata"] = response["metadata"]
                        model_metrics.append(eval)

                    # Flatten JSON -> pd.DataFrame
                    model_df = pd.json_normalize(model_metrics, sep="_")
                    # Reorder columns so "question" is always first
                    cols = ["question"] + [
                        c for c in model_df.columns if c != "question"
                    ]
                    model_df = model_df[cols]
                    filename_pattern = r"evals-(.*).json"
                    match = re.match(
                        filename_pattern, json_file, re.DOTALL
                    )  # MATCH 100% possible (if conditions are correct)
                    model_df.to_csv(
                        os.path.join(
                            metrics_result_path, f"metrics-{match.group(1)}_k={k}.csv"
                        ),
                        index=None,
                    )
                    log.info(
                        f"Collecting summary for model: {match.group(1)} for k={k}..."
                    )
                    # Add one run
                    summary_df = pd.concat(
                        [summary_df, summarize_eval(model_df, match.group(1))],
                        ignore_index=True,
                    )
                    log.info("✓ Done")
            # Write to a CSV file
            summary_df.to_csv(
                os.path.join(
                    metrics_result_path,
                    f"overall-metrics_k={k}.csv",
                ),
                index=None,
            )

    # Run each LLM -> generate response
    elif len(args) == 4 and ("prompt" in args and "--bucket" in args):
        bucket_name = args[sys.argv.index("--bucket") + 1]
        env = ENVLoader()
        runner = RunLLMPrompts()
        runner.collect_prompts()  # collect all prompts
        _, filenames = runner.run_all_models(n_samples=10)  # run for all models
        for filename in filenames:
            try:
                gcp_storage.upload_file(
                    local_file_path=filename,
                    bucket_name=bucket_name,  # could change in your machine
                    blob_name=f"results/{filename}",
                )
                log.info(f"Copied file {filename} to GCP storage: results/")
            except Exception as err:
                log.error(f"Error in uploading file: {err}")
        # Finally copy all log files
        log_dir = "logs/"
        if os.path.exists(log_dir):
            log_filenames = os.listdir("logs/")
            for log_filename in log_filenames:
                try:
                    gcp_storage.upload_file(
                        local_file_path=log_filename,
                        bucket_name=bucket_name,  # could change in your machine
                        blob_name=f"logs/{log_filename}",
                    )
                    print(f"Copied file {log_filename} to GCP storage: logs/")
                except Exception as err:
                    print(f"Error in uploading file: {err}")
        else:
            print("Skipping LOG file(s) upload as log/ isn't found")
    else:
        log.warning("Unknown argument used")
