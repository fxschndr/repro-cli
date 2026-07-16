import argparse
import pymupdf
import pymupdf.layout
import pymupdf4llm
import pathlib
import openai
import subprocess
import time
import pandas as pd
import os
import re
import requests
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Literal
from typing import List
from typing import Optional
from git import Repo
from bs4 import BeautifulSoup


load_dotenv(override=True)
client = OpenAI()

class ResearchPaperReproducibility(BaseModel):
    title: str
    article_number: int
    code_required: Literal["Yes", "No", "Unknown"]
    code_available: Literal["Yes", "Partially", "No", "Unknown", "N/A"]
    code_url: str
    data_required: Literal["Yes", "No", "Unknown"]
    data_available: Literal["Yes", "Partially", "No", "Unknown", "N/A"]
    data_url: str
    hardware_required: Literal["Yes", "No", "Unknown", "N/A"]
    hardware_notes: str
    claims: str

class Step(BaseModel):
    description: str
    command: Optional[str] = None
    optional: bool = False
    user_instruction: Optional[str] = None

class ExecutionPlan(BaseModel):
    setup: List[Step]
    execution: List[Step]
    repo_empty: bool = False
    hardware_requirements: str

timestamp = pd.Timestamp.now().isoformat()

total_start = time.time()

result_log = {

    "paper_id": None, #
    "timestamp": timestamp,#
    "outcome": None,            # final outcome 
    
    "title": None,          #
    "article_number": None,     #
    "code_required": None,#
    "code_available": None,#
    "code_url": None,#
    "data_required": None,#
    "data_available": None,#
    "data_url": None,#
    "hardware_required": None,#
    "hardware_notes": None,#
    "claims": None,#

    "repo_url": None,           # final giturl used for repo cloning
    "repo_hardware_requirements": None, #
    "failed_step": None,      # description of last step that failed #
    "success": None,
    "retries": 0,

    #time
    "analysis_duration_s": None,        #duration the llm analyzes the paper
    "plan_duration_s": None,            #total duration of planning, initial and fix if
    "execution_duration_s": None,       #duration of execution starting from confirm init plan
    "total_duration_s": None,           #total duuration of the pipeline

     # token usage
    "tokens_input": None,   #
    "tokens_output": None,  #
    "tokens_total": None,   #
}


execution_log = {
    # id
    "paper_id": None, 
    "timestamp": timestamp,
    "repo_url": None,#

    # outcome
    "success": None,#
    "retries": 0,   #
    "repo_empty": None,#

    # hardware
    "hardware_0": None,#
    "hardware_1": None,#
    "hardware_2": None,#
    "hardware_3": None,#

    # failed steps (human readable)
    "failed_step_0": None,#
    "failed_step_1": None,#          # just the description string
    "failed_step_2": None,#
    "failed_step_3": None,#

    # errors 
    "error_0": None,#
    "error_1": None,#               
    "error_2": None,#
    "error_3": None,#

    # plan sizes (human readable)
    "plan_initial_len": None,     # len(setup) + len(execution)
    "plan_fix_1_len": None,#
    "plan_fix_2_len": None,#
    "plan_fix_3_len": None,#

    # full JSON (for evtl programmatic analysis)
    "plan_initial_json": None,#
    "plan_fix_1_json": None,#
    "plan_fix_2_json": None,#
    "plan_fix_3_json": None,#
    "failed_step_0_json": None,#
    "failed_step_1_json": None,#
    "failed_step_2_json": None,#
    "failed_step_3_json": None,#
}


token_usage = {"input": 0, "output": 0, "total": 0}

def track_tokens(usage):
    token_usage["input"] += usage.input_tokens
    token_usage["output"] += usage.output_tokens
    token_usage["total"] += usage.input_tokens + usage.output_tokens


def convertToMd(path, paper_id):
    print("Converting PDF to Markdown")

    md_text = pymupdf4llm.to_markdown(path,ignore_graphics=True , ignore_images=True, header=False, footer=False)
    pathlib.Path(f"{paper_id}.md").write_bytes(md_text.encode()) 
    return md_text



analysis_message = """
You are an expert in analyzing research papers for reproducibility.

Your task is to extract structured information based on the provided markdown text. Images are intentionally omitted

STRICT RULES:

- Do NOT guess information but use contextual clues from the paper
- Prefer "Yes" or "No" over "Unknown" when evidence exists, even if indirect.
- Do not default to "Unknown" when a reasonable interpretation can be made.
- If something is not stated or cannot be inferred, return "Unknown"

GENERAL:

- Extract the title EXACTLY as written
- Extract URLs exactly if present
- Output must strictly follow the given schema

CRITERIA RULES:

code_required:
- "Yes" only if code is required to reproduce the research
- "No" only if no code is needed e.g. the research is purely theoretical/descriptive, or a user study
- Otherwise "Unknown"

code_available:
- "Yes" only if there is a URL to the published code repository (e.g., GitHub) 
- "Partially" if only parts are available
- "Upon_request" only if stated code is available upon request from the authors
- "No" if code is required and there is no link to the code 
- "N/A", Code is not required (code_required = No).
- Otherwise "Unknown"

code_url:
- Direct URL to the published code repository (e.g., GitHub), if available; empty otherwise.

data_required:
- "Yes" if datasets are needed for reproducing the research
- "No" if no data is needed, e.g User Study
- Otherwise "Unknown"

data_available:
- "Yes" only if a repository/link to the data is found or a public dataset is used and mentioned
- "Partially" if only parts are available
- "Upon_request" only if explicitly stated data is available upon request from the authors
- "No" only if data is required but there is no url to the Repository or public dataset used
- "N/A", Data is not required (code_required = No).
- Otherwise "Unknown"

data_url:
- Direct URL to the dataset (e.g., GitHub, Kaggle), if available; empty otherwise.

hardware_required:
- "Yes" if special hardware is explicitly required (e.g., sensors, wearables, GPUs)
- "No" if standard computing is sufficient
- "N/A", Not applicable (e.g., pure literature review).
- Otherwise "Unknown"

hardware_notes: short description of the hardware used and its role in the setup.

claims_notes:
- the claims from the paper, describing what should be reproduced 
- Copy EXACT sentences from the text
- Do NOT paraphrase

title: The full title of the paper, copied exactly as published. This must not be paraphrased or modified.

article_number: The article number assigned by the publisher (ACM IMWUT article number).

EXAMPLES:

-repo url e.g. github found and mentioned that the code is published there -> code_avaiable:"Yes"
-url found and mentioned that the code and data is published there -> code_avaiable:"Yes", data_available:"Yes"
-code_available:"Yes" -> code_required:"Yes"
-public dataset used and mentioned -> data_available:"Yes"
-code required but no code_url found -> code available:"No"
-data required but no url or mention of public dataset -> data_available:"No"
-implementation, models,...  described -> code_required:"Yes"
-results depend on collected or external data -> data_required:"Yes"


"""


def analyzePaper(md_text):

    user_message = f"""
    Extract reproducibility information from the following paper.

    PAPER TEXT:
    {md_text}
    """

    start_time = time.time()
    response = client.responses.parse(
        model="gpt-5.2",
        # gpt-5-nano
        reasoning={"effort": "high"},
        input=[
            {
                "role": "developer",
                "content": analysis_message,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": user_message,
                    }
                ]
            }
        ],
        text_format= ResearchPaperReproducibility
    )
    end_time = time.time()

    # print(response.output_parsed)
    print(response.usage)
    track_tokens(response.usage)

    duration = end_time - start_time
    print(f"Total request time: {duration} seconds")

    #output results and usage 

    # results_path = "results.csv"

    # results_df = pd.DataFrame([response.output_parsed.model_dump()])

    # if os.path.exists(results_path):
    #     results_df.to_csv(results_path, mode="a", header=False, index=False)
    # else:
    #     results_df.to_csv(results_path, mode="w", header=True, index=False)

    usage_path = "usage.csv"

    usage_df = pd.DataFrame([response.usage.model_dump()])

    if os.path.exists(usage_path):
        usage_df.to_csv(usage_path, mode="a", header=False, index=False)
    else:
        usage_df.to_csv(usage_path, mode="w", header=True, index=False)

    result : ResearchPaperReproducibility = response.output_parsed

    #logging
    
    result_log["analysis_duration_s"] = duration

    result_log["title"] = result.title
    result_log["article_number"] = result.article_number
    result_log["code_required"] = result.code_required
    result_log["code_available"] = result.code_available
    result_log["code_url"] = result.code_url
    result_log["data_required"] = result.data_required
    result_log["data_available"] = result.data_available
    result_log["data_url"] = result.data_url
    result_log["hardware_required"] = result.hardware_required
    result_log["hardware_notes"] = result.hardware_notes
    result_log["claims"] = result.claims

    return result


def clone_repo(repo_url: str, base_dir: str = "repos") -> Path:
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)

    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    repo_path = base_path / repo_name

    if repo_path.exists():
        print("Repository already exists.")
        return repo_path
    try:   
        Repo.clone_from(repo_url, repo_path, depth=1)
        print(f"Cloned into {repo_path}")
        return repo_path

    except Exception as e:
        print(f"Cloning failed: {e}")
        return None


def validate_github_repo(url):
    pattern = r"^https://github\.com/[\w\-]+/[\w\-\.]+"
    return re.match(pattern, url)

def scrape_repo_from_io(url):

    r = requests.get(url, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")

    for a in soup.find_all("a", href=True):

        href = a["href"]

        if "github.com" in href and "github.io" not in href:
            return href

    return url



def find_readmes(repo_path):
    readmes = []

    for file in Path(repo_path).rglob("*"):
        name = file.name.lower()

        if name.startswith("readme"):
            readmes.append(file)

    return readmes


def load_readmes(readme_paths):
    contents = []

    for path in readme_paths:
        try:
            text = path.read_text(errors="ignore")

            contents.append(
                f"\n### {path}\n{text}"
            )

        except Exception:
            pass

    return "\n".join(contents)



developer_message = """
You are an expert system for generating executable plans from software repositories.

Your task is to analyze repository structure and documentation and produce a structured execution plan which can be run step by step using python subprocess.

STRICT RULES:

- output ONLY the valid schema
- follow the schema exactly
- follow the readme instructions provided as documentation
- Do NOT invent files that are not present use the provided tree for information
- Prefer CPU-based execution
- Be concise and practical
- ALWAYS activate a virtual environment like venv or conda before installing an environment + dependencies and activate it before running any code 


REPO EVALUATION:

- If repository is empty or not usable, set repo_empty = true and leave steps empty

STEP RULES:

Each step must contain:
- description: short explanation
- command: a valid shell command OR null if not applicable
- optional: true/false
- user_instruction: null OR a clear instruction for the user

Use user_instruction when:
- manual download is required
- login/authentication or api key is required
- external setup cannot be automated


STRUCTURE:

- setup: environment + dependencies
- execution: running the main code


HARDWARE:

- Detect if GPU or other hardware is required
- Set hardware_requirements to a short string like:
  "cpu"
  "gpu recommended"
  "gpu required"
  "external hardware"

DIRECTORY CONTEXT RULES:
- All commands will be executed with the repository root as the Current Working Directory (cwd).
- DO NOT use `cd` to enter the repository folder. You are already inside it.
- Use relative paths based on the provided structure.

ENVIRONMENT INFORMATION:
- python, pip, and conda are already installed
- system runs Fedora Linux
- commands are run using the following code:

subprocess.run(
    ["bash", "-lc", step.command],
    text=True,
    capture_output=True,
    cwd=repo_path
    )

OUTPUT FORMAT:
- use provided schema
"""


def analyze_repo(repo_path, tree, readme_text):
   
    user_message = f"""
    Repository structure:
    {tree}

    Documentation:
    {readme_text}

    Task:
    Generate an execution plan for this repository.
    """


    start_time = time.time()
    response = client.responses.parse(
        model="gpt-5.2",
        # gpt-5-nano
        reasoning={"effort": "high"},
        input=[
            {
                "role": "developer",
                "content": developer_message,
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        text_format= ExecutionPlan
    )
    end_time = time.time()

    # print(response.output_parsed)
    print(response.usage)

    track_tokens(response.usage)


    duration = end_time - start_time
    print(f"Total request time: {duration} seconds")

    #output results and usage 

    # results_path = "results.csv"

    # results_df = pd.DataFrame([response.output_parsed.model_dump()])

    # if os.path.exists(results_path):
    #     results_df.to_csv(results_path, mode="a", header=False, index=False)
    # else:
    #     results_df.to_csv(results_path, mode="w", header=True, index=False)

    usage_path = "usage.csv"

    usage_df = pd.DataFrame([response.usage.model_dump()])

    if os.path.exists(usage_path):
        usage_df.to_csv(usage_path, mode="a", header=False, index=False)
    else:
        usage_df.to_csv(usage_path, mode="w", header=True, index=False)

    result : ExecutionPlan = response.output_parsed

    return result


def print_steps(plan):
    for phase_name, steps in [("SETUP", plan.setup), ("EXECUTION", plan.execution)]:
        print(f"\n{'='*50}")
        print(f" PHASE: {phase_name}")
        print(f"{'='*50}")
        for i, step in enumerate(steps, 1):
            print(f"\n[Step {i}]")
            print(f"  Description    : {step.description}")
            print(f"  Command        : {step.command or 'N/A'}")
            print(f"  Optional       : {step.optional}")
            if step.user_instruction:
                print(f"!User Action! : {step.user_instruction}")

    print(f"\n{'='*50}")
    print(f"  Hardware       : {plan.hardware_requirements}")
    print(f"  Repo Empty     : {plan.repo_empty}")
    print(f"{'='*50}\n")


def execute_plan(plan, repo_path):
    print("\nSETUP PHASE")

    for step in plan.setup:
        success, error = run_step(step, repo_path)

        if not success:
            if step.optional:
                print("Optional step failed -> continuing")
                continue
            if error == "skip":
                print("Step skipped by user")
                continue
            if error == "abort":
                print("Execution ended by user")
                return False, step, error

            return False, step, error

    print("\nEXECUTION PHASE")

    for step in plan.execution:
        success, error = run_step(step, repo_path)

        if not success:
            if step.optional:
                print("Optional step failed -> continuing")
                continue
            if error == "skip":
                print("Step skipped by user")
                continue
            if error == "abort":
                print("Execution ended by user")
                return False, step, error

            return False, step, error

    return True, None, None



def run_step(step, repo_path):
    """
    Run a single step.
    Returns:
        (True, None)           — success
        (False, error_output)  — failure, with captured error
    """
    if step.user_instruction:
        print("Manual action required:")
        print(step.user_instruction)
        if step.command:
            print(f"Run this when ready:\n{step.command}")
            confirm = input("Press Enter when done or c to abort")
            if confirm in ["c"]: return False, "abort"

        else:
            confirm = input("Press Enter when done or c to abort")
            if confirm in ["c"]: return False, "abort"
            return True, None

    if not step.command or step.command.strip() == "true":
        print(f"No command, skipping.")
        return True, None

    print("Running:" + step.command)
    print("Description:" + step.description)
    confirm = input("Run this step? (Y/n) or press c to abort: ").lower()
    if confirm in ["c"]: return False, "abort"
    if confirm not in ["", "y"]:
        print("Skipped by user")
        return False, "skip"

    # result = subprocess.run(
    #     step.command,
    #     shell=True,
    #     executable="/bin/bash",
    #     text=True,
    #     capture_output=True,
    #     cwd=repo_path
    # )

    result = subprocess.run(
    ["bash", "-lc", step.command],
    text=True,
    capture_output=True,
    cwd=repo_path
    )

    if result.returncode == 0:
        print(f"Done!")
        if result.stdout:
            print(result.stdout)
        return True, None

    error_output = (result.stdout + "\n" + result.stderr).strip()
    print(f"Failed with exit code {result.returncode}")
    print(f"Error:\n{error_output}")
    return False, error_output



def fix_plan_with_llm(failed_step, error_output, plan, tree, readme_text):

    fix_message = f"""
    A step in the execution plan failed. Return a revised ExecutionPlan starting with fixing the failed step.

    FAILED STEP:
    {failed_step.model_dump_json(indent=2)}

    ERROR OUTPUT:
    {error_output}

    ORIGINAL PLAN:
    {plan.model_dump_json(indent=2)}

    REPOSITORY STRUCTURE:
    {tree}

    DOCUMENTATION:
    {readme_text}

    RULES:
    - Return a valid ExecutionPlan
    - the plan should contain only the steps from the failed step onwards (including the fix)
    - execution should contain the original execution steps unchanged if setup failed
    - You may add, remove, or modify steps to fix the error
    - if already setup use the same environment as the original plan
    """

    response = client.responses.parse(
        model="gpt-5.2",
        reasoning={"effort": "high"},
        input=[
            {"role": "developer", "content": developer_message},
            {"role": "user", "content": fix_message}
        ],
        text_format=ExecutionPlan
    )

    print(response.usage)
    track_tokens(response.usage)

    usage_path = "usage.csv"

    usage_df = pd.DataFrame([response.usage.model_dump()])

    if os.path.exists(usage_path):
        usage_df.to_csv(usage_path, mode="a", header=False, index=False)
    else:
        usage_df.to_csv(usage_path, mode="w", header=True, index=False)


    result : ExecutionPlan = response.output_parsed

    return result

# Update outcome and write logs to csv

def log_result(outcome: str):
        
        result_log["outcome"] = outcome

        result_log["total_duration_s"] = time.time() - total_start

        result_log["tokens_input"] = token_usage["input"]
        result_log["tokens_output"] = token_usage["output"]
        result_log["tokens_total"] = token_usage["total"]

        results_path = "results.csv"

        res = pd.DataFrame([result_log])

        execution_path = "execution.csv"

        ex = pd.DataFrame([execution_log])

        if os.path.exists(results_path):
            res.to_csv(results_path, mode="a", header=False, index=False)
        else:
            res.to_csv(results_path, mode="w", header=True, index=False)

        if os.path.exists(execution_path):
            ex.to_csv(execution_path, mode="a", header=False, index=False)
        else:
            ex.to_csv(execution_path, mode="w", header=True, index=False)

        print(f"Result logged: {outcome}")



def main():
   
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="Path to PDF file")
    args = parser.parse_args()

    pdf_path = args.pdf

    if not os.path.exists(pdf_path):
        print(f"pdf-not-found")
        log_result("pdf-not-found")
        return

    paper_id = Path(pdf_path).stem 
    result_log["paper_id"] = paper_id
    execution_log["paper_id"] = paper_id
    
    md_text = convertToMd(pdf_path, paper_id)

    print("Analyzing Paper...")

    result = analyzePaper(md_text)  #get result from llm paper analysis

    if result.code_required == "No":
        print("No code required, Reproducibility must be checked by practically repating the steps from the paper.")
        
        if result.code_url != "":
            print("But url found continuing with analyzing given code")    
        else:
            log_result("code-not-required")
            return

    if result.code_available != "Yes":
        print("There is no code available to reproduce the results from the paper so the program cannot proceed.")

        if result.code_url != "":
            print("But url found continuing with analyzing given code")    
        else:
            log_result("code-not-available")
            return

    

# Code required/given and github repo

    if not validate_github_repo(result.code_url):
        print("Scanning page for github repository")
        url = scrape_repo_from_io(result.code_url)
    else:
        url = result.code_url

    # if not url or "github" not in url:
    #     print("The code can be found at: " + result.code_url)
    #     log_result("non-github-repo", result.code_url)
    # return
    
    # manual url correction for adashark md failure
    # url="https://github.com/MarvinMartin24/ADAShark-Human-Performance/"  

    print("The following repo has been found:" + url)
    # print("Would you like to clone it? (Y/n)")
    result_log["repo_url"] = url
    execution_log["repo_url"] = url


    # clone = input("Clone repository? (Y/n): ").strip().lower()

    # if clone in ["", "y", "yes"]:
    #     print("cloning...")
    #     repo_path = clone_repo(repo_url=url)
    #     if repo_path is None:
    #         log_result("cloning-failed") 
    #         return
    #     print("Repository cloned successfully.")
    #     print()
                
    #     print("Repository structure:")
    #     tree = subprocess.check_output(["tree", "-L", "2", repo_path]).decode()
    #     print(tree)

    # else:
    #     print("Skipping repository cloning.")
    #     log_result("abort-before-cloning")
    #     return


    print("cloning...")
    repo_path = clone_repo(repo_url=url)
    if repo_path is None:
        log_result("cloning-failed") 
        return
    print("Repository cloned successfully.")
    print()
    
    
    print("Repository structure:")
    tree = subprocess.check_output(["tree", "-L", "2", repo_path]).decode()
    print(tree)
    
    # continue_repo = input("Continue with analyzing repository and generating execution plan? (Y/n): ").strip().lower()

    # if continue_repo not in ["", "y", "yes"]:
    #     print("Abort before analysing and plan generation.")
    #     log_result("abort-after-git-clone")
    #     return
        
    print("searching for readmes")
    readme_paths = find_readmes(repo_path)
    readme_text = load_readmes(readme_paths)


    print("analyzing repository...")
    start_time = time.time()
    plan = analyze_repo(repo_path, tree, readme_text)
    end_time = time.time()

    result_log["plan_duration_s"] = end_time - start_time
    execution_log["plan_initial_json"] = plan.model_dump_json()
    execution_log["plan_initial_len"] = len(plan.execution) + len(plan.setup)
    execution_log["hardware_0"] = plan.hardware_requirements
    result_log["repo_hardware_requirements"] = plan.hardware_requirements
    execution_log["repo_empty"] = plan.repo_empty

    print_steps(plan) 

    if plan.repo_empty:
        log_result("repo-unusable")
        return

    continue_execute = input("Continue with running execution plan? (Y/n): ").strip().lower()
    if continue_execute not in ["", "y", "yes"]:
        print("init plan not run")
        log_result("abort-after-init-plan")
        return
    exec_start = time.time()
    success, failed_step, error = execute_plan(plan, repo_path)

    
    execution_log["retries"] = 0
    execution_log["success"] = success
    result_log["success"] = success
    result_log["retries"] = 0

    if not success:
        print(f"\nX Plan failed at step: {failed_step.description}")
        print(f"Error: {error}")
        execution_log["failed_step_0"] = failed_step.description
        execution_log["failed_step_0_json"] = failed_step.model_dump_json()
        execution_log["error_0"] = error
        
         

        retries = 0 
        max_retries = 3    

        if error == "abort": retries = max_retries

        while retries<max_retries:   

            continue_fix = input("Continue with fixing execution plan? (Y/n): ").strip().lower()
            if continue_fix in ["", "y", "yes"]:
                print("fixing...")

                start_time = time.time()
                plan = fix_plan_with_llm(failed_step, error, plan, tree, readme_text)
                end_time = time.time()

                result_log["plan_duration_s"] += end_time - start_time

                print_steps(plan) 

                if plan.repo_empty:
                    result_log["execution_duration_s"] = time.time() - exec_start
                    log_result("repo-unusable")
                    return
                
                success, failed_step, error = execute_plan(plan, repo_path)
                retries+=1


                execution_log[f"plan_fix_{retries}_json"] = plan.model_dump_json()
                execution_log[f"plan_fix_{retries}_len"] = len(plan.execution) + len(plan.setup)
                execution_log[f"hardware_{retries}"] = plan.hardware_requirements

                result_log["repo_hardware_requirements"] = plan.hardware_requirements

                execution_log["retries"] = retries
                execution_log["success"] = success
                result_log["success"] = success
                result_log["retries"] = retries

                if not success:
                    print(f"\nX Plan failed at step: {failed_step.description}")
                    print(f"Error: {error}")

                    execution_log[f"failed_step_{retries}"] = failed_step.description
                    result_log["failed_step"] = failed_step.description
                    execution_log[f"failed_step_{retries}_json"] = failed_step.model_dump_json()
                    execution_log[f"error_{retries}"] = error

                    if error == "abort": retries = max_retries

                else:
                    print("success")
                    result_log["execution_duration_s"] = time.time() - exec_start
                    log_result("fix-plan-success")
                    return
                    
            else:
                print("abort retry")

                execution_log["retries"] = retries
                execution_log["success"] = success
                result_log["success"] = success
                result_log["retries"] = retries
                   
                result_log["execution_duration_s"] = time.time() - exec_start
                log_result("fix-execution-abort")
                return



        print(f"unsuccessful abort after {retries} retries ")
        
        execution_log["retries"] = retries
        execution_log["success"] = success
        result_log["success"] = success
        result_log["retries"] = retries

        result_log["execution_duration_s"] = time.time() - exec_start
        log_result("fix-plan-abort")

        return

    print("init plan success")
    result_log["execution_duration_s"] = time.time() - exec_start
    log_result("init-plan-run-success")


if __name__ == '__main__':
    main()
