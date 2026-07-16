# repro-cli

Semi-automated pipeline for reproducing research papers using LLMs. Takes a paper PDF, extracts metadata, clones the linked repository, generates an execution plan, and runs it with a retry loop on errors.

Developed as part of a Bachelor's thesis at the University of Siegen.

The system was developed and tested under Python Version 3.12.12 using the following environment:

 `$ conda create --name <env> --file requirements.txt`
 
 `$ conda activate <env>`

Insert your OpenAI API key into the `.env` file or create one with: `"OPENAI_API_KEY=sk-your-key-here"` inside.

The tool can be started using the following command:

 `$ python reprocli.py path/to/paper.pdf`
 
 ## Output

- `results.csv` — outcome and extracted metadata per paper
- `execution.csv` — detailed execution log, plans, errors, retries
- `usage.csv` — OpenAI API token usage
- `repos/` — cloned repositories
