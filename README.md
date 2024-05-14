# UXR Chatbot (AskUXR clone)

This chatbot uses RAG (Retrieval-Augmented Generation) to answer questions based on the [UX Research Playbook](https://pages.github.ibm.com/reops/ux-research/research-practice-playbook/overview/). It's a clone of the [AskUXR tool](https://askuxr.dal1a.cirrus.ibm.com/) released in December 2023 under the GenAI for UXR mission sponsored by Karel Vredenburg and developed by Carlos Rosemberg and Gobind Bakhshi, with the support of the UI&E team led by Gord Davison. See [more information about AskUXR](https://pages.github.ibm.com/reops/ux-research/ux-research-tools/askuxr).


## How it works

For learning how RAG works, please refer to [this explanation on watsonx documentation](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-rag.html?context=wx&audience=wdp).

To run this RAG tool locally:

1. Run the environment setup (see below).
2. To add your own content, generate TXT files in the data-input folder. In this solution, the file/folder replicates the same structure on the UXR website.
3. (Optional) For better results, adjust the content in those files for better LLM ingestion. Learn more [here](https://w3.ibm.com/w3publisher/adapting-content-for-ai).
4. (Optional) Run ingest.py (`python ingest.py`) to chunk the content and generate the vector indexes (vector database used by the solution). Do this whenever you add/remove TXT files or change their content.
5. Run `streamlit run main.py` to open the application. A streamlit dialog may appear in the terminal window if this is the first time ever it runs, just hit Esc or Enter.


Feel free to change it to your use case, and in case of any questions feel free to reach out to Carlos Rosemberg @Carlos Rosemberg or [carlos.rosemberg@ibm.com](mailto:carlos.rosemberg@ibm.com).

## Environment setup
Follow the steps below to run this app locally on your machine.

### Check requirements

- [Python >= 3.9](https://www.python.org/downloads/) installed
- [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) 
- A [watsonx account](https://dataplatform.cloud.ibm.com/wx/home?context=wx)
- An [IBM Cloud Api Key](https://cloud.ibm.com/iam/apikeys)
- (Optional) An Airtable API key if you want to store user feedback on Airtable.

### Clone the Repository

Using the terminal (MacOs) go to the folder you want the project folder to be and type the command:
```
git clone git@github.ibm.com:carlos-rosemberg/askuxrclone.git
```

### Create the virtual environment
```
python -m venv .venv
```

### Activate the Virtual environment
```
source .venv/bin/activate
```

### Install dependencies
```
pip install -r requirements_local.txt
```
(Note: the `requirements.txt` file is for deployment in production, feel free to ignore.)

### .env file
Create a .env file with the following content:
```
API_KEY=<your IBM Cloud API key>
IBM_CLOUD_URL=https://us-south.ml.cloud.ibm.com
PROJECT_ID=<Your watsonx project ID>
AIRTABLE_API_KEY=<Your Airtable key if you want to capture user feedback on Airtable>
DEV_MODE_FLAG=True
```

### Run the app locally
   ```
   streamlit run main.py
   ```
