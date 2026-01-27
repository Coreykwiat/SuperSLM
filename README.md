# SuperSLM
SuperSLM offers the capabilities to train a Hyper-Focused Small Language Model (SLM) intended for offline local use. The intended use for this script is to have a dedicated Server/VM running the SuperSLM_Main.py.  Once the server is running to interact with it the user would simply send a remote query over Port 5005 and then receive the response back to the originating machine over port 5006.  

There is no front end to this server as it is intended to be interacted with remotely, however a user can query it locally through its interactive local shell!  


### System Requirements
CPU: 2 Cores
Memory: 9GB Minimum 

### Software Requirements
Ollama installed with llama3 or other model [https://ollama.com/]  
all-MiniLM-L6-v2 [https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2]  
Python Libraries (outlined in requirements.txt)  
    -torch  
    -sentence-transformers  
    -ollama  
    -PyMuPDF  
    -python-docx  
    -pandas  
    -openpyxl  
    -beautifulsoup4  


## How To Use
