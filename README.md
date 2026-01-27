# SuperSLM
SuperSLM offers the capabilities to train a Hyper-Focused Small Language Model (SLM) intended for offline local use to minimize breakage in workflow. The intended use for this script is to have a dedicated Server/VM running the SuperSLM_Main.py.  Once the server is running to interact with it the user would simply send a remote query over Port 5005 and then receive the response back to the originating machine over port 5006. While this does utilize LLamav3 and all-MiniLM-L6-v2 these are used for the purpose of sentence parsing and provide human readable output, by default SuperSLM does not have any knowledge base.  

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

### Supported Training Formats:  
    -CSV
    -PDF
    -EXCEL
    -WORD
    -TXT
    -HTML


## How To Use

## To start the SuperSLM Server:  
"python3 SuperSLM_Main.py"

### Accessing server shell and shell commands:  
  ~ Hit **Enter** to initiate interactive server shell  
  ~ Type **background** to exit interactive shell and drop back to purely a remote listening server
  ~ Type **refresh** to retrain model
  ~ Tyep **exit** or **q** to stop server

### How To Train Your Hyper-Focused Model:  
Upon the first time run of SuperSLM.py if the directory "docs/" does not already exist it will automatically create this directory in same directory as the SuperSLM.py script.  

**If the directory does not exist and you want to train the model upon startup create the docs/ directory manually and place all documents into this directory**  

If the model is running and you would liek to update its knowledge base enter the interactive shell and run the command **refresh**  
Everytime the server restarts if there are new documents in the "docs/" directory, upon startup the SuperSLM server will automatically retrain itself.

### How to send data to server  
The provided script will provide all of the required capabilities to interact with the server remotely!

Method 1 Provided Script:  
python3 send_and_receive.py "Hello World" <--- Query  
or  
echo "hello world" | python3 send_and_receive.py <--- The script will read standard input and send it over port 5005 and then initate a listener on port 5006 to receive the response  

Method 2 Netcat:  
To Send:  
echo "hello world" | nc <IP> 5005  

To Receive:  
nc -l 5006

