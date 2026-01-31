from openai import OpenAI
import os 
import json
import pdfplumber
from PIL import Image
from credentials import credentials
import pytesseract

# initialise client
client = OpenAI(api_key=credentials.OpenAI_API_KEY)

# parsing functions
def parse_pdf(path):
    """
    input:
    path: filepath 
    """
    with pdfplumber.open(path) as pdf: 
        return "\n".join((p.extract_text() or "") for p in pdf.pages)
    
def parse_image(path):
    """
    input:
    path: filepath 
    """
    pytesseract.pytesseract.tesseract_cmd = credentials.tesseract_cmd
    return pytesseract.image_to_string(Image.open(path))

def parse_mp3(path):
    """
    input:
    path: filepath 
    """
    with open(path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            file=audio_file,
            model="gpt-4o-mini-transcribe"
        )
    return transcript.text

# tools for the llm (openai function calling)
tools = [
    {"type": "function",
     "function": {
         "name": "parse_pdf",
         "description": "extracts text from a pdf using pdfplumber",
         # JSON schema defining the functions input arguments
         "parameters": { 
             "type": "object",
             "properties": {
                 "path": {"type": "string"}
             },
             "required": ["path"]
         }
     } 
},
{
    "type": "function",
     "function": {
         "name": "parse_image",
         "description": "extracts text from an image using pytesseract ocr",
         "parameters": {
             "type": "object",
             "properties": {
                 "path": {"type": "string"}
             },
             "required": ["path"]
         }
     } 
},
{
    "type": "function",
     "function": {
         "name": "parse_mp3",
         "description": "Transcribes text from an mp3 file using whisper",
         "parameters": {
             "type": "object",
             "properties": {
                 "path": {"type": "string"}
             },
             "required": ["path"]
         }
     } 
}
]

# LLM decides which tools to use
def extract_with_llm(path):
    """
    LLM decides which tool to parse text, then parser is used to extract text from file.

    input:
    path: filepath 
    """
    task = f"choose the correct tool for this file: {path}"
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user", "content":task}],
        tools=tools,
        tool_choice="auto" # allows model to pick
    )

    if not r.choices[0].message.tool_calls:
        print(f"no tool chosen for {path}, skipping")
        return None

    # print(r) 
    call = r.choices[0].message.tool_calls[0]

    if call.function.name == "parse_pdf":
        print("pdf path chosen")
        return parse_pdf(path)
    
    if call.function.name == "parse_image":
        print("image path chosen")
        return parse_image(path)
    
    if call.function.name == "parse_mp3":
        print("mp3 path chosen")
        return parse_mp3(path)
    
template = """
# {Title}
## Date: {date}

### Key Points: {key_points}

### Action Items: {action_items}

### Next Steps: {next_steps}


"""
    
def convert_to_markdown(text):
    """
    Function returns a json with 2 keys, file name and markdown (filled in template).
    These are based off the content of the extracted text.

    Input:
    text: extracted content of file
    """
    prompt = f"""
    Fill the following markdown template using the text that is passed in as well
    create a suitable file name in kebab case based on passed in text. 
    DO NOT ADD ANY EXTRA SECTIONS TO THE END OF THE TEMPLATE FILE 

    Template:
    {template}

    Content: {text}
    """
    
    # MD document Enforced using pydantic? 
    system_instructions = """
    Return a valid JSON object with this shape:
    {
    "filename": "kebab-case-string",
    "markdown": "filled in markdown template document. DO NOT ADD ANY EXTRA SECTIONS" 
    }
    """

    r = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type":"json_object"}, # Forces JSON 
        messages=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": prompt}
        ]
    )
    # print(r)
    return json.loads(r.choices[0].message.content)

# natural language trigger
def parse_request(request):
    """
    Function extracts the directory of files to be processed from user prompt

    Input:
    request: User prompt that contains the folder path for files to be processed.
    """
    prompt = f"""
Extract the input folder from the user's request.
Return JSON only with:
{{"input_dir": "string"}}

Instruction:
{request}
"""
    
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type":"json_object"},
        messages=[{"role":"user", "content":prompt}]
    )
    return json.loads(r.choices[0].message.content)


# main pipeline
def run(request):
    params = parse_request(request)

    # Limitation - using the same directory as input files
    input_dir = params["input_dir"]

    for fname in os.listdir(input_dir):
        ext = fname.lower().split(".")[-1]
        if ext not in ["pdf", "jpg", "png", "mp3"]:
            (print("File type not supported"))
            continue

        path = os.path.join(input_dir, fname)
        print("processing", path)

        extracted = extract_with_llm(path)
        
        if extracted is None:
            continue

        result = convert_to_markdown(extracted)

        # Save markdown 
        directory_path = os.path.dirname(path)
        output_path = os.path.join(directory_path, result.get("filename") + ".md")

        with open(output_path, "w") as w:
            w.write(result["markdown"])

        print("saved", output_path)