-- Document to markdown pipeline --

This pipeline processes PDFs, images, and MP3 files, extracts their content and converts it into a structured markdown file using an OpenAI LLM.

-- How it works -- 

    - Automatic file type detection (LLM):
        - PDFs are parsed using pdfplumber
        - images are parsed using tesseract
        - MP3 are parsed using the OpenAIs audio transcription API 

    - Parsed/extracted text is then formatted into a predefined template: 
        # {Title}

        ## Date: {date}

        ### Key Points: {key_points}

        ### Action Items: {action_items}

        ### Next Steps: {next_steps}

    - In the same step, a kebab case file name is generated based on content extracted

    - Lastly, the trigger is a natural language message containing the folder path. The markdown files generated are saved in the same folder as the input files.


 -- Usage --
 
    Use the main.py file:

    import functions

    # run the pipeline
    run("process all the folders and create markdowns of content within in ./sample_docs")

-- requires --

    credentials.py file containing:
        openAI_API_KEY = "key"
        tesseract_cmd = r"path to tesseract application"


