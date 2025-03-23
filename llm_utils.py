import subprocess
import logging

# Configure logging for debugging and traceability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def call_llm(model_name: str, prompt: str) -> str:
    """
    Call a local LLM via the Ollama CLI using the provided prompt.
    Returns the model's response as a string.
    """
    logging.info(f"Calling model '{model_name}' with prompt (first 50 chars): {prompt[:50]}...")
    try:
        result = subprocess.run(
            ['ollama', 'run', model_name],
            input=prompt.encode('utf-8'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            # remove the timeout or set it to something large
            # timeout=120
        )
        response = result.stdout.decode('utf-8').strip()
        logging.info("Received response from LLM.")
        return response
    except subprocess.TimeoutExpired:
        logging.error("LLM call timed out.")
        return "Error: LLM call timed out."
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8')
        logging.error(f"LLM call failed: {error_msg}")
        return f"Error: {error_msg}"

def simulate_llm_conversation(data_summary: str, model1: str, model2: str) -> (str, str):
    """
    Simulate a conversation between two LLMs.
    The first model provides an analysis, and the second expands upon that analysis.
    Returns a tuple of responses (response1, response2).
    """
    prompt1 = f"Analyze the following sensor data summary and provide detailed insights:\n{data_summary}"
    response1 = call_llm(model1, prompt1)
    prompt2 = f"Based on the analysis below:\n{response1}\nProvide further detailed suggestions and actionable insights."
    response2 = call_llm(model2, prompt2)
    return response1, response2
