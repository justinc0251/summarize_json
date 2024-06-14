# summarize_json

CSEN 193 Undergraduate Research

I created a Flask application to process uploaded JSON files containing article data. The app summarizes the content using a BART model, analyzes the summarized text for emotions, and generates a relevant image using Stable Diffusion. Finally, it displays the summarized text, emotions, and generated image on the web page.

### Installation

1. Clone the repository:

2. Set the REPLICATE_API_TOKEN environment variable:
   ```bash
   export REPLICATE_API_TOKEN=r8_LUq**********************************

3. Run the app:
   ```bash
   python summarizejson.py
