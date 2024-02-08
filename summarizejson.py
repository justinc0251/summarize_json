from flask import Flask, render_template, request
from transformers import BartTokenizer, BartForConditionalGeneration
import json
from openai import OpenAI

client = OpenAI(api_key="")

app = Flask(__name__)

# Initialize the BART model for summarization
bart_model_name = "facebook/bart-large-cnn"
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
bart_summarizer = BartForConditionalGeneration.from_pretrained(bart_model_name)


def generate_image_with_dalle(text_description):
    try:
        response = client.images.generate(
            prompt=text_description + "generate text only in english",
            n=1,  # Number of images to generate
            size="1024x1024",
        )
        # Extract the image URL or data from the response
        image_url = response.data[0].url
        return image_url
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

# Function to summarize a given text using BART
def summarize_with_bart(input_text):
    if not input_text.strip():
        return ""

    input_ids = bart_tokenizer.encode(
        input_text, return_tensors="pt", max_length=1024, truncation=True
    )
    summary = bart_summarizer.generate(input_ids, max_length=50, min_length=25)
    summarized_text = bart_tokenizer.decode(summary[0], skip_special_tokens=True)
    return summarized_text


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded_file = request.files["file"]

        if uploaded_file and uploaded_file.filename.endswith(".json"):
            json_data = json.load(uploaded_file)
            summaries = []

            if isinstance(json_data, list):
                for item in json_data:
                    if "content" in item:
                        original_text = item["content"]
                        article_title = item["title"]
                        summarized_text = summarize_with_bart(original_text)
                        image_url = generate_image_with_dalle(summarized_text)
                        summaries.append(
                            (article_title, original_text, summarized_text, image_url)
                        )

            return render_template("index.html", original_texts=summaries)

    return render_template("index.html", original_texts=None)


if __name__ == "__main__":
    app.run(debug=True)
