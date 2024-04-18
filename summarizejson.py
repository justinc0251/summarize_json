from flask import Flask, render_template, request
from transformers import BartTokenizer, BartForConditionalGeneration
import json
import text2emotion as te
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont, ImageOps
import requests
from io import BytesIO
import base64

client = OpenAI(api_key="")

app = Flask(__name__)

bart_model_name = "facebook/bart-large-cnn"
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
bart_summarizer = BartForConditionalGeneration.from_pretrained(bart_model_name)


def get_default_font(size=250):
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except IOError:
        font = ImageFont.load_default()
    return font


def generate_image_with_dalle(text_description, summary_text):
    try:
        response = client.images.generate(
            prompt=text_description,
            n=1,
            size="256x256",
        )
        image_url = response.data[0].url
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        image_width, image_height = image.size
        black_strip_height = 80
        top_strip = Image.new("RGB", (image_width, black_strip_height), "black")
        image = ImageOps.expand(
            image, border=(0, black_strip_height, 0, 0), fill="black"
        )
        draw = ImageDraw.Draw(image)
        font = get_default_font(250)
        text_position = (10, 10)
        draw.text(text_position, summary_text, fill="white", font=font)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        base64_img = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{base64_img}"
    except Exception as e:
        print(f"Error generating image with text overlay: {e}")
        return None


def summarize_with_bart(input_text):
    if not input_text.strip():
        return ""
    input_ids = bart_tokenizer.encode(
        input_text, return_tensors="pt", max_length=1024, truncation=True
    )
    summary = bart_summarizer.generate(input_ids, max_length=35, min_length=25)
    summarized_text = bart_tokenizer.decode(summary[0], skip_special_tokens=True)
    return summarized_text


def process_text(input_text):
    emotions = te.get_emotion(input_text)
    return emotions


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
                        emotions = process_text(summarized_text)
                        image_url = generate_image_with_dalle(
                            article_title, summarized_text
                        )
                        summaries.append(
                            (
                                article_title,
                                original_text,
                                summarized_text,
                                emotions,
                                image_url,
                            )
                        )
            return render_template("index.html", original_texts=summaries)
    return render_template("index.html", original_texts=None)


if __name__ == "__main__":
    app.run(debug=True)
