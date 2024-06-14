from flask import Flask, render_template, request
from transformers import BartTokenizer, BartForConditionalGeneration
import json
import text2emotion as te
from PIL import Image, ImageDraw, ImageFont, ImageOps
import requests
from io import BytesIO
import base64
from textwrap import wrap
import replicate

app = Flask(__name__)

bart_model_name = "facebook/bart-large-cnn"
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
bart_summarizer = BartForConditionalGeneration.from_pretrained(bart_model_name)


def generate_image_with_stable_diffusion(article_title, summarized_text):
    try:
        output = replicate.run(
            "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
            input={
                "width": 768,
                "height": 768,
                "prompt": "A simple and clear image of: " + article_title + " " + summarized_text + ".",
                "negative_prompt": "text, words, letters, captions, labels, signs, written, people, humans, person, human figures, faces, hands, arms, legs, feet, fingers, toes, eyes, mouth, body parts",
                "scheduler": "DPMSolverMultistep",
                "num_outputs": 1,
                "guidance_scale": 7.5,
                "num_inference_steps": 50,
            },
        )
        image_url = output[0]
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        image_width, image_height = image.size
        black_strip_height = 300
        top_strip = Image.new("RGB", (image_width, black_strip_height), "black")
        image = ImageOps.expand(
            image, border=(0, black_strip_height, 0, 0), fill="black"
        )
        draw = ImageDraw.Draw(image)

        # Load a higher quality font
        font_path = "./arial.ttf"
        font_size = 48  # Adjust the font size as needed
        font = ImageFont.truetype(font_path, font_size)

        text_position = (10, 10)
        wrapped_text = "\n".join(wrap(summarized_text, 30))
        draw.multiline_text(text_position, wrapped_text, font=font, fill="white")

        # Render the text with anti-aliasing
        image = image.convert("RGBA")
        txt = Image.new("RGBA", image.size, (255, 255, 255, 0))
        d = ImageDraw.Draw(txt)
        d.multiline_text(
            text_position, wrapped_text, font=font, fill=(255, 255, 255, 255)
        )
        image = Image.alpha_composite(image, txt)

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
    summary = bart_summarizer.generate(input_ids, max_length=40, min_length=25)
    summarized_text = bart_tokenizer.decode(summary[0], skip_special_tokens=True)

    # Find the last period in the summarized text
    last_period_index = summarized_text.rfind(".")

    # If a period is found, truncate the text after the last period
    if last_period_index != -1:
        summarized_text = summarized_text[: last_period_index + 1]

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
                        image_url = generate_image_with_stable_diffusion(
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
