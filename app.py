from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)

# Load the model and tokenizer
model_name = "MBZUAI/LaMini-Flan-T5-783M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = torch.device("cpu")
model.to(device)

@app.route("/", methods=["GET", "POST"])
def index():
    output_text = ""
    if request.method == "POST":
        name = request.form["name"]
        experience = request.form["experience"]
        job_title = request.form["job_title"]
        skills = request.form["skills"]

        prompt = f"""
        Create a detailed and professional resume and cover letter based on the following:

        Name: {name}
        Experience: {experience}
        Job Title: {job_title}
        Skills: {skills}

        Format the output with clear sections: 'RESUME' and 'COVER LETTER'. 
        Do not include placeholder text like [Your Name], [Company Name], etc.
        """

        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
        )

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return render_template("index.html", output=output_text)

if __name__ == "__main__":
    app.run(debug=True)
