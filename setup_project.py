import os

# Folder structure
folders = [
    "templates",
    "static",
    "data"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

# ---------------- app.py ----------------
app_code = """from flask import Flask, render_template, request, redirect
import pandas as pd
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Load internships dataset
DATA_FILE = "data/internships.csv"
internships = pd.read_csv(DATA_FILE)

# Load SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

def recommend_internships(user_text, top_n=5):
    job_descriptions = internships['description'].tolist()
    job_embeddings = model.encode(job_descriptions, convert_to_tensor=True)
    user_embedding = model.encode([user_text], convert_to_tensor=True)

    scores = util.cos_sim(user_embedding, job_embeddings)[0]
    internships['score'] = scores.cpu().tolist()
    top_matches = internships.sort_values(by="score", ascending=False).head(top_n)
    return top_matches

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "resume_text" in request.form:
            resume_text = request.form["resume_text"]
        else:
            resume_text = " ".join([
                request.form.get("name", ""),
                request.form.get("education", ""),
                request.form.get("skills", ""),
                request.form.get("interests", "")
            ])

        recommendations = recommend_internships(resume_text)
        return render_template("results.html", recommendations=recommendations.to_dict(orient="records"))

    return render_template("index.html")

@app.route("/admin")
def admin():
    return render_template("admin.html", internships=internships.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True)
"""

with open("app.py", "w", encoding="utf-8") as f:
    f.write(app_code)

# ---------------- index.html ----------------
index_html = """<!DOCTYPE html>
<html>
<head>
    <title>AI Internship Recommender</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <h1>AI Internship Recommender</h1>
    <form method="POST">
        <h2>Upload Resume Text</h2>
        <textarea name="resume_text" placeholder="Paste your CV here..."></textarea>
        <button type="submit">Find Internships</button>
    </form>
    <hr>
    <h2>Or Fill Manually</h2>
    <form method="POST">
        <input type="text" name="name" placeholder="Your Name"><br>
        <input type="text" name="education" placeholder="Education"><br>
        <input type="text" name="skills" placeholder="Skills (comma separated)"><br>
        <input type="text" name="interests" placeholder="Interests"><br>
        <button type="submit">Find Internships</button>
    </form>
</body>
</html>
"""

with open("templates/index.html", "w", encoding="utf-8") as f:
    f.write(index_html)

# ---------------- results.html ----------------
results_html = """<!DOCTYPE html>
<html>
<head>
    <title>Results - AI Internship Recommender</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <h1>Recommended Internships</h1>
    {% for job in recommendations %}
        <div class="job-card">
            <h2>{{ job['title'] }} - {{ job['company'] }}</h2>
            <p><b>Domain:</b> {{ job['domain'] }} | <b>Location:</b> {{ job['location'] }}</p>
            <p><b>Duration:</b> {{ job['duration_months'] }} months | <b>Stipend:</b> ₹{{ job['stipend'] }}</p>
            <p><b>Skills Required:</b> {{ job['skills_required'] }}</p>
            <p>{{ job['description'] }}</p>
            <a href="{{ job['apply_link'] }}" target="_blank">Apply Now</a>
        </div>
    {% endfor %}
    <a href="/">🔙 Back</a>
</body>
</html>
"""

with open("templates/results.html", "w", encoding="utf-8") as f:
    f.write(results_html)

# ---------------- admin.html ----------------
admin_html = """<!DOCTYPE html>
<html>
<head>
    <title>Admin Panel - AI Internship Recommender</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <h1>Admin Panel</h1>
    <table border="1">
        <tr>
            <th>Title</th>
            <th>Company</th>
            <th>Domain</th>
            <th>Location</th>
            <th>Skills</th>
        </tr>
        {% for job in internships %}
        <tr>
            <td>{{ job['title'] }}</td>
            <td>{{ job['company'] }}</td>
            <td>{{ job['domain'] }}</td>
            <td>{{ job['location'] }}</td>
            <td>{{ job['skills_required'] }}</td>
        </tr>
        {% endfor %}
    </table>
    <a href="/">🔙 Back</a>
</body>
</html>
"""

with open("templates/admin.html", "w", encoding="utf-8") as f:
    f.write(admin_html)

# ---------------- style.css ----------------
style_css = """body {
    font-family: Arial, sans-serif;
    margin: 20px;
    background: #f9f9f9;
}
h1 {
    color: #2c3e50;
}
form, .job-card, table {
    background: white;
    padding: 15px;
    margin: 15px 0;
    border-radius: 8px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
button, a {
    background: #2c3e50;
    color: white;
    padding: 8px 12px;
    text-decoration: none;
    border: none;
    border-radius: 5px;
}
button:hover, a:hover {
    background: #34495e;
}
"""

with open("static/style.css", "w", encoding="utf-8") as f:
    f.write(style_css)

# ---------------- internships.csv ----------------
csv_data = """id,title,company,domain,location,mode,duration_months,stipend,skills_required,description,apply_link
1,AI/ML Intern,DesignStudio,AI/ML,Jaipur,Hybrid,6,10000,"NLP;Python;Pandas","Work on AI/ML projects, using NLP ,Python ,Pandas. Contribute to real-world solutions at DesignStudio.",https://example.com/apply/1
2,Data Science Intern,NexGen Solutions,Data Science,Bengaluru,Remote,4,12000,"Python;Statistics;Tableau;PowerBI","Work on Data Science projects, using Python, Tableau, and PowerBI. Contribute to real-world solutions at NexGen.",https://example.com/apply/2
3,Web Development Intern,RoboWorks,Web Development,Kochi,Hybrid,6,15000,"React;Node.js;CSS;REST","Work on Web Development projects, using React, Node.js, CSS, REST. Build scalable solutions.",https://example.com/apply/3
4,Marketing Intern,HUL,Marketing,Mumbai,Onsite,3,8000,"SEO;Social Media;Content Writing","Work on Marketing projects, managing SEO, Social Media, and Content Creation. Gain real-world exposure.",https://example.com/apply/4
5,Finance Intern,Deloitte,Finance,Delhi,Onsite,4,12000,"Excel;Financial Analysis;Accounting","Work on Finance projects, using Excel and financial modeling techniques. Contribute to Deloitte.",https://example.com/apply/5
"""

with open("data/internships.csv", "w", encoding="utf-8") as f:
    f.write(csv_data)

print("✅ Project setup complete! Run with: python app.py")

