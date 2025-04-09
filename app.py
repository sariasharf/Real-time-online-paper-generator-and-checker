#openai.api_key = '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'
#gpt-3.5-turbo-instruct
from flask import Flask, render_template, request, send_file,Response
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import openai
from fuzzywuzzy import fuzz
from reportlab.pdfgen import canvas
from io import BytesIO
from PyPDF2 import PdfReader
import os
import csv
from fpdf import FPDF
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer


app = Flask(__name__)

# Set your OpenAI API key
openai.api_key = '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'
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['UPLOAD_FOLDER'] = 'questionsdata'

# Step 1: Choose a pre-trained model architecture
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

# Step 3: Load the pre-trained model
try:
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
except OSError:
    print(f"Model not found locally. Downloading {model_name}...")
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Step 4: Define a tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

def clean_text(text):
    """
    Removes special characters from the given text
    """
    cleaned_text = ""
    for char in text:
        if char.isalnum() or char.isspace():
            cleaned_text += char
    return cleaned_text


@app.route('/')
def index():
    return render_template('index.html')

@ app.route('/signup')
def firbase():
    """ The signup page for the app """
    return render_template('signup.html')


@ app.route('/login')
def login():
    """ The login page for the app """
    return render_template('login.html')


@ app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    return render_template('dashboard.html')
@ app.route('/dashqa', methods=['GET', 'POST'])
def dashqa():
    return render_template('dashqa.html')
@ app.route('/dashqab', methods=['GET', 'POST'])
def dashqab():
    return render_template('dashboardqab.html')
'''
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    # Save the uploaded file
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Extract text from the uploaded PDF
    pdf_text = extract_text_from_pdf(file_path)
'''
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    # **Add this block to check the file extension**
    _, file_extension = os.path.splitext(file.filename)
    if file_extension.lower() != '.pdf':
        return render_template('error.html', message="Unsupported file type. Please upload a PDF file.")

    # Save the uploaded file
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Extract text from the uploaded PDF
    pdf_text = extract_text_from_pdf(file_path)

    # ... rest of your existing code ...

    # Get user input for question generation
    num_questions = int(request.form.get('num_questions', 1))

    # Get additional details from the form
    school_name = request.form.get('school_name')
    subject_name = request.form.get('subject_name')
    total_marks = request.form.get('total_marks')
    student_name = request.form.get('student_name')
    program = request.form.get('program')
    semester = request.form.get('semester')
    obtained_marks = request.form.get('obtained_marks')
    teacher_name = request.form.get('teacher_name')

    # Generate questions using OpenAI
    questions = generate_questions(pdf_text, num_questions)

    # Create a PDF file with the generated questions
    pdf_file_path = create_pdf(questions, school_name, subject_name, total_marks, student_name, program, semester, obtained_marks, teacher_name)
    # Ensure pdf_file_path is an absolute path
    pdf_file_path_absolute = os.path.abspath(pdf_file_path)

    return render_template('result.html', questions=questions, pdf_file=pdf_file_path_absolute)


@app.route('/uploadi', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    num_questions = int(request.form['num_questions'])
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Generate questions using OpenAI API
    pdf_text = extract_text_from_pdf(file_path)
    generated_content = generate_questions(pdf_text, num_questions)

    # Split the generated_content list into individual questions
    generated_questions = []
    for content in generated_content:
        generated_questions.extend(content.split('\n'))

    return render_template('generated_questions.html', generated_content=generated_questions)


@app.route('/submit_answers', methods=['POST'])
def submit_answers():
    # Get user's name from the form data
    user_name = request.form.get('user_name')
    submitted_answers = request.form.getlist('submitted_answers[]')
    original_answers = request.form.getlist('original_answers[]')

    # Initialize a list to store accuracy scores and results
    results = []
    all_data = []
    total_marks = 0

    # Compare each user answer with the original answer
    for user_answer, original_answer in zip(submitted_answers, original_answers):
        # Send the user answer and original answer to OpenAI for comparison
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",  # Choose the appropriate OpenAI engine
            prompt=f"User Answer: {user_answer}\nOriginal Answer: {original_answer}",
            max_tokens=50,
            stop=None,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # Get the completion's text from the response
        completion_text = response.choices[0].text.strip()

        # Calculate accuracy based on completion text similarity
        similarity_score = fuzz.ratio(original_answer.lower(), completion_text.lower())
        accuracy = similarity_score / 100  # Convert similarity score to a value between 0 and 1

        # Assign marks based on accuracy
        if accuracy >= 0.95:
            marks = 10
        elif accuracy >= 0.85:
            marks = 9
        elif accuracy >= 0.75:
            marks = 8
        elif accuracy >= 0.65:
            marks = 7
        elif accuracy >= 0.55:
            marks = 6
        elif accuracy >= 0.45:
            marks = 5
        elif accuracy >= 0.35:
            marks = 4
        elif accuracy >= 0.25:
            marks = 3
        elif accuracy >= 0.15:
            marks = 2
        elif accuracy >= 0.05:
            marks = 1
        else:
            marks = 0

        # Append the accuracy score to the results list
        results.append(marks)
        all_data.append([original_answer, user_answer, completion_text, accuracy])
        total_marks += marks


    # Save results to CSV
    with open('results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Question', 'User Answer', 'Orignal Answer', 'Accuracy'])
        writer.writerows(all_data)
    # Calculate total marks obtained and total marks possible
    total_marks_possible = len(original_answers) * 10

    return render_template('results.html', results=results, user_name=user_name, total_marks=total_marks,total_marks_possible=total_marks_possible)

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text


def generate_questions(passage, num_questions):
    # Define a specific prompt to guide OpenAI in generating questions
    prompt = f"Generate{num_questions} subjective questions about the main purpose of the following passage:\n{passage}\n"

    # Use OpenAI API to generate questions
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=150,

        stop=None
    )

    # Extract generated questions from OpenAI response
    raw_questions = [item['text'].strip() for item in response['choices']]

    # Remove duplicate questions
    unique_questions = list(set(raw_questions))

    # Limit the number of questions to the user's specified amount
    sliced_questions = unique_questions[:num_questions]

    return sliced_questions
#changes made

@app.route('/generate', methods=['POST'])
def generate():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    # Save the uploaded file
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Extract text from the uploaded PDF
    pdf_text = extract_text_from_pdf(file_path)

    # Get user input for question generation

    num_questions = int(request.form.get('num_questions', 1))
    # Define a prompt with placeholders for questions and answers
    prompt = f"Generate {num_questions} diverse and unique subjective questions and their answers about the main purpose of the following passage:\n{pdf_text}\nQuestions:\n"


    # Use OpenAI API to generate questions and answers
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=1000,  # Adjust as needed
        n=1,  # Generate a single response
        stop=None
    )

    # Extract generated content from OpenAI response
    if 'choices' not in response or not response['choices']:
        print("Invalid OpenAI response structure")
        return "Invalid response structure", []

    generated_content = response['choices'][0]['text']
    print(generated_content)

    # formatted_questions, answers = generate_questions_and_answers_from_file(pdf_text, num_questions)
    # print("Formatted Questions:", formatted_questions)
    # print("Answers:", answers)
    return render_template('generate_file.html', generated_content=generated_content)
    # return render_template('generate_file.html', questions=formatted_questions, answers=answers)

def generate_questions_and_answers_from_file(passage, num_questions):
    # Read the content from the file

    # Define a prompt with placeholders for questions and answers
    prompt = f"Generate {num_questions} diverse and unique subjective questions and their answers about the main purpose of the following passage:\n{passage}\nQuestions:\n"

    # Use OpenAI API to generate questions and answers
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=1000,  # Adjust as needed
        n=1,  # Generate a single response
        stop=None
    )

    # Extract generated content from OpenAI response
    if 'choices' not in response or not response['choices']:
        print("Invalid OpenAI response structure")
        return "Invalid response structure", []

    generated_content = response['choices'][0]['text']
    print(generated_content)

    # Split the content into questions and answers
    questions_and_answers = generated_content.split("Answers:")
    print(questions_and_answers)

    # Extract questions
    if len(questions_and_answers) >= 2:
        questions = [question.strip() for question in questions_and_answers[0].strip().split("\n") if question.strip()]

        questions = questions_and_answers[0].strip().split("\n")
        answers = questions_and_answers[1].strip().split("\n")
    else:
        print("Invalid OpenAI response structure")
        questions = []
        answers = []
    # Format questions with question numbers and new lines
    formatted_questions = '\n'.join([f"{i+1}. {question}" for i, question in enumerate(questions)])

    return generated_content


def create_pdf(questions,school_name, subject_name, total_marks, student_name, program, semester, obtained_marks, teacher_name):


    # Create a BytesIO buffer to hold the PDF content
    pdf_file_path = "generated_questions.pdf"

    # Create a PDF canvas
    pdfmetrics.registerFont(
        TTFont('Arial', 'C:/Windows/Fonts/arial.ttf'))  # Replace 'arial.ttf' with the path to your Arial font file

    # Create a PDF document
    pdf = SimpleDocTemplate(pdf_file_path, pagesize=letter)
    styles = getSampleStyleSheet()

    # Create tables for details_data
    left_details_data = [
        ['Quiz', ''],
        ['School Name', school_name],
        ['Subject Name', subject_name],
        ['Total Marks', str(total_marks)],
        ['Student Name', student_name],
    ]
    right_details_data = [
        ['', ''],
        ['Program', program],
        ['Semester', semester],
        ['Obtained Marks', str(obtained_marks)],
        ['Teacher Name', teacher_name],
    ]

    left_details_table = Table(left_details_data, colWidths=[100, 200])
    right_details_table = Table(right_details_data, colWidths=[100, 200])

    left_details_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), 'grey'),
                                            ('TEXTCOLOR', (0, 0), (-1, 0), 'white'),
                                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                            ('FONTNAME', (0, 0), (-1, 0), 'Arial'),
                                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                            ('BACKGROUND', (0, 1), (-1, -1), 'beige'),
                                            ]))

    right_details_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), 'grey'),
                                             ('TEXTCOLOR', (0, 0), (-1, 0), 'white'),
                                             ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                             ('FONTNAME', (0, 0), (-1, 0), 'Arial'),
                                             ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                             ('BACKGROUND', (0, 1), (-1, -1), 'beige'),
                                             ]))

    # Combine left and right details tables horizontally
    combined_details_table = Table([
        [left_details_table, right_details_table]
    ], colWidths=[300, 300])  # Adjust the widths as needed

    # Create a table for questions
    questions_data = [['Questions']]
    for question in questions:
        questions_data.append([question])

    questions_table = Table(questions_data, colWidths=[500])  # Adjust the width as needed

    # Build the PDF document
    flowables = [
        combined_details_table,
        Spacer(20, 20),  # Add some space between the tables
        questions_table,
    ]
    pdf.build(flowables)

    return pdf_file_path



@app.route('/download', methods=['POST'])
def download_pdf():
    # Get form data
    school_name = request.form['school_name']
    subject_name = request.form['subject_name']
    total_marks = request.form['total_marks']
    student_name = request.form['student_name']
    program = request.form.get('program')
    semester = request.form.get('semester')
    obtained_marks = request.form.get('obtained_marks')
    teacher_name = request.form.get('teacher_name')

    # Get the PDF content from the hidden input field
    questions = request.form.getlist('questions[]')

    # Create a PDF file with the additional details and questions
    pdf_file_path = create_pdf(questions, school_name, subject_name, total_marks, student_name, program, semester, obtained_marks, teacher_name)

    # Send the PDF file for download
    return send_file(pdf_file_path, as_attachment=True)


@app.route("/question", methods=["GET", "POST"])
def question():
    if request.method == "POST":
        file = request.files["file"]
        text = file.read().decode("utf-8")
        cleaned_text = clean_text(text)

        question = request.form["question"]

        inputs = tokenizer.encode_plus(question, cleaned_text, return_tensors="pt", max_length=512, truncation=True)

        outputs = model(**inputs)

        answer_start = outputs.start_logits.argmax(dim=-1).item()
        answer_end = outputs.end_logits.argmax(dim=-1).item()
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end + 1]))

        return render_template("resultans.html", question=question, answer=answer)
    else:
        return render_template("question.html")

if __name__ == '__main__':
    # Create 'uploads' directory if not exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(debug=True)
