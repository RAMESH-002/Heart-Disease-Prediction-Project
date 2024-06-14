import os
from io import BytesIO
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, Response
from flask_pymongo import PyMongo
from werkzeug.security import check_password_hash, generate_password_hash
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

app = Flask(__name__)
app.config['MONGO_URI'] = 'mongodb://localhost:27017/heardiseasedb'
mongo = PyMongo(app)
app.secret_key = 'your_secret_key'

averages = {
    "Age": [58, 52],
    "Chest Pain": [1.5, 1],
    "BP": [167.5, 140.8],
    "Cholesterol": [234, 203.9],
    "Fasting Blood Sugar": [143.4, 79.9],
    "Resting ECG": [1.17, 1.42],
    "Thalach": [171.3, 139.5],
    # "Exang": [0.78, 0],
    # "Oldpeak": [2.8, 0],
    # "Slope": [1.33, 1.17],
    # "Ca": [1, 0.22],
    # "Thal": [1.22, 0.78]
}

columns = ["Age", "Gender"] + [col for col in averages if col != "Age"]

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/signup_page')
@app.route('/register')
def register():
    return render_template('signup.html')

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form.get('username')
    name = request.form.get('name')
    email = request.form.get('email')
    phone = request.form.get('phone')
    age = request.form.get('age')
    password = request.form.get('password')
    confirm_password = request.form.get('confirm_password')

    # Check if passwords match
    if password != confirm_password:
        flash("Passwords do not match!")
        return redirect(url_for('welcome'))
    users = mongo.db.users

    existing_user = users.find_one({'username': username})
    if existing_user:
        flash("User already exists!")
        return redirect(url_for('welcome'))


    # Check if user already exists
    plain_password = password
    

    users.insert_one({'username': username, 'name': name, 'email': email, 'phone': phone, 'age': age, 'password': plain_password})



    flash("User registered successfully!")
    return redirect(url_for('welcome'))
@app.route('/signup_page')
def signup_page():
    return render_template('signup.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    print(f"Logging in with: {username}, {password}")  # DEBUG
    users = mongo.db.users

    login_user = users.find_one({'username': username})
    if login_user and login_user['password'] == password:
        session['user'] = login_user['username']
        return redirect(url_for('predict'))
    



    flash("Invalid credentials!")
    return redirect(url_for('welcome'))

@app.before_request
def check_user():
    # Check if user is accessing protected route
    if request.endpoint in ['predict', 'result']:
        if 'user' not in session:
            return "Please log in first!"
app.secret_key = 'your_secret_key'

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None

    if request.method == 'POST':
        new_instance = [float(request.form[col]) if col in averages else request.form[col] for col in columns]
        prediction = get_prediction(new_instance)

    return render_template('index.html', columns=columns, averages=averages, prediction=prediction)


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        # Fetch the user entered values and the prediction
        new_instance = {}
        for col in columns:
            value = request.form.get(col, "N/A")
            if col in averages and value.replace(".", "", 1).isdigit():
                value = float(value)
            new_instance[col] = value

        prediction = predict([new_instance[col] for col in columns])
        if prediction == "Yes":
            custom_message = "Based on the provided information, there's a high likelihood of heart disease. Please consult with a medical professional."
        else:
            custom_message = "Based on the provided information, you seem to be in good heart health. However, regular checkups are always recommended."
        # Storing prediction and user's input
        users = mongo.db.users
        users.update_one(
            {"username": session['user']},
            {"$set": {"prediction": prediction, "message": custom_message, "inputs": new_instance}}
        )

        return render_template('result.html', prediction=prediction, message=custom_message)


@app.route('/generate_pdf')
def generate_pdf():
    users = mongo.db.users
    user = users.find_one({"username": session['user']})
    prediction = user.get('prediction', 'N/A')

    # Fetch the user input values
    new_instance = {col: request.form[col] for col in columns}

    # Render the template with user inputs and prediction
    rendered = render_template('pdf_template.html', prediction=prediction, inputs=new_instance)
    
    pdf = BytesIO()
    pisa.CreatePDF(BytesIO(rendered.encode('utf-8')), pdf)
    
    return Response(pdf.getvalue(), mimetype='application/pdf', headers={'Content-Disposition':'inline;filename=prediction.pdf'})


@app.route('/download_pdf')
def download_pdf():
    username = session['user']
    user_data = mongo.db.users.find_one({"username": username})
    prediction = user_data.get("prediction", "N/A")
    new_instance = user_data.get("inputs", {})

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), rightMargin=72, leftMargin=72, topMargin=24, bottomMargin=18)
    Story = []

    styles = getSampleStyleSheet()

    # Main title at the top
    Story.append(Paragraph("<u>Heart Disease Prediction Results</u>", styles["Title"]))
    Story.append(Spacer(1, 12))

    # Username display directly below the title
    Story.append(Paragraph(f"For: <strong>{username}</strong>", styles["Heading2"]))
    Story.append(Spacer(1, 24))

    # Creating the data for the table
    data = [['Parameter', 'Value']]
    for col in columns:
        value = new_instance.get(col, "N/A")
        data.append([col, str(value)])

    # Adjusting the column widths and row heights for the table
    colWidths = [250, 250]
    rowHeights = [30] * (len(columns) + 1)
    rowHeights[0] = 35

    table = Table(data, colWidths=colWidths, rowHeights=rowHeights)

    # Styling the table
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.red),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 16),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 14),
        ('BACKGROUND', (0, 1), (-1, -1), colors.transparent),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('GRID', (0, 0), (-1, -1), 1.5, colors.darkred)
    ])
    table.setStyle(style)

    Story.append(table)
    Story.append(Spacer(1, 24))

    # Adding prediction and its message with enhanced aesthetics
    Story.append(Paragraph(f"Prediction: <strong>{prediction}</strong>", styles["Heading2"]))
    Story.append(Spacer(1, 12))
    if prediction == "Yes":
        message = "Based on the provided information, there's a high likelihood of heart disease. Please consult with a medical professional."
    else:
        message = "Based on the provided information, you seem to be in good heart health. However, regular checkups are always recommended."
    Story.append(Paragraph(message, styles["Heading3"]))  # Slightly increased prominence of this message

    # Slightly larger heart health quote
    quote = "Remember: Early detection and timely treatment of heart disease can save lives."
    Story.append(Spacer(1, 12))
    Story.append(Paragraph(quote, styles["Italic"]))

    doc.build(Story, onFirstPage=add_background, onLaterPages=add_background)

    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name=f"{username}_prediction.pdf", mimetype="application/pdf")


def predict(instance):
    attributes_without_gender = [instance[i] for i, col in enumerate(columns) if col in averages]
    
    diff_heart_disease = sum([abs(val - averages[col][0]) for col, val in zip(averages, attributes_without_gender)])

    diff_normal = sum([abs(val - averages[col][1]) for col, val in zip(averages, attributes_without_gender)])  # Renamed diff_no_heart_disease to diff_normal
    
    if diff_heart_disease < diff_normal:
        return "Yes"
    else:
        return "No"

def add_background(canvas, doc):
    canvas.drawImage("static/heartpicture.jpg", 0, 0, width=landscape(letter)[0], height=landscape(letter)[1])
@app.route('/prediction-guide')
def prediction_guide():
    return render_template('prediction-guide.html', active_page='prediction_guide')

@app.route('/heart-health-tips')
def heart_health_tips():
    return render_template('heart-health-tips.html', active_page='heart_health_tips')

@app.route('/resources')
def resources():
    return render_template('resources.html', active_page='resources')

@app.route('/contact-us')
def contact_us():
    return render_template('contact-us.html', active_page='contact_us')



if __name__ == '__main__':
    app.run(debug=True)
