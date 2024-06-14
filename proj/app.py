from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_pymongo import PyMongo
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
# Flask app initialization
app = Flask(__name__)
app.config['MONGO_URI'] = 'mongodb://localhost:27017/heardiseasedb'
mongo = PyMongo(app)
app.secret_key = 'your_secret_key'

# Load and preprocess data
print("Loading and preprocessing data...")
data = pd.read_csv('C:\\Users\\Ramesh\\Desktop\\heart2.csv')
columns = list(data.columns)
columns.remove('target')
X = data.drop('target', axis=1)
y = data['target']
print("Data loaded successfully.")
print(f"Number of samples: {len(data)}")
print(f"Number of features: {X.shape[1]}")
print(f"Columns: {', '.join(columns)}")
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Define individual classifiers
decision_tree = DecisionTreeClassifier()
random_forest = RandomForestClassifier()
logistic_regression = LogisticRegression(max_iter=10000)
svm = SVC(probability=True)

# Define a function to compute sensitivity and specificity
def get_sensitivity_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity

# Train individual models and compute metrics
metrics = {}
best_accuracy = 0
best_model_name = ""

for model_name, model in [("decision_tree", decision_tree), ("random_forest", random_forest),
                          ("logistic_regression", logistic_regression), ("svm", svm)]:
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    sensitivity, specificity = get_sensitivity_specificity(y_val, predictions)
    weight = (accuracy + sensitivity + specificity) / 3
    metrics[model_name] = {"accuracy": accuracy, "sensitivity": sensitivity, 
                           "specificity": specificity, "weight": weight}
    print(f"{model_name} - Accuracy: {accuracy:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, Weight: {weight:.4f}")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = model_name

print(f"\nBest model: {best_model_name} with accuracy {best_accuracy:.4f}")

# Extract weights for ensemble classifier
weights = [metrics[model]["weight"] for model in ["decision_tree", "random_forest", "logistic_regression", "svm"]]
print("\nEnsemble Classifier Training")

# Set up the ensemble classifier using derived weights
ensemble_classifier = VotingClassifier(estimators=[
    ('decision_tree', decision_tree),
    ('random_forest', random_forest),
    ('logistic_regression', logistic_regression),
    ('svm', svm)
], voting='soft', weights=weights)

# Train the ensemble classifier on the full training data
ensemble_classifier.fit(X_train_full, y_train_full)



# ... [Your Flask routes and other code here]


ATTRIBUTE_DESCRIPTIONS = {
    'age': 'How old are you?',
    'gender': 'Your gender. Choose 0 for female and 1 for male.',
    'cp': "Type of chest pain you experience. It's a scale from 0-3; higher numbers indicate more severe pain.",
    'trestbps': 'Your resting blood pressure. Normal is usually below 120/80 mmHg.',
    'chol': 'Your cholesterol level. A good level is usually below 200 mg/dl.',
    'fbs': 'Is your fasting sugar above 120 mg/dl? Choose 1 for Yes and 0 for No.',
    'restecg': 'Your ECG measurement when resting. It\'s a scale from 0-2.',
    'thalach': 'Your maximum heart rate during exercise.',
    'exang': 'Do you get chest pain when you exercise? Choose 1 for Yes and 0 for No.',
    'oldpeak': 'Difference in your ECG when resting and exercising.',
    'slope': 'Change in your heart activity during exercise. It\'s a scale from 0-2.',
    'ca': 'How many of your main vessels are colored in a test? It\'s a number between 0-3.',
    'thal': 'A test related to blood disorder thalassemia. It\'s a scale from 0-3.'
}





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

# Route for heart disease prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        details = [
            float(request.form['age']),
            float(request.form['gender']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal'])
        ]
        prediction_result = ensemble_classifier.predict([details])[0]

        if prediction_result == 1:
            session['prediction'] = "Yes"
        else:
            session['prediction'] = "No"

        session['form_data'] = {col: request.form.get(col) for col in columns}
        return redirect(url_for('result'))
    
    return render_template('index.html')
# Route for displaying prediction resul
@app.route('/result', methods=['GET'])
def result():
    if 'form_data' not in session or 'prediction' not in session:
        flash("Please submit the form first.")
        return redirect(url_for('predict'))

    new_instance = session.get('form_data')
    prediction = session.get('prediction')

    y_pred = ensemble_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100

    custom_message = "Based on the provided information, there's a high likelihood of heart disease. Please consult with a medical professional." if prediction == "Yes" else "Based on the provided information, you seem to be in good heart health. However, regular checkups are always recommended."

    users = mongo.db.users
    users.update_one(
        {"username": session['user']},
        {"$set": {"prediction": prediction, "message": custom_message, "inputs": new_instance}}
    )

    return render_template('result.html', prediction=prediction, message=custom_message, accuracy=accuracy)
 
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
    doc = SimpleDocTemplate(pdf, pagesize=landscape(letter), rightMargin=72, leftMargin=72, topMargin=24, bottomMargin=18)

    # Define styles
    styles = getSampleStyleSheet()
    style_normal = styles['Normal']

    # Define a gray background color for the entire PDF
    background_color = colors.HexColor('#CCCCCC')  # Adjust the color code as needed

    # Create a Story
    Story = []

    # Add a background color to the entire PDF
    Story.append(Paragraph('<font color="white">Background color</font>', style_normal))
    doc.addPageTemplates([PageTemplate(id='background', frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='normal')], background=background_color)])

    # Add rendered content to the PDF
    Story.append(Paragraph(rendered, style_normal))

    # Build the PDF
    doc.build(Story)

    pdf.seek(0)
    return Response(pdf.getvalue(), mimetype='application/pdf', headers={'Content-Disposition':'inline;filename=prediction.pdf'})


@app.route('/download_pdf')
def download_pdf():
    username = session['user']
    user_data = mongo.db.users.find_one({"username": username})
    prediction = user_data.get("prediction", "N/A")
    new_instance = user_data.get("inputs", {})

    # Fetch the background color dynamically from the database or set it as a constant value
    background_color = user_data.get("background_color", '#f0f0f0')

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
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FFA07A')),  # Background color for the header row (light red)
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),                # Text color for the header row
        ('BACKGROUND', (0, 1), (1, -1), colors.HexColor('#FFB6C1')), # Background color for the first two columns (light pink)
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),               # Text color for the data rows
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),                       # Center alignment for all cells
        ('GRID', (0, 0), (-1, -1), 1, colors.black),                 # Grid lines for all cells
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

@app.route('/submit_contact', methods=['POST'])
def submit_contact():
    name = request.form['name']
    email = request.form['email']
    message = request.form['message']

    # Insert the data into the MongoDB's contact_messages collection
    contact_messages = mongo.db.contact_messages
    contact_messages.insert_one({
        'name': name,
        'email': email,
        'message': message
    })

    flash('Thank you for contacting us. We will get back to you soon!', 'success')
    return redirect(url_for('contact_us'))
@app.route('/contact-us')
def contact_us():
    contact_messages = mongo.db.contact_messages
    try:
        last_message = contact_messages.find().sort("_id", -1).limit(1).next() # Get the last message submitted
    except StopIteration:
        last_message = None
    
    return render_template('contact-us.html', active_page='contact_us', last_message=last_message)


if __name__ == '__main__':
    app.run(debug=True)