# Resume Analyzer

This application analyzes resumes against job descriptions to:
1. Calculate a match score
2. Generate personalized email templates
3. Generate personalized LinkedIn messages
4. Suggest improvements to make your resume a better fit for the job

## Features

- Upload PDF or DOCX resumes
- Paste job descriptions
- Get match scores using TF-IDF vectorization and cosine similarity
- Receive personalized email and LinkedIn message templates
- View missing keywords and improvement suggestions

## Setup Instructions

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. Clone this repository or download the source code

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv env
   ```

3. Activate the virtual environment:
   - Windows:
     ```
     env\Scripts\activate
     ```
   - macOS/Linux:
     ```
     source env/bin/activate
     ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Running the Application

1. Start the application:
   ```
   python main.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:8000
   ```

3. Use the application:
   - Upload your resume (PDF or DOCX format)
   - Paste the job description
   - Click "Analyze Match"
   - View your results and customized messages

## How It Works

The application uses Natural Language Processing techniques to analyze the content of your resume against the job description:

1. Text extraction from PDF or DOCX files
2. Keyword extraction using TF-IDF (Term Frequency-Inverse Document Frequency)
3. Cosine similarity calculation to determine match percentage
4. Smart template generation based on identified keywords and match score
5. Analysis of missing keywords and generation of improvement suggestions

## License

MIT 