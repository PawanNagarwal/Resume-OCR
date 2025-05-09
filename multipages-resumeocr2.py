import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import os
import json
import base64
from openai import OpenAI
import mimetypes
import tempfile
import time 

# Set page configuration
st.set_page_config(page_title="Multi-Pages CandidNexus", page_icon="📄", layout="wide")

# Initialize OpenAI client - Using a placeholder for the API key
api_key = os.getenv('OPENAI_API_KEY')

def add_experience_status(resume_data):
    """
    Add an 'Experience status' field to the resume data based on 'Work Experience'.
    
    Classification rules:
    - If 'Work Experience' is None or an empty list, set status to 'fresher'
    - If all job titles contain internship indicators ('intern', 'internship', etc.), set status to 'fresher'
    - If any job title doesn't contain internship indicators, set status to 'experienced'
    
    Args:
        resume_data (dict): The original resume data dictionary
        
    Returns:
        dict: Resume data with added 'Experience status' field
    """
    # Create a copy of the input data to avoid modifying the original
    updated_resume = resume_data.copy()
    
    work_exp = updated_resume.get('Work Experience')
    
    # Define internship indicators - words that would identify a position as an internship
    internship_indicators = ['intern', 'internship', 'trainee']
    
    # Check if Work Experience is None, an empty list, or a string
    if (work_exp is None or 
        isinstance(work_exp, list) and len(work_exp) == 0 or
        isinstance(work_exp, str)):
        updated_resume['Experience status'] = 'fresher'
    else:
        # Initialize flag assuming all positions are internships
        is_only_intern = True
        
        try:
            for exp in work_exp:
                # Check if exp is a dictionary before using .get()
                if not isinstance(exp, dict):
                    # If an entry isn't a dictionary, assume it's not an internship
                    is_only_intern = False
                    break
                
                # Get job title and handle None values
                job_title = exp.get('job title', '') or ''
                job_title = job_title.lower()
                
                # Check if any of the internship indicators are in the job title
                is_internship_position = any(indicator in job_title for indicator in internship_indicators)
                
                # If any position isn't an internship, candidate is experienced
                if not is_internship_position:
                    is_only_intern = False
                    break
                    
        except Exception as e:
            # Handle any other unexpected errors
            print(f"Error processing work experience: {e}")
            is_only_intern = True  # Default to fresher if there's an error
        
        # Set status based on internship check
        if is_only_intern:
            updated_resume['Experience status'] = 'fresher'
        else:
            updated_resume['Experience status'] = 'experienced'
    
    return updated_resume

# Create a function to reset the application state
def reset_app_state():
    if 'results' in st.session_state:
        st.session_state['results'] = None
    if 'temp_file_path' in st.session_state and st.session_state['temp_file_path']:
        try:
            os.remove(st.session_state['temp_file_path'])
        except:
            pass
        st.session_state['temp_file_path'] = None

def convert_pdf_to_images(pdf_path, output_folder, dpi=800, image_format="png"):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    page_count = pdf_document.page_count

    # Get the base filename without extension
    base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    output_files = []
    # Calculate the zoom factor based on DPI (72 is the base DPI for PDFs)
    zoom_factor = dpi / 72
    
    # Create a status message for page processing
    page_status = st.empty()

    # Process each page in the PDF
    for page_num in range(page_count):
        page_status.text(f"Processing page {page_num+1} of {page_count}...")
        page = pdf_document.load_page(page_num)
        # Create a matrix for increased resolution criteria specific to the required parameters
        mat = fitz.Matrix(zoom_factor, zoom_factor)
        # Render page to an image
        pix = page.get_pixmap(matrix=mat, alpha=False)
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        output_filename = f"{base_filename}_page_{page_num+1}.{image_format}"
        output_path = os.path.join(output_folder, output_filename)
        img.save(output_path, quality=95 if image_format.lower() == "jpg" else None)
        output_files.append(output_path)
    
    # Clear the status message
    page_status.empty()
    pdf_document.close()
    return output_files

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_image(image_path, output_folder, dpi=800, image_format="jpg"):
    """Process an image file by copying it to the output folder"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Get the base filename without extension
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = f"{base_filename}_processed.{image_format}"
    output_path = os.path.join(output_folder, output_filename)
    img = Image.open(image_path)
    img.save(output_path, quality=95 if image_format.lower() == "jpg" else None)
    return [output_path]

def resume_ocr(image_paths=None, api_key=None):
    if not api_key:
        return {"error": "OpenAI API key is required"}
    
    client = OpenAI(api_key=api_key)
    
    system_prompt = """  
    You are a resume parsing assistant. Extract the following information from the resume as accurately and completely as possible:  
    - Full Name  
    - Phone Number  
    - Email Address    
    - LinkedIn Profile (if link is available)    
    - Github Profile (if link is available)   
    - Location (city, state)  
    - Skills (as a list)  
    - Education (degree, institution name, dates, CGPA/Percentage)   
    - Work Experience (company name, job title, dates, and full description for each position)  
    - Personal Projects (project name, full description)  
    - Certifications (if any)    
    - Recent Job Title  
    - Profile Summary (A concise summary generated using previous work experience and key skills, limited to 100 words)  

    ### **Profile Summary Guidelines:**  
    - **Analyze previous work experience** (including job titles and descriptions).     
    - **Identify key skills** from the "Skills" section and **previous work experience descriptions**.      
    - **Summarize professional expertise**, emphasizing **relevant achievements and technical strengths**.       
    - Keep it **professional, clear, and within 100 words**.        
        
    Format the output as a valid JSON object with these fields.    
    For fields where information is not available, use null.     
    For list fields like skills, return an empty list if not available.
    
    If the resume is spread across multiple pages, consolidate information from all pages, 
    ensuring no duplication and comprehensive coverage.
    """
    
    try:
        messages = [{"role": "system", "content": system_prompt}]
        
        # If image paths are provided, use vision capabilities of GPT-4o
        if image_paths and len(image_paths) > 0:
            # Prepare content for multiple images
            content = [
                {"type": "text", "text": "Please extract the required information from this multi-page resume. Make sure to consolidate information across all pages without duplication:"}
            ]
            
            # Add each image to the content
            for image_path in image_paths:
                base64_image = encode_image_to_base64(image_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })

            messages.append({
                "role": "user",
                "content": content
            })
        else:
            return {"error": "Image paths must be provided"}
        
        # Show status for API call
        api_status = st.empty()
        api_status.text("Extracting information from resume...")
            
        # Call GPT-4o API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        # Clear status
        api_status.empty()
            
        # Extract and parse the JSON response
        parsed_response = json.loads(response.choices[0].message.content)
        return add_experience_status(parsed_response)
            
    except Exception as e:
        return {"error": str(e)}

def get_file_type(file_path):
    """Determine if the file is a PDF or an image"""
    mime_type, _ = mimetypes.guess_type(file_path)
    
    if mime_type is None:
        # Check file extension manually if mime_type detection fails
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.pdf']:
            return 'application/pdf'
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            return 'image'
        else:
            return None
    
    if mime_type.startswith('application/pdf'):  
        return 'application/pdf'  
    elif mime_type.startswith('image/'):  
        return 'image'
    else:     
        return None

def process_resume_multi(file_path, api_key, output_folder=None, dpi=800, image_format="jpg"):
    """Process a resume file (PDF or image) and extract information"""
    # Create temporary directory if output_folder not provided
    if output_folder is None:
        output_folder = tempfile.mkdtemp()
        
    # Determine file type
    file_type = get_file_type(file_path)
    
    if file_type is None:
        return {"error": "Unsupported file type. Please provide a PDF or image file."}
    
    # Use a spinner instead of progress bar
    with st.spinner("Processing resume..."):
        # Create temporary images based on file type
        if file_type == 'application/pdf':
            image_paths = convert_pdf_to_images(file_path, output_folder, dpi, image_format)
        elif file_type == 'image':
            image_paths = process_image(file_path, output_folder, dpi, image_format)
        else:
            return {"error": "Unsupported file type. Please provide a PDF or image file."}

        if not image_paths:
            return {"error": "Failed to process the file"}
            
        # Parse resume
        result = resume_ocr(image_paths, api_key)
        
        # Clean up by removing the temporary image files
        for image_path in image_paths:
            try:
                os.remove(image_path)
            except Exception:
                pass
                
        # Try to remove the temp directory
        try:
            if os.path.exists(output_folder) and not os.listdir(output_folder):
                os.rmdir(output_folder)
        except Exception:
            pass
        
    return result

# UI Components
st.title("Multi-Pages CandidNexus")
st.write("Upload a resume PDF file or JPEG image.")

# API Key input
if not api_key:
    api_key = st.text_input("Enter OpenAI API Key", type="password")

# Create a session state to store the processed results
if 'results' not in st.session_state:
    st.session_state['results'] = None

# Create a session state for the temp file path
if 'temp_file_path' not in st.session_state:
    st.session_state['temp_file_path'] = None

# File uploader
uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "jpg", "jpeg", "png"], label_visibility="hidden", 
                               help="Limit 200MB per file • PDF, JPG, JPEG",
                               on_change=reset_app_state)  # Reset when a new file is uploaded

# When a file is uploaded, save it temporarily but don't process yet
if uploaded_file is not None and st.session_state['temp_file_path'] is None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        st.session_state['temp_file_path'] = tmp_file.name

# Add analyze button only if a file is uploaded
if uploaded_file is not None:
    analyze_button = st.button("Analyze Resume")
    
    # Process the resume only when the analyze button is clicked
    if analyze_button and api_key and st.session_state['temp_file_path']:
        start = time.time()
        result = process_resume_multi(st.session_state['temp_file_path'], api_key)
        end = time.time()
        execution_time = end - start
        st.session_state['results'] = result
        st.write(f"Time taken: {execution_time:.4f} seconds")

# Display results if available
if st.session_state['results'] is not None:
    if "error" in st.session_state['results']:
        st.error(f"Error: {st.session_state['results']['error']}")
    else:
        st.success("Extracted Information")
        st.json(st.session_state['results'])
else:
    if uploaded_file is None:
        st.info("Please upload a resume PDF file or JPEG image file to begin.")
