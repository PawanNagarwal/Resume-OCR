import streamlit as st
import fitz  # PyMuPDF
from PIL import Image   
import os
import json
import base64
from openai import OpenAI
import tempfile

# Set page title and configuration
st.set_page_config(
    page_title="Resume OCR",
    page_icon="📄",
    layout="wide"
)

# Set your OpenAI API key here or use environment variable
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Initialize OpenAI client
def initialize_openai():
    client = OpenAI(api_key=OPENAI_API_KEY)
    return client

# Convert PDF to images
def convert_pdf_to_images(pdf_path, output_folder, image_format="png"):
    # Fixed DPI at 800
    dpi = 800
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    # Get the base filename without extension
    base_filename = os.path.splitext(os.path.basename(pdf_path))[0]

    output_files = []

    # Calculate the zoom factor based on DPI (72 is the base DPI for PDFs)
    zoom_factor = dpi / 72

    # Process each page in the PDF
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)

        # Create a matrix for increased resolution criteria specific to the required paramters
        mat = fitz.Matrix(zoom_factor, zoom_factor)

        # Render page to an image
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Create output filename
        output_filename = f"{base_filename}_page_{page_num+1}.{image_format}"
        output_path = os.path.join(output_folder, output_filename)

        # Save the image
        img.save(output_path, quality=95 if image_format.lower() == "jpg" else None)
        output_files.append(output_path)
        with st.spinner(f"Converting page {page_num+1}/{pdf_document.page_count}..."):
            st.info(f"Converted page {page_num+1}/{pdf_document.page_count}")

    pdf_document.close()
    return output_files

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def resume_ocr(image_path=None, client=None):
    if not client:
        return {"error": "OpenAI client not initialized. Please check your API key."}
        
    system_prompt = """
    You are a resume parsing assistant. Extract the following information from the resume as accurately and completely as possible:
    - Full Name
    - Phone Number
    - Email Address
    - LinkedIn Profile (if link is available)
    - GIthub Profile (if link is available)
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
    """
    try:
        messages = [{"role": "system", "content": system_prompt}]
        
        # If image path is provided, use vision capabilities of GPT-4o
        if image_path:
            base64_image = encode_image_to_base64(image_path)
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please extract the required information from this resume:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            })

        else:
            return {"error": "Image path must be provided"}

        # Call GPT-4o API
        with st.spinner("Extracting information from resume..."):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                response_format={"type": "json_object"}
            )

        # Extract and parse the JSON response
        parsed_response = json.loads(response.choices[0].message.content)
        return parsed_response
        
    except Exception as e:
        return {"error": str(e)}

def remove_image_files(image_paths):
    """Remove all image files from the provided paths"""
    for image_path in image_paths:
        try:
            os.remove(image_path)
        except Exception as e:
            st.error(f"Error removing image {image_path}: {e}")

def process_resume_pdf(pdf_path, temp_dir, client, image_format="jpg"):
    with st.spinner("Converting PDF to image..."):
        image_paths = convert_pdf_to_images(pdf_path, temp_dir, image_format)

    if not image_paths:
        return {"error": "Failed to convert pdf to image"}
    
    # st.info("E Resume...")
    result = resume_ocr(image_path=image_paths[0], client=client)
    
    # Clean up by removing the temporary image files
    remove_image_files(image_paths)
    
    return result

def process_image_file(image_path, client):
    """Process a JPEG/JPG image file directly"""
    with st.spinner("Processing image..."):
        result = resume_ocr(image_path=image_path, client=client)
    return result

# Save uploaded file to temp location
def save_uploaded_file(uploaded_file, temp_dir):
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path

# Main app function
def main():
    st.title("Resume OCR")
    st.write("Upload a resume PDF file or JPEG image.")

    # File uploader with expanded file types
    uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "jpg", "jpeg"])

    # Initialize OpenAI client
    client = initialize_openai()

    if uploaded_file and client:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the uploaded file to the temporary directory
            temp_file_path = save_uploaded_file(uploaded_file, temp_dir)
            
            # Check file extension to determine processing method
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            
            if st.button("Analyze Resume"):
                with st.spinner("Processing..."):
                    if file_extension in ['.jpg', '.jpeg']:
                        # Direct processing for image files
                        result = process_image_file(temp_file_path, client)
                    elif file_extension == '.pdf':
                        # Process PDF files
                        result = process_resume_pdf(temp_file_path, temp_dir, client, image_format="jpg")
                    else:
                        result = {"error": "Unsupported file format"}
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    # Display only the JSON tab
                    st.success("Extracted Information")
                    
                    # Display Raw JSON Data
                    st.subheader("Raw JSON Data")
                    st.json(result)
                    
                    # Add download button for JSON
                    json_str = json.dumps(result, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name="resume_parsed.json",
                        mime="application/json",
                    )
    elif not client:
        st.error("OpenAI client initialization failed. Please ensure the API key is set correctly via environment variable.")
    elif not uploaded_file:
        st.info("Please upload a resume PDF file or JPEG image file to begin.")

if __name__ == "__main__":
    main()