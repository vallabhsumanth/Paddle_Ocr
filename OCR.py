import sys
import time
import paddle
import os
import re
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from tqdm import tqdm
from PIL import Image
import io

# Function to normalize text (remove extra spaces, lowercase)
def normalize_text(text):
    return re.sub(r'\s+', ' ', text.strip()).lower()

# Check if PaddlePaddle is using GPU
print("GPU available:", paddle.is_compiled_with_cuda())

# Initialize PaddleOCR for English with optimized settings
ocr = PaddleOCR(
    use_angle_cls=True,      # Enable for better text orientation detection
    lang='en',              # English language
    use_gpu=True,           # Explicitly use GPU
    rec_batch_num=16,       # Increased for faster processing
    use_dilation=True,      # Improve detection of small/dense text
    det_db_score_mode='fast',  # Faster processing
    det_db_unclip_ratio=2.2,  # Adjusted for better text region detection
    det_db_box_thresh=0.6,  # Higher threshold for accurate detection
    det_db_thresh=0.3,      # Balanced for accuracy
    use_space_char=True     # Improve recognition of spaces
)

# PDF path
pdf_path = '/workspace/Paddle_OCR/testing_tribe/CALAX_Concession Agreement 10July2015.pdf'

# Verify PDF file exists
if not os.path.exists(pdf_path):
    print(f"Error: PDF file not found at {pdf_path}")
    sys.exit(1)

# Convert PDF to images
start_time = time.time()
try:
    print("Converting PDF to images...")
    images = convert_from_path(pdf_path, dpi=200, thread_count=12, fmt='png')  # Lower DPI, more threads
    print(f"Converted {len(images)} pages from PDF.")
except Exception as e:
    print(f"Error converting PDF: {e}")
    print("Ensure you have the required libraries installed and the PDF is not corrupted.")
    sys.exit(1)

# Output file
output_file = '/workspace/Paddle_OCR/ocr_results.txt'

with open(output_file, 'w', encoding='utf-8') as f:
    f.write(f"OCR Results for {pdf_path}\n")
    f.write(f"Total Pages: {len(images)}\n\n")

# Process each page and collect OCR results
all_texts = []
combined_text = []  # Store cleaned text for search
total_word_count = 0  # Track total words
low_confidence_words = 0  # Track words with low confidence
confidence_threshold = 0.8  # Confidence below this indicates potential error

for page_num in tqdm(range(len(images)), desc="Scanning pages"):
    img = images[page_num]
    # Log image dimensions
    img_width, img_height = img.size
    print(f"Page {page_num + 1}: Image size {img_width}x{img_height} pixels.")

    # Convert image to bytes in-memory
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Run OCR on in-memory image
    try:
        result = ocr.ocr(img_byte_arr, cls=True)  
        text_count = sum(len(line) for line in result if result) if result else 0
        print(f"Page {page_num + 1}: Detected {text_count} text items.")
    except Exception as e:
        print(f"Error on page {page_num + 1}: {e}")
        result = []

    # Collect text and compute metrics
    texts = []
    if result and isinstance(result, list):
        for line in result:
            for word_info in line:
                _, (text, confidence) = word_info
                # Clean and normalize text for search
                text_clean = normalize_text(text.strip())
                if text_clean:
                    texts.append(text_clean)
                    combined_text.append(text_clean)
                    # Count words
                    words = text_clean.split()
                    total_word_count += len(words)
                    # Check confidence for error estimation
                    if confidence < confidence_threshold:
                        low_confidence_words += len(words)
    else:
        print(f"No OCR results for page {page_num + 1} (possibly blank or image-only).")

    all_texts.append(texts)

# Calculate error rate
error_rate = (low_confidence_words / total_word_count * 100) if total_word_count > 0 else 0

# Combine texts into paragraphs
combined_paragraph_clean = ' '.join(combined_text)
with open(output_file, 'a', encoding='utf-8') as f:
    f.write("Combined Text (Cleaned, Normalized):\n")
    f.write(f"{combined_paragraph_clean}\n\n")
    f.write(f"Accuracy Metrics:\n")
    f.write(f"Total Word Count: {total_word_count}\n")
    f.write(f"Low Confidence Words (below {confidence_threshold}): {low_confidence_words}\n")
    f.write(f"Estimated Error Rate: {error_rate:.2f}%\n")

# Verify text collection
print(f"Total cleaned text lines collected: {len(combined_text)}")
print(f"Total Word Count: {total_word_count}")
print(f"Low Confidence Words: {low_confidence_words}")
print(f"Estimated Error Rate: {error_rate:.2f}%")

end_time = time.time()
elapsed_time = end_time - start_time

# Print and save total duration
duration_output = f"OCR completed in {elapsed_time:.2f} seconds for {len(images)} pages (GPU: {paddle.is_compiled_with_cuda()})\n"
print(duration_output)
with open(output_file, 'a', encoding='utf-8') as f:
    f.write(duration_output)

# Search functionality
search_query = input("Enter search term (spaces are okay): ")
normalized_query = normalize_text(search_query)
if not normalized_query:
    print("No valid search term provided.")
    sys.exit(1)

# Search output file
search_output_file = '/workspace/Paddle_OCR/search_results.txt'
with open(search_output_file, 'w', encoding='utf-8') as f:
    f.write(f"Search Results for '{search_query}' in {pdf_path}\n\n")

# Search for the query in OCR results
found = False
with open(search_output_file, 'a', encoding='utf-8') as f:
    for page_num, texts in enumerate(all_texts):
        page_text = ' '.join(texts)  # Combine all text on the page
        normalized_page_text = normalize_text(page_text)
        if normalized_query in normalized_page_text:
            found = True
            result_output = f"Page {page_num + 1}:\n"
            for text in texts:
                if normalized_query in normalize_text(text):
                    result_output += f"- {text}\n"
            result_output += "\n"
            print(result_output.strip())
            f.write(result_output)

    if not found:
        no_result_output = f"No matches found for '{search_query}'.\n"
        print(no_result_output.strip())
        f.write(no_result_output)

print(f"Search results saved to {search_output_file}")
sys.exit()