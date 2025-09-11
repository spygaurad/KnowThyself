# webapp.py (likely located in your_project_root/kyllm_agent/)
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import mimetypes

app = FastAPI()

# Configure CORS
# IMPORTANT: In a production environment, restrict allow_origins to your frontend's domain.
# For local development, "*" is common but less secure.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Define the base directory where your content folders are located.
# This path is relative to the location of webapp.py
# If webapp.py is in 'KNOWYOURLLM/kyllm_agent/', then:
# '..' goes to 'KNOWYOURLLM/'
# 'src/files/documents' then leads to 'KNOWYOURLLM/src/files/documents/'
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'files', 'documents')

# A secure mapping of front-end tab names to their actual server-side directory paths.
# This prevents malicious users from requesting arbitrary file paths.
ALLOWED_FOLDERS = {
    "documentation": os.path.join(BASE_DIR, "documentation"),
    "workflows": os.path.join(BASE_DIR, "workflows"),
    "tools": os.path.join(BASE_DIR, "tools"),


    # Add any other folders you want to expose if they exist under src/files/documents/
}

# --- ONLY Ensure folders exist; DO NOT create dummy files ---
# This loop just ensures that the directories mapped in ALLOWED_FOLDERS exist on the server.
# It will NOT create any files. Your actual files should be present in these folders.
for folder_name, folder_path in ALLOWED_FOLDERS.items():
    os.makedirs(folder_path, exist_ok=True) # Ensure folders exist. This does not create files.


# --- API Endpoints ---

@app.get("/api/files/list")
async def list_files(folder: str = Query(..., description="The name of the folder (e.g., documentation, workflows, tools)")):
    """
    Lists files within a specified allowed folder.
    """
    if folder not in ALLOWED_FOLDERS:
        raise HTTPException(status_code=400, detail="Invalid or unsupported folder name.")

    folder_path = ALLOWED_FOLDERS[folder]
    if not os.path.isdir(folder_path):
        raise HTTPException(status_code=404, detail="Folder not found on server.")

    try:
        # List only files, exclude directories
        # files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        # return {"files": files}
        pdf_files = [
            f for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(".pdf")
        ]
        return {"files": pdf_files}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/api/files/content")
async def get_file_content(
    folder: str = Query(..., description="The name of the folder"),
    filename: str = Query(..., description="The name of the file")
):
    """
    Serves the content of a specific file from an allowed folder.
    Handles different MIME types for PDF, text, CSV, MD.
    """
    if folder not in ALLOWED_FOLDERS:
        raise HTTPException(status_code=400, detail="Invalid or unsupported folder name.")

    folder_path = ALLOWED_FOLDERS[folder]
    file_path = os.path.join(folder_path, filename)

    # Security check: Crucial to prevent path traversal attacks (e.g., ../../secret.txt)
    # This ensures that 'file_path' is strictly within 'folder_path'.
    if not os.path.commonpath([os.path.realpath(file_path), os.path.realpath(folder_path)]) == os.path.realpath(folder_path):
        raise HTTPException(status_code=403, detail="Access denied: Attempted directory traversal.")

    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found.")

    # Determine content type based on file extension
    mimetype, _ = mimetypes.guess_type(file_path)
    if mimetype is None:
        mimetype = 'application/octet-stream' # Default for unknown types

    # For text-based files, read content and return as PlainTextResponse
    # For PDFs and other binary files, use FileResponse
    if mimetype.startswith('text/') or mimetype == 'application/json' or mimetype == 'application/xml':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return PlainTextResponse(content, media_type=mimetype)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading text file: {e}")
    else:
        # FastAPI's FileResponse automatically handles streaming and headers
        return FileResponse(path=file_path, media_type=mimetype, filename=filename)


@app.get("/api/files/results")
async def get_results_html(
    filename: str = Query(..., description="HTML filename inside the results folder (e.g., report.html)")
):
    """
    Serve an HTML file from the fixed 'results' folder.
    Only .html / .htm files are allowed.
    """
    results_dir = os.path.join(BASE_DIR, "results")
    if not results_dir or not os.path.isdir(results_dir):
        raise HTTPException(status_code=500, detail="Results folder is not configured or missing on server.")

    # Require .html/.htm extension
    lowered = filename.lower()
    if not (lowered.endswith(".html") or lowered.endswith(".htm")):
        raise HTTPException(status_code=415, detail="Only .html or .htm files are allowed.")

    # Build and sanitize path
    file_path = os.path.join(results_dir, filename)
    folder_real = os.path.realpath(results_dir)
    file_real = os.path.realpath(file_path)

    # Prevent directory traversal
    try:
        if os.path.commonpath([file_real, folder_real]) != folder_real:
            raise HTTPException(status_code=403, detail="Access denied: Attempted directory traversal.")
    except ValueError:
        # Raised if paths are on different drives (Windows edge case)
        raise HTTPException(status_code=403, detail="Access denied: Invalid path.")

    if not os.path.exists(file_real) or not os.path.isfile(file_real):
        raise HTTPException(status_code=404, detail="File not found in results folder.")

    # Serve inline as HTML
    return FileResponse(
        path=file_real,
        media_type="text/html; charset=utf-8",
        filename=os.path.basename(file_real),
    )


@app.get("/get_model")
def read_root():
    return {"Hello": "World"}
