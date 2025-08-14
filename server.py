from streamlit.web import bootstrap
import os

def main():
    # Get port from Vercel environment variable
    port = int(os.environ.get("PORT", 8501))
    
    # Configure Streamlit
    bootstrap.run(
        "app.py",  # Changed to app.py
        command_line=None,
        args=[
            "--server.port", str(port),
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false"
        ],
        flag_options={}
    )

if __name__ == "__main__":
    main()
