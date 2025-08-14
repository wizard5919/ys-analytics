from streamlit.web import bootstrap
import os

def main():
    port = int(os.environ.get("PORT", 8501))
    bootstrap.run(
        "Home.py",  # Changed to Home.py
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
