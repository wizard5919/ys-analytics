from streamlit.web import bootstrap

def main():
    bootstrap.run(
        "app.py",
        command_line=None,
        args=[],
        flag_options={}
    )

if __name__ == "__main__":
    main()
